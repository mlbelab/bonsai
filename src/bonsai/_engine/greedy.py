"""Greedy merge engine over one parent with C children (maths ref §8; plan M2 Task 6a; A8, A11, A18, A20, A21b).

The host owns the tree under construction (``NodeTable``: parent, float64 t, node messages in the
compute dtype, mirrored row-wise on the device) and the parent's float64 accumulators
(``ParentAcc``, A8). Candidate pairs live in a host store with their score, times, bound slopes and
scoring epoch; the selection is the pure ``select_candidate``. Every round decides like
``bonsai._ref.greedy.decide``: merge the best pair iff its dLogL > 1e-6 (TH:2428; A21b, no zero-root
branch), ties to the lower (i, j). Device pair scores that did not converge or fell back
(``~converged | fallback``, A11) are re-scored on the host with the fp64 reference before the store
sees them. Units: 2x log-likelihoods (A1). Bounds off (``lazy=False`` or C ≤ 10): exhaustive rounds, every
candidate stale; on: stale candidates are re-scored in the order of their A20 bound. kNN candidates while C > 50
with the A19 rebuild schedule; ``merges_per_round`` at the root (A20). A merge whose device dLogL is within its
rounding resolution of the 1e-6 threshold is decided on the fp64 host re-score (A24).
"""

import time
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from bonsai._ref import pair as ref_pair
from bonsai._ref import star as ref_star
from bonsai._ref.greedy import Event

from .array_tree import _bucket
from .host_acc import ParentAcc
from .knn import knn, knn_coords, knn_query
from .pair import pair_score
from .star import STAR_T_MAX, star_em

# Constants (plan "Constants", Revision 5; ``_ref.greedy`` keeps its own copies, a test asserts equality)
ACCEPT_DLOGL = 1e-6  # merge iff the best dLogL > this (TH:2428, A4, A21b)
ADD_AS_CHILD_T = 1e-6  # t_1a < this with child 1 internal: t_1a := 0 (TH:2467-2476, A18)
REDO_TIME_FRAC, REDO_TIME_MIN = 0.05, 10  # redoTimeAt = max(⌊0.05·C⌋, 10) (TH:1154, 2354)
REDO_NN_FRAC, REDO_NN_MIN = 0.1, 10  # kNN rebuild schedule (TH:1951, A19); Task 6b
NN_MIN_CHILDREN = 50  # kNN only when C > 50 (MA:176, TH:1888); Task 6b
LAZY_MIN_CHILDREN = 10  # bounds only when C > 10 (TH:1040); Task 6b
STAR_TOL, STAR_MAX_SWEEPS = 1e-4, 1000  # star EM, with STAR_T_MAX = 1e8 (TH:4992-5004)
ZERO_EDGE_PASSES = 8  # cap on collapse + redo passes (A18); Task 7
K_NN = 10  # use_knn (config.py:75); Task 6b

__all__ = [
    "STAR_T_MAX",
    "GreedyConfig",
    "GreedyStats",
    "NodeTable",
    "greedy_star",
    "pair_round_resolution",
    "select_candidate",
]


@dataclass(frozen=True)
class GreedyConfig:
    mode: str = "sequential"  # pair mode (A2); A17 R1 decides the default
    k: int = K_NN
    knn: bool = True  # kNN candidates while C > NN_MIN_CHILDREN (A19)
    lazy: bool = True  # bounds while C > LAZY_MIN_CHILDREN (A20)
    rescore_top: int = 256  # M: rows per rescore batch (A17 R2)
    merges_per_round: int = 1  # m, root pass only (A17 R3, A20)
    new_batch: int = 64  # rows per batch of new pairs, scored right after a merge or rebuild when bounds are on
    star_tol: float = STAR_TOL
    false_stop_check: bool = False  # diagnostic (A24): at each stop, host-re-score every near-threshold candidate


class GreedyStats(NamedTuple):
    events: list
    n_pair_scores: int  # device pair rows scored (padding excluded)
    n_host_rescored: int  # rows re-scored by the fp64 reference: flagged (A11) plus near-threshold (A24)
    n_rescore_batches: int
    n_rescored: int  # rows re-scored in the rescore batches (the rest of n_pair_scores: new pairs)
    n_rebuilds: int  # kNN index builds, the initial one included
    n_reopts: int
    n_flagged_used_unrescored: int  # guard: flagged device rows stored without a host re-score; must stay 0
    n_threshold_rescored: int  # chosen pairs re-scored on the host because d − 1e-6 ≤ their resolution (A24)
    # False-stop diagnostic (cfg.false_stop_check; never changes decisions):
    n_stop_events: int = 0  # stops examined
    n_false_stop_events: int = 0  # of those, stops with at least one band row whose fp64 d > 1e-6
    n_band_rows: int = 0  # candidates re-scored on the host at stops (inside the resolution band)
    stop_check_s: float = 0.0  # wall time spent in the diagnostic


@eqx.filter_jit(donate="all")
def _set_rows(buf, idx, rows):
    """buf[idx] = rows on the device; the buffer is donated, so a write does not copy it."""
    return buf.at[idx].set(rows)


class NodeTable:
    """Host tree under construction plus a device mirror of the node messages.

    Ids: leaves 0..N−1, root N, new nodes N+1, N+2, … in merge order, never reused (``_ref.greedy``'s
    convention; ``parent`` −1 at the root, −2 for removed nodes). ``x``, ``v`` (cap, G) in the compute
    dtype: data and variance at cells, (x_n, 1/W_n) at internal nodes. ``t`` float64 (the edge above).
    """

    def __init__(self, x, var, cap: int | None = None, dtype=jnp.float32):
        x, var = np.asarray(x, np.float64), np.asarray(var, np.float64)
        N, G = x.shape
        cap = _bucket(2 * N + 1) if cap is None else cap
        self.dtype = np.dtype(dtype)
        self.parent = np.full(cap, -2, np.int64)
        self.parent[:N], self.parent[N] = N, -1
        self.t = np.zeros(cap)
        self.is_cell = np.zeros(cap, bool)
        self.is_cell[:N] = True
        self.x = np.zeros((cap, G), self.dtype)
        self.v = np.ones((cap, G), self.dtype)
        self.x[:N] = x - x.mean(axis=0)  # centred in float64 before the cast (pair_score's contract)
        self.v[:N] = var
        self.n = N + 1
        self.dev_x, self.dev_v = jnp.asarray(self.x), jnp.asarray(self.v)

    def children(self, p: int) -> list[int]:
        """Children of ``p``, increasing id."""
        return [int(c) for c in np.flatnonzero(self.parent[: self.n] == p)]

    def add(self, x64, v64, parent: int, t: float) -> int:
        """New internal node with message (x64, v64), rounded once to the compute dtype; returns its id."""
        a = self.n
        if a == len(self.parent):
            self._grow()
        self.x[a], self.v[a] = x64, v64
        self.parent[a], self.t[a], self.n = parent, t, a + 1
        self.sync(a)
        return a

    def _grow(self) -> None:
        """Reallocate every array at the next bucket (ids are never reused, so redo starry can exceed 2N+1)."""
        old, cap = len(self.parent), _bucket(len(self.parent) + 1)
        for name, fill in (("parent", -2), ("t", 0), ("is_cell", False), ("x", 0), ("v", 1)):
            a = getattr(self, name)
            b = np.full((cap, *a.shape[1:]), fill, a.dtype)
            b[:old] = a
            setattr(self, name, b)
        self.dev_x, self.dev_v = jnp.asarray(self.x), jnp.asarray(self.v)

    def sync(self, a: int) -> None:
        """Copy host row ``a`` (x, v) to the device mirror."""
        self.dev_x = _set_rows(self.dev_x, jnp.asarray([a]), jnp.asarray(self.x[a : a + 1]))
        self.dev_v = _set_rows(self.dev_v, jnp.asarray([a]), jnp.asarray(self.v[a : a + 1]))

    def x64(self, ids) -> np.ndarray:
        return self.x[ids].astype(np.float64)

    def v64(self, ids) -> np.ndarray:
        return self.v[ids].astype(np.float64)

    def weights64(self, ids) -> np.ndarray:
        """1/(v + t) in float64 on the host from the stored messages: the same value every time (A8 same-weight rule)."""
        return 1.0 / (self.v[ids].astype(np.float64) + self.t[ids][..., None])


def _round_terms(w, xbar, xs, W) -> tuple[float, float]:
    """(Σ_g ρ_g, Σ_g T_g) of one 3-star term (plan "Near-tie rule", r_round), summed over genes."""
    r = xbar - xs
    X = np.abs(xs).max(axis=0)
    T = (np.abs(np.log(w)) + w * r * r).sum(axis=0) + np.abs(np.log(W))
    rho = ((np.abs(np.log(w)) + 2) + w * (6 * r * r + 12 * np.abs(r) * X)).sum(axis=0) + (np.abs(np.log(W)) + 4) + 6 * T
    return float(rho.sum()), float(T.sum())


def pair_round_resolution(x1, v1, t1, x2, v2, t2, xR, vR, xr, Wr, t3, eps: float) -> float:
    """r_round of the plan's near-tie rule: the rounding resolution of a pair's dLogL computed at precision ``eps``.

    eps·[Σρ(new) + Σρ(old) + Σ(T(new) + T(old)) + 9·ΣT(new)], all inputs float64 (G,) except the scalars t1, t2
    and ``t3`` = (t_1a, t_2a, t_aR) the pair's times; "new" = the 3-star at ``t3`` (x̄ = its mean), "old" = the
    weights (1/(v1 + t1), 1/(v2 + t2), 1/vR) with x̄ = x_r. The near-tie rule adds r_in = |f(û) − f(u)|, which needs
    the fp64 score itself; the engine's A24 test uses this estimate and the host re-score supplies f(û).

    Returned: 1.25·r_round. The factor 1.25 covers the input casts to the compute dtype (A24, first order): each cast
    is one relative rounding of eps/2, so per gene and term |δ log w|, |δ log W| ≤ eps/2 and |δr| ≤ (eps/2)·X, while
    r_round already allows ≥ 2·eps for each of those (2 roundings in w, 4 in W, 6 in x̄); the casts thus add ≤ ¼ of
    r_round term by term. The times t3 drop out to first order (envelope theorem: dLogL is stationary in them).
    """
    xs, vs = np.stack([x1, x2, xR]), np.stack([v1, v2, vR])
    w = 1.0 / (vs + np.asarray(t3)[:, None])
    W = w.sum(axis=0)
    rho_n, T_n = _round_terms(w, (w * xs).sum(axis=0) / W, xs, W)
    w_old = np.stack([1.0 / (v1 + t1), 1.0 / (v2 + t2), 1.0 / vR])
    rho_o, T_o = _round_terms(w_old, xr, xs, Wr)
    return 1.25 * eps * (rho_n + rho_o + T_n + T_o + 9 * T_n)


def select_candidate(d, ub, fresh, ij, M: int) -> tuple[str, np.ndarray]:
    """One selection step over the live candidates (A20, A21b; pure).

    ``d``, ``ub`` (n,): stored score and upper bound (+inf for a candidate marked stale);
    ``fresh`` (n,): scored at the current epoch; ``ij`` (n, 2) canonical pairs. The best eligible
    (fresh, finite d) candidate by (d desc, ij asc) is accepted once d ≥ every stale bound
    (equality accepts, A20). Otherwise ("rescore", the M stale candidates with the largest bound by
    (ub desc, ij asc)); ("none", []) when nothing is eligible or stale. A NaN bound counts as +inf (rescored first).
    """
    ub = np.where(np.isnan(ub), np.inf, ub)
    stale = np.flatnonzero(~fresh)
    elig = np.flatnonzero(fresh & np.isfinite(d))
    if len(elig):  # O(n): the largest d, then the lowest (i, j) key among the rows tied at it
        top = elig[d[elig] == d[elig].max()]
        best = top[np.argmin(ij[top, 0] * (int(ij.max()) + 1) + ij[top, 1])]
        if not len(stale) or d[best] >= ub[stale].max():
            return "accept", np.array([best])
    if not len(stale):
        return "none", np.array([], np.int64)
    u = ub[stale]
    if len(stale) > M:  # O(n): partition on ub, then on the combined (i, j) key among the ties at the M-th bound
        thr = np.partition(u, len(u) - M)[len(u) - M]
        tie = stale[u == thr]
        need = M - int(np.sum(u > thr))
        if len(tie) > need:
            key = ij[tie, 0] * (int(ij.max()) + 1) + ij[tie, 1]
            tie = tie[np.argpartition(key, need - 1)[:need]]
        stale = np.r_[stale[u > thr], tie]
        u = ub[stale]
    top = stale[np.lexsort((ij[stale, 1], ij[stale, 0], -u))]
    return "rescore", top


class _Store:
    """Candidate pairs of one parent: score, times, bound slopes, scoring epoch (−1: stale, UB = +inf)."""

    def __init__(self):
        self.ij = np.zeros((0, 2), np.int64)
        self.d = np.zeros(0)
        self.t3 = np.zeros((0, 3))
        self.a = np.zeros(0)
        self.b = np.zeros(0)
        self.epoch = np.zeros(0, np.int64)
        self.Dx = np.zeros(0)
        self.DW = np.zeros(0)
        self.host = np.zeros(0, bool)

    def add(self, ij) -> None:
        n = len(ij)
        self.ij = np.vstack([self.ij, np.asarray(ij, np.int64).reshape(n, 2)])
        self.d, self.a, self.b = (np.r_[z, np.full(n, np.nan)] for z in (self.d, self.a, self.b))
        self.Dx, self.DW = np.r_[self.Dx, np.zeros(n)], np.r_[self.DW, np.zeros(n)]
        self.t3 = np.vstack([self.t3, np.full((n, 3), np.nan)])
        self.epoch, self.host = np.r_[self.epoch, np.full(n, -1)], np.r_[self.host, np.zeros(n, bool)]

    def drop(self, nodes) -> None:
        self.keep(~np.isin(self.ij, nodes).any(axis=1))

    def keep(self, keep) -> None:
        for name in ("ij", "d", "t3", "a", "b", "epoch", "Dx", "DW", "host"):
            setattr(self, name, getattr(self, name)[keep])

    def ub(self, Dx: float, DW: float) -> np.ndarray:
        return np.where(self.epoch < 0, np.inf, self.d + self.a * (Dx - self.Dx) + self.b * (DW - self.DW))


def _movement(old, new) -> tuple[float, float]:
    """(δx, δW) of a parent move (A20): ‖(x_new − x_old)√W_old‖₂, ‖(W_new − W_old)/W_old‖₂, fp64."""
    (x0, W0), (x1, W1) = old, new
    return float(np.linalg.norm((x1 - x0) * np.sqrt(W0))), float(np.linalg.norm((W1 - W0) / W0))


def greedy_star(
    table: NodeTable, p: int, outside, is_root: bool, cfg: GreedyConfig = GreedyConfig(), trace: list | None = None
) -> GreedyStats:
    r"""Greedily merge the children of ``p`` in place (maths ref §8; ``_ref.greedy.greedy_star`` semantics).

    ``outside`` = (x_{pp\p}, w̃) float64, the message of everything outside p as seen from p (None at
    the root), in the table's centred frame. Children's times must be set (star EM by the caller). Per round: at the
    root, star re-optimisation when counter > redoTimeAt = max(⌊0.05·C⌋, 10) (recomputed at each one, A21); else the
    best candidate is certified by ``select_candidate`` in rescore batches of ``cfg.rescore_top`` rows and merged iff
    its dLogL > 1e-6, with the A18 snap (t_1a < 1e-6 and child 1 internal → t_1a := 0, else the same for child 2).
    Bounds (A20): with ``cfg.lazy`` and C > 10 a stored score keeps the bound d + a·ΔDx + b·ΔDW; otherwise, and after
    a re-optimisation, every candidate is stale (UB = +inf), so the round is exhaustive. With bounds on, new pairs
    are scored right after the merge (``cfg.new_batch`` rows). Candidates (A19): all pairs, or with ``cfg.knn`` and
    C > 50 the symmetrised kNN pairs, rebuilt when merges since the last build > max(⌊0.1·C⌋, 10), after a
    re-optimisation and once before stopping; a new node's pairs come from an exact ``knn_query`` into a freed row.
    At the root up to m − 1 = ``cfg.merges_per_round`` − 1 further fresh, disjoint pairs with dLogL > 1e-6 are merged
    in the same round, by (d desc, ij asc), with their stored times. Stops at ≤ 3 children (root) / ≤ 2 (other).
    ``trace``: as ``_ref.greedy``, one entry per round before the decision, (scores {(i, j): (dLogL, t3, host, x_R,
    v_R)} of the fresh candidates, {child: t}, x_r, W_r, counter) with (x_r, W_r) the float64 AIR, ``host`` whether
    the score is the fp64 host re-score, (x_R, v_R) the rest as sent to the device; re-optimisation rounds have no
    scores. p's row ends with its message from its children.
    """
    mode, M = cfg.mode, cfg.rescore_top
    m = cfg.merges_per_round if is_root else 1
    dt = table.dtype
    kids = table.children(p)
    events: list[Event] = []
    n = {
        "scores": 0,
        "host": 0,
        "batches": 0,
        "rescored": 0,
        "rebuilds": 0,
        "reopts": 0,
        "unrescored": 0,
        "thr": 0,
        "stop": 0,
        "false": 0,
        "band": 0,
        "stop_s": 0.0,
    }
    eps = float(np.finfo(dt).eps)
    store = _Store()
    mv = {"Dx": 0.0, "DW": 0.0, "epoch": 0}
    sent: dict = {}  # trace only: (x_R, v_R) per pair as last sent to the device
    nn: dict | None = None  # kNN state: device buffer, host validity mask, child -> row; None: all pairs
    since, redo_nn_at = 0, 0  # merges since the last kNN build, and its threshold

    def build_acc() -> ParentAcc:
        acc = ParentAcc.from_children(table.weights64(kids), table.x64(kids))
        if outside is not None:
            acc.add(outside[1], outside[0])
        return acc

    def move(old) -> None:
        dx, dW = _movement(old, acc.message())
        mv["Dx"] += dx
        mv["DW"] += dW
        mv["epoch"] += 1

    def lazy() -> bool:
        return cfg.lazy and len(kids) > LAZY_MIN_CHILDREN

    def score(idx: np.ndarray, rows: int) -> None:
        """Score store rows ``idx`` (≤ rows) in one device batch padded to ``rows``; flagged rows re-scored on the host."""
        P = len(idx)
        i, j = store.ij[idx, 0], store.ij[idx, 1]
        pad = np.r_[np.arange(P), np.zeros(rows - P, np.int64)]  # repeat row 0; padded results ignored
        ip, jp = i[pad], j[pad]
        xr, Wr = acc.message()
        w1, w2 = table.weights64(ip), table.weights64(jp)
        xR, WR = acc.rest(w1, table.x64(ip), w2, table.x64(jp))
        xR, vR = xR.astype(dt), (1.0 / WR).astype(dt)
        s = pair_score(
            table.dev_x[ip],
            table.dev_v[ip],
            jnp.asarray(table.t[ip].astype(dt)),
            table.dev_x[jp],
            table.dev_v[jp],
            jnp.asarray(table.t[jp].astype(dt)),
            jnp.asarray(xR),
            jnp.asarray(vR),
            jnp.asarray(xr.astype(dt)),
            jnp.asarray(Wr.astype(dt)),
            mode,
        )
        d, t3, a, b = (np.array(z, np.float64)[:P] for z in (s.dlogl, s.t, s.a, s.b))
        flag = ~np.asarray(s.converged)[:P] | np.asarray(s.fallback)[:P]
        host = np.zeros(P, bool)
        for r in np.flatnonzero(flag):  # A11: no decision rests on an unconverged device solve
            d[r], t3[r], a[r], b[r] = host_score(i[r], j[r])
            host[r] = True
        n["unrescored"] += int(np.sum(flag & ~host))
        n["scores"] += P
        n["host"] += int(host.sum())
        store.d[idx], store.t3[idx], store.a[idx], store.b[idx], store.host[idx] = d, t3, a, b, host
        store.epoch[idx], store.Dx[idx], store.DW[idx] = mv["epoch"], mv["Dx"], mv["DW"]  # bound refreshed (A3, A20)
        if trace is not None:  # the rest exactly as sent to the device, for the trace
            sent.update({(int(i[r]), int(j[r])): (xR[r], vR[r]) for r in range(P)})

    def host_score(c1: int, c2: int):
        """(d, t3, a, b) of pair (c1, c2) by the fp64 reference at the host's inputs (A11, A24)."""
        xr, Wr = acc.message()
        ref = ref_pair.pair_score(
            table.x64(c1), table.v64(c1), table.t[c1], table.x64(c2), table.v64(c2), table.t[c2], xr, Wr, mode
        )
        a, b = ref_pair.ub_slopes(ref, xr, Wr) if np.isfinite(ref.dlogl) else (0.0, 0.0)
        return ref.dlogl, ref.t, a, b

    def resolution(r: int) -> float:
        """The fp32 rounding resolution of store row r's device score (``pair_round_resolution``)."""
        i, j = (int(c) for c in store.ij[r])
        xr, Wr = acc.message()
        w1, w2 = table.weights64(i), table.weights64(j)
        xR, WR = acc.rest(w1, table.x64(i), w2, table.x64(j))
        args = (table.x64(i), table.v64(i), table.t[i], table.x64(j), table.v64(j), table.t[j], xR, 1.0 / WR, xr, Wr)
        return pair_round_resolution(*args, store.t3[r], eps)

    def threshold_rescore(r: int) -> bool:
        """A24: re-score row r on the host (fp64) if its device d > 1e-6 is within its resolution of 1e-6; True if so."""
        if store.host[r] or not store.d[r] > ACCEPT_DLOGL:
            return False
        i, j = (int(c) for c in store.ij[r])
        if store.d[r] - ACCEPT_DLOGL > resolution(r):
            return False
        store.d[r], store.t3[r], store.a[r], store.b[r] = host_score(i, j)
        store.host[r] = True  # fresh at the current epoch, as before
        n["thr"] += 1
        n["host"] += 1
        return True

    def count_false_stops() -> None:
        """Count false stops (diagnostic; reads the store, never writes it).

        Host-re-scores EVERY candidate that could be a false stop: fresh device rows with 1e-6 − d ≤ r(p), and stale
        rows whose bound UB lies in (1e-6 − r(p), 1e-6]. A stop is flagged if any of them has fp64 d > 1e-6.
        """
        t0 = time.perf_counter()
        fresh = store.epoch == mv["epoch"]
        val = np.where(fresh, store.d, store.ub(mv["Dx"], mv["DW"]))  # device score, or the bound of a stale row
        ok = (fresh & ~store.host) | (~fresh & (store.epoch >= 0))
        cand = np.flatnonzero(ok & np.isfinite(val) & (val <= ACCEPT_DLOGL))
        rows = [r for r in cand if ACCEPT_DLOGL - val[r] <= resolution(r)]
        hits = [host_score(*(int(c) for c in store.ij[r]))[0] > ACCEPT_DLOGL for r in rows]
        n["stop"] += 1
        n["band"] += len(rows)
        n["false"] += int(any(hits))
        n["stop_s"] += time.perf_counter() - t0

    def score_new() -> None:
        """With bounds on, score the never-scored candidates now (batches of ``new_batch``); else certify does."""
        if lazy():
            todo = np.flatnonzero(np.isnan(store.d))
            for k in range(0, len(todo), cfg.new_batch):
                score(todo[k : k + cfg.new_batch], cfg.new_batch)

    def certify() -> int | None:
        while True:
            kind, idx = select_candidate(store.d, store.ub(mv["Dx"], mv["DW"]), store.epoch == mv["epoch"], store.ij, M)
            if kind != "rescore":
                return int(idx[0]) if kind == "accept" else None
            score(idx, M)
            n["batches"] += 1
            n["rescored"] += len(idx)

    def set_candidates(ij: np.ndarray) -> None:
        """Candidate set := the canonical pairs ``ij``; retained pairs keep their stored scores, new ones unscored."""
        cap = len(table.parent)  # live: the table may have grown since the last call
        key, have = ij[:, 0] * cap + ij[:, 1], store.ij[:, 0] * cap + store.ij[:, 1]
        store.keep(np.isin(have, key))
        new = np.setdiff1d(key, have)
        store.add(np.c_[new // cap, new % cap])

    def all_pairs() -> np.ndarray:
        return np.array([(c1, c2) for k, c1 in enumerate(kids) for c2 in kids[k + 1 :]], np.int64).reshape(-1, 2)

    def coords(ids) -> np.ndarray:
        xp, Wp = acc.message()
        return knn_coords(table.x64(ids), 1.0 / table.v64(ids), table.t[ids], xp, Wp).astype(dt)

    def rebuild() -> None:
        """A19: kNN index over all current children (bucket(C) rows), symmetrised pairs as the candidate set."""
        nonlocal nn, since, redo_nn_at
        C = len(kids)
        buf = np.zeros((_bucket(C), table.x.shape[1]), dt)
        buf[:C] = coords(kids)
        valid = np.arange(len(buf)) < C
        ids = np.asarray(kids)
        dev = jnp.asarray(buf)
        nb = ids[np.asarray(knn(dev, cfg.k, valid))[:C]]
        ij = np.unique(np.sort(np.c_[np.repeat(ids, cfg.k), nb.ravel()], axis=1), axis=0)
        row_id = np.full(len(buf), -1, np.int64)
        row_id[:C] = ids
        nn = {"buf": dev, "valid": valid, "row": dict(zip(kids, range(C), strict=True)), "id": row_id}
        set_candidates(ij)
        n["rebuilds"] += 1
        since, redo_nn_at = 0, max(int(REDO_NN_FRAC * C), REDO_NN_MIN)

    def update_candidates() -> None:
        """After a round's merges: all pairs at C ≤ 50, a kNN rebuild when due (A19); then score the new pairs."""
        nonlocal nn
        if nn is not None and len(kids) <= NN_MIN_CHILDREN:
            nn = None
            set_candidates(all_pairs())
        elif nn is not None and since > redo_nn_at:
            rebuild()
        score_new()

    def reopt() -> None:
        nonlocal acc
        C = len(kids)
        Cb = _bucket(C)
        xs, vs = np.zeros((Cb, table.x.shape[1]), dt), np.ones((Cb, table.x.shape[1]), dt)
        xs[:C], vs[:C] = table.x[kids], table.v[kids]
        res = star_em(jnp.asarray(xs), jnp.asarray(vs), jnp.asarray(np.arange(Cb) < C), cfg.star_tol, STAR_MAX_SWEEPS)
        if bool(res.converged):
            t_new = np.asarray(res.t, np.float64)[:C]
        else:  # upstream's BFGS fallback in the fp64 reference (TH:5005-5016)
            t_new = ref_star.star_em(table.x64(kids), table.v64(kids), cfg.star_tol, STAR_MAX_SWEEPS)[0]
        old = acc.message()
        table.t[kids] = t_new
        acc = build_acc()
        move(old)
        store.epoch[:] = -1  # every child time changed: UB := +inf (A20)
        n["reopts"] += 1
        events.append(Event("reopt", -1, -1, -1, np.nan, (np.nan, np.nan, np.nan)))
        if nn is not None:
            rebuild()  # A19: after every re-optimisation
        score_new()

    def merge(i: int, j: int, t3, dlogl: float) -> None:
        nonlocal kids, since
        t1a, t2a, t_ar = (float(u) for u in t3)
        if t1a < ADD_AS_CHILD_T and not table.is_cell[i]:  # child 1 first (TH:2467-2476, A18)
            t1a = 0.0
        elif t2a < ADD_AS_CHILD_T and not table.is_cell[j]:
            t2a = 0.0
        x2, v2 = table.x64([i, j]), table.v64([i, j])
        xa, Wa = ref_star.node_message(x2, v2, np.array([t1a, t2a]))  # host fp64, rounded once by add
        old = acc.message()
        acc.remove(table.weights64(i), table.x64(i))  # weights at the stored times: the ones added
        acc.remove(table.weights64(j), table.x64(j))
        a = table.add(xa, 1.0 / Wa, p, t_ar)
        table.parent[[i, j]] = a
        table.t[i], table.t[j] = t1a, t2a
        acc.add(table.weights64(a), table.x64(a))
        move(old)
        events.append(Event("merge", i, j, a, dlogl, (t1a, t2a, t_ar)))
        kids = [c for c in kids if c not in (i, j)]
        store.drop([i, j])
        since += 1
        if nn is None:
            store.add([(c, a) for c in kids])  # canonical: a is the highest id
        else:  # A19: a's coordinate into i's freed row, queried while that row is still invalid, then marked valid
            r = nn["row"].pop(i)
            nn["valid"][[r, nn["row"].pop(j)]] = False
            q = coords([a])
            nn["buf"] = _set_rows(nn["buf"], jnp.asarray([r]), jnp.asarray(q))
            nb = np.asarray(knn_query(nn["buf"], nn["valid"], jnp.asarray(q[0]), cfg.k))
            store.add([(c, a) for c in np.sort(nn["id"][nb])])
            nn["valid"][r], nn["row"][a], nn["id"][r] = True, r, a
        kids.append(a)

    def picks(best: int) -> list[int]:
        """``best`` plus up to m − 1 further fresh, disjoint candidates with d > 1e-6 by (d desc, ij asc) (A20)."""
        out, used = [best], set(store.ij[best].tolist())
        limit = min(m, len(kids) - 3)  # m > 1 only at the root, which stops at 3 children
        if limit > 1:
            fresh = np.flatnonzero((store.epoch == mv["epoch"]) & (store.d > ACCEPT_DLOGL))
            for r in fresh[np.lexsort((store.ij[fresh, 1], store.ij[fresh, 0], -store.d[fresh]))]:
                if len(out) == limit:
                    break
                if r == best or used & set(store.ij[r].tolist()):
                    continue
                threshold_rescore(r)  # A24 for the extra pairs too
                if store.d[r] > ACCEPT_DLOGL:
                    out.append(int(r))
                    used.update(store.ij[r].tolist())
        return out

    # ponytail: remaining per-merge host costs, ~100 ms/merge at C = 1e5 (reviewer, Task 6b): _Store.drop/add
    # (np.isin + array copies over all candidates), the kids list filter, select_candidate's O(n) passes. M4 item.
    acc = build_acc()
    if cfg.knn and len(kids) > NN_MIN_CHILDREN:
        rebuild()
    else:
        store.add(all_pairs())
    score_new()
    counter, redo_at = 0, max(int(REDO_TIME_FRAC * len(kids)), REDO_TIME_MIN)
    while len(kids) > (3 if is_root else 2):
        if is_root and counter > redo_at:  # TH:1926-1927 (A21: on any round)
            if trace is not None:
                trace.append(({}, {c: float(table.t[c]) for c in kids}, *acc.message(), counter))
            reopt()
            counter, redo_at = 0, max(int(REDO_TIME_FRAC * len(kids)), REDO_TIME_MIN)
            continue
        if not lazy():
            store.epoch[:] = -1  # bounds off: every candidate stale, UB := +inf, an exhaustive round (A20)
        best = certify()
        while best is not None and threshold_rescore(best):
            best = certify()  # A24, then A20: a host score is accepted only if still fresh-best over every bound
        if trace is not None:
            fresh = np.flatnonzero(store.epoch == mv["epoch"])
            scores = {}
            for r in fresh:
                ij = (int(store.ij[r, 0]), int(store.ij[r, 1]))
                scores[ij] = (store.d[r], store.t3[r].copy(), bool(store.host[r]), *sent[ij])
            trace.append((scores, {c: float(table.t[c]) for c in kids}, *acc.message(), counter))
        if best is None or not store.d[best] > ACCEPT_DLOGL:
            if nn is not None and since > 0:  # A19: one rebuild before stopping
                rebuild()
                score_new()
                continue
            if cfg.false_stop_check:
                count_false_stops()
            break
        for i, j, t3, dlogl in [(*store.ij[r].tolist(), store.t3[r].copy(), float(store.d[r])) for r in picks(best)]:
            merge(i, j, t3, dlogl)
        counter += 1
        update_candidates()
    xp, Wp = ref_star.node_message(table.x64(kids), table.v64(kids), table.t[kids])
    table.x[p], table.v[p] = xp, 1.0 / Wp
    table.sync(p)
    return GreedyStats(
        events,
        n["scores"],
        n["host"],
        n["batches"],
        n["rescored"],
        n["rebuilds"],
        n["reopts"],
        n["unrescored"],
        n["thr"],
        n["stop"],
        n["false"],
        n["band"],
        n["stop_s"],
    )
