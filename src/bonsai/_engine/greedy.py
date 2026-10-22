"""Greedy merge engine over one parent with C children (maths ref §8; plan M2 Task 6a; A8, A11, A18, A20, A21b).

The host owns the tree under construction (``NodeTable``: parent, float64 t, node messages in the
compute dtype, mirrored row-wise on the device) and the parent's float64 accumulators
(``ParentAcc``, A8). Candidate pairs live in a host store with their score, times, bound slopes and
scoring epoch; the selection is the pure ``select_candidate``. Every round decides like
``bonsai._ref.greedy.decide``: merge the best pair iff its dLogL > 1e-6 (TH:2428; A21b, no zero-root
branch), ties to the lower (i, j). Device pair scores that did not converge or fell back
(``~converged | fallback``, A11) are re-scored on the host with the fp64 reference before the store
sees them. Units: 2x log-likelihoods (A1). Task 6a: exhaustive rounds (every candidate stale each
round); Task 6b adds lazy bounds, kNN candidates and merges_per_round.
"""

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

__all__ = ["STAR_T_MAX", "GreedyConfig", "GreedyStats", "NodeTable", "greedy_star", "select_candidate"]


@dataclass(frozen=True)
class GreedyConfig:
    mode: str = "sequential"  # pair mode (A2); A17 R1 decides the default
    k: int = K_NN
    knn: bool = True  # kNN candidates while C > NN_MIN_CHILDREN (Task 6b)
    lazy: bool = True  # bounds while C > LAZY_MIN_CHILDREN (Task 6b)
    rescore_top: int = 256  # M: rows per rescore batch (A17 R2)
    merges_per_round: int = 1  # m (A17 R3; Task 6b)
    new_batch: int = 64  # rows per batch of new-node pairs (Task 6b)
    star_tol: float = STAR_TOL


class GreedyStats(NamedTuple):
    events: list
    n_pair_scores: int  # device pair rows scored (padding excluded)
    n_host_rescored: int  # flagged rows re-scored by the fp64 reference
    n_rescore_batches: int
    n_rebuilds: int
    n_reopts: int
    n_flagged_used_unrescored: int  # guard: flagged device rows stored without a host re-score; must stay 0


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
            raise ValueError("NodeTable is full")
        self.x[a], self.v[a] = x64, v64
        self.parent[a], self.t[a], self.n = parent, t, a + 1
        self.sync(a)
        return a

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


def select_candidate(d, ub, fresh, ij, M: int) -> tuple[str, np.ndarray]:
    """One selection step over the live candidates (A20, A21b; pure).

    ``d``, ``ub`` (n,): stored score and upper bound (+inf for a candidate marked stale);
    ``fresh`` (n,): scored at the current epoch; ``ij`` (n, 2) canonical pairs. The best eligible
    (fresh, finite d) candidate by (d desc, ij asc) is accepted once d ≥ every stale bound
    (equality accepts, A20). Otherwise ("rescore", the M stale candidates with the largest bound by
    (ub desc, ij asc)); ("none", []) when nothing is eligible or stale.
    """
    stale = np.flatnonzero(~fresh)
    elig = np.flatnonzero(fresh & np.isfinite(d))
    if len(elig):
        best = elig[np.lexsort((ij[elig, 1], ij[elig, 0], -d[elig]))[0]]
        if not len(stale) or d[best] >= ub[stale].max():
            return "accept", np.array([best])
    if not len(stale):
        return "none", np.array([], np.int64)
    u = ub[stale]
    # ponytail: when every stale UB is +inf (each exhaustive 6a round) the threshold keeps all of them and the
    # lexsort below costs O(n log n); Task 6b must partition on a combined (ub desc, ij asc) key instead.
    if len(stale) > M:  # O(n) partition, then exact (ub desc, ij asc) among ties at the M-th bound
        thr = np.partition(u, len(u) - M)[len(u) - M]
        stale, u = stale[u >= thr], u[u >= thr]
    top = stale[np.lexsort((ij[stale, 1], ij[stale, 0], -u))[:M]]
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
        keep = ~np.isin(self.ij, nodes).any(axis=1)
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
    the root), in the table's centred frame. Children's times must be set (star EM by the caller). Per round: at the root, star
    re-optimisation when counter > redoTimeAt = max(⌊0.05·C⌋, 10) (recomputed at each one); else every
    candidate is marked stale and certified by ``select_candidate`` in rescore batches of
    ``cfg.rescore_top`` rows, and the best pair is merged iff its dLogL > 1e-6, with the A18 snap
    (t_1a < 1e-6 and child 1 internal → t_1a := 0, else the same for child 2). Stops at ≤ 3 children
    (root) / ≤ 2 (other). ``trace``: as ``_ref.greedy``, one entry per round before the decision,
    (scores {(i, j): (dLogL, t3, host, x_R, v_R)}, {child: t}, x_r, W_r, counter) with (x_r, W_r) the float64
    AIR, ``host`` whether the score is the fp64 host re-score, (x_R, v_R) the rest as sent to the device;
    re-optimisation rounds have no scores. p's row ends with its message from its children.
    ponytail: Task 6a reads only ``mode``, ``rescore_top`` and ``star_tol``; every round is exhaustive
    whatever ``knn``/``lazy``/``merges_per_round``/``new_batch`` say, until Task 6b adds those branches.
    """
    mode, M = cfg.mode, cfg.rescore_top
    dt = table.dtype
    kids = table.children(p)
    events: list[Event] = []
    n = {"scores": 0, "host": 0, "batches": 0, "reopts": 0, "unrescored": 0}
    store = _Store()
    mv = {"Dx": 0.0, "DW": 0.0, "epoch": 0}
    sent: dict = {}  # trace only: (x_R, v_R) per pair as last sent to the device

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

    def score(idx: np.ndarray) -> None:
        """Score store rows ``idx`` (≤ M) in one padded device batch; flagged rows re-scored on the host."""
        P = len(idx)
        i, j = store.ij[idx, 0], store.ij[idx, 1]
        pad = np.r_[np.arange(P), np.zeros(M - P, np.int64)]  # repeat row 0; padded results ignored
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
            c1, c2 = i[r], j[r]
            ref = ref_pair.pair_score(
                table.x64(c1), table.v64(c1), table.t[c1], table.x64(c2), table.v64(c2), table.t[c2], xr, Wr, mode
            )
            d[r], t3[r] = ref.dlogl, ref.t
            a[r], b[r] = ref_pair.ub_slopes(ref, xr, Wr) if np.isfinite(ref.dlogl) else (0.0, 0.0)
            host[r] = True
        n["unrescored"] += int(np.sum(flag & ~host))
        n["scores"] += P
        n["host"] += int(host.sum())
        store.d[idx], store.t3[idx], store.a[idx], store.b[idx], store.host[idx] = d, t3, a, b, host
        store.epoch[idx], store.Dx[idx], store.DW[idx] = mv["epoch"], mv["Dx"], mv["DW"]
        if trace is not None:  # the rest exactly as sent to the device, for the trace
            sent.update({(int(i[r]), int(j[r])): (xR[r], vR[r]) for r in range(P)})

    def certify() -> int | None:
        while True:
            kind, idx = select_candidate(store.d, store.ub(mv["Dx"], mv["DW"]), store.epoch == mv["epoch"], store.ij, M)
            if kind != "rescore":
                return int(idx[0]) if kind == "accept" else None
            score(idx)
            n["batches"] += 1

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
        store.epoch[:] = -1  # every child time changed
        n["reopts"] += 1
        events.append(Event("reopt", -1, -1, -1, np.nan, (np.nan, np.nan, np.nan)))

    def merge(r: int) -> None:
        nonlocal kids
        i, j = (int(c) for c in store.ij[r])
        t1a, t2a, t_ar = (float(u) for u in store.t3[r])
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
        events.append(Event("merge", i, j, a, float(store.d[r]), (t1a, t2a, t_ar)))
        kids = [c for c in kids if c not in (i, j)]
        store.drop([i, j])
        store.add([(c, a) for c in kids])  # canonical: a is the highest id; scored when certified
        kids.append(a)

    acc = build_acc()
    store.add([(c1, c2) for k, c1 in enumerate(kids) for c2 in kids[k + 1 :]])
    counter, redo_at = 0, max(int(REDO_TIME_FRAC * len(kids)), REDO_TIME_MIN)
    while len(kids) > (3 if is_root else 2):
        if is_root and counter > redo_at:  # TH:1926-1927 (A21: on any round)
            if trace is not None:
                trace.append(({}, {c: float(table.t[c]) for c in kids}, *acc.message(), counter))
            reopt()
            counter, redo_at = 0, max(int(REDO_TIME_FRAC * len(kids)), REDO_TIME_MIN)
            continue
        store.epoch[:] = -1  # exhaustive round (Task 6a): every candidate stale, UB := +inf
        best = certify()
        if trace is not None:
            fresh = np.flatnonzero(store.epoch == mv["epoch"])
            scores = {}
            for r in fresh:
                ij = (int(store.ij[r, 0]), int(store.ij[r, 1]))
                scores[ij] = (store.d[r], store.t3[r].copy(), bool(store.host[r]), *sent[ij])
            trace.append((scores, {c: float(table.t[c]) for c in kids}, *acc.message(), counter))
        if best is None or not store.d[best] > ACCEPT_DLOGL:
            break
        merge(best)
        counter += 1
    xp, Wp = ref_star.node_message(table.x64(kids), table.v64(kids), table.t[kids])
    table.x[p], table.v[p] = xp, 1.0 / Wp
    table.sync(p)
    return GreedyStats(events, n["scores"], n["host"], n["batches"], 0, n["reopts"], n["unrescored"])
