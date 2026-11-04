"""Float64 gate re-timer (amendments A16, A16b, A16c, A16f, A16g): ``python -m bench.retime IN.npz OUT.npz``.

Two parts, cached separately by ``bench.gate`` (A16f):

- ``lbfgs`` (the endpoint): maximises L_tree over s = log t of every non-root edge with scipy
  L-BFGS-B (ftol 0, gtol 1e-12) on the engine's float64 ``tree_loglik_grad`` (needs
  ``JAX_ENABLE_X64=1``; ``bench.gate`` runs this module in such a subprocess). After any stop with
  status 0, 1 or 2 it restarts from the stopping point while a restart raises L by more than the
  bound, at most 10 restarts (A16b, A16c).
- ``accept`` (A16f, always recomputed from the endpoint): the dense free-set Hessian H = −∇²L in s
  from batched ``jax.jvp`` columns of the engine gradient, symmetrised and eigendecomposed (on the
  GPU when there is one); c = Uᵀ·pg. gap = ``gap_terms``: ½·Σ c_i²/λ_i over λ_i > τ (τ = 1e-10·scale)
  + max(Σ over λ_i ≤ τ of the measured fp64 ``_ref`` gain along ±u_i projected to the box
  (``line_gain``), the joint gain of a bounded L-BFGS on span{u_i : λ_i ≤ τ} when there are ≥ 2)
  + the joint gain of a projected L-BFGS on ``_ref`` L over the free edges with s < −9 (near the floor
  L ≈ L∞ − a·eˢ: the Newton term gives ½·a·eˢ of the true gain ≈ a·eˢ; A16f detail). Terms may
  double-count, which only errs towards INVALID. It is an estimate, not a rigorous bound.

- ``settle`` (A16g, the saddle escape, on top of the unchanged A16f acceptance): while ``needs_escape`` (the gap
  is the only failure and the flat/negative-direction plus floor terms together exceed the bound; the Newton term
  never counts), ``escape_step`` moves to the best point the A16f searches of those terms measure (``escape_point``),
  resumes ``lbfgs`` from there (same options and restart rule), and the escaped endpoint is accepted again; at most
  MAX_ESCAPES escapes, then INVALID (also if an escape makes no progress or lowers L).

Accepted iff every scipy run stopped with status 0, 1 or 2, the last restart gained ≤ the bound,
gap ≤ the bound, and the engine's L agrees with ``_ref.tree`` to 1e-10 relative. The bound is
1e-2·(1e-6·N·G): each arm's error is ≤ 1% of the A1 slack. Anything else is INVALID with the reason.

The input file holds ``parent``, ``t``, ``x``, ``var`` and optionally an endpoint (``Endpoint`` fields
prefixed ``ep_``), which skips ``lbfgs``; the output holds the ``RetimeResult`` and the endpoint. A freshly
computed endpoint is also written to ``OUT.ep.npz`` before the acceptance runs (A16f detail), so a crash in the
acceptance does not lose the L-BFGS work. With ``esc_flat``/``esc_floor`` in the input, ``main`` instead runs one
``escape_step`` from the given endpoint and writes the escaped endpoint (unprefixed) to OUT.
"""

# mypy: disable-error-code="return-value"
# File-wide because the only two hits (eigh: numpy's EighResult vs tuple; escape_point: best[1] as object) sit in
# functions hashed by bench.gate.acceptance_hash/escape_hash, whose source must not change (not even a line ignore).

import os
import sys
import time
from dataclasses import asdict, dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from bonsai._engine.array_tree import ArrayTree
from bonsai._engine.opt_times import LOG_T_MAX, LOG_T_MIN
from bonsai._engine.tree import tree_loglik_grad
from bonsai._ref import tree as ref_tree

OPTIONS = {"ftol": 0.0, "gtol": 1e-12, "maxiter": 15000, "maxfun": 30000}  # per run (A16b)
MAX_RESTARTS = 10
GAP_FRAC = 1e-2  # bound = GAP_FRAC · 1e-6 · N · G (A1 slack per entry is 1e-6)
L_MATCH_REL = 1e-10  # |L_engine64 − L_ref| / |L_ref| above this → INVALID
CURV_REL = 1e-10  # τ = CURV_REL·scale: eigenvalues ≤ τ get a measured line gain instead of the Newton term
LG_ALPHA_MIN, LG_GRID = 1e-3, 40  # line_gain: LG_GRID log-spaced steps from LG_ALPHA_MIN to the box limit
FLOOR_S = -9.0  # free edges with s below this get the joint floor-gain term
# JVP batch: at most this share of the free device memory (env BONSAI_HESS_MEM_FRAC overrides) ...
HESS_MEM_FRAC = float(os.environ.get("BONSAI_HESS_MEM_FRAC", "0.25"))
HESS_CPU_BYTES = 2 * 2**30  # ... or this budget when the device reports none (CPU)
HESS_MAX_BATCH = 512
MAX_ESCAPES = 5  # A16g
ESCAPE_COLS = ("L", "gap", "gap_newton", "gap_flat", "gap_floor", "converged")  # one row per escape: its acceptance


@dataclass(frozen=True)
class Endpoint:
    """L-BFGS-B endpoint (``lbfgs``); cached by ``bench.gate`` whatever its status (A16f)."""

    s: np.ndarray  # (n_edges,) log t of the non-root edges (parent >= 0, in node order)
    L_start: float  # L_tree at the start (times clipped into [e⁻¹⁵, e¹⁵])
    nit: int  # scipy iterations, summed over all runs
    restarts: int  # restarts after the first run (≤ MAX_RESTARTS)
    extra_dL: float  # L gain of the last restart (inf if there was none)
    status_ok: bool  # every run stopped with scipy status 0, 1 or 2
    message: str  # scipy status and message of the last run
    history: np.ndarray  # (runs, 3): nit, engine L at the run's end, scipy status ((0, 3) if not recorded)


@dataclass(frozen=True)
class RetimeResult:
    """Outcome of ``accept`` (A16f); ``status`` is "converged" only if every acceptance test passed."""

    t: np.ndarray  # (n,) node-indexed re-timed branch lengths, 0 at the root
    L: float  # L_tree at t (A1 convention, `_ref.tree`)
    L_start: float
    gap: float  # gap_newton + gap_flat + gap_floor
    gap_newton: float  # ½·Σ c²/λ over λ > τ
    gap_flat: float  # max(gap_flat_sum, gap_flat_joint)
    gap_flat_sum: float  # Σ measured line gains along the eigenvectors with λ ≤ τ
    gap_flat_joint: float  # joint L-BFGS gain on their span (0 unless n_flat ≥ 2)
    gap_floor: float  # joint L-BFGS gain over the free edges with s < FLOOR_S
    n_floor: int  # those edges
    lambda_min: float  # smallest eigenvalue of the symmetrised free-set Hessian (nan if the free set is empty)
    n_flat: int  # eigenvalues ≤ τ
    n_free: int
    asym: float  # max|H − Hᵀ| / max|H| before symmetrising
    sum_abs_pg: float  # Σ|t·∂L/∂t| over the free set
    extra_dL: float
    nit: int
    restarts: int
    hess_s: float  # wall time of the Hessian build (s)
    eigh_s: float  # wall time of the eigendecomposition (s)
    status: str  # "converged" or "invalid"
    reason: str  # why the result is invalid ("" if converged)


def free_set(s: np.ndarray, pg: np.ndarray) -> np.ndarray:
    """Interior coordinates plus bound coordinates whose gradient points into the box."""
    return ~(((s <= LOG_T_MIN) & (pg <= 0)) | ((s >= LOG_T_MAX) & (pg >= 0)))


def line_gain(L, s: np.ndarray, p: np.ndarray) -> float:
    """Largest measured gain L(s + α·u) − L(s), α ≥ 0, over the projected directions u of ±p (A16c, A16f).

    Per sign: zero the components that push an active bound outward, renormalise, cap the step at
    the box: α_max is where the first component reaches its bound (the box limit along u). Evaluate L on LG_GRID log-spaced steps from min(1e-3, α_max) to α_max, then a
    bounded Brent search between the best grid point's neighbours. It never stops at the first
    non-increase, which rounding can fake. Returns ≥ 0.
    """
    L0, best = L(s), 0.0
    for v in (p, -p):
        v = np.where(((s <= LOG_T_MIN) & (v < 0)) | ((s >= LOG_T_MAX) & (v > 0)), 0.0, v)
        if not np.any(v):
            continue
        u = v / np.linalg.norm(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            room = np.where(u > 0, (LOG_T_MAX - s) / u, np.where(u < 0, (LOG_T_MIN - s) / u, np.inf))
        amax = float(np.min(room))
        if amax <= 0:
            continue
        grid = np.geomspace(min(LG_ALPHA_MIN, amax), amax, LG_GRID)
        f = np.array([L(s + a * u) for a in grid])
        k = int(np.argmax(f))
        lo, hi = (grid[k - 1] if k else 0.0), grid[min(k + 1, LG_GRID - 1)]
        r = minimize_scalar(lambda z, u=u: -L(s + z * u), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        best = max(best, f[k] - L0, -r.fun - L0)
    return best


def joint_gain(Lg, s: np.ndarray, B: np.ndarray, bounds=None) -> float:
    """Gain of an L-BFGS-B maximisation of L over s + B·y from y = 0 (``Lg(z)`` = (L, ∂L/∂z)); ≥ 0.

    With ``bounds`` (per y) they keep s + B·y in the box; without, L is evaluated at the box projection of
    s + B·y (gradient zero on clipped coordinates), so every evaluated point is feasible.
    """
    L0 = Lg(s)[0]

    def neg(y):
        z = s + B @ y
        zc = np.clip(z, LOG_T_MIN, LOG_T_MAX)
        L, g = Lg(zc)
        return -L, -(B.T @ np.where(z == zc, g, 0.0))

    r = minimize(neg, np.zeros(B.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds, options=OPTIONS)
    return max(0.0, -float(r.fun) - L0)


def gap_terms(w, U, pg, tau: float, s: np.ndarray, fi: np.ndarray, L, Lg) -> dict:
    """The A16f gap terms (a dict, see below).

    Inputs: the eigenpairs (w, U) of the free-set H, the free pg, the point s (all edges), the free indices
    ``fi`` and the fp64 ``L``/``Lg`` (value; value and gradient in s).

    newton = ½·Σ c_i²/λ_i over λ_i > ``tau`` (c = Uᵀ·pg; exact for a quadratic); flat_sum = Σ ``line_gain``
    along each u_i with λ_i ≤ ``tau``; flat_joint = ``joint_gain`` on their span when there are ≥ 2;
    floor = ``joint_gain`` over the free coordinates with s < FLOOR_S, box-bounded.
    """
    c = U.T @ pg
    pos = w > tau
    flat = np.flatnonzero(~pos)
    Uf = np.zeros((len(s), len(flat)))
    Uf[fi] = U[:, flat]
    floor = fi[s[fi] < FLOOR_S]
    Bf = np.eye(len(s))[:, floor]
    return {
        "newton": 0.5 * float(np.sum(c[pos] ** 2 / w[pos])),
        "flat_sum": float(sum(line_gain(L, s, Uf[:, k]) for k in range(len(flat)))),
        "flat_joint": joint_gain(Lg, s, Uf) if len(flat) >= 2 else 0.0,
        "floor": joint_gain(Lg, s, Bf, [(LOG_T_MIN - v, LOG_T_MAX - v) for v in s[floor]]) if len(floor) else 0.0,
        "n_flat": len(flat),
        "n_floor": len(floor),
    }


def _grad_scale(parent: np.ndarray, t: np.ndarray, R: ref_tree.TreeGrad) -> float:
    """Max over edges of Σ_g |terms of ∂L/∂s_i| = t·(1 + (x_ex − x)²/T)/T, T = t + 1/W + 1/W_ex."""
    e = np.flatnonzero(parent >= 0)
    T = t[e, None] + 1.0 / R.W[e] + 1.0 / R.W_ex[e]
    return float(np.max(np.sum(t[e, None] * (1.0 + (R.x_ex[e] - R.x[e]) ** 2 / T) / T, axis=1)))


def times_of(parent, s: np.ndarray) -> np.ndarray:
    """Node-indexed t with exp(s) on the non-root edges, 0 at the root."""
    parent = np.asarray(parent)
    out = np.zeros(len(parent))
    out[parent >= 0] = np.exp(s)
    return out


def _objective(parent, t, x, var):
    """(e, engine, grad_s): the edges, the engine's (L, ∂L/∂s) on the host, and ∂L/∂s as a jax function."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError("bench.retime needs JAX_ENABLE_X64=1 (run it via bench.gate.retime_subprocess)")
    e = np.flatnonzero(np.asarray(parent) >= 0)
    tree = ArrayTree.from_arrays(np.asarray(parent, dtype=np.int64), t, x, var, dtype=jnp.float64)
    n_pad = tree.t.shape[0]

    def engine(s):
        tt = np.zeros(n_pad)
        tt[e] = np.exp(s)
        ev = tree_loglik_grad(eqx.tree_at(lambda tr: tr.t, tree, jnp.asarray(tt)))
        return ev.L, np.asarray(ev.grad, dtype=np.float64)[e] * tt[e]

    ej = jnp.asarray(e)

    def grad_s(s):
        tt = jnp.zeros(n_pad).at[ej].set(jnp.exp(s))
        return tree_loglik_grad(eqx.tree_at(lambda tr: tr.t, tree, tt)).grad[ej] * tt[ej]

    return e, engine, grad_s


def _bound(parent, x) -> float:
    n_cells = int(sum(1 for k in ref_tree.postorder(np.asarray(parent))[1] if not k))
    return GAP_FRAC * 1e-6 * n_cells * np.shape(x)[1]


def lbfgs(parent, t, x, var) -> Endpoint:
    """L-BFGS-B with the A16b/c restart loop from the stage tree's times ``t`` (node-indexed)."""
    x, var = np.asarray(x, dtype=np.float64), np.asarray(var, dtype=np.float64)
    e, engine, _ = _objective(parent, t, x, var)
    bound, history = _bound(parent, x), []

    def neg(s):
        L, g = engine(s)
        return -L, -g

    def run(s0):
        r = minimize(neg, s0, jac=True, method="L-BFGS-B", bounds=[(LOG_T_MIN, LOG_T_MAX)] * len(e), options=OPTIONS)
        history.append((r.nit, -r.fun, r.status))
        return r

    s = np.log(np.clip(np.asarray(t, dtype=np.float64)[e], np.exp(LOG_T_MIN), np.exp(LOG_T_MAX)))
    L_start = ref_tree.tree_loglik(np.asarray(parent), times_of(parent, s), x, var)
    res = run(s)
    nit, restarts, extra_dL, status_ok = res.nit, 0, np.inf, res.status in (0, 1, 2)
    while status_ok and restarts < MAX_RESTARTS:  # A16b/c: restart while a restart gains > bound
        nxt = run(res.x)
        nit, restarts, extra_dL = nit + nxt.nit, restarts + 1, res.fun - nxt.fun
        res, status_ok = nxt, nxt.status in (0, 1, 2)
        if extra_dL <= bound:
            break
    return Endpoint(
        s=np.asarray(res.x, dtype=np.float64),
        L_start=float(L_start),
        nit=int(nit),
        restarts=restarts,
        extra_dL=float(extra_dL),
        status_ok=bool(status_ok),
        message=f"scipy status {res.status} ({res.message})",
        history=np.array(history, dtype=np.float64).reshape(-1, 3),
    )


def _jvp_batch(f, s: np.ndarray) -> int:
    """Columns per batched-JVP call, from the compiled temp memory.

    A linear fit fixed + B·per of the temp memory at B = 1 and 2, against HESS_MEM_FRAC of the free device
    memory (HESS_CPU_BYTES if the device reports none).
    """

    def temp(B):
        mem = f.lower(jnp.asarray(s), jnp.zeros((B, len(s)))).compile().memory_analysis()
        return int(getattr(mem, "temp_size_in_bytes", 0)) + 16 * B * len(s)

    t1, t2 = temp(1), temp(2)
    per = max(t2 - t1, 1)
    try:
        st = jax.devices()[0].memory_stats() or {}
        budget = HESS_MEM_FRAC * (st["bytes_limit"] - st["bytes_in_use"])
    except (KeyError, TypeError):
        budget = HESS_CPU_BYTES
    return int(np.clip((budget - (t1 - per)) // per, 1, HESS_MAX_BATCH))


def free_hessian(grad_s, s: np.ndarray, fi: np.ndarray) -> np.ndarray:
    """Dense H = −∂(grad_s)/∂s on the free indices ``fi`` (rows and columns), unsymmetrised, by batched JVPs.

    The batch size comes from memory (``_jvp_batch``); the last batch is padded, so there is one compiled shape.
    """
    n_e, nf = len(s), len(fi)
    fij = jnp.asarray(fi)
    f = jax.jit(lambda z, V: jax.vmap(lambda v: jax.jvp(grad_s, (z,), (v,))[1][fij])(V))
    B = min(_jvp_batch(f, s), nf)
    sj, H = jnp.asarray(s), np.empty((nf, nf))
    for j0 in range(0, nf, B):
        cols = fi[j0 : j0 + B]
        V = np.zeros((B, n_e))
        V[np.arange(len(cols)), cols] = 1.0
        H[:, j0 : j0 + len(cols)] = -np.asarray(f(sj, jnp.asarray(V)))[: len(cols)].T
    return H


def eigh(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs (ascending) of the symmetric ``H``: ``jnp.linalg.eigh`` on a GPU, NumPy otherwise."""
    if jax.default_backend() == "gpu":
        w, U = jnp.linalg.eigh(jnp.asarray(H))
        return np.asarray(w), np.asarray(U)
    return np.linalg.eigh(H)


def accept(parent, x, var, ep: Endpoint) -> RetimeResult:
    """A16f acceptance of the endpoint ``ep`` (memoised by ``bench.gate`` under the endpoint and acceptance code)."""
    parent = np.asarray(parent, dtype=np.int64)
    x, var = np.asarray(x, dtype=np.float64), np.asarray(var, dtype=np.float64)
    s = np.asarray(ep.s, dtype=np.float64)
    t_fin = times_of(parent, s)
    e, engine, grad_s = _objective(parent, t_fin, x, var)
    bound = _bound(parent, x)
    R = ref_tree.tree_loglik_grad(parent, t_fin, x, var)
    L_eng = engine(s)[0]
    pg = (R.grad * t_fin)[e]
    fi = np.flatnonzero(free_set(s, pg))
    tau = CURV_REL * _grad_scale(parent, t_fin, R)

    def L(z):
        return ref_tree.tree_loglik(parent, times_of(parent, z), x, var)

    def Lg(z):
        tz = times_of(parent, z)
        r = ref_tree.tree_loglik_grad(parent, tz, x, var)
        return r.L, (r.grad * tz)[e]

    t0 = time.perf_counter()
    H = free_hessian(grad_s, s, fi) if len(fi) else np.zeros((0, 0))
    t1 = time.perf_counter()
    hmax = float(np.max(np.abs(H), initial=0.0))
    asym = float(np.max(np.abs(H - H.T), initial=0.0)) / hmax if hmax > 0 else 0.0
    w, U = eigh(0.5 * (H + H.T))
    t2 = time.perf_counter()
    g = gap_terms(w, U, pg[fi], tau, s, fi, L, Lg)
    flat = max(g["flat_sum"], g["flat_joint"])
    gap = g["newton"] + flat + g["floor"]
    reasons = [
        msg
        for bad, msg in [
            (not ep.status_ok, ep.message),
            (abs(L_eng - R.L) > L_MATCH_REL * abs(R.L), f"engine/reference L mismatch {L_eng - R.L:.3e}"),
            (
                ep.status_ok and not ep.extra_dL <= bound,
                f"{ep.restarts} restarts, the last raised L by {ep.extra_dL:.3e}",
            ),
            (
                not gap <= bound,
                f"gap {gap:.3e} > {bound:.3e} (Newton {g['newton']:.3e}, flat {flat:.3e}, floor {g['floor']:.3e})",
            ),
        ]
        if bad
    ]
    return RetimeResult(
        t=t_fin,
        L=R.L,
        L_start=ep.L_start,
        gap=gap,
        gap_newton=g["newton"],
        gap_flat=flat,
        gap_flat_sum=g["flat_sum"],
        gap_flat_joint=g["flat_joint"],
        gap_floor=g["floor"],
        n_floor=g["n_floor"],
        lambda_min=float(w[0]) if len(w) else float("nan"),
        n_flat=g["n_flat"],
        n_free=len(fi),
        asym=asym,
        sum_abs_pg=float(np.abs(pg[fi]).sum()),
        extra_dL=ep.extra_dL,
        nit=ep.nit,
        restarts=ep.restarts,
        hess_s=t1 - t0,
        eigh_s=t2 - t1,
        status="invalid" if reasons else "converged",
        reason="; ".join(reasons),
    )


def sole_gap_failure(r: RetimeResult, bound: float) -> bool:
    """The gap is ``r``'s only failure: its reason is exactly the gap clause ``accept`` writes from r's own terms.

    ``accept`` joins every failing test (bad scipy stop, exhausted restarts, engine/reference L mismatch, gap), so an
    exact match rules all the others out. A stored flag would change ``RetimeResult`` and so ``acceptance_hash``,
    invalidating every memoised A16f acceptance; this reconstruction keeps them valid.
    """
    flat = f"flat {r.gap_flat:.3e}, floor {r.gap_floor:.3e}"
    return r.reason == f"gap {r.gap:.3e} > {bound:.3e} (Newton {r.gap_newton:.3e}, {flat})"


def needs_escape(r: RetimeResult, bound: float) -> bool:
    """A16g (controller ruling): the gap is the sole failure and flat + floor > ``bound`` (combined).

    The Newton term never counts: its gain is the positive-curvature remainder the L-BFGS restart loop handles.
    """
    return sole_gap_failure(r, bound) and r.gap_flat + r.gap_floor > bound


def escape_flags(r: RetimeResult) -> tuple[bool, bool]:
    """(flat, floor): the searches ``escape_point`` runs, those whose A16f term in ``r`` gained anything."""
    return bool(r.gap_flat > 0), bool(r.gap_floor > 0)


def escape_point(parent, x, var, ep: Endpoint, flat: bool, floor: bool) -> np.ndarray:
    """A16g: the highest-L point measured by the selected A16f searches (``ep.s`` if none gains).

    ``flat``: ``line_gain`` along each eigenvector of the free-set H with λ ≤ τ and, with ≥ 2, ``joint_gain`` on their
    span (the dense H is rebuilt: the memoised acceptance keeps no eigenvectors); ``floor``: ``joint_gain`` over the free
    edges with s < FLOOR_S. The searches are A16f's own; the best of their fp64 ``_ref`` evaluations is kept.
    """
    parent = np.asarray(parent, dtype=np.int64)
    x, var = np.asarray(x, dtype=np.float64), np.asarray(var, dtype=np.float64)
    s = np.asarray(ep.s, dtype=np.float64)
    t = times_of(parent, s)
    e, _, grad_s = _objective(parent, t, x, var)
    R = ref_tree.tree_loglik_grad(parent, t, x, var)
    fi = np.flatnonzero(free_set(s, (R.grad * t)[e]))
    best = [R.L, s]

    def keep(v, z):
        if v > best[0]:
            best[:] = [v, np.array(z)]
        return v

    def L(z):
        return keep(ref_tree.tree_loglik(parent, times_of(parent, z), x, var), z)

    def Lg(z):
        tz = times_of(parent, z)
        r = ref_tree.tree_loglik_grad(parent, tz, x, var)
        return keep(r.L, z), (r.grad * tz)[e]

    if flat and len(fi):
        H = free_hessian(grad_s, s, fi)
        w, U = eigh(0.5 * (H + H.T))
        fl = np.flatnonzero(w <= CURV_REL * _grad_scale(parent, t, R))
        Uf = np.zeros((len(s), len(fl)))
        Uf[fi] = U[:, fl]
        for k in range(len(fl)):
            line_gain(L, s, Uf[:, k])
        if len(fl) >= 2:
            joint_gain(Lg, s, Uf)
    cols = fi[s[fi] < FLOOR_S]
    if floor and len(cols):
        joint_gain(Lg, s, np.eye(len(s))[:, cols], [(LOG_T_MIN - v, LOG_T_MAX - v) for v in s[cols]])
    return best[1]


def escape_step(parent, x, var, ep: Endpoint, flat: bool, floor: bool) -> Endpoint:
    """A16g: ``lbfgs`` (its options and restart loop) resumed from ``escape_point``; start log clip(exp z) = z to rounding.

    The escaped endpoint keeps ``ep``'s L_start and sums nit and history over all runs; its other fields are the last
    run's. If no search gained (the point is ``ep.s``) ``ep`` itself is returned: ``settle`` reports no progress.
    """
    z = escape_point(parent, x, var, ep, flat, floor)
    if np.array_equal(z, ep.s):
        return ep
    new = lbfgs(parent, times_of(parent, z), x, var)
    return replace(new, L_start=ep.L_start, nit=ep.nit + new.nit, history=np.vstack([ep.history, new.history]))


def settle(acc, step, ep: Endpoint, bound: float, r=None, n: int = 0, rows=(), L0=None, save=None):
    """A16g escape loop: (final acceptance, escape state (ep, n, rows, L0)).

    ``acc(ep)`` is the A16f acceptance (``r`` is ``acc(ep)`` if already known), ``step(ep, r)`` one escape. While
    ``needs_escape`` and fewer than MAX_ESCAPES escapes were taken: escape, ``save(state)`` (before the acceptance), accept
    again. ``rows`` (ESCAPE_COLS) records the acceptance after each escape, L0 the L before the first; ``n``, ``rows``, ``L0``
    continue a stored state (``n > len(rows)``: the last escape's acceptance is still missing). INVALID with the reason
    if an escape made no progress (the point is the endpoint), lowered L by more than 1e-10·|L| (rounding of a sum of
    that size is far below; also checked on resume, against the last recorded L), or still needs an escape at the cap.
    """

    def lowered(prev, L):
        return f"escape lowered L by {prev - L:.3e}" if prev is not None and L < prev - 1e-10 * abs(prev) else ""

    rows = [tuple(row) for row in rows]
    L_prev = (rows[-1][0] if rows else L0) if r is None and n > 0 else None  # resuming a stored escape state
    r = acc(ep) if r is None else r
    L0 = r.L if L0 is None else L0
    stop = lowered(L_prev, r.L)
    while True:
        if len(rows) < n:
            rows.append(tuple(float(getattr(r, c)) for c in ESCAPE_COLS[:-1]) + (float(r.status == "converged"),))
        if stop or n >= MAX_ESCAPES or not needs_escape(r, bound):
            break
        new = step(ep, r)
        if np.array_equal(new.s, ep.s):
            stop = "escape made no progress"
            break
        L_prev, ep, n = r.L, new, n + 1
        if save is not None:
            save((ep, n, np.array(rows, dtype=np.float64).reshape(-1, len(ESCAPE_COLS)), L0))
        r = acc(ep)
        stop = lowered(L_prev, r.L)
    if stop or needs_escape(r, bound):
        why = stop or f"{n} escapes, the flat/floor gain still exceeds the bound"
        r = replace(r, status="invalid", reason="; ".join(filter(None, [r.reason, why])))
    return r, (ep, n, np.array(rows, dtype=np.float64).reshape(-1, len(ESCAPE_COLS)), L0)


def retime(parent, t, x, var) -> RetimeResult:
    """Re-time a fixed topology in float64 (A16), accept it (A16f) with the saddle escape (A16g).

    ``x``/``var`` node-indexed (leaf rows read).
    """
    bound = _bound(parent, x)
    return settle(
        lambda e: accept(parent, x, var, e),
        lambda e, r: escape_step(parent, x, var, e, *escape_flags(r)),
        lbfgs(parent, t, x, var),
        bound,
    )[0]


def escape_fields(state, prefix="esc_") -> dict:
    """The npz fields of an escape state (ep, n, rows, L0): the endpoint's fields and n, rows, L0, all ``prefix``ed."""
    ep, n, rows, L0 = state
    return {prefix + k: v for k, v in asdict(ep).items()} | {prefix + "n": n, prefix + "rows": rows, prefix + "L0": L0}


def read_escape(z, prefix="esc_"):
    """The escape state in an open npz written with ``escape_fields``, or ``None`` if it holds none."""
    if prefix + "n" not in z.files:
        return None
    return _fields(z, Endpoint, prefix), int(z[prefix + "n"]), z[prefix + "rows"], float(z[prefix + "L0"])


def _fields(z, cls, prefix=""):
    arrays = {"t", "s", "history"}
    return cls(**{k: z[prefix + k] if k in arrays else z[prefix + k].item() for k in cls.__dataclass_fields__})


def save(path, r: RetimeResult, ep: Endpoint) -> None:
    """Write ``r`` and its endpoint (fields prefixed ``ep_``) as an npz (the subprocess result file)."""
    np.savez(path, **asdict(r), **{f"ep_{k}": v for k, v in asdict(ep).items()})


def load(path) -> tuple[RetimeResult, Endpoint]:
    """Read a file written by ``save``."""
    with np.load(path) as z:
        return _fields(z, RetimeResult), _fields(z, Endpoint, "ep_")


def load_endpoint(path) -> Endpoint:
    """An endpoint stored unprefixed (``bench.gate``'s cache entry)."""
    with np.load(path) as z:
        return _fields(z, Endpoint)


def main(argv: list[str]) -> None:
    """``IN.npz OUT.npz``: the endpoint (given, or from ``lbfgs`` and saved to OUT.ep.npz first), then ``accept``.

    With ``esc_flat``/``esc_floor`` (A16g): one ``escape_step`` from the given endpoint instead, written to OUT.
    """
    inp, out = argv
    with np.load(inp) as z:
        a = (z["parent"], z["t"], z["x"], z["var"])
        ep = _fields(z, Endpoint, "ep_") if "ep_s" in z.files else None
        flags = (bool(z["esc_flat"]), bool(z["esc_floor"])) if "esc_flat" in z.files else None
    if flags is not None:  # A16g: one escape step, the escaped endpoint is the output
        assert ep is not None, "an escape input carries its endpoint (gate.escape_subprocess)"
        np.savez(out, **asdict(escape_step(a[0], a[2], a[3], ep, *flags)))
        return
    if ep is None:
        ep = lbfgs(*a)
        np.savez(endpoint_path(out), **asdict(ep))
    save(out, accept(a[0], a[2], a[3], ep), ep)


def endpoint_path(out) -> str:
    """Where ``main`` saves a fresh endpoint before the acceptance runs: OUT.ep.npz."""
    return str(out)[: -len(".npz")] + ".ep.npz" if str(out).endswith(".npz") else f"{out}.ep.npz"


if __name__ == "__main__":
    main(sys.argv[1:])
