"""Float64 gate re-timer (amendments A16, A16b, A16c): ``python -m bench.retime IN.npz OUT.npz``.

Maximises L_tree over s = log t of every non-root edge with scipy L-BFGS-B (ftol 0, gtol 1e-12),
using the engine's float64 ``tree_loglik_grad`` (needs ``JAX_ENABLE_X64=1``; ``bench.gate`` runs
this module in such a subprocess). Acceptance:

- restarts: after any stop with status 0, 1 or 2 the optimiser restarts from the stopping point
  while the restart raises L by more than the bound, at most 10 restarts; the last restart's gain
  must be ≤ the bound (A16b, A16c). Any other status (e.g. NaN) is INVALID;
- the estimated remaining gap on the free set by CG with central-FD Hessian-vector products
  (``remaining_gap``: Newton gap, or on non-positive curvature the CG model gain plus a measured
  line-search gain) ≤ the bound. It is an estimate, not a rigorous bound;
- the engine's L agrees with ``_ref.tree`` to 1e-10 relative at the final times.

The bound is 1e-2·(1e-6·N·G): each arm's error is ≤ 1% of the A1 slack. Anything else is INVALID.
"""

import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass

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
MAX_CG = 2000  # CG cap = max(MAX_CG, n_free) (A16c)
FD_H = 1e-4  # step in s (along a unit direction) of the central-FD Hessian-vector product
L_MATCH_REL = 1e-10  # |L_engine64 − L_ref| / |L_ref| above this → INVALID
GAP_FRAC = 1e-2  # bound = GAP_FRAC · 1e-6 · N · G (A1 slack per entry is 1e-6)
CG_GAP_REL, CG_GAP_WINDOW, CG_RES_GAP, CG_RES = 1e-3, 50, 1e-3, 1e-6  # pre-declared CG stop rule
CURV_REL = 1e-10  # pᵀHp ≤ CURV_REL·‖p‖²·scale: FD noise ≈ eps64·scale/h ≈ 2e-12·scale, ×50


@dataclass(frozen=True)
class RetimeResult:
    """Outcome of one ``retime`` call (A16); ``status`` is "converged" only if every acceptance test passed."""

    t: np.ndarray  # (n,) node-indexed re-timed branch lengths, 0 at the root
    L: float  # L_tree at t (A1 convention, `_ref.tree`)
    L_start: float  # L_tree at the optimiser's start (times clipped into [e⁻¹⁵, e¹⁵])
    gap: float  # estimated remaining gain (``remaining_gap``)
    sum_abs_pg: float  # Σ|t·∂L/∂t| over the free set
    extra_dL: float  # L gain of the last restart (≤ the bound unless the restarts ran out)
    cg_iters: int
    cg_converged: bool
    curvature_stop: bool  # CG met non-positive curvature; gap includes the measured line-search gain
    nit: int  # scipy iterations, summed over all runs
    restarts: int  # L-BFGS-B restarts after the first run (≤ MAX_RESTARTS)
    status: str  # "converged" or "invalid"
    reason: str  # why the result is invalid ("" if converged)


def free_set(s: np.ndarray, pg: np.ndarray) -> np.ndarray:
    """Interior coordinates plus bound coordinates whose gradient points into the box."""
    return ~(((s <= LOG_T_MIN) & (pg <= 0)) | ((s >= LOG_T_MAX) & (pg >= 0)))


def line_gain(L: Callable[[np.ndarray], float], s: np.ndarray, p: np.ndarray) -> float:
    """Largest measured gain L(s + α·u) − L(s), α ≥ 0, over the projected directions u of ±p (A16c).

    Per sign: zero the components that push an active bound outward, renormalise, cap the step at
    the box; then double α from 1e-3 while L improves and finish with a bounded Brent search on
    [0, 2α]. Returns ≥ 0.
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
        a = min(1e-3, amax)
        fa = L(s + a * u)
        while a < amax:
            a2 = min(2 * a, amax)
            f2 = L(s + a2 * u)
            if f2 <= fa:
                break
            a, fa = a2, f2
        r = minimize_scalar(
            lambda z, u=u: -L(s + z * u), bounds=(0.0, min(2 * a, amax)), method="bounded", options={"xatol": 1e-12}
        )
        best = max(best, fa - L0, -r.fun - L0)
    return best


def remaining_gap(
    pg: np.ndarray,
    free: np.ndarray,
    hvp: Callable[[np.ndarray], np.ndarray],
    scale: float,
    max_cg: int = MAX_CG,
    line_search: Callable[[np.ndarray], float] | None = None,
) -> tuple[float, int, bool, bool]:
    """Estimated remaining gain on ``free``; returns (gap, cg_iters, cg_converged, curvature_stop).

    CG on H·d = pg, H = −∇²L; the gap is ½·pgᵀd (the Newton gap, exact for a quadratic). ``hvp(v)``
    = H·v only ever sees vectors that are zero outside ``free``, and its output is projected to
    ``free``. Converged when the gap changes ≤ 1e-3 relative over the last 50 iterations with
    ‖r‖ ≤ 1e-3·‖pg‖, or ‖r‖ ≤ 1e-6·‖pg‖; not converged at the cap. On curvature
    pᵀHp ≤ 1e-10·‖p‖²·scale (A16b, Steihaug-style) CG stops: gap = the CG model gain so far plus
    ``line_search(p)``, the measured gain of an exact line search along ±p from the re-timed point,
    and the result counts as converged; without ``line_search`` it does not. Either way the gap
    is an estimate (the residual beyond p is not counted), not a rigorous bound.
    """
    b = np.where(free, pg, 0.0)
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return 0.0, 0, True, False
    d, r = np.zeros_like(b), b.copy()
    p, rr, gaps = r.copy(), float(r @ r), [0.0]
    for it in range(1, max_cg + 1):
        Hp = np.where(free, hvp(p), 0.0)
        c = float(p @ Hp)
        if c <= CURV_REL * float(p @ p) * scale:
            if line_search is None:
                return gaps[-1], it, False, True
            return gaps[-1] + line_search(p), it, True, True
        a = rr / c
        d += a * p
        r -= a * Hp
        rn = float(r @ r)
        gaps.append(0.5 * float(b @ d))
        res = np.sqrt(rn)
        if res <= CG_RES * nb:
            return gaps[-1], it, True, False
        if (
            len(gaps) > CG_GAP_WINDOW + 1
            and abs(gaps[-1] - gaps[-1 - CG_GAP_WINDOW]) <= CG_GAP_REL * abs(gaps[-1])
            and res <= CG_RES_GAP * nb
        ):
            return gaps[-1], it, True, False
        p = r + (rn / rr) * p
        rr = rn
    return gaps[-1], max_cg, False, False


def _grad_scale(parent: np.ndarray, t: np.ndarray, R: ref_tree.TreeGrad) -> float:
    """Max over edges of Σ_g |terms of ∂L/∂s_i| = t·(1 + (x_ex − x)²/T)/T, T = t + 1/W + 1/W_ex."""
    e = np.flatnonzero(parent >= 0)
    T = t[e, None] + 1.0 / R.W[e] + 1.0 / R.W_ex[e]
    return float(np.max(np.sum(t[e, None] * (1.0 + (R.x_ex[e] - R.x[e]) ** 2 / T) / T, axis=1)))


def retime(parent, t, x, var) -> RetimeResult:
    """Re-time a fixed topology in float64 (A16). ``x``/``var`` node-indexed (only leaf rows read)."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError("bench.retime needs JAX_ENABLE_X64=1 (run it via bench.gate.retime_subprocess)")
    parent = np.asarray(parent, dtype=np.int64)
    x, var = np.asarray(x, dtype=np.float64), np.asarray(var, dtype=np.float64)
    e = np.flatnonzero(parent >= 0)
    n_cells = int(sum(1 for k in ref_tree.postorder(parent)[1] if not k))
    bound = GAP_FRAC * 1e-6 * n_cells * x.shape[1]
    tree = ArrayTree.from_arrays(parent, t, x, var, dtype=jnp.float64)
    n_pad = tree.t.shape[0]

    def engine(s):
        tt = np.zeros(n_pad)
        tt[e] = np.exp(s)
        ev = tree_loglik_grad(eqx.tree_at(lambda tr: tr.t, tree, jnp.asarray(tt)))
        return ev.L, np.asarray(ev.grad, dtype=np.float64)[e] * tt[e]

    def neg(s):
        L, g = engine(s)
        return -L, -g

    def times(s):
        out = np.zeros(len(parent))
        out[e] = np.exp(s)
        return out

    def run(s0):
        return minimize(neg, s0, jac=True, method="L-BFGS-B", bounds=[(LOG_T_MIN, LOG_T_MAX)] * len(e), options=OPTIONS)

    s = np.log(np.clip(np.asarray(t, dtype=np.float64)[e], np.exp(LOG_T_MIN), np.exp(LOG_T_MAX)))
    L_start = ref_tree.tree_loglik(parent, times(s), x, var)
    res = run(s)
    nit, restarts, extra_dL, status_ok = res.nit, 0, np.inf, res.status in (0, 1, 2)
    while status_ok and restarts < MAX_RESTARTS:  # A16b/c: restart while a restart gains > bound
        nxt = run(res.x)
        nit, restarts, extra_dL = nit + nxt.nit, restarts + 1, res.fun - nxt.fun
        res, status_ok = nxt, nxt.status in (0, 1, 2)
        if extra_dL <= bound:
            break
    s = res.x
    t_fin = times(s)
    R = ref_tree.tree_loglik_grad(parent, t_fin, x, var)
    L_eng = engine(s)[0]
    pg = (R.grad * t_fin)[e]
    free = free_set(s, pg)

    def hvp(v):  # step h along the unit direction: the FD noise is then ≈ eps64·scale/h per unit ‖v‖²
        nv = float(np.linalg.norm(v))
        u = v / nv
        return -nv * (engine(s + FD_H * u)[1] - engine(s - FD_H * u)[1]) / (2 * FD_H)

    def ref_L(z):
        return ref_tree.tree_loglik(parent, times(z), x, var)

    gap, cg_iters, cg_ok, curv = remaining_gap(
        pg,
        free,
        hvp,
        _grad_scale(parent, t_fin, R),
        max_cg=max(MAX_CG, int(free.sum())),
        line_search=lambda p: line_gain(ref_L, s, p),
    )
    reasons = [
        msg
        for bad, msg in [
            (not status_ok, f"scipy status {res.status} ({res.message})"),
            (abs(L_eng - R.L) > L_MATCH_REL * abs(R.L), f"engine/reference L mismatch {L_eng - R.L:.3e}"),
            (status_ok and not extra_dL <= bound, f"{restarts} restarts, the last raised L by {extra_dL:.3e}"),
            (not cg_ok, f"CG not converged in {cg_iters} iterations"),
            (not gap <= bound, f"gap {gap:.3e} > {bound:.3e}"),
        ]
        if bad
    ]
    return RetimeResult(
        t=t_fin,
        L=R.L,
        L_start=L_start,
        gap=gap,
        sum_abs_pg=float(np.abs(pg[free]).sum()),
        extra_dL=float(extra_dL),
        cg_iters=cg_iters,
        cg_converged=cg_ok,
        curvature_stop=curv,
        nit=nit,
        restarts=restarts,
        status="invalid" if reasons else "converged",
        reason="; ".join(reasons),
    )


def save(path, r: RetimeResult) -> None:
    """Write ``r`` as an npz (the subprocess result file)."""
    np.savez(path, **asdict(r))


def load(path) -> RetimeResult:
    """Read a result written by ``save``."""
    with np.load(path) as z:
        return RetimeResult(**{k: z[k] if k == "t" else z[k].item() for k in z.files})


if __name__ == "__main__":
    with np.load(sys.argv[1]) as z:
        r = retime(z["parent"], z["t"], z["x"], z["var"])
    save(sys.argv[2], r)
