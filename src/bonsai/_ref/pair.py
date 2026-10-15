"""fp64 NumPy reference: 2-leaf time, 3-leaf pair score (sequential/joint, A2), upper-bound slopes.

Maths ref §5-§7. G-vectors are 1-D arrays; times and log-likelihoods in 2x units (A1). The
optimisers here are tight (they define the optimum), not upstream's loose scoring tolerances.
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq, minimize

from .star import star_loglik_grad, two_leaf_dlogl  # two_leaf_dlogl re-exported

LOG_T_MIN, LOG_T_MAX = -16.1180, 10.0  # TH:4826, 4864 (t in [1.0e-7, 2.2e4])


def two_leaf_time(x1, v1, x2, v2) -> tuple[float | None, bool]:
    """argmax_t of L_2(t) = −Σ_g [log T + d²/T] (getOptTime2LeafTree, TH:4893-4923), to fp64 precision."""
    S, d2 = v1 + v2, (x1 - x2) ** 2
    if two_leaf_dlogl(0.0, S, d2) <= 0:
        return 0.0, True  # exact zero clamp (TH:4899-4901)
    lb, ub = 0.0, 1.0
    while two_leaf_dlogl(ub, S, d2) >= 0:  # ×10 bracket expansion (TH:4905-4916)
        lb, ub = ub, 10.0 * ub
        if lb > 1e6:  # upstream raises first once f(ub) > 1e9, reachable only for Σd² ≳ 1e11 (not normalised data)
            return None, False
    t, res = brentq(two_leaf_dlogl, lb, ub, args=(S, d2), xtol=1e-300, rtol=4 * np.finfo(float).eps, full_output=True)
    return t, res.converged


def init_times(t1: float, t2: float) -> tuple[list[float], list[float]]:
    """CalcTInit (HE:787-793): (sequential [t12, tau], joint [t1 − tau, t2 − tau, tau]), tau = min/2."""
    tau = min(t1, t2) / 2
    return [t1 + t2 - 2 * tau, tau], [t1 - tau, t2 - tau, tau]


class PairScore(NamedTuple):
    dlogl: float  # ell_new − ell_old (TH:5403); −inf if both optimisers failed
    t: np.ndarray  # (t_1a, t_2a, t_aR)
    converged: bool
    xs: np.ndarray  # (3, G) child means (x_1, x_2, x_R)
    vs: np.ndarray  # (3, G) child variances (v_1, v_2, 1/W_R)
    w_old: np.ndarray  # (3, G) old weights (w_1, w_2, W_R)


def rest_of_parent(xr, Wr, x1, w1, x2, w2):
    """R = parent AIR minus children 1 and 2 (TH:5372-5374, 2281-2282)."""
    Wm1 = Wr - w1
    WR = Wm1 - w2
    xR = ((xr * Wr - w1 * x1) - w2 * x2) / WR
    return xR, WR


def _neg_ell(times, xs, vs):
    ell, grad = star_loglik_grad(xs, vs, times)
    return -ell, -grad


def seq_objective(u, t12, xs, vs):
    """(−ell, gradient) of the 3-star in u = (t_1a, log t_aR), t_2a = t12 − t_1a (TH:5219-5226)."""
    tt = np.array([u[0], t12 - u[0], np.exp(u[1])])
    fv, g = _neg_ell(tt, xs, vs)
    return fv, np.array([g[0] - g[1], tt[2] * g[2]])


def joint_objective(s, xs, vs):
    """(−ell, gradient) of the 3-star in s = log t (TH:5241-5246)."""
    tt = np.exp(s)
    fv, g = _neg_ell(tt, xs, vs)
    return fv, tt * g


def pair_score(x1, v1, t1, x2, v2, t2, xr, Wr, mode: str = "sequential") -> PairScore:
    """3-leaf score of merging children 1, 2 of a parent with AIR message (xr, Wr) (TH:5362-5408).

    Sequential falls back to joint when t_12 has no bracket or when ``_minimise_box`` fails its KKT
    test after all restarts; upstream falls back on scipy's ``success=False`` at its loose tolerance
    (docs/deviations.md). Raises ValueError when the parent has no third child (W_R not positive).
    """
    w1, w2 = 1.0 / (t1 + v1), 1.0 / (t2 + v2)
    if np.any((Wr - w1) - w2 <= 4 * np.finfo(float).eps * Wr):  # 2-child parent: W_R is rounding noise, ≤ eps·W_r
        raise ValueError("pair_score: the parent needs at least 3 children (W_r − w_1 − w_2 is not positive)")
    xR, WR = rest_of_parent(xr, Wr, x1, w1, x2, w2)
    xs, vs = np.stack([x1, x2, xR]), np.stack([v1, v2, 1.0 / WR])
    seq0, joint0 = init_times(t1, t2)
    times, ok = np.array(joint0), False
    if mode == "sequential":
        t12, ok = two_leaf_time(x1, v1, x2, v2)
        if ok and t12 is not None:
            u, ok = _minimise_box(
                lambda u: seq_objective(u, t12, xs, vs),
                [0.5 * t12, np.log(max(seq0[1], 1e-4))],
                [(0.0, t12), (LOG_T_MIN, LOG_T_MAX)],
            )
            times = np.array([u[0], t12 - u[0], np.exp(u[1])])
    elif mode != "joint":
        raise ValueError(f"mode must be 'sequential' or 'joint', got {mode!r}")
    if not ok:  # joint mode, or the sequential fallback (TH:4822-4831)
        s, ok = _minimise_box(
            lambda s: joint_objective(s, xs, vs), np.log(np.maximum(joint0, 1e-4)), [(LOG_T_MIN, LOG_T_MAX)] * 3
        )
        times = np.exp(s)
    if not ok:
        return PairScore(-np.inf, np.array(joint0), False, xs, vs, np.stack([w1, w2, WR]))
    ell_new, _ = star_loglik_grad(xs, vs, times)
    w_old = np.stack([w1, w2, WR])
    ell_old, _ = star_loglik_grad(xs, vs, times, xn=xr, Wn=Wr, w=w_old)  # TH:5400-5402; times unused (w given)
    return PairScore(ell_new - ell_old, times, True, xs, vs, w_old)


KKT_RTOL = 1e-8  # reference optimum: max|projected grad| <= 1e-8·(1 + |f|), ~1e4 × the fp64 noise floor


def _minimise_box(f, u0, box, restarts: int = 5):
    """Tight L-BFGS-B, restarted from its own result until the projected-gradient KKT test passes.

    scipy's own ``success`` flag is not used: with ftol at machine precision it often reports a
    line-search stop at a point that is optimal, and sometimes stops early (seen in joint mode).
    """
    lo, hi = np.array(box, dtype=float).T
    u = np.clip(np.asarray(u0, dtype=float), lo, hi)
    for _ in range(restarts):
        u = minimize(
            f, u, jac=True, method="L-BFGS-B", bounds=box, options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 10_000}
        ).x
        fv, g = f(u)
        pg = np.where(((u <= lo) & (g > 0)) | ((u >= hi) & (g < 0)), 0.0, g)
        if np.max(np.abs(pg)) <= KKT_RTOL * (1.0 + abs(fv)):
            return u, True
    return u, False


def ub_gradients(score: PairScore, xr: np.ndarray, Wr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(∂dLogL/∂x_r, ∂dLogL/∂W_r) per gene at frozen optimal times (TH:5166-5208)."""
    xs, vs, (w1, w2, WR) = score.xs, score.vs, score.w_old
    wa = 1.0 / (vs + score.t[:, None])
    w1a, w2a, wRa = wa
    rho_a = wRa / wa.sum(axis=0)
    rho_r = WR / Wr
    x1, x2, xR = xs
    v12, v1R, v2R = (x1 - x2) ** 2, (x1 - xR) ** 2, (x2 - xR) ** 2
    da = w1a * w2a * v12 + w1a * wRa * v1R + w2a * wRa * v2R
    dr = w1 * w2 * v12 + w1 * WR * v1R + w2 * WR * v2R
    dW = (
        wRa * (1 - rho_a)
        - WR * (1 - rho_r)
        + rho_a * (w1a * w2a * v12 - (1 - rho_a) * da)
        - rho_r * (w1 * w2 * v12 - (1 - rho_r) * dr)
    ) / WR**2 + (2 * (xr - xR) / WR) * (
        rho_a * (w1a * (x1 - xR) + w2a * (x2 - xR)) - rho_r * (w1 * (x1 - xR) + w2 * (x2 - xR))
    )
    dx = 2 * (Wr / WR) * ((x1 - xR) * (w1a * rho_a - w1 * rho_r) + (x2 - xR) * (w2a * rho_a - w2 * rho_r))
    return dx, dW


def ub_slopes(score: PairScore, xr: np.ndarray, Wr: np.ndarray) -> tuple[float, float]:
    """(a, b) of UB = dLogL + a·eps_x + b·eps_W: a = ‖dx/√W_r‖, b = ‖dW·W_r‖ (TH:5200, 5211, 1522)."""
    dx, dW = ub_gradients(score, xr, Wr)
    return float(np.linalg.norm(dx / np.sqrt(Wr))), float(np.linalg.norm(dW * Wr))
