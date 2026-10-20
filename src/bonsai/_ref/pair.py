"""fp64 NumPy reference: 2-leaf time, 3-leaf pair score (sequential/joint, A2), upper-bound slopes.

Maths ref §5-§7. G-vectors are 1-D arrays; times and log-likelihoods in 2x units (A1). The
optimisers here are tight: they compute the exact optimum of upstream's objective (to fp64
resolution), and upstream's solves (tol 1e-4 when scoring, 1e-9 for merges) sit within their own
stopping gap of it.
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq, minimize

from .star import node_message, star_loglik_grad, two_leaf_dlogl  # two_leaf_dlogl re-exported

PAIR_LOG_T_MIN, PAIR_LOG_T_MAX = -16.1180, 10.0  # TH:4824, 4864 (t in [1.0e-7, 2.2e4])
EPS64 = float(np.finfo(np.float64).eps)
# Reference stopping rule: every free coordinate has |g_i| <= (KKT_EPS_FACTOR + G)·eps64·Σ_g|terms of g_i|
# (``_scales``), i.e. its gradient is zero to what fp64 resolves: <= 8 rounded ops per gene term, +1 for
# the chain rule (g_1 − g_2 or t·g), <= G − 1 for the gene sum, and <= 6 for t = fl(exp(u)) and
# t_2a = fl(t12 − t_1a) (a relative eps change of a time moves each term, a product of <= 3 weights with
# |dlog w/dlog t| <= 1, by <= 3·eps·|term|). G + 14 <= G + 16.
KKT_EPS_FACTOR = 16


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
    """CalcTInit (HE:788-794): (sequential [t12, tau], joint [t1 − tau, t2 − tau, tau]), tau = min/2."""
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


def _scales(tt, xs, vs):
    """(Σ|terms| of ell, Σ_g|terms| of dell/dt_c per time): fp64 rounding scales (TH:5093-5114).

    The scales of ``star_loglik_grad``'s terms. r = x_n − x_c is formed from numbers of size |x|, so its
    rounding moves w·r² by up to 2·w·r·eps·(|x_n| + |x_c|): budgeted as a cross term.
    """
    w = 1.0 / (vs + tt[:, None])
    xn, Wn = node_message(xs, vs, tt)
    r = np.abs(xn - xs)
    wd2 = w * (r**2 + 2 * r * (np.abs(xn) + np.abs(xs)))
    fs = float(np.sum(np.abs(np.log(w)) + wd2) + np.sum(np.abs(np.log(Wn))))
    return fs, np.sum(w * (wd2 + 1.0 + w / Wn), axis=1)


def seq_objective(u, t12, xs, vs, scales: bool = False):
    """(−ell, gradient) of the 3-star in u = (t_1a, log t_aR), t_2a = t12 − t_1a (TH:5219-5226).

    ``scales``: also (value scale, per-coordinate gradient scale) of ``_scales``, chain rule to u.
    """
    tt = np.array([u[0], t12 - u[0], np.exp(u[1])])
    fv, g = _neg_ell(tt, xs, vs)
    out = fv, np.array([g[0] - g[1], tt[2] * g[2]])
    if not scales:
        return out
    fs, gs = _scales(tt, xs, vs)
    return *out, fs, np.array([gs[0] + gs[1], tt[2] * gs[2]])


def joint_objective(s, xs, vs, scales: bool = False):
    """(−ell, gradient) of the 3-star in s = log t (TH:5241-5246); ``scales`` as in ``seq_objective``."""
    tt = np.exp(s)
    fv, g = _neg_ell(tt, xs, vs)
    if not scales:
        return fv, tt * g
    fs, gs = _scales(tt, xs, vs)
    return fv, tt * g, fs, tt * gs


def pair_score(x1, v1, t1, x2, v2, t2, xr, Wr, mode: str = "sequential") -> PairScore:
    """3-leaf score of merging children 1, 2 of a parent with AIR message (xr, Wr) (TH:5362-5408).

    Sequential falls back to joint when t_12 has no bracket or when ``_minimise_box`` fails its
    resolution-scaled KKT test after all restarts; upstream falls back on scipy's ``success=False``
    at its loose tolerance (docs/deviations.md). Raises ValueError when the parent has no third child (W_R not positive).
    """
    w1, w2 = 1.0 / (t1 + v1), 1.0 / (t2 + v2)
    if np.any((Wr - w1) - w2 <= 4 * np.finfo(float).eps * Wr):  # 2-child parent: W_R is rounding noise, ≤ eps·W_r
        raise ValueError("pair_score: the parent needs at least 3 children (W_r − w_1 − w_2 is not positive)")
    xR, WR = rest_of_parent(xr, Wr, x1, w1, x2, w2)
    xs, vs = np.stack([x1, x2, xR]), np.stack([v1, v2, 1.0 / WR])
    seq0, joint0 = init_times(t1, t2)
    rtol = (KKT_EPS_FACTOR + xs.shape[1]) * EPS64
    times, ok = np.array(joint0), False
    if mode == "sequential":
        t12, ok = two_leaf_time(x1, v1, x2, v2)
        if ok and t12 is not None:
            u, ok = _minimise_box(
                lambda u, sc=False: seq_objective(u, t12, xs, vs, sc),
                [0.5 * t12, np.log(max(seq0[1], 1e-4))],
                [(0.0, t12), (PAIR_LOG_T_MIN, PAIR_LOG_T_MAX)],
                rtol,
            )
            times = np.array([u[0], t12 - u[0], np.exp(u[1])])
    elif mode != "joint":
        raise ValueError(f"mode must be 'sequential' or 'joint', got {mode!r}")
    if not ok:  # joint mode, or the sequential fallback (TH:4822-4831)
        s, ok = _minimise_box(
            lambda s, sc=False: joint_objective(s, xs, vs, sc),
            np.log(np.maximum(joint0, 1e-4)),
            [(PAIR_LOG_T_MIN, PAIR_LOG_T_MAX)] * 3,
            rtol,
        )
        times = np.exp(s)
    if not ok:
        return PairScore(-np.inf, np.array(joint0), False, xs, vs, np.stack([w1, w2, WR]))
    ell_new, _ = star_loglik_grad(xs, vs, times)
    w_old = np.stack([w1, w2, WR])
    ell_old, _ = star_loglik_grad(xs, vs, times, xn=xr, Wn=Wr, w=w_old)  # TH:5400-5402; times unused (w given)
    return PairScore(ell_new - ell_old, times, True, xs, vs, w_old)


def _kkt(f, u, lo, hi):
    """(r, f, value scale, gradient, free mask): r = max over free coordinates of |g_i|/gscale_i.

    A coordinate on a bound with an outward gradient, or in a zero-width box (t_12 = 0), is not free.
    """
    fv, g, fs, gs = f(u, True)
    # (hi > lo) is defensive: a zero-width box (t_12 = 0) already has u on both bounds, so only an
    # exactly zero gradient could leave it free, and the polish would then divide by h = 0
    free = (hi > lo) & ~(((u <= lo) & (g > 0)) | ((u >= hi) & (g < 0)))
    return float(np.max(np.where(free, np.abs(g) / gs, 0.0))), fv, fs, g, free


def _polish(f, u, lo, hi, rtol, iters: int = 20):
    """Projected Newton on the free coordinates with a forward-FD Hessian of the analytic gradient.

    L-BFGS-B stops where f's decreases fall below its rounding (ftol), which can leave a gradient
    far above its own rounding, and it crawls along a log t tail heading for PAIR_LOG_T_MIN (gradient
    and curvature ∝ t). The gradient stays resolved there, so a step is kept when it lowers r
    without raising f beyond the rounding of two evaluations, 2·rtol·fscale.
    """
    r, fv, fs, g, free = _kkt(f, u, lo, hi)
    for _ in range(iters):
        if r <= rtol:
            break
        idx = np.flatnonzero(free)
        H = np.empty((len(idx), len(idx)))
        for a, i in enumerate(idx):
            h = min(1e-5 * max(1.0, abs(u[i])), 0.5 * (hi[i] - lo[i]))
            e = np.zeros_like(u)
            e[i] = h if u[i] - lo[i] < hi[i] - u[i] else -h  # step into the box
            H[:, a] = (f(u + e)[1][idx] - g[idx]) / e[i]
        H = 0.5 * (H + H.T)
        try:
            np.linalg.cholesky(H)  # not positive definite: no Newton step (the solve has failed here)
        except np.linalg.LinAlgError:
            break
        un = u.copy()
        un[idx] = np.clip(u[idx] - np.linalg.solve(H, g[idx]), lo[idx], hi[idx])
        rn, fn, fsn, gn, freen = _kkt(f, un, lo, hi)
        if rn >= r or fn > fv + 2 * rtol * max(fs, fsn):
            break
        u, r, fv, fs, g, free = un, rn, fn, fsn, gn, freen
    return u, r


def _minimise_box(f, u0, box, rtol, restarts: int = 5):
    """Tight L-BFGS-B, then ``_polish``, restarted until the resolution-scaled KKT test r <= rtol passes.

    ``f(u, scales)`` as ``seq_objective``. scipy's own ``success`` flag is not used: with ftol at
    machine precision it often reports a line-search stop at a point that is optimal.
    """
    lo, hi = np.array(box, dtype=float).T
    u = np.clip(np.asarray(u0, dtype=float), lo, hi)
    for _ in range(restarts):
        u = minimize(
            f, u, jac=True, method="L-BFGS-B", bounds=box, options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 10_000}
        ).x
        u, r = _polish(f, u, lo, hi, rtol)
        if r <= rtol:
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
