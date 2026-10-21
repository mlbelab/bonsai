"""fp64 NumPy reference: node messages, star (one parent, C children) log-likelihood terms, star EM.

Maths ref §2-§3, §9. Children arrays have shape (C, G); ``t`` has shape (C,). Units: 2x (A1).
"""

import numpy as np
from scipy.optimize import brentq, minimize


def node_message(x: np.ndarray, var: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x_n, W_n) from the children (TH:5257-5294): w_c = 1/(var_c + t_c), W = Σw, x = Σ w x / W."""
    w = 1.0 / (var + np.asarray(t, dtype=np.float64)[:, None])
    W = w.sum(axis=0)
    return (w * x).sum(axis=0) / W, W


def star_loglik_grad(
    x: np.ndarray,
    var: np.ndarray,
    t: np.ndarray,
    xn: np.ndarray | None = None,
    Wn: np.ndarray | None = None,
    w: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """ell_n and dell_n/dt_c of a star (TH:5063-5115).

    ``xn``/``Wn``/``w`` override the parent message and child weights, as upstream does when they
    are passed in (TH:5077; used by the pair score's "old" term); the gradient is only meaningful
    when they are the ones implied by ``t``. ``xn`` and ``Wn`` must be given together or not at
    all (upstream always passes ``xr_g``/``W_g`` as a pair, never one alone).
    """
    if (xn is None) != (Wn is None):
        raise ValueError("star_loglik_grad: xn and Wn must be given together, or neither")
    if w is None:
        w = 1.0 / (var + np.asarray(t, dtype=np.float64)[:, None])
    if xn is None:
        Wn = w.sum(axis=0)
        xn = (w * x).sum(axis=0) / Wn
    assert Wn is not None  # guaranteed by the pairing check above; narrows the type for mypy
    d2 = (xn - x) ** 2
    ell = float(np.sum(np.log(w).sum(axis=0) - np.log(Wn) - (w * d2).sum(axis=0)))
    grad = np.sum(w * (w * d2 - 1.0 + w / Wn), axis=1)  # TH:5107, 5114
    return ell, grad


def as_if_root(
    xn: np.ndarray, Wn: np.ndarray, t: float, xp: np.ndarray, Wp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """AIR message of a node from its own message and its parent's AIR message (TH:5313-5331)."""
    wbar = 1.0 / (t + 1.0 / Wn)
    Wx = Wp - wbar
    if np.max(Wx) < 1e-12:  # parent fully determined by this node (TH:5321)
        return xn.copy(), Wn.copy()
    xx = (xp * Wp - wbar * xn) / Wx
    wt = 1.0 / (t + 1.0 / Wx)
    W = Wn + wt
    return (xn * Wn + wt * xx) / W, W


def two_leaf_dlogl(t: float, S: np.ndarray, d2: np.ndarray) -> float:
    """f(t) = Σ_g (d² − T)/T², T = t + S (der2LeafTree, TH:4927-4930)."""
    T = t + S
    return float(np.sum((-1.0 + d2 / T) / T))


def _star_dlogl(t: float, var_c: np.ndarray, d2: np.ndarray, W: np.ndarray) -> float:
    """Upstream's derLogLikStar (TH:4815-4818): Σ_g w(w d² − 1 + 1/(T W)), T = var_c + t, w = 1/T, x̄ and W frozen."""
    T = var_c + t
    w = 1.0 / T
    return float(np.dot(w, w * d2 - 1.0 + 1.0 / (T * W)))


def _child_time(t0: float, var_c: np.ndarray, d2: np.ndarray, W: np.ndarray) -> float:
    """One child's sweep step (findOptimalTSingleCell, TH:4790-4811).

    0 if D(0) < 0 (strict, TH:4796), else upstream's ×10 bracket from max(t0, 0.01) and a tight brentq
    (upstream: xtol = rtol = 1e-4).
    """
    if _star_dlogl(0.0, var_c, d2, W) < 0:
        return 0.0
    lb, ub = 0.0, max(t0, 0.01)
    while _star_dlogl(ub, var_c, d2, W) >= 0:
        lb, ub = ub, 10.0 * ub
    return brentq(_star_dlogl, lb, ub, args=(var_c, d2, W), xtol=1e-14, rtol=4 * np.finfo(float).eps)


def star_em(
    x: np.ndarray, var: np.ndarray, tol: float = 1e-4, max_sweeps: int = 1000
) -> tuple[np.ndarray, float, bool]:
    """Star times by upstream's Jacobi fixed point (optimiseTStar, TH:4983-5027; maths ref §9).

    ``x``, ``var``: (C, G) children. Start t = 1; per sweep every child solves dell/dt_c = 0 with the
    parent message (x̄, W) frozen at the current t (findOptimalTSingleCell, TH:4790-4811): t_c = 0 if
    the derivative at 0 is < 0 (strict), else upstream's ×10 bracket from max(t_c, 0.01) and a tight
    brentq (upstream: xtol = rtol = 1e-4). Stops when max_c |Δt_c| < ``tol`` (absolute), max t ≥ 1e8,
    or after ``max_sweeps``. Returns (t (C,), ell, converged), converged iff the sweeps met ``tol``.
    Otherwise upstream's fallback: BFGS on log t from the last iterate (TH:5009; clamped children
    start at log 1e-4, upstream's own floor at TH:4823, not at log 0 = −inf), its t and ell returned
    with converged False; if BFGS fails, (ones, −1e9, False) (TH:5011-5016).
    """
    C = x.shape[0]
    t = np.ones(C)
    converged = False
    sweeps = 0
    while not converged and t.max() < 1e8 and sweeps < max_sweeps:
        sweeps += 1
        xn, W = node_message(x, var, t)
        t_new = np.array([_child_time(t[c], var[c], (xn - x[c]) ** 2, W) for c in range(C)])
        converged = bool(np.max(np.abs(t_new - t)) < tol)
        t = t_new
    if not converged:

        def neg(s):
            ell, g = star_loglik_grad(x, var, np.exp(s))
            return -ell, -np.exp(s) * g

        res = minimize(neg, np.log(np.maximum(t, 1e-4)), jac=True)
        if not res.success:
            return np.ones(C), -1e9, False
        t = np.exp(res.x)
    return t, star_loglik_grad(x, var, t)[0], converged
