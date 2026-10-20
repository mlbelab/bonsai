"""fp64 NumPy reference: node messages and star (one parent, C children) log-likelihood terms.

Maths ref §2-§3. Children arrays have shape (C, G); ``t`` has shape (C,). Units: 2x (A1).
"""

import numpy as np


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
