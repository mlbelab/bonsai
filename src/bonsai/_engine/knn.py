"""Exact blocked k-nearest neighbours in upstream's metric (maths ref §8 "kNN candidates")."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax

from .numerics import MATMUL_PRECISION

MIN_TIME = 1e-4  # children are kept at least this far from the parent (TH:2120)


def knn_coords(x, W, t, xp, Wp) -> np.ndarray:
    """Upstream's kNN coordinates of C children (getNNPairs, TH:2115-2131; TH:868-888), float64 host.

    Each child's as-if-root message with t ← max(t, 1e-4) (getLtqsAsIfRoot_vectorized, TH:5335-5349,
    no 1e-12 guard), minus the parent's AIR mean, scaled to unit L2 norm per child (squared
    Euclidean = 2 − 2·cosine). x, W: (C, G) children; t: (C,); xp, Wp: (G,) the parent's AIR
    message (from ``ParentAcc``). Host float64 (A8): W_p − w̄_c cancels when one child dominates.
    Computed once per index rebuild; cast the result to float32 for ``knn``. A child whose vector
    is exactly zero stays zero (upstream divides 0/0 here and then crashes on sklearn's NaN check,
    TH:2133-2134; spec amendment A13).
    """
    x, W = np.asarray(x, dtype=np.float64), np.asarray(W, dtype=np.float64)
    tt = np.maximum(np.asarray(t, dtype=np.float64), MIN_TIME)[:, None]
    wbar = 1.0 / (tt + 1.0 / W)
    Wx = Wp - wbar
    xx = (xp * Wp - wbar * x) / Wx
    wt = 1.0 / (tt + 1.0 / Wx)
    v = (x * W + wt * xx) / (W + wt) - xp
    norm = np.sqrt((v * v).sum(axis=1, keepdims=True))
    return np.where(norm > 0, v / np.where(norm > 0, norm, 1.0), 0.0)


@eqx.filter_jit
def knn(points, k: int, block: int = 1024):
    """Indices (C, k) of each point's k nearest other points by squared Euclidean distance.

    Exact: every query block is compared with all points in one matmul (explicit HIGHEST
    precision, no TF32). Self-matches are excluded; ties keep the lower index first (lax.top_k).
    """
    C = points.shape[0]
    if not 0 < k < C:
        raise ValueError(f"k must satisfy 0 < k < C, got k={k}, C={C}")
    n_blocks = -(-C // block)
    sq = (points * points).sum(axis=1)
    rows = jnp.arange(n_blocks * block).reshape(n_blocks, block)
    col = jnp.arange(C)

    def one(r):
        q = points[jnp.minimum(r, C - 1)]
        d = sq[jnp.minimum(r, C - 1)][:, None] + sq[None, :] - 2.0 * jnp.matmul(q, points.T, precision=MATMUL_PRECISION)
        d = jnp.where(col[None, :] == r[:, None], jnp.inf, d)
        return lax.top_k(-d, k)[1]

    return lax.map(one, rows).reshape(-1, k)[:C]
