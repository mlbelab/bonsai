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
    message (from ``ParentAcc``). Host float64: W_p − w̄_c cancels when one child dominates.
    Computed once per index rebuild; cast the result to float32 for ``knn``. A child whose vector
    is exactly zero stays zero (upstream divides 0/0 here and then crashes on sklearn's NaN check,
    TH:2136-2137; see docs/deviations.md).
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


def knn(points, k: int, valid=None, block: int = 1024):
    """Indices (C, k) of each point's k nearest *valid* other points by squared Euclidean distance.

    Host wrapper: ``valid`` (C,), defaulting to all points valid, must be concrete so the row
    count below can be checked eagerly. Raises ``ValueError`` unless ``0 < k < count(valid)``
    (``k == count(valid)`` would silently return an invalid row as the last neighbour). Invalid
    columns never come back as a neighbour (+inf distance); invalid rows' own result is arbitrary
    (callers ignore it). Delegates to the jitted, bucket-shaped ``_knn``.
    """
    C = points.shape[0]
    valid = np.ones(C, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    n_valid = int(valid.sum())
    if not 0 < k < n_valid:
        raise ValueError(f"k must satisfy 0 < k < count(valid), got k={k}, count(valid)={n_valid}")
    return _knn(points, k, jnp.asarray(valid), block)


@eqx.filter_jit
def _knn(points, k: int, valid, block: int):
    """Indices (C, k) of each point's k nearest other valid points by squared Euclidean distance.

    Exact up to fp32 distance rounding: every query block is compared with all points in one
    matmul (explicit HIGHEST precision, no TF32). Invalid rows are zeroed first so a padded NaN
    row cannot poison a valid row's distance; self-matches and invalid columns are then masked to
    +inf. Ties keep the lower index first (lax.top_k).
    """
    C = points.shape[0]
    points = jnp.where(valid[:, None], points, 0.0)
    n_blocks = -(-C // block)
    sq = (points * points).sum(axis=1)
    rows = jnp.arange(n_blocks * block).reshape(n_blocks, block)
    col = jnp.arange(C)

    def one(r):
        q = points[jnp.minimum(r, C - 1)]
        d = sq[jnp.minimum(r, C - 1)][:, None] + sq[None, :] - 2.0 * jnp.matmul(q, points.T, precision=MATMUL_PRECISION)
        d = jnp.where((col[None, :] == r[:, None]) | ~valid[None, :], jnp.inf, d)
        return lax.top_k(-d, k)[1]

    return lax.map(one, rows).reshape(-1, k)[:C]


def knn_query(points, valid, q, k: int) -> jnp.ndarray:
    """Indices of the ``k`` valid rows of ``points`` nearest to a new vector ``q`` (G,).

    Exact (one masked matvec, no blocking): used to place a freshly merged node into an existing
    kNN buffer without rebuilding it (A19). The caller queries with the new node's own buffer row
    still marked invalid, then flips that row to valid afterward; ``q`` itself is never excluded
    by this function, so a query against a buffer whose own row is already valid can return that
    row (a zero-distance self-match) rather than its intended neighbours — the A19 ordering
    (query, then validate) is what keeps this from happening, not this function. ``k`` may equal
    ``count(valid)``. Ties keep the lower index first (lax.top_k).
    """
    valid_host = np.asarray(valid, dtype=bool)  # count on the host array: no host->device->host
    n_valid = int(valid_host.sum())
    if not 0 < k <= n_valid:
        raise ValueError(f"k must satisfy 0 < k <= count(valid), got k={k}, count(valid)={n_valid}")
    return _knn_query(points, jnp.asarray(valid_host), q, k)


@eqx.filter_jit
def _knn_query(points, valid, q, k: int):
    d = ((points - q[None, :]) ** 2).sum(axis=1)
    d = jnp.where(valid, d, jnp.inf)
    return lax.top_k(-d, k)[1]
