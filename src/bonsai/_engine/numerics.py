"""Precision policy of the JAX engine (spec §1 "Precision", amendment A8).

Device kernels run in the dtype of their inputs (float32 by default) in the default, x64-off
process. Long accumulations that must be float64 (a parent's Σw and Σw·x over many children, the
total log-likelihood) live on the host as float64 NumPy arrays (arriving in Task 12). Reductions
over genes or children on the device use :func:`dsum`, a compensated (double-float) pairwise sum,
so their error does not grow with the number of terms. Matmuls pass ``precision=MATMUL_PRECISION``:
JAX lowers float32 matmuls to TF32 on GPUs by default, which is not float32 (pattern from rose
``rose.core.numerics``, internal).
"""

import jax
import jax.numpy as jnp

MATMUL_PRECISION = jax.lax.Precision.HIGHEST


def _two_sum(a, b):
    """A + b = s + e exactly (Knuth's TwoSum; needs IEEE arithmetic without reassociation).

    ``e`` is forced to 0 where ``s`` is non-finite: inf − inf (overflow or an infinite input) would
    otherwise turn a legitimate ±inf result into NaN. NaN inputs still propagate to NaN through ``s``.
    """
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, jnp.where(jnp.isfinite(s), e, 0)


def dsum(a, axis: int = -1):
    """Sum along ``axis`` in the dtype of ``a`` with double-float compensation.

    Pairwise tree of TwoSum steps carrying (hi, lo): the result is the exact sum rounded once,
    order-independent up to O(eps²)·log2(n)·Σ|a| (fixed pairwise tree), in place of the
    (n−1)·eps·Σ|a| bound of a plain recursive sum. Differentiable (the error terms have zero
    derivative). Overflow or an infinite input gives ±inf (not NaN); NaN inputs, or opposite-signed
    infinities, give NaN, matching ``jnp.sum``.
    """
    a = jnp.moveaxis(a, axis, -1)
    n = a.shape[-1]
    m = 1 << max(0, n - 1).bit_length()
    hi = jnp.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, m - n)])
    lo = jnp.zeros_like(hi)
    while hi.shape[-1] > 1:
        s, e = _two_sum(hi[..., 0::2], hi[..., 1::2])
        hi, lo = _two_sum(s, e + lo[..., 0::2] + lo[..., 1::2])
    return hi[..., 0] + lo[..., 0]
