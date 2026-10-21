"""Batched star kernels: child weights, node messages, per-node ell/gradient, star EM (maths ref §2-§3, §9).

Shapes: S stars × C children (padded, ``mask``) × G genes, all in the compute dtype (A8). Sums
over children and genes use the compensated ``dsum``. A parent whose message must survive many
merges keeps its Σw, Σw·x on the host (``host_acc.ParentAcc``), fed with ``child_weights``.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from .numerics import dsum

STAR_T_MAX = 1e8  # sweeps stop once max t reaches this (TH:4992)
# upper brackets u0·10^k, k = 0..12, u0 = max(t_c, 0.01): 10^12·0.01 = 1e10 > STAR_T_MAX, so a child
# whose bracket is not found ends the sweeps through STAR_T_MAX (upstream expands ×10 unboundedly)
N_STAR_BRACKET = 13
N_STAR_ROOT = 64  # bisection steps: the bracket shrinks to ub·2^-64 ≈ 5e-20·ub, below eps32 and eps64 relative


@eqx.filter_jit
def child_weights(var, t):
    """w_c = 1/(var_c + t_c) (TH:5257-5294), the exact values a host accumulator adds and removes."""
    return 1.0 / (var + t[..., None])


@eqx.filter_jit
def node_message(x, var, t, mask):
    """(x_n, W_n) of each star from its children (TH:5257-5294). Padded children may hold any x."""
    m = mask[..., None]
    w = jnp.where(m, child_weights(var, t), 0.0)
    W = dsum(w, axis=1)
    return dsum(jnp.where(m, w * x, 0.0), axis=1) / W, W


@eqx.filter_jit
def star_loglik_grad(x, var, t, mask):
    """ell_n (S,) and dell_n/dt_c (S, C) (TH:5063-5115). Padded children: zero gradient."""
    m = mask[..., None]
    w = jnp.where(m, child_weights(var, t), 1.0)
    W = dsum(jnp.where(m, w, 0.0), axis=1)
    xn = dsum(jnp.where(m, w * x, 0.0), axis=1) / W
    r = xn[:, None, :] - x  # residuals, never Σw x² − W x² (maths ref §3)
    per = jnp.where(m, jnp.log(w) - w * r * r, 0.0)
    ell = dsum(dsum(per, axis=1), axis=-1) - dsum(jnp.log(W), axis=-1)
    grad = dsum(jnp.where(m, w * (w * r * r - 1.0 + w / W[:, None, :]), 0.0), axis=-1)
    return ell, grad


class StarEM(NamedTuple):
    t: Float[Array, " C"]  # star times; padded children 0
    sweeps: Int[Array, ""]  # Jacobi sweeps run
    converged: Bool[Array, ""]  # the last sweep changed every t by < tol (absolute)


def _child_time(t0, v, d2, W):
    """Root of derLogLikStar in t (findOptimalTSingleCell, TH:4790-4811), x̄ and W frozen.

    D(t) = Σ_g w(w d² − 1 + w/W), w = 1/(v + t): t = 0 if D(0) < 0 (strict), else the first upper
    bracket u0·10^k with D < 0 and bisection on it; no bracket: the last one (≥ 1e10).
    """
    dt = v.dtype

    def f(t):
        w = 1.0 / (v + t)
        return dsum(w * (w * d2 - 1.0 + w / W))

    ubs = jnp.maximum(t0, 0.01) * (10.0 ** jnp.arange(N_STAR_BRACKET)).astype(dt)
    neg = jax.vmap(f)(ubs) < 0
    k = jnp.argmax(neg)
    lb0, ub0 = jnp.where(k == 0, 0.0, ubs[k - 1]).astype(dt), ubs[k]

    def body(_, st):
        lo, hi = st
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        return jnp.where(pos, mid, lo), jnp.where(pos, hi, mid)

    lo, hi = lax.fori_loop(0, N_STAR_ROOT, body, (lb0, ub0))
    t = jnp.where(jnp.any(neg), 0.5 * (lo + hi), ubs[-1])
    return jnp.where(f(jnp.zeros((), dt)) < 0, 0.0, t).astype(dt)


@eqx.filter_jit
def star_em(x, var, mask, tol=1e-4, max_sweeps=1000) -> StarEM:
    """Star times by upstream's Jacobi fixed point (optimiseTStar, TH:4983-5027; maths ref §9).

    ``x``, ``var``: (C, G) children in the compute dtype (centred, as ``pair_score``); ``mask`` (C,)
    marks valid children (padded rows may hold NaN). Start t = 1; each sweep solves every child's
    dell/dt_c = 0 with (x̄, W) of the valid children frozen (``_child_time``), in a ``lax.while_loop``
    while max|Δt| ≥ ``tol``, max t < 1e8 and sweeps < ``max_sweeps``. ``converged`` iff the first
    condition ended it; otherwise the caller falls back to the fp64 reference (``_ref.star.star_em``,
    upstream's BFGS fallback, TH:5005-5016). ``tol``, ``max_sweeps``: static.
    """
    dt = x.dtype

    def sweep(t):
        xn, W = node_message(x[None], var[None], t[None], mask[None])
        d2 = (xn - x) ** 2
        tn = jax.vmap(_child_time, in_axes=(0, 0, 0, None))(t, var, d2, W[0])
        return jnp.where(mask, tn, 0.0)

    def cond(st):
        t, dmax, k = st
        return (dmax >= tol) & (jnp.max(t) < STAR_T_MAX) & (k < max_sweeps)

    def body(st):
        t, _, k = st
        tn = sweep(t)
        return tn, jnp.max(jnp.abs(tn - t)), k + 1

    init = (jnp.where(mask, 1.0, 0.0).astype(dt), jnp.asarray(jnp.inf, dt), jnp.zeros((), jnp.int32))
    t, dmax, k = lax.while_loop(cond, body, init)
    return StarEM(t, k, dmax < tol)
