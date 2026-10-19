"""Batched star kernels: child weights, node messages and per-node ell/gradient (maths ref §2-§3).

Shapes: S stars × C children (padded, ``mask``) × G genes, all in the compute dtype (A8). Sums
over children and genes use the compensated ``dsum``. A parent whose message must survive many
merges keeps its Σw, Σw·x on the host (``host_acc.ParentAcc``), fed with ``child_weights``.
"""

import equinox as eqx
import jax.numpy as jnp

from .numerics import dsum


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
