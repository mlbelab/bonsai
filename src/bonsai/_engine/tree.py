"""Whole-tree log-likelihood and gradient by level-wise post-/pre-order passes (maths ref §4).

Everything runs in the compute dtype of the tree (float32 by default, no x64; amendment A8):
- sums over a node's children are sums of positive weights (and of w·x), with no subtraction;
- the pre-order pass never forms W_A − w_c (the cancellation of maths ref §2): the message of
  "everything but child c" is built from exclusive prefix and suffix sums over c's siblings plus
  the message from above the parent, again sums only;
- sums over genes use the compensated ``dsum``; the total L_tree is summed over nodes in float64
  on the host (``TreeEval.L``);
- the per-edge terms of ell_n are added over a node's children by a plain scatter-add
  (``ell.at[].add``), not ``dsum``: its (C − 1)·eps error fits the per-term budget of the tests
  while C ≲ 10; nodes with many children take the host-accumulator path (M2).
On GPUs the ``.at[].add`` scatters accumulate in a nondeterministic order, so results may differ
in the last bits between runs (tests use rounding bounds, never bitwise equality).
Cost per pass: levels × width × G. Nodes with very many children (a star root) belong to the
host-accumulator path of the greedy engine (M2), not here.
"""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float

from .array_tree import ArrayTree
from .numerics import dsum


class TreeEval(NamedTuple):
    ell: Float[Array, " n"]  # per-node term ell_n (0 at leaves and padding)
    grad: Float[Array, " n"]  # dL/dt of each node's parent edge (0 at root/padding)
    x: Float[Array, "n G"]  # downward messages (centred)
    W: Float[Array, "n G"]
    x_air: Float[Array, "n G"]  # as-if-root messages (centred)
    W_air: Float[Array, "n G"]

    @property
    def L(self) -> float:
        """L_tree (A1 convention): Σ_n ell_n summed in float64 on the host."""
        return float(np.sum(np.asarray(self.ell, dtype=np.float64)))


def _window(tree: ArrayTree, child, off, first, k):
    """Edges of level k: child indices, valid mask, masked-safe parent indices, segment starts."""
    c = lax.dynamic_slice_in_dim(child, off[k], tree.width)
    m = jnp.arange(tree.width) < off[k + 1] - off[k]
    f = lax.dynamic_slice_in_dim(first, off[k], tree.width)
    return c, m, jnp.where(m, tree.parent[c], 0), f


def _excl_scan(v, first):
    """Exclusive prefix sums of v (width, G) within segments that start where ``first`` is True."""

    def op(a, b):
        return a[0] | b[0], jnp.where(b[0][..., None], b[1], a[1] + b[1])

    incl = lax.associative_scan(op, (first, v))[1]
    return jnp.where(first[:, None], 0.0, jnp.roll(incl, 1, axis=0))


def _excl_siblings(v, first, m):
    """Σ over the other members of each segment (prefix + suffix, never total − own)."""
    v = jnp.where(m[:, None], v, 0.0)
    last = jnp.roll(first, -1).at[-1].set(True)
    return _excl_scan(v, first) + _excl_scan(v[::-1], last[::-1])[::-1]


@eqx.filter_jit
def tree_loglik_grad(tree: ArrayTree) -> TreeEval:
    """Per-node ell_n, dL/dt of every edge and the node messages, in the tree's dtype."""
    dt = tree.x.dtype
    n, G = tree.x.shape
    drop = n  # out-of-range scatter index: masked lanes write nothing (mode="drop")
    leaf = tree.is_leaf[:, None]

    def up(k, carry):
        S_W, S_X, ell = carry
        c, m, p, first = _window(tree, tree.up_child, tree.up_off, tree.up_first, k)
        W_safe = jnp.where(leaf[c], 1.0, S_W[c])
        xc = jnp.where(leaf[c], tree.x[c], S_X[c] / W_safe)
        vc = jnp.where(leaf[c], tree.var[c], 1.0 / W_safe)
        w = jnp.where(m[:, None], 1.0 / (vc + tree.t[c][:, None]), 1.0)  # w_c = 1/(sigma²_c + t_c)
        tgt = jnp.where(m, p, drop)
        S_W = S_W.at[tgt].add(jnp.where(m[:, None], w, 0.0), mode="drop")
        S_X = S_X.at[tgt].add(jnp.where(m[:, None], w * xc, 0.0), mode="drop")
        Wp = jnp.where(m[:, None], S_W[p], 1.0)  # parents of level k are complete now
        r = S_X[p] / Wp - xc
        # ell_n per gene (TH:5093-5105); −log W_p is added once per parent, on its first edge
        term = jnp.log(w) - w * r * r - jnp.where(first[:, None], jnp.log(Wp), 0.0)
        return S_W, S_X, ell.at[tgt].add(dsum(jnp.where(m[:, None], term, 0.0)), mode="drop")

    zeros = jnp.zeros((n, G), dt)
    S_W, S_X, ell = lax.fori_loop(0, tree.n_up, up, (zeros, zeros, jnp.zeros(n, dt)))
    W = jnp.where(leaf, 1.0 / tree.var, jnp.where(S_W > 0, S_W, 1.0))
    X = jnp.where(leaf, tree.x, S_X / W)
    V = jnp.where(leaf, tree.var, 1.0 / W)

    def down(k, carry):
        wt, xo, g = carry  # wt[p], xo[p]: weight and mean of everything outside p's subtree, seen from p
        c, m, p, first = _window(tree, tree.dn_child, tree.dn_off, tree.dn_first, k)
        w = 1.0 / (V[c] + tree.t[c][:, None])  # the same w_c the post-order pass summed
        W_ex = _excl_siblings(w, first, m) + wt[p]  # W_{\c}: siblings + above, no subtraction
        x_ex = (_excl_siblings(w * X[c], first, m) + wt[p] * xo[p]) / jnp.where(m[:, None], W_ex, 1.0)
        T = tree.t[c][:, None] + V[c] + 1.0 / W_ex
        d = x_ex - X[c]
        gc = dsum(jnp.where(m[:, None], (-1.0 + d * d / T) / T, 0.0))  # der2LeafTree, TH:607-609
        tgt = jnp.where(m, c, drop)
        wt = wt.at[tgt].set(1.0 / (tree.t[c][:, None] + 1.0 / W_ex), mode="drop")  # w̃ (TH:613)
        return wt, xo.at[tgt].set(x_ex, mode="drop"), g.at[tgt].set(gc, mode="drop")

    wt, xo, g = lax.fori_loop(0, tree.n_dn, down, (zeros, zeros, jnp.zeros(n, dt)))
    W_air = W + wt
    return TreeEval(ell, g, X, W, (X * W + wt * xo) / W_air, W_air)
