"""Whole-tree log-likelihood and gradient by block-scheduled post-/pre-order passes (maths ref §4, A14).

Everything runs in the compute dtype of the tree (float32 by default, no x64; amendment A8):
- sums over a node's children are sums of positive weights (and of w·x), with no subtraction;
- the pre-order pass never forms W_A − w_c (the cancellation of maths ref §2): the message of
  "everything but child c" is built from exclusive prefix and suffix sums over c's siblings plus
  the message from above the parent, again sums only. A parent's children may span several
  blocks: each block adds the partial sums of its first sibling group over earlier blocks (a loop
  carry) and of its last group over later blocks (``tails``, one backward loop before the
  pre-order loop), each a disjoint partial sum, so no subtraction either;
- the per-edge terms of ell_n need the parent's final message, so the pre-order loop adds them;
- sums over genes use the compensated ``dsum``; the total L_tree is summed over nodes in float64
  on the host (``TreeEval.L``);
- the per-edge terms of ell_n are added over a node's children by a plain scatter-add
  (``ell.at[].add``), not ``dsum``: its (C − 1)·eps error fits the per-term budget of the tests
  while C ≲ 10; nodes with many children take the host-accumulator path (M2).
On GPUs the ``.at[].add`` scatters accumulate in a nondeterministic order, so results may differ
in the last bits between runs (tests use rounding bounds, never bitwise equality).
Cost per pass: (blocks × B) × G ≤ (n + levels·B)·G, plus B × G in ``tails`` for each block that
starts mid-group (only where a sibling group crosses a block boundary). Nodes with very many
children (a star root) belong to the host-accumulator path of the greedy engine (M2), not here.
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


def _window(tree: ArrayTree, child, off, j):
    """Block j: start, child indices, valid mask and masked-safe parent indices."""
    s = off[j]
    c = lax.dynamic_slice_in_dim(child, s, tree.B)
    m = jnp.arange(tree.B) < off[j + 1] - s
    return s, c, m, jnp.where(m, tree.parent[c], 0)


def _excl_scan(v, first):
    """Exclusive prefix sums of v (B, G) within segments that start where ``first`` is True."""

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
    n, G = tree.parent.shape[0], tree.x.shape[1]
    drop = n  # out-of-range scatter index: masked lanes write nothing (mode="drop")
    leaf = tree.is_leaf[:, None]

    def up(j, carry):
        S_W, S_X = carry
        _, c, m, p = _window(tree, tree.up_child, tree.up_off, j)
        W_safe = jnp.where(leaf[c], 1.0, S_W[c])
        xc = jnp.where(leaf[c], tree.x[tree.leaf_row[c]], S_X[c] / W_safe)
        vc = jnp.where(leaf[c], tree.var[tree.leaf_row[c]], 1.0 / W_safe)
        w = 1.0 / (vc + tree.t[c][:, None])  # w_c = 1/(sigma²_c + t_c)
        tgt = jnp.where(m, p, drop)
        return S_W.at[tgt].add(w, mode="drop"), S_X.at[tgt].add(w * xc, mode="drop")

    zeros = jnp.zeros((n, G), dt)
    S_W, S_X = lax.fori_loop(0, tree.n_up, up, (zeros, zeros))
    W = jnp.where(leaf, 1.0 / tree.var[tree.leaf_row], jnp.where(S_W > 0, S_W, 1.0))
    X = jnp.where(leaf, tree.x[tree.leaf_row], S_X / W)

    def edges(j):
        """Pre-order block j: start, children, mask, parents, segment starts, sigma²_c, w_c, w_c·x_c."""
        s, c, m, p = _window(tree, tree.dn_child, tree.dn_off, j)
        f = lax.dynamic_slice_in_dim(tree.dn_first, s, tree.B) & m
        Vc = jnp.where(leaf[c], tree.var[tree.leaf_row[c]], 1.0 / W[c])  # node variance, as summed above
        w = 1.0 / (Vc + tree.t[c][:, None])  # the same w_c the post-order pass summed
        return s, c, m, p, f, Vc, w, w * X[c]

    def tails(k, T):  # T[j]: Σ over blocks > j of the sibling group still open at block j's end
        j = tree.n_dn - 1 - k

        def add_tail(T):
            _, _, m, _, f, _, w, wx = edges(j)
            head = ((jnp.cumsum(f) == 0) & m)[:, None]  # block j's leading part of that group
            cont = ~jnp.any(f)  # block j is all one group, continued from block j − 1
            add = lambda v, Tj: jnp.sum(jnp.where(head, v, 0.0), axis=0) + jnp.where(cont, Tj, 0.0)
            return T[0].at[j - 1].set(add(w, T[0][j])), T[1].at[j - 1].set(add(wx, T[1][j]))

        # only a block that starts mid-group has a head; otherwise T[j − 1] keeps its initial 0
        return lax.cond(tree.dn_first[tree.dn_off[j]], lambda T: T, add_tail, T)

    nb = tree.dn_off.shape[0]
    T_W, T_X = lax.fori_loop(0, tree.n_dn - 1, tails, (jnp.zeros((nb, G), dt), jnp.zeros((nb, G), dt)))

    def down(j, carry):
        wt, xo, g, ell, pre_W, pre_X = carry  # wt[p], xo[p]: weight and mean outside p's subtree, seen from p
        s, c, m, p, f, Vc, w, wx = edges(j)
        fb = f.at[0].set(True)  # the block start opens a group too
        seg = jnp.cumsum(fb)
        head = ((jnp.cumsum(f) == 0) & m)[:, None]  # group continued from block j − 1
        last = ((seg == seg[tree.dn_off[j + 1] - s - 1]) & m)[:, None]  # group open at block end
        cont = ~jnp.any(f)

        def excl(v, pre, tail):  # Σ over the other members of each sibling group, in any block
            return _excl_siblings(v, fb, m) + jnp.where(head, pre, 0.0) + jnp.where(last, tail, 0.0)

        nxt = lambda v, pre: jnp.sum(jnp.where(last, v, 0.0), axis=0) + jnp.where(cont, pre, 0.0)
        W_ex = excl(w, pre_W, T_W[j]) + wt[p]  # W_{\c}: siblings + above, no subtraction
        x_ex = (excl(wx, pre_X, T_X[j]) + wt[p] * xo[p]) / jnp.where(m[:, None], W_ex, 1.0)
        T = tree.t[c][:, None] + Vc + 1.0 / W_ex
        d = x_ex - X[c]
        gc = dsum(jnp.where(m[:, None], (-1.0 + d * d / T) / T, 0.0))  # der2LeafTree, TH:607-609
        r = X[p] - X[c]
        # ell_n per gene (TH:5093-5105); −log W_p is added once per parent, on its first edge
        term = jnp.log(w) - w * r * r - jnp.where(f[:, None], jnp.log(W[p]), 0.0)
        tgt, tp = jnp.where(m, c, drop), jnp.where(m, p, drop)
        wt = wt.at[tgt].set(1.0 / (tree.t[c][:, None] + 1.0 / W_ex), mode="drop")  # w̃ (TH:613)
        ell = ell.at[tp].add(dsum(jnp.where(m[:, None], term, 0.0)), mode="drop")
        g = g.at[tgt].set(gc, mode="drop")
        return wt, xo.at[tgt].set(x_ex, mode="drop"), g, ell, nxt(w, pre_W), nxt(wx, pre_X)

    zG, zn = jnp.zeros(G, dt), jnp.zeros(n, dt)
    wt, xo, g, ell, _, _ = lax.fori_loop(0, tree.n_dn, down, (zeros, zeros, zn, zn, zG, zG))
    W_air = W + wt
    return TreeEval(ell, g, X, W, (X * W + wt * xo) / W_air, W_air)
