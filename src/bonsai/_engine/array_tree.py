"""``ArrayTree``: a rooted tree with leaf data as fixed-shape arrays (spec §1 "State")."""

import warnings

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from bonsai._ref.newick import parse_newick
from bonsai._ref.preprocess import preprocess
from bonsai._ref.tree import postorder

# Default block width B (A14): a pass schedules <= n + levels·B rows in n/B + levels loop steps; 256
# keeps levels·B small next to n on large trees while bounding the number of steps.
BLOCK = 256


def _bucket(n: int) -> int:
    """Next power of two >= n (bounds the number of compiled shapes)."""
    return 1 << max(0, int(n) - 1).bit_length()


def _blocks(level_off: np.ndarray, B: int) -> np.ndarray:
    """Block boundaries: each level ``level_off[k]:level_off[k+1]`` cut into runs of at most B."""
    starts = [np.arange(a, b, B) for a, b in zip(level_off[:-1], level_off[1:], strict=True)]
    return np.r_[np.concatenate(starts), level_off[-1]]


class ArrayTree(eqx.Module):
    """Topology, branch lengths and leaf data, padded to power-of-two shapes.

    Node rows ``>= n_nodes`` are padding: never referenced by an edge, zero gradient. ``x``/``var``
    hold leaf data only, (bucket(N), G); ``leaf_row[v]`` is leaf v's row (0 for other nodes). ``x``
    is the leaf means minus the per-gene centre (L_tree is invariant under a per-gene shift, flat
    root prior; centring keeps float32 residuals x_p − x_c accurate). The centre is kept exactly
    as a float32 pair ``centre[0] + centre[1]`` (no float64 on the device, A8; see ``centre64``).
    Block schedules (A14): ``up_child`` lists edges (by child) ordered by the parent's height
    (post-order levels), ``dn_child`` by the parent's depth (pre-order levels) with each parent's
    edges contiguous and ``dn_first`` marking the first. Block j is entries ``off[j]:off[j+1]``:
    at most ``B`` edges, all of one level. Shapes depend only on (bucket(n), bucket(N), G, B and the
    bucketed block count), not on the tree's shape.
    """

    parent: Int[Array, " n"]
    t: Float[Array, " n"]
    x: Float[Array, "N G"]
    var: Float[Array, "N G"]
    leaf_row: Int[Array, " n"]
    is_leaf: Bool[Array, " n"]
    centre: Float[Array, "2 G"]
    up_child: Int[Array, " e"]
    up_off: Int[Array, " b"]
    n_up: Int[Array, ""]
    dn_child: Int[Array, " e"]
    dn_first: Bool[Array, " e"]
    dn_off: Int[Array, " b"]
    n_dn: Int[Array, ""]
    n_nodes: Int[Array, ""]  # an array, not static: trees of one bucket share a compiled kernel
    B: int = eqx.field(static=True)

    @classmethod
    def from_arrays(cls, parent, t, x, var, dtype=jnp.float32, block: int = BLOCK) -> "ArrayTree":
        """Build from node-indexed numpy arrays (only the leaf rows of ``x``/``var`` are read).

        ``block`` is the block width B, capped at bucket(n) for small trees.
        """
        parent = np.asarray(parent, dtype=np.int64)
        n = len(parent)
        order, kids = postorder(parent)
        if len(kids[int(np.flatnonzero(parent < 0)[0])]) == 1:  # as _ref.tree.tree_loglik_grad
            raise ValueError("a unary root makes the tree gradient undefined (W_{\\c} = 0)")
        is_leaf = np.array([not k for k in kids])
        height = np.zeros(n, dtype=np.int64)
        for v in order:
            if kids[v]:
                height[v] = 1 + max(height[c] for c in kids[v])
        depth = np.zeros(n, dtype=np.int64)
        for v in order[::-1]:
            for c in kids[v]:
                depth[c] = depth[v] + 1
        n_pad = _bucket(n)
        B = min(block, n_pad)
        child = np.flatnonzero(parent >= 0)
        pc = parent[child]
        up = child[np.lexsort((child, pc, height[pc]))]
        up_off = _blocks(np.searchsorted(height[parent[up]] - 1, np.arange(height.max() + 1)), B)
        dn = child[np.lexsort((child, pc, depth[pc]))]
        dn_first = np.r_[True, parent[dn][1:] != parent[dn][:-1]]
        dn_off = _blocks(np.searchsorted(depth[parent[dn]], np.arange(depth.max() + 1)), B)
        nb_pad = _bucket(max(len(up_off), len(dn_off)) - 1)
        e_pad = n_pad + B  # room for a full block after the last edge (dynamic_slice clamps)

        def pad(a, size, fill):
            out = np.full((size, *np.shape(a)[1:]), fill, dtype=np.asarray(a).dtype)
            out[: len(a)] = a
            return out

        leaves = np.flatnonzero(is_leaf)
        xl = np.asarray(np.asarray(x)[leaves], dtype=np.float64)  # leaf rows only: no (n, G) float64 temporaries
        centre = xl.mean(axis=0)
        c_hi = centre.astype(np.float32)
        c_lo = centre - c_hi  # cast to dtype below: exact pair in float64, rounded once in float32
        leaf_row = np.zeros(n, dtype=np.int64)
        leaf_row[leaves] = np.arange(len(leaves))
        N_pad = _bucket(len(leaves))
        t_c = np.where(parent >= 0, np.asarray(t, dtype=np.float64), 0.0)
        i32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.int32))
        return cls(
            parent=i32(pad(parent, n_pad, -1)),
            t=jnp.asarray(pad(t_c, n_pad, 0.0), dtype=dtype),
            x=jnp.asarray(pad(xl - centre, N_pad, 0.0), dtype=dtype),
            var=jnp.asarray(pad(np.asarray(np.asarray(var)[leaves], dtype=np.float64), N_pad, 1.0), dtype=dtype),
            leaf_row=i32(pad(leaf_row, n_pad, 0)),
            is_leaf=jnp.asarray(pad(is_leaf, n_pad, False)),
            centre=jnp.asarray(np.stack([c_hi, c_lo]), dtype=dtype),
            up_child=i32(pad(up, e_pad, 0)),
            up_off=i32(pad(up_off, nb_pad + 1, up_off[-1])),
            n_up=i32(len(up_off) - 1),
            dn_child=i32(pad(dn, e_pad, 0)),
            dn_first=jnp.asarray(pad(dn_first, e_pad, False)),
            dn_off=i32(pad(dn_off, nb_pad + 1, dn_off[-1])),
            n_dn=i32(len(dn_off) - 1),
            n_nodes=i32(n),
            B=B,
        )

    @classmethod
    def from_upstream(cls, adata, dtype=jnp.float32) -> tuple["ArrayTree", list[str]]:
        """ArrayTree of the upstream result in ``adata.uns["bonsai"]``; see :func:`upstream_arrays`."""
        parent, t, x, var, names = upstream_arrays(adata)
        return cls.from_arrays(parent, t, x, var, dtype=dtype), names


def centre64(tree: ArrayTree) -> np.ndarray:
    """The per-gene centre subtracted from the leaves, float64 on the host."""
    c = np.asarray(tree.centre, dtype=np.float64)
    return c[0] + c[1]


def upstream_arrays(adata):
    """Node-indexed float64 (parent, t, x, var, names) of an upstream result in ``adata.uns["bonsai"]``.

    The result comes from ``tl.reconstruct`` or ``io.read_results``. Rows follow upstream's vertInd (vertInfo.txt, posterior .npy rows); the topology comes from
    ``edges`` (edgeInfo.txt), branch lengths at full precision from ``newick`` (edgeInfo.txt
    keeps 9 digits, TH:3161-3163), leaf data from ``preprocess`` with the run's
    ``zscore_cutoff``/``rescale_by_var`` (defaults if the run config is unknown);
    ``names[i]`` = vertName of row i. Raises ``ValueError`` on a cell-set, gene-set or topology
    mismatch.
    """
    res = adata.uns["bonsai"]
    nodes = res["nodes"].sort_values("vertInd")
    if not (nodes["vertInd"].to_numpy() == np.arange(len(nodes))).all():
        raise ValueError("uns['bonsai']['nodes'].vertInd is not a 0..n-1 permutation")
    names = [str(v) for v in nodes["name"]]
    row = {v: i for i, v in enumerate(names)}
    parent = np.full(len(names), -1, dtype=np.int64)
    parent[res["edges"]["child"].to_numpy()] = res["edges"]["parent"].to_numpy()
    p_nwk, t_nwk, n_nwk = parse_newick(res["newick"])
    if sorted(n_nwk) != sorted(names):
        raise ValueError("newick node names differ from uns['bonsai']['nodes']")
    idx = np.array([row[v] for v in n_nwk])
    t = np.zeros(len(names))
    t[idx] = t_nwk
    p_from_nwk = np.full(len(names), -1, dtype=np.int64)
    p_from_nwk[idx] = np.where(p_nwk >= 0, idx[np.maximum(p_nwk, 0)], -1)
    if not np.array_equal(p_from_nwk, parent):
        raise ValueError("newick and edges describe different topologies")
    _, kids = postorder(parent)
    leaf_names = [v for i, v in enumerate(names) if not kids[i]]
    if sorted(leaf_names) != sorted(adata.obs_names):
        raise ValueError("the tree's leaves and adata.obs_names are not the same cells")
    cfg = res.get("config")
    if not cfg:
        warnings.warn(
            "uns['bonsai']['config'] not found; assuming the run's defaults "
            "(zscore_cutoff=1.0, rescale_by_var=True). A run with different settings "
            "and no recorded config would silently reproduce the wrong leaf arrays.",
            stacklevel=2,
        )
        cfg = {}
    leaves = preprocess(adata, cfg.get("zscore_cutoff", 1.0), cfg.get("rescale_by_var", True))
    if list(adata.var_names[leaves.genes]) != list(res["genes_used"]):
        raise ValueError("genes kept by preprocessing differ from uns['bonsai']['genes_used']")
    cell = {c: i for i, c in enumerate(adata.obs_names)}
    G = leaves.x.shape[1]
    x, var = np.zeros((len(names), G)), np.ones((len(names), G))
    for i, v in enumerate(names):
        if not kids[i]:
            x[i], var[i] = leaves.x[cell[v]], leaves.var[cell[v]]
    return parent, t, x, var, names
