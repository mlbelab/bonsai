import numpy as np
import pytest
from ruamel.yaml import YAML

from bonsai import io, pp
from bonsai._ref.newick import parse_newick
from bonsai._ref.tree import postorder, tree_loglik

from ._trees import random_leaf_data, random_tree
from .golden import FIXTURE, GOLDEN, load_tree
from .test_ref_tree import _fp64_tol


def _host(tree):
    return {k: np.asarray(getattr(tree, k)) for k in ("parent", "up_child", "up_first", "up_off", "dn_child", "dn_off")}


def _assert_schedules_valid(tree, parent):
    """Every edge appears exactly once in each schedule, in dependency order (brief Step 1 invariants)."""
    h = _host(tree)
    n = len(parent)
    assert tree.parent.shape[0] == 1 << (n - 1).bit_length() and int(tree.n_nodes) == n  # power-of-two bucket
    assert np.array_equal(h["parent"][:n], parent) and np.all(h["parent"][n:] == -1)
    _, kids = postorder(parent)
    height = np.zeros(n, int)
    for v in postorder(parent)[0]:
        if kids[v]:
            height[v] = 1 + max(height[c] for c in kids[v])
    n_up, n_dn = int(tree.n_up), int(tree.n_dn)
    dn_first = np.asarray(tree.dn_first)
    up = [h["up_child"][h["up_off"][k] : h["up_off"][k + 1]] for k in range(n_up)]
    assert sorted(np.concatenate(up).tolist()) == sorted(np.flatnonzero(parent >= 0).tolist())
    for k, edges in enumerate(up):  # level k = parents of height k+1, each parent's edges contiguous
        assert np.all(height[parent[edges]] == k + 1)
        first = h["up_first"][h["up_off"][k] : h["up_off"][k + 1]]
        assert first.sum() == len(np.unique(parent[edges]))
    seen = {int(np.flatnonzero(parent < 0)[0])}
    for k in range(n_dn):  # pre-order: a parent's AIR is final before its children's level
        edges = h["dn_child"][h["dn_off"][k] : h["dn_off"][k + 1]]
        assert set(parent[edges].tolist()) <= seen
        first = dn_first[h["dn_off"][k] : h["dn_off"][k + 1]]  # parents contiguous, first edge marked
        assert first.sum() == len(np.unique(parent[edges])) == len(np.unique(np.cumsum(first)))
        seen |= set(edges.tolist())
    assert len(seen) == n
    assert tree.width >= max(len(e) for e in up) and tree.width & (tree.width - 1) == 0


@pytest.mark.parametrize("max_children", [2, 5])
def test_schedules_cover_every_edge_once_in_dependency_order(max_children):
    from bonsai._engine.array_tree import ArrayTree

    rng = np.random.default_rng(0)
    parent, t = random_tree(37, rng, max_children=max_children)
    x, var = random_leaf_data(parent, 3, rng)
    tree = ArrayTree.from_arrays(parent, t, x, var)
    _assert_schedules_valid(tree, parent)


def test_schedules_handle_unary_node_and_arbitrary_numbering():
    """Review focus 3/4 context: a unary node, and numbering that is neither children-before-parents
    nor root-last (`random_tree` always gives both; the golden and real upstream trees need not)."""
    from bonsai._engine.array_tree import ArrayTree

    rng = np.random.default_rng(5)
    parent, t = random_tree(11, rng, max_children=3)
    kids = postorder(parent)[1]
    v = int(np.flatnonzero([bool(k) for k in kids])[0])  # an internal node
    u = len(parent)  # splice a unary node u between v and its parent
    parent, t = np.r_[parent, parent[v]], np.r_[t, t[v] / 2]
    parent[v], t[v] = u, t[v] / 2
    n = len(parent)
    perm = rng.permutation(n)  # arbitrary numbering: root need not land last, parents need not exceed children
    new_parent, new_t = np.full(n, -1, dtype=parent.dtype), np.zeros(n)
    for old in range(n):
        new_parent[perm[old]] = perm[parent[old]] if parent[old] >= 0 else -1
        new_t[perm[old]] = t[old]
    assert int(np.flatnonzero(new_parent < 0)[0]) != n - 1  # the permutation did move the root
    x, var = random_leaf_data(new_parent, 3, rng)
    tree = ArrayTree.from_arrays(new_parent, new_t, x, var)
    _assert_schedules_valid(tree, new_parent)


def test_leaves_are_centred_and_rounded_once():
    import jax.numpy as jnp

    from bonsai._engine.array_tree import ArrayTree, centre64

    rng = np.random.default_rng(1)
    parent, t = random_tree(10, rng)
    x, var = random_leaf_data(parent, 4, rng)
    x += 30.0  # uncentred, like upstream's x (mu_g added, maths ref §1)
    tree = ArrayTree.from_arrays(parent, t, x, var, dtype=jnp.float32)
    leaf = np.asarray(tree.is_leaf)[:19]
    centre = x[:10].mean(axis=0)
    assert tree.x.dtype == tree.centre.dtype == jnp.float32  # no float64 on the device (A8)
    # the float32 pair hi + lo holds the float64 centre to eps32²·|c| (lo is rounded once)
    assert np.all(np.abs(centre64(tree) - centre) <= np.finfo(np.float32).eps ** 2 * np.abs(centre))
    np.testing.assert_array_equal(np.asarray(tree.x)[:19][leaf], (x[:10] - centre).astype(np.float32))


@pytest.mark.x64
def test_float64_centre_is_exact():
    """In float64 the pair hi + lo is the float64 centre exactly (lo = centre − hi is exact, Sterbenz)."""
    import jax.numpy as jnp

    from bonsai._engine.array_tree import ArrayTree, centre64

    rng = np.random.default_rng(1)
    parent, t = random_tree(10, rng)
    x, var = random_leaf_data(parent, 4, rng)
    x += 30.0
    tree = ArrayTree.from_arrays(parent, t, x, var, dtype=jnp.float64)
    np.testing.assert_array_equal(centre64(tree), x[:10].mean(axis=0))


def test_unary_root_raises():
    """As the fp64 reference: a one-child root leaves W_{\\c} = 0 at its child (gradient 0/0)."""
    from bonsai._engine.array_tree import ArrayTree

    x, var = np.zeros((4, 2)), np.ones((4, 2))
    with pytest.raises(ValueError, match="unary root"):
        ArrayTree.from_arrays(np.array([2, 2, 3, -1]), np.full(4, 0.5), x, var)


def _golden_config() -> dict:
    """The golden run's own recorded ``zscore_cutoff``/``rescale_by_var`` (`config.yaml`)."""
    cfg = YAML(typ="safe").load((GOLDEN / "config.yaml").read_text())
    return {"zscore_cutoff": cfg["zscore_cutoff"], "rescale_by_var": cfg["rescale_by_var"]}


def _golden_adata():
    a = pp.read_sanity(FIXTURE)
    io.read_results(GOLDEN, a)
    a.uns["bonsai"]["config"] = _golden_config()  # the real recorded run config, not a guessed default
    return a


def test_from_upstream_golden_rows_follow_vertind():
    from bonsai._engine.array_tree import ArrayTree

    a = _golden_adata()
    tree, names = ArrayTree.from_upstream(a)
    nodes = a.uns["bonsai"]["nodes"].sort_values("vertInd")
    assert names == list(nodes["name"].astype(str)) and int(tree.n_nodes) == 127
    edges = a.uns["bonsai"]["edges"]
    np.testing.assert_array_equal(np.asarray(tree.parent)[edges["child"]], edges["parent"])
    assert tree.x.shape[1] == len(a.uns["bonsai"]["genes_used"])


def test_from_upstream_schedules_valid_on_the_real_golden_tree():
    """The schedule invariants, checked on upstream's own (non-synthetic) topology."""
    from bonsai._engine.array_tree import ArrayTree, upstream_arrays

    a = _golden_adata()
    parent = upstream_arrays(a)[0]
    tree, _ = ArrayTree.from_upstream(a)
    _assert_schedules_valid(tree, parent)


def test_from_upstream_arrays_match_independent_fp64_reference():
    """`upstream_arrays`'s (parent, t, x, var) must agree with the fp64 reference, not just itself:
    `tree_loglik` on them must reproduce upstream's own ``finalLoglik`` (within the derived fp64
    tolerance used throughout `tests/test_ref_tree.py`), and `t` must equal the newick's own
    full-precision times exactly (not e.g. the coarser `%.8e` text in `edgeInfo.txt`, whose
    1.5e-15 relative effect on the loglik is far under that tolerance and would not be caught by
    the loglik check alone).
    """
    from bonsai._engine.array_tree import upstream_arrays

    a = _golden_adata()
    parent, t, x, var, names = upstream_arrays(a)
    L, golden = tree_loglik(parent, t, x, var), load_tree(GOLDEN)
    assert abs(L - golden.loglik) <= _fp64_tol(parent, t, x, var)

    _, t_nwk, n_nwk = parse_newick((GOLDEN / "tree.nwk").read_text())
    row = {v: i for i, v in enumerate(names)}
    t_from_newick = np.zeros(len(names))
    t_from_newick[[row[v] for v in n_nwk]] = t_nwk
    np.testing.assert_array_equal(t, t_from_newick)


def test_from_upstream_fallback_defaults_match_this_runs_recorded_config():
    """The hardcoded fallback defaults (zscore_cutoff=1.0, rescale_by_var=True) happen to equal this
    golden run's own recorded config (`config.yaml`), so dropping the recorded config and taking the
    fallback path must still reproduce upstream's `finalLoglik`. Also how mutant (a) (default
    `rescale_by_var=False`) is caught: `test_from_upstream_uses_recorded_config_not_defaults` and
    `test_from_upstream_arrays_match_independent_fp64_reference` both always set/have an explicit
    `rescale_by_var`, so neither exercises the fallback value itself.
    """
    from bonsai._engine.array_tree import upstream_arrays

    a = _golden_adata()
    del a.uns["bonsai"]["config"]
    with pytest.warns(UserWarning, match="config"):
        parent, t, x, var, _ = upstream_arrays(a)
    L, golden = tree_loglik(parent, t, x, var), load_tree(GOLDEN)
    assert abs(L - golden.loglik) <= _fp64_tol(parent, t, x, var)


def test_from_upstream_independent_of_nodes_row_order():
    """`upstream_arrays` must sort by vertInd, not rely on `nodes` already being in that order."""
    from bonsai._engine.array_tree import ArrayTree

    a = _golden_adata()
    tree, names = ArrayTree.from_upstream(a)
    b = a.copy()
    b.uns["bonsai"]["nodes"] = b.uns["bonsai"]["nodes"].sample(frac=1, random_state=3).reset_index(drop=True)
    tree_b, names_b = ArrayTree.from_upstream(b)
    assert names == names_b
    for k in ("parent", "t", "x", "var", "centre"):
        np.testing.assert_array_equal(np.asarray(getattr(tree, k)), np.asarray(getattr(tree_b, k)))


def test_from_upstream_uses_recorded_config_not_defaults():
    """A non-default recorded config (here zscore_cutoff=-1, keep-all) must actually be used."""
    from bonsai._engine.array_tree import ArrayTree
    from bonsai._ref.preprocess import preprocess

    a = _golden_adata()
    bad = a.copy()
    bad.uns["bonsai"]["config"] = {"zscore_cutoff": -1, "rescale_by_var": True}
    leaves = preprocess(bad, -1, True)
    bad.uns["bonsai"]["genes_used"] = list(bad.var_names[leaves.genes])
    tree, _ = ArrayTree.from_upstream(bad)
    assert tree.x.shape[1] == len(bad.uns["bonsai"]["genes_used"]) > len(a.uns["bonsai"]["genes_used"])


def test_from_upstream_warns_when_config_is_missing():
    """`io.read_results` on its own carries no recorded config; silently assuming defaults must be audible."""
    from bonsai._engine.array_tree import ArrayTree

    a = _golden_adata()
    del a.uns["bonsai"]["config"]  # as `io.read_results` actually leaves it: no "config" key
    with pytest.warns(UserWarning, match="config"):
        ArrayTree.from_upstream(a)


def test_from_upstream_independent_of_obs_order_and_rejects_cell_mismatch():
    """Review focus 2: AnnData row order must not matter; missing or extra cells must raise."""
    from bonsai._engine.array_tree import ArrayTree

    a = _golden_adata()
    tree, _ = ArrayTree.from_upstream(a)
    perm = np.random.default_rng(2).permutation(a.n_obs)
    tree_b, _ = ArrayTree.from_upstream(a[perm].copy())
    for k in (
        "parent",
        "t",
        "x",
        "var",
        "centre",
    ):  # identical kernel inputs (the kernel itself may round differently between GPU runs: scatter-add order)
        assert np.array_equal(np.asarray(getattr(tree, k)), np.asarray(getattr(tree_b, k)))
    with pytest.raises(ValueError, match="not the same cells"):
        ArrayTree.from_upstream(a[1:].copy())


def test_from_upstream_rejects_inconsistent_results():
    from bonsai._engine.array_tree import ArrayTree

    a = _golden_adata()
    bad = a.copy()
    bad.uns["bonsai"]["genes_used"] = bad.uns["bonsai"]["genes_used"][1:]
    with pytest.raises(ValueError, match="genes_used"):
        ArrayTree.from_upstream(bad)
    bad = a.copy()
    e = bad.uns["bonsai"]["edges"].copy()
    e.loc[e.index[-1], "parent"] = e["parent"].iloc[0]  # move one edge
    bad.uns["bonsai"]["edges"] = e
    with pytest.raises(ValueError, match="different topologies"):
        ArrayTree.from_upstream(bad)
