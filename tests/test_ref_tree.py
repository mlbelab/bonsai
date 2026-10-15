import json

import numpy as np
import pytest

from bonsai._ref.newick import parse_newick
from bonsai._ref.tree import _upward, postorder, tree_loglik

from ._bounds import EPS64, loglik_scale, tree_height
from ._trees import leaf_lca_cov, random_leaf_data, random_tree, reroot
from ._upstream import upstream_leaves, upstream_loglik
from .golden import FIXTURE, GOLDEN, load_tree


def _fp64_tol(parent, t, x, var, n_orders: int = 2) -> float:
    """Two fp64 evaluations of L_tree that sum in different orders differ by ≤ n·8·(1+H)·eps64·A."""
    _, X, V, _, _ = _upward(parent, t, x, var)
    return n_orders * 8 * (1 + tree_height(parent)) * EPS64 * loglik_scale(parent, t, X, 1.0 / V)


@pytest.mark.parametrize("max_children", [2, 4])
def test_loglik_equals_brute_force_gaussian_marginal(max_children):
    """L_tree = Σ_g [−log det C − log(1ᵀC⁻¹1) − (yᵀC⁻¹y − (1ᵀC⁻¹y)²/1ᵀC⁻¹1)] (maths ref §4).

    C = Brownian covariance (root-to-LCA path lengths) + diag(sigma²): the flat-root-prior marginal
    likelihood, times 2, plus (N−1)·log 2π per gene (upstream's convention).
    Tolerance: dense solves/log-det have forward error ≤ 8·N·eps64·cond(C) relative to the summed
    magnitudes |log det C| + |log a| + |Q| per gene (standard LU/Cholesky backward-error bound).
    """
    rng = np.random.default_rng(1)
    parent, t = random_tree(7, rng, max_children=max_children)
    x, var = random_leaf_data(parent, 4, rng)
    leaves = np.flatnonzero([not k for k in postorder(parent)[1]])
    C0 = leaf_lca_cov(parent, t, leaves)
    want, scale = 0.0, 0.0
    for g in range(x.shape[1]):
        C, y, one = C0 + np.diag(var[leaves, g]), x[leaves, g], np.ones(len(leaves))
        Ci = np.linalg.inv(C)
        a, b = one @ Ci @ one, one @ Ci @ y
        terms = np.array([np.linalg.slogdet(C)[1], np.log(a), y @ Ci @ y - b * b / a])
        want -= terms.sum()
        scale += 8 * len(leaves) * EPS64 * np.linalg.cond(C) * np.abs(terms).sum()
    assert abs(tree_loglik(parent, t, x, var) - want) <= scale


def test_loglik_invariant_under_rerooting_and_gene_shift():
    """Flat root prior ⇒ L_tree depends neither on the root position nor on a per-gene shift of all x."""
    rng = np.random.default_rng(2)
    parent, t = random_tree(12, rng)
    x, var = random_leaf_data(parent, 5, rng)
    L = tree_loglik(parent, t, x, var)
    tol = _fp64_tol(parent, t, x, var)
    for r in np.flatnonzero([bool(k) for k in postorder(parent)[1]]):  # every internal node
        p2, t2 = reroot(parent, t, r)
        assert abs(tree_loglik(p2, t2, x, var) - L) <= tol
    shift = rng.normal(0, 30, size=x.shape[1])
    assert abs(tree_loglik(parent, t, x + shift, var) - L) <= tol + 8 * EPS64 * 30 * np.abs(x).sum()


def test_unary_node_contributes_zero():
    """A node with one child has ell = log w − log W − 0 = 0 (it appears after rerooting)."""
    parent = np.array([-1, 0, 1, 1, 0])  # ((a:0.2, b:0.4)n1:0.3, c:0.5)root
    t = np.array([0.0, 0.3, 0.2, 0.4, 0.5])
    x, var = random_leaf_data(parent, 3, np.random.default_rng(3))
    p_u = np.array([-1, 5, 1, 1, 0, 0])  # node 5 is unary between root and node 1, edge 0.3 split 0.1 + 0.2
    t_u = np.array([0.0, 0.2, 0.2, 0.4, 0.5, 0.1])
    xu, vu = np.vstack([x, np.zeros(3)]), np.vstack([var, np.ones(3)])
    assert abs(tree_loglik(p_u, t_u, xu, vu) - tree_loglik(parent, t, x, var)) <= _fp64_tol(parent, t, x, var)


def _golden_arrays():
    """Golden tree (full-precision newick times) and upstream's own preprocessed leaf arrays."""
    parent, t, names = parse_newick((GOLDEN / "tree.nwk").read_text())
    x, var, _, _ = upstream_leaves(FIXTURE, "delta_vmax.txt", "d_delta_vmax.txt", True)
    cells = json.loads((GOLDEN / "metadata.json").read_text())["cellIds"]  # read_and_filter column order
    row = {c: i for i, c in enumerate(cells)}
    X, V = np.zeros((len(parent), x.shape[1])), np.ones((len(parent), x.shape[1]))
    for i, v in enumerate(names):
        if v in row:
            X[i], V[i] = x[row[v]], var[row[v]]
    return parent, t, X, V


def test_golden_loglik_matches_upstream_final_loglik():
    """Spec gate (M1): the fp64 reference matches upstream to 1e-10 relative; also within the
    derived fp64 bound 2·8·(1+H)·eps64·A (two summation orders). On this fixture upstream's
    finalLoglik equals L_tree on the full-precision newick times (it is computed in memory)."""
    parent, t, X, V = _golden_arrays()
    L = tree_loglik(parent, t, X, V)
    final = load_tree(GOLDEN).loglik
    assert abs(L - final) <= 1e-10 * abs(final)
    assert abs(L - final) <= _fp64_tol(parent, t, X, V)


@pytest.mark.parametrize("max_children", [2, 3])
def test_loglik_matches_upstream_on_random_topologies(max_children):
    """Same quantity through upstream's object tree (Tree.calcLogLComplete) on random topologies."""
    rng = np.random.default_rng(4)
    parent, t = random_tree(40, rng, max_children=max_children)
    x, var = random_leaf_data(parent, 25, rng)
    L, want = tree_loglik(parent, t, x, var), upstream_loglik(parent, t, x, var)
    assert abs(L - want) <= 1e-10 * abs(want)
    assert abs(L - want) <= _fp64_tol(parent, t, x, var)


def test_postorder_rejects_forests_and_cycles():
    with pytest.raises(ValueError, match="exactly one root"):
        postorder(np.array([-1, -1, 0]))
    with pytest.raises(ValueError, match="cycle"):
        postorder(np.array([-1, 2, 1]))
