"""JAX tree pass vs the fp64 references and upstream (M1 gates).

float32 cases run in the default (x64-off) lane; the float64 cases (``x64`` marker) check that the
kernel implements exactly the reference formulas, at eps64 bounds.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bench.run import count_compilations
from bench.simulators.brownian import simulate

from bonsai._engine.array_tree import ArrayTree, centre64, upstream_arrays
from bonsai._engine.numerics import dsum
from bonsai._engine.star import child_weights, node_message, star_loglik_grad
from bonsai._engine.tree import tree_loglik_grad
from bonsai._ref import star as rstar
from bonsai._ref.preprocess import preprocess
from bonsai._ref.tree import postorder
from bonsai._ref.tree import tree_loglik_grad as ref_grad

from ._bounds import EPS32, input_rounding, rounded, tree_height, tree_tols
from ._trees import bm_tree, random_leaf_data, random_tree
from ._upstream import upstream_loglik, upstream_opt_times
from .golden import GOLDEN, load_tree
from .test_array_tree import _golden_adata

F64 = pytest.param(jnp.float64, marks=pytest.mark.x64, id="float64")
F32 = pytest.param(jnp.float32, id="float32")


def _check(parent, t, x, var, dtype, block=None):
    """Kernel vs reference on the inputs the kernel actually sees (centred, rounded to dtype)."""
    kw = {} if block is None else {"block": block}
    tree = ArrayTree.from_arrays(parent, t, x, var, dtype=dtype, **kw)
    eps = float(jnp.finfo(dtype).eps)
    np_dt = np.dtype(dtype)
    xc = rounded(x - centre64(tree), np_dt)
    vr, tr = rounded(var, np_dt), rounded(t, np_dt)
    ref = ref_grad(parent, tr, xc, vr)
    e = tree_loglik_grad(tree)
    tol_L, tol_g = tree_tols(parent, tr, ref, eps)
    n = len(parent)
    assert e.ell.dtype == dtype and e.grad.dtype == dtype
    assert abs(e.L - ref.L) <= tol_L
    g = np.asarray(e.grad)
    assert np.all(np.abs(g[:n] - ref.grad) <= tol_g) and np.all(g[n:] == 0.0)
    assert np.all(np.asarray(e.ell)[n:] == 0.0)
    _check_messages(tree, parent, tr, vr, e, eps)
    return tree, e


def _node_x(tree, n):
    """The kernel's own centred leaf data as node-indexed float64 rows (internal rows 0)."""
    leaf = np.asarray(tree.is_leaf)[:n]
    x = np.zeros((n, tree.x.shape[1]))
    x[leaf] = np.asarray(tree.x, dtype=np.float64)[np.asarray(tree.leaf_row)[:n][leaf]]
    return x


def _check_messages(tree, parent, tr, vr, e, eps):
    """Node and as-if-root messages (tree_tols' message bounds): below height H, relative error of
    W and absolute error of the convex-combination means are ≤ (H + 1)(C + 3)·eps (·max|x_leaf|);
    the pre-order pass adds H levels and the final AIR combination one more: (2H + 2)(C + 3)·eps.
    Means are not shift-invariant, so the reference gets the kernel's own centred leaf rows (in
    float32, x − centre64 can differ from them by a double rounding)."""
    n, H = len(parent), tree_height(parent)
    ref = ref_grad(parent, tr, _node_x(tree, n), vr)
    C = max(len(k) for k in postorder(parent)[1])
    X = np.abs(ref.x).max()
    rho, rho_air = (H + 1) * (C + 3) * eps, (2 * H + 2) * (C + 3) * eps
    f64 = lambda a: np.asarray(a, dtype=np.float64)[:n]
    assert np.all(np.abs(f64(e.W) - ref.W) <= rho * ref.W)
    assert np.all(np.abs(f64(e.x) - ref.x) <= rho * X)
    assert np.all(np.abs(f64(e.W_air) - ref.W_air) <= rho_air * ref.W_air)
    assert np.all(np.abs(f64(e.x_air) - ref.x_air) <= rho_air * X)


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize(("n", "max_children"), [(5, 2), (40, 2), (300, 2), (40, 5)])
def test_tree_loglik_grad_matches_reference(dtype, n, max_children):
    rng = np.random.default_rng(n + max_children)
    parent, t = random_tree(n, rng, max_children=max_children)
    x, var = random_leaf_data(parent, 7, rng)
    _check(parent, t, x + 30.0, var, dtype)  # +30: uncentred input, as upstream's x


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("block", [1, 2, 4])
def test_small_blocks_split_levels_and_sibling_groups(dtype, block):
    """A14: with B smaller than a level and than a sibling group, a parent's children fall into
    several blocks; the sibling sums (carried/tail partial sums) and the once-per-parent −log W_p term must still
    give the reference L, gradient and messages."""
    rng = np.random.default_rng(17)
    parent, t = random_tree(60, rng, max_children=5)
    x, var = random_leaf_data(parent, 7, rng)
    tree, _ = _check(parent, t, x, var, dtype, block=block)
    off, dn = np.asarray(tree.dn_off)[: int(tree.n_dn) + 1], np.asarray(tree.dn_child)
    starts = dn[off[1:-1]]  # first edge of every block but the first
    assert np.any(parent[starts] == parent[dn[off[1:-1] - 1]])  # some parent's children do span blocks


def test_arbitrary_node_numbering():
    """Leaves need not be nodes 0..N−1 (upstream's vertInd interleaves them): leaf data is read
    through ``leaf_row``."""
    rng = np.random.default_rng(21)
    parent, t = random_tree(25, rng, max_children=3)
    perm = rng.permutation(len(parent))
    new_parent = np.where(parent >= 0, perm[np.maximum(parent, 0)], -1)[np.argsort(perm)]
    x, var = random_leaf_data(parent, 5, rng)
    inv = np.argsort(perm)
    _check(new_parent, t[inv], x[inv], var[inv], jnp.float32)


def test_three_leaf_trees():
    """Smallest trees: a cherry plus a leaf (binary) and a 3-leaf star."""
    rng = np.random.default_rng(18)
    for parent in (np.array([3, 3, 4, 4, -1]), np.array([3, 3, 3, -1])):
        t = rng.uniform(0.1, 1.0, len(parent))
        t[-1] = 0.0
        x, var = random_leaf_data(parent, 5, rng)
        _check(parent, t, x, var, jnp.float32)


def test_identical_leaves():
    """All leaves equal: every residual is 0 up to rounding of the centre, so dL/dt = Σ_g −1/T < 0."""
    rng = np.random.default_rng(19)
    parent, t = random_tree(30, rng, max_children=3)
    x, var = random_leaf_data(parent, 6, rng)
    x[:], var[:] = x[0] + 3.0, var[0]
    _, e = _check(parent, t, x, var, jnp.float32)
    assert np.all(np.asarray(e.grad)[: len(parent)][parent >= 0] < 0)


def test_zero_length_edge_with_tiny_variance():
    """Maths ref §2 hazard: t = 0 and sigma² = 1e-6 (no-SD input) make w_c ≫ its siblings, so
    W_A − w_c would cancel. Mitigation: the pre-order pass never subtracts; W_{\\c} is built from
    the siblings' weights and the message from above (exclusive prefix + suffix sums)."""
    rng = np.random.default_rng(7)
    parent, t = random_tree(30, rng)
    x, var = random_leaf_data(parent, 7, rng)
    var[3], t[3] = 1e-6, 0.0
    _, e = _check(parent, t, x, var, jnp.float32)
    assert np.all(np.isfinite(np.asarray(e.grad)))


def test_unary_and_wide_nodes():
    """A unary node (rerooting artefact) and a 9-child node (upstream polytomy)."""
    rng = np.random.default_rng(8)
    parent = np.array([10] * 9 + [11, 12, 12, -1])  # leaves 0-8 under node 10; leaf 9 under unary 11
    t = rng.uniform(0.0, 0.5, len(parent))
    t[-1] = 0.0
    x, var = random_leaf_data(parent, 5, rng)
    _check(parent, t, x, var, jnp.float32)


def _broom(spine: int, bristles: int) -> np.ndarray:
    """A chain of ``spine`` internal nodes, each holding one leaf, with ``bristles`` leaves at the bottom."""
    n_leaves = spine - 1 + bristles
    chain = n_leaves + np.arange(spine)  # chain[0] is the root
    parent = np.full(n_leaves + spine, -1)
    parent[chain[1:]] = chain[:-1]
    parent[: spine - 1] = chain[:-1]
    parent[spine - 1 : n_leaves] = chain[-1]
    return parent


def test_padding_and_compile_signatures():
    """A14: trees of one (n, N, block count) bucket share one compiled kernel whatever their shape
    (a random binary tree and a broom: different heights, level widths and child counts), and give
    the unpadded reference values; padding rows carry zero gradient (checked in _check)."""
    rng = np.random.default_rng(9)
    jax.clear_caches()
    shapes = [random_tree(30, rng)[0], _broom(10, 21)]  # 59 and 40 nodes, 30 leaves each
    trees = []
    for parent in shapes:
        t = rng.exponential(0.5, len(parent))
        x, var = random_leaf_data(parent, 6, rng)
        trees.append(ArrayTree.from_arrays(parent, t, x, var))
    assert tree_height(shapes[0]) != tree_height(shapes[1])
    assert max(len(k) for k in postorder(shapes[1])[1]) == 21
    shape = lambda tr: ([a.shape for a in jax.tree.leaves(tr)], tr.B)
    assert shape(trees[0]) == shape(trees[1])  # same buckets: only the values differ
    with count_compilations() as c:
        for tr in trees:
            tree_loglik_grad(tr).ell.block_until_ready()
    assert c.count == 1
    with count_compilations() as c:
        for tr in trees:
            tree_loglik_grad(tr).ell.block_until_ready()
    assert c.count == 0  # repeated calls in a seen bucket reuse the compiled kernel
    for parent in shapes:
        t = rng.exponential(0.5, len(parent))
        t[parent < 0] = 0.0
        x, var = random_leaf_data(parent, 6, rng)
        _check(parent, t, x, var, jnp.float32)  # padded evaluation equals the unpadded reference


def _pad(tree: ArrayTree, rows: int, width: int) -> ArrayTree:
    """``tree`` with ``rows`` extra node rows, ``rows`` extra NaN leaf-data rows (never read) and
    blocks ``width`` times wider, padded as ``ArrayTree.from_arrays`` pads (edge entries 0, a real
    node, with ``first`` False)."""

    def pad(a, k, fill):
        return jnp.concatenate([a, jnp.full((k, *a.shape[1:]), fill, a.dtype)])

    n_pad = tree.parent.shape[0] + rows
    e = n_pad + width * tree.B - tree.up_child.shape[0]
    off = lambda a: pad(a, rows, a[-1])
    return dataclasses.replace(
        tree,
        parent=pad(tree.parent, rows, -1),
        t=pad(tree.t, rows, 0.0),
        x=pad(tree.x, rows, jnp.nan),
        var=pad(tree.var, rows, jnp.nan),
        leaf_row=pad(tree.leaf_row, rows, 0),
        is_leaf=pad(tree.is_leaf, rows, False),
        up_child=pad(tree.up_child, e, 0),
        up_off=off(tree.up_off),
        dn_child=pad(tree.dn_child, e, 0),
        dn_first=pad(tree.dn_first, e, False),
        dn_off=off(tree.dn_off),
        B=width * tree.B,
    )


def test_extra_padding_changes_nothing():
    """Padded edge entries name node 0 (a real node, ``first`` False) and a block's window reads
    past its block into the next block's edges, so the kernel masks by idx < off[j + 1]. With 64
    extra rows, and with those plus 4× wider blocks, L, the gradient and the messages meet the same
    reference bounds and padding rows stay zero. Not bitwise: wider blocks change the scan shapes,
    and GPU scatter-adds are unordered."""
    rng = np.random.default_rng(14)
    parent, t = random_tree(40, rng, max_children=3)
    x, var = random_leaf_data(parent, 7, rng)
    tree, _ = _check(parent, t, x, var, jnp.float32, block=8)  # several blocks per level
    n = len(parent)
    xc, vr, tr = (rounded(a) for a in (x - centre64(tree), var, t))
    ref = ref_grad(parent, tr, xc, vr)
    tol_L, tol_g = tree_tols(parent, tr, ref, EPS32)
    for width in (1, 4):
        e = tree_loglik_grad(_pad(tree, 64, width))
        g = np.asarray(e.grad)
        assert abs(e.L - ref.L) <= tol_L and np.all(np.abs(g[:n] - ref.grad) <= tol_g)
        assert np.all(g[n:] == 0.0) and np.all(np.asarray(e.ell)[n:] == 0.0)
        _check_messages(tree, parent, tr, vr, e, EPS32)


@pytest.mark.parametrize("dtype", [F32, F64])
def test_child_weights(dtype):
    """w = 1/(var + t) in the inputs' dtype: one addition and one division, |Δw| ≤ 2·eps·w. These
    are the exact weights a host accumulator adds and later removes (A8 same-weight rule), so
    node_message's W must be their compensated sum."""
    rng = np.random.default_rng(15)
    var, t = rng.uniform(1e-6, 1, (3, 6, 9)), rng.uniform(0, 1, (3, 6))
    t[0, 0] = 0.0
    eps = float(jnp.finfo(dtype).eps)
    w = child_weights(jnp.asarray(var, dtype), jnp.asarray(t, dtype))
    assert w.dtype == dtype
    vr, tr = rounded(var, np.dtype(dtype)), rounded(t, np.dtype(dtype))
    w_ref = 1 / (vr + tr[..., None])
    assert np.all(np.abs(np.asarray(w, dtype=np.float64) - w_ref) <= 2 * eps * w_ref)
    x = jnp.asarray(rng.normal(size=var.shape), dtype)
    _, W = node_message(x, jnp.asarray(var, dtype), jnp.asarray(t, dtype), jnp.ones((3, 6), bool))
    np.testing.assert_array_equal(np.asarray(W), np.asarray(dsum(w, axis=1)))


def test_node_message_ignores_nonfinite_padded_children():
    """Padded children may hold garbage: NaN or inf x in masked slots must not reach (x_n, W_n)."""
    rng = np.random.default_rng(20)
    x, var, t = rng.normal(size=(2, 4, 3)), rng.uniform(0.1, 1, (2, 4, 3)), rng.uniform(0, 1, (2, 4))
    mask = np.array([[True, True, False, False], [True, True, True, False]])
    bad = x.copy()
    bad[~mask] = np.nan
    bad[1, 3, 0] = np.inf
    f = lambda a: jnp.asarray(a, jnp.float32)
    want = node_message(f(np.where(mask[..., None], x, 0.0)), f(var), f(t), jnp.asarray(mask))
    got = node_message(f(bad), f(var), f(t), jnp.asarray(mask))
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_gene_sums_are_compensated():
    """A10: gradient sums over genes use ``dsum``. Two children with w = 1 (var = t = 0.5) and
    x = ±a_g, a_g = k/64 (k < 4096), make every per-gene gradient term a_g² − 1/2 exact in float32
    (star: w(w r² − 1 + w/W), r = ∓a_g, W = 2; tree: (−1 + d²/T)/T, d = ∓2a_g, T = 2). Zero-a genes
    (term −1/2) cancel the sum to S ∈ [0, 1/2), so a compensated sum, rounded once, errs by
    ≤ eps·|S| + 2⌈log2 G⌉·eps²·Σ|terms| (dsum docstring), while a plain float32 sum loses the low bits
    of partial sums of size ~Σ|terms|."""
    rng = np.random.default_rng(16)
    a = rng.integers(256, 4096, 64) / 64.0
    a = np.r_[a, np.zeros(int(2 * np.sum(a**2)) - len(a))]
    terms = a**2 - 0.5
    S, G = terms.sum(), len(a)  # exact in float64: dyadic terms with 2^-12 resolution
    tol = EPS32 * abs(S) + 2 * np.ceil(np.log2(G)) * EPS32**2 * np.abs(terms).sum()
    x = np.stack([a, -a])
    var, t = np.full((2, G), 0.5), np.full(2, 0.5)
    f = lambda v: jnp.asarray(v, jnp.float32)
    _, g = star_loglik_grad(f(x[None]), f(var[None]), f(t[None]), jnp.ones((1, 2), bool))
    assert np.all(np.abs(np.asarray(g, dtype=np.float64) - S) <= tol)
    tree = ArrayTree.from_arrays(np.array([2, 2, -1]), np.r_[t, 0.0], np.r_[x, x[:1]], np.r_[var, var[:1]])
    assert np.all(np.abs(np.asarray(tree_loglik_grad(tree).grad[:2], dtype=np.float64) - S) <= tol)


@pytest.mark.parametrize("dtype", [F32, F64])
def test_star_kernels_match_reference(dtype):
    """Bounds (all in the kernel's precision, A8): the parent message is two compensated sums and a
    division, so W has relative error ≤ 4·eps and x_n absolute error δx ≤ 6·eps·max|x|; per gene
    term ≤ 9 roundings (8 own + the gene/child dsum) plus the residual error 2|r|·δx."""
    rng = np.random.default_rng(10)
    S, C, G = 3, 6, 9
    x, var, t = rng.normal(size=(S, C, G)), rng.uniform(0.05, 1, (S, C, G)), rng.uniform(0, 1, (S, C))
    mask = np.ones((S, C), bool)
    mask[1, 4:] = False
    eps = float(jnp.finfo(dtype).eps)
    c = lambda a: jnp.asarray(a, dtype)
    xn, Wn = node_message(c(x), c(var), c(t), jnp.asarray(mask))
    ell, g = star_loglik_grad(c(x), c(var), c(t), jnp.asarray(mask))
    assert ell.dtype == g.dtype == xn.dtype == dtype
    for s in range(S):
        m = mask[s]
        xs, vs, ts = (rounded(a, np.dtype(dtype)) for a in (x[s][m], var[s][m], t[s][m]))
        rx, rW = rstar.node_message(xs, vs, ts)
        re, rg = rstar.star_loglik_grad(xs, vs, ts)
        w = 1 / (vs + ts[:, None])
        r = np.abs(rx - xs)
        dx = 6 * eps * np.abs(xs).max()
        tol_ell = np.sum(9 * eps * (1 + np.abs(np.log(w))) + w * (2 * r * dx + 9 * eps * r**2))
        tol_ell += np.sum(9 * eps * (1 + np.abs(np.log(rW))))
        assert abs(float(ell[s]) - re) <= tol_ell
        tol_g = np.sum(w * (w * (2 * r * dx + 9 * eps * r**2) + 9 * eps * (1 + w / rW)), axis=1)
        assert np.all(np.abs(np.asarray(g[s], dtype=np.float64)[m] - rg) <= tol_g)
        assert np.all(np.asarray(g[s])[~m] == 0.0)
        np.testing.assert_allclose(np.asarray(Wn[s], dtype=np.float64), rW, rtol=4 * eps)
        assert np.all(np.abs(np.asarray(xn[s], dtype=np.float64) - rx) <= dx)


@pytest.mark.x64
def test_float32_inputs_stay_float32_with_x64_enabled():
    """A process where another library set JAX_ENABLE_X64=1. float32 inputs must
    still compute in float32 (no promotion through default dtypes or Python scalars) and meet the
    default-lane bounds."""
    rng = np.random.default_rng(13)
    parent, t = random_tree(40, rng, max_children=3)
    x, var = random_leaf_data(parent, 7, rng)
    tree, e = _check(parent, t, x, var, jnp.float32)
    assert tree.x.dtype == tree.centre.dtype == jnp.float32
    assert {a.dtype for a in (e.ell, e.grad, e.x, e.W, e.x_air, e.W_air)} == {jnp.dtype(jnp.float32)}
    f = lambda a: jnp.asarray(a, jnp.float32)
    xs, vs, ts = rng.normal(size=(2, 4, 5)), rng.uniform(0.1, 1, (2, 4, 5)), rng.uniform(0, 1, (2, 4))
    ell, g = star_loglik_grad(f(xs), f(vs), f(ts), jnp.ones((2, 4), bool))
    assert ell.dtype == g.dtype == jnp.float32


def test_golden_tree_matches_upstream_final_loglik():
    """M1 gate: L_tree of the golden tree. The float32 kernel vs upstream's finalLoglik within the
    kernel bound (tree_tols, eps32) plus the first-order effect of rounding the inputs to float32
    (input_rounding): the kernel sees float32 data, upstream float64."""
    a = _golden_adata()  # with the run's recorded config
    tree, _ = ArrayTree.from_upstream(a)
    parent, t, x, var, _ = upstream_arrays(a)
    x = x - centre64(tree)
    ref = ref_grad(parent, t, x, var)
    final = load_tree(GOLDEN).loglik
    tol_L, _ = tree_tols(parent, t, ref, EPS32)
    assert abs(tree_loglik_grad(tree).L - final) <= tol_L + input_rounding(parent, t, var, ref, EPS32)
    assert abs(ref.L - final) <= 1e-10 * abs(final)


def test_simulated_fixed_topology_with_upstream_optimised_times():
    """M1 gate: on a fixed simulated topology with upstream's optTimes branch lengths, the fp64
    reference matches upstream's Tree.calcLogLComplete to 1e-10 relative and the fp32 kernel
    matches it within tree_tols(eps32) + input_rounding(eps32)."""
    rng = np.random.default_rng(11)
    parent, t, x, var = bm_tree(48, 120, rng)
    t_up, nit = upstream_opt_times(parent, t, x, var)
    assert nit < 100  # upstream stopped on its own rule, not on maxiter
    L_up = upstream_loglik(parent, t_up, x, var)
    tree = ArrayTree.from_arrays(parent, t_up, x, var, dtype=jnp.float32)
    xc = x - centre64(tree)
    ref = ref_grad(parent, t_up, xc, var)
    assert abs(ref.L - L_up) <= 1e-10 * abs(L_up)
    tol_L, _ = tree_tols(parent, t_up, ref, EPS32)
    assert abs(tree_loglik_grad(tree).L - L_up) <= tol_L + input_rounding(parent, t_up, var, ref, EPS32)


def test_simulator_topology_through_preprocess_with_upstream_times():
    """M1 gate on the benchmark path: the simulator's true topology, its Sanity-like output
    through ``preprocess``, upstream's optTimes branch lengths; same bounds as above."""
    sim = simulate(48, 300, seed=4)
    leaves = preprocess(sim.adata)  # cells cell0..cell47 = nodes 0..47
    n, G = len(sim.parent), leaves.x.shape[1]
    x, var = np.zeros((n, G)), np.ones((n, G))
    x[:48], var[:48] = leaves.x, leaves.var
    t_up, nit = upstream_opt_times(sim.parent, sim.t, x, var)
    assert nit < 100
    L_up = upstream_loglik(sim.parent, t_up, x, var)
    tree = ArrayTree.from_arrays(sim.parent, t_up, x, var)
    ref = ref_grad(sim.parent, t_up, x - centre64(tree), var)
    assert abs(ref.L - L_up) <= 1e-10 * abs(L_up)
    tol_L, _ = tree_tols(sim.parent, t_up, ref, EPS32)
    assert abs(tree_loglik_grad(tree).L - L_up) <= tol_L + input_rounding(sim.parent, t_up, var, ref, EPS32)


def _gpu_check(gpu_device):
    rng = np.random.default_rng(12)
    parent, t = random_tree(300, rng)
    x, var = random_leaf_data(parent, 64, rng)
    with jax.default_device(gpu_device):
        tree, e = _check(parent, t, x, var, jnp.float32)
    assert e.ell.devices() == {gpu_device}


@pytest.mark.gpu
def test_tree_loglik_grad_on_gpu(gpu_device):
    _gpu_check(gpu_device)
