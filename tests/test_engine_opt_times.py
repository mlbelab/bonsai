"""Device L-BFGS branch-length optimisation (maths ref §10), float32 in the default lane (A8)."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from bonsai._engine.array_tree import ArrayTree, centre64, upstream_arrays
from bonsai._engine.opt_times import LOG_T_MAX, LOG_T_MIN, T_SNAP, optimise_times
from bonsai._ref.tree import tree_loglik_grad as ref_grad

from ._bounds import EPS32, EPS64, rounded, tree_tols
from ._trees import bm_tree, random_tree
from ._upstream import upstream_opt_times
from .test_array_tree import _golden_adata

# Stopping parameter, not a tolerance. The float32 objective resolves ΔL only to ~1e-5 here (every
# per-gene term is rounded to eps32), so line searches stall near max|d(−L)/ds| ≈ 3e-3 on this problem
# (seen in planning); 1e-2 lets the gtol rule, not max_steps, end the run.
GTOL = 1e-2


def _problem(seed=5, n=40, G=20):
    parent, t, x, var = bm_tree(n, G, np.random.default_rng(seed))
    tree = ArrayTree.from_arrays(parent, t, x, var)
    return parent, t, rounded(x - centre64(tree)), rounded(var), tree  # the inputs the kernel sees


def _times(res, n):
    return np.asarray(res.tree.t, dtype=np.float64)[:n]


def _at_bounds(parent, tt, eps):
    """Edges at the lower/upper box bound. The kernel returns t = exp(±15) rounded to its dtype, so
    a bound edge is recognised within 4 roundings of exp(±15), far inside the box otherwise."""
    edge = parent >= 0
    low = edge & (tt <= np.exp(LOG_T_MIN) * (1 + 4 * eps))
    high = edge & (tt >= np.exp(LOG_T_MAX) * (1 - 4 * eps))
    return low, high, edge & ~low & ~high


def test_gtol_stop_fires_and_times_are_first_order_optimal():
    """A9: with optimistix's Cauchy rule disabled (rtol = atol = 0) only the gtol rule, which
    reads optimistix's internal ``state.f_info.grad``, can stop the run before max_steps; an
    optimistix change that breaks it fails here. KKT at the returned t, with the fp64 reference
    gradient: interior |t·dL/dt| ≤ GTOL + t·tol_g (tol_g: kernel vs reference, tree_tols);
    s at the lower bound ⇒ dL/dt ≤ tol_g; at the upper bound ⇒ dL/dt ≥ −tol_g."""
    parent, t, x, var, tree = _problem()
    res = optimise_times(tree, rtol=0.0, atol=0.0, gtol=GTOL, max_steps=2000, snap_zero=False)
    assert res.converged and int(res.steps) < 2000
    tt = _times(res, len(parent))
    ref = ref_grad(parent, tt, x, var)
    _, tol_g = tree_tols(parent, tt, ref, EPS32)
    low, high, free = _at_bounds(parent, tt, EPS32)
    assert np.all(np.abs(tt * ref.grad)[free] <= GTOL + (tt * tol_g)[free])
    assert np.all(ref.grad[low] <= tol_g[low]) and np.all(ref.grad[high] >= -tol_g[high])
    assert res.L >= ref_grad(parent, rounded(t), x, var).L  # never worse than the start


def test_reaches_the_fp64_reference_optimum_within_its_value_resolution():
    """Reference optimum: scipy L-BFGS-B at machine precision on the fp64 reference, same box. The
    float32 objective is resolved to tol_L (tree_tols), so L_ref(t_jax) ≥ L_ref(t_ref) − 2·tol_L.
    With gtol = 0 and the Cauchy rule off, only max_steps can end the run: the status must say so."""
    parent, t, x, var, tree = _problem(seed=8)
    edge = parent >= 0

    def f(s):
        tt = np.zeros(len(parent))
        tt[edge] = np.exp(s)
        r = ref_grad(parent, tt, x, var)
        return -r.L, -(tt * r.grad)[edge]

    s0 = np.log(np.clip(rounded(t)[edge], 1e-4, 1e4))
    opt = minimize(
        f,
        s0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(LOG_T_MIN, LOG_T_MAX)] * int(edge.sum()),
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10_000},
    )
    assert opt.success
    t_ref = np.zeros(len(parent))
    t_ref[edge] = np.exp(opt.x)
    best = ref_grad(parent, t_ref, x, var)
    res = optimise_times(tree, rtol=0.0, atol=0.0, gtol=0.0, max_steps=500, snap_zero=False)
    assert not res.converged and int(res.steps) == 500
    L_jax = ref_grad(parent, _times(res, len(parent)), x, var).L
    tol_L, _ = tree_tols(parent, t_ref, best, EPS32)
    assert L_jax >= best.L - 2 * tol_L


def test_defaults_are_at_least_as_good_as_upstream_opt_times():
    """Same start, upstream's settings (tol 1e-6, maxiter 100; TH:3444-3484); upstream must stop on
    its own rule. Its rule bounds the last step, not the distance to the optimum, so the slack is
    the A1 per-entry gate tolerance, 1e-6 per cell and gene (in 2× units), plus the float32
    kernel's value resolution tol_L: L_ref(jax) ≥ L_ref(upstream) − 1e-6·N·G − tol_L."""
    parent, t, x, var, tree = _problem(seed=6, n=48, G=120)
    t_up, nit = upstream_opt_times(parent, rounded(t), x, var)
    assert nit < 100
    res = optimise_times(tree)
    # This case hits the step cap (the step budget, not the gtol rule, ends the run: M3 decision,
    # spec A10 correction).
    assert not res.converged and int(res.steps) == 100
    L_up = ref_grad(parent, t_up, x, var)
    L_jax = ref_grad(parent, _times(res, len(parent)), x, var).L
    tol_L, _ = tree_tols(parent, t_up, L_up, EPS32)
    assert L_jax >= L_up.L - 1e-6 * 48 * 120 - tol_L


def test_golden_tree_at_least_as_good_as_upstream_opt_times():
    """Upstream's golden topology and data, both optimisers from t = 1 on every edge (a fixed start
    far from the optimum; the golden times are already optTimes' fixed point). Same inputs (the
    float32-rounded centred data the kernel sees), both scored by the fp64 reference; upstream
    must stop on its own rule. Slack as above: 1e-6·N·G (A1) + tol_L (float32 value resolution)."""
    a = _golden_adata()
    parent, _, x, var, _ = upstream_arrays(a)
    t1 = np.where(parent >= 0, 1.0, 0.0)
    tree = ArrayTree.from_arrays(parent, t1, x, var)
    x, var = rounded(x - centre64(tree)), rounded(var)
    t_up, nit = upstream_opt_times(parent, t1, x, var)
    assert nit < 100
    res = optimise_times(tree)
    assert res.converged and int(res.steps) < 100
    L_up = ref_grad(parent, t_up, x, var)
    tol_L, _ = tree_tols(parent, t_up, L_up, EPS32)
    N, G = a.n_obs, x.shape[1]
    assert ref_grad(parent, _times(res, len(parent)), x, var).L >= L_up.L - 1e-6 * N * G - tol_L


def test_snap_sets_tiny_times_to_exact_zero():
    """Upstream sets t < 1e-6 + 1e-6 to 0 after optimising (TH:3467); the upstream literal, not
    the module constant, so a wrong ``T_SNAP`` fails here."""
    parent, t, x, var, tree = _problem(seed=7, n=30, G=6)  # few genes: some edges are driven to 0
    res = optimise_times(tree, max_steps=300)
    tt = np.asarray(res.tree.t)[: len(parent)]
    assert T_SNAP == 1e-6 + 1e-6
    assert np.all((tt == 0.0) | (tt >= 1e-6 + 1e-6)) and np.any(tt[parent >= 0] == 0.0)


def test_capped_run_is_flagged_and_never_goes_uphill():
    """A run stopped by max_steps (2: the start and one line-search step) is not converged. The
    Armijo search accepts no rise of the float32 objective, so the true L may fall only by the
    objective's rounding at both ends: L_ref(end) ≥ L_ref(start) − tol_L(start) − tol_L(end)."""
    parent, t, x, var, tree = _problem()
    res = optimise_times(tree, rtol=0.0, atol=0.0, gtol=GTOL, max_steps=2, snap_zero=False)
    assert not res.converged and int(res.steps) == 2
    t0 = np.where(parent >= 0, np.clip(rounded(t), 1e-4, 1e4), 0.0)
    tt = _times(res, len(parent))
    start, end = ref_grad(parent, t0, x, var), ref_grad(parent, tt, x, var)
    tol_0, _ = tree_tols(parent, t0, start, EPS32)
    tol_1, _ = tree_tols(parent, tt, end, EPS32)
    assert end.L >= start.L - tol_0 - tol_1


def test_optimum_on_the_lower_bound():
    """A cherry with identical leaf data and tiny variances wants t = 0 on both edges: d(−L)/ds ≈
    −t·G/(2σ²) stays ≈ −3G at s = −15, so both edges end exactly on the lower bound (s0 ≥ log 1e-4,
    so the iterates reached it). KKT there: dL/dt ≤ tol_g; the other edges as in the gtol test."""
    rng = np.random.default_rng(9)
    parent, t, x, var = bm_tree(40, 20, rng)
    a, b = next((i, j) for i in range(40) for j in range(i + 1, 40) if parent[i] == parent[j])
    x[b], var[a], var[b] = x[a], 1e-7, 1e-7
    tree = ArrayTree.from_arrays(parent, t, x, var)
    x, var = rounded(x - centre64(tree)), rounded(var)
    res = optimise_times(tree, rtol=0.0, atol=0.0, gtol=GTOL, max_steps=2000, snap_zero=False)
    assert res.converged and int(res.steps) < 2000
    tt = _times(res, len(parent))
    ref = ref_grad(parent, tt, x, var)
    _, tol_g = tree_tols(parent, tt, ref, EPS32)
    low, high, free = _at_bounds(parent, tt, EPS32)
    assert low[a] and low[b] and not high.any()
    assert np.all(ref.grad[low] <= tol_g[low])
    assert np.all(np.abs(tt * ref.grad)[free] <= GTOL + (tt * tol_g)[free])


def test_overshooting_iterate_is_projected_not_frozen():
    """Branch lengths ~1e-6 with σ² = 1e-7: optima near s = −14, so line-search trial points cross
    s = −15. A zero-slope clip froze such edges outside the box (12 edges at the bound with
    t·dL/dt up to 2.1 here, reported as converged). With the projected gradient no bound edge may
    point inward beyond the solver's own threshold, t·dL/dt ≤ GTOL + t·tol_g, and a run reported
    as converged must be first-order optimal on the free edges too."""
    rng = np.random.default_rng(0)
    parent, t = random_tree(30, rng, t_scale=3e-6)
    var = np.full((len(parent), 20), 1e-7)
    z = np.zeros_like(var)
    for c in range(len(parent) - 2, -1, -1):  # parents have larger indices; the root is last
        z[c] = z[parent[c]] + np.sqrt(t[c]) * rng.normal(size=20)
    x = z + np.sqrt(var) * rng.normal(size=z.shape)
    tree = ArrayTree.from_arrays(parent, t, x, var)
    x, var = rounded(x - centre64(tree)), rounded(var)
    res = optimise_times(tree, rtol=0.0, atol=0.0, gtol=GTOL, max_steps=300, snap_zero=False)
    tt = _times(res, len(parent))
    ref = ref_grad(parent, tt, x, var)
    _, tol_g = tree_tols(parent, tt, ref, EPS32)
    low, high, free = _at_bounds(parent, tt, EPS32)
    assert low.any() and not high.any()  # the bound is active: the case is not vacuous
    assert np.all((tt * ref.grad)[low] <= GTOL + (tt * tol_g)[low])
    assert not res.converged or np.all(np.abs(tt * ref.grad)[free] <= GTOL + (tt * tol_g)[free])


def test_eval_masks_only_the_outward_gradient_at_each_bound():
    """`_eval`'s box mask (opt_times.py:39-40): at s = LOG_T_MAX the outward gradient (g < 0, the
    edge still wants more time) is zeroed, the inward one (g > 0) is kept; symmetric at LOG_T_MIN
    (outward is g > 0, inward g < 0). Deleting either clause makes this fail (tried as a mutant)."""
    from bonsai._engine import opt_times
    from bonsai._engine.tree import tree_loglik_grad as eng_grad

    # Upper bound: leaf 0 is a huge, near-exact outlier (t* far beyond the box) -> outward (g < 0)
    # at LOG_T_MAX; untouched edges keep their ordinary inward (g > 0) pull.
    parent, _, x, var, _ = _problem(seed=3, n=8, G=5)
    x[0], var[0] = x[0] + 1e4, np.float32(1e-6)
    tree = ArrayTree.from_arrays(parent, np.zeros(len(parent)), x, var)
    x, var = rounded(x - centre64(tree)), rounded(var)
    tree = ArrayTree.from_arrays(parent, np.zeros(len(parent)), x, var)
    args = (tree, eng_grad(tree).ell)
    s_max = jnp.full_like(jnp.asarray(tree.t), LOG_T_MAX)
    _, g_max = opt_times._eval(s_max, args)
    g_max = np.asarray(g_max)[: len(parent)]
    assert g_max[0] == 0.0
    assert np.any(g_max[parent >= 0] > 0.0)

    # Lower bound: a cherry with identical data and tiny variance wants t = 0 -> outward (g > 0)
    # at LOG_T_MIN; the other edges keep their ordinary inward (g < 0) pull.
    parent, t, x, var = bm_tree(10, 5, np.random.default_rng(9))
    a, b = next((i, j) for i in range(10) for j in range(i + 1, 10) if parent[i] == parent[j])
    x[b], var[a], var[b] = x[a], 1e-7, 1e-7
    tree = ArrayTree.from_arrays(parent, t, x, var)
    x, var = rounded(x - centre64(tree)), rounded(var)
    tree = ArrayTree.from_arrays(parent, t, x, var)
    args = (tree, eng_grad(tree).ell)
    s_min = jnp.full_like(jnp.asarray(tree.t), LOG_T_MIN)
    _, g_min = opt_times._eval(s_min, args)
    g_min = np.asarray(g_min)[: len(parent)]
    assert g_min[a] == 0.0 and g_min[b] == 0.0
    assert np.any(g_min[parent >= 0] < 0.0)


@pytest.mark.x64
def test_float64_run_is_first_order_optimal_at_upstream_gtol():
    """The same kernel in float64 reaches upstream's gtol = 1e-6 (unreachable in float32, A10): gtol
    rule only (Cauchy off), KKT as in the float32 test with tol_g at eps64."""
    parent, t, x, var = bm_tree(40, 20, np.random.default_rng(5))
    tree = ArrayTree.from_arrays(parent, t, x, var, dtype=jnp.float64)
    res = optimise_times(tree, rtol=0.0, atol=0.0, gtol=1e-6, max_steps=2000, snap_zero=False)
    assert res.converged and int(res.steps) < 2000 and res.tree.t.dtype == jnp.float64
    tt = _times(res, len(parent))
    ref = ref_grad(parent, tt, x - centre64(tree), var)
    _, tol_g = tree_tols(parent, tt, ref, EPS64)
    low, high, free = _at_bounds(parent, tt, EPS64)
    assert np.all(np.abs(tt * ref.grad)[free] <= 1e-6 + (tt * tol_g)[free])
    assert np.all(ref.grad[low] <= tol_g[low]) and np.all(ref.grad[high] >= -tol_g[high])
