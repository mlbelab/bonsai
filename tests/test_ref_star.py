import numpy as np
import pytest

from bonsai._ref import star as rstar
from bonsai._ref.star import as_if_root, node_message, star_em, star_loglik_grad, two_leaf_dlogl
from bonsai._ref.tree import tree_loglik_grad

from ._bounds import EPS64, ell_scale, fd_dell_dt, tree_height, tree_tols
from ._trees import random_leaf_data, random_tree, reroot


def _star(rng, C=5, G=6):
    return rng.normal(size=(C, G)), rng.uniform(0.05, 1.0, (C, G)), rng.uniform(0.0, 1.0, C)


def _fd(f, t, c, h):
    tp, tm = t.copy(), t.copy()
    tp[c] += h
    tm[c] -= h
    return (f(tp) - f(tm)) / (2 * h)


def test_node_message_is_the_product_of_child_gaussians():
    """Π_c N(x_c; y, var_c + t_c) ∝ N(y; x_n, 1/W_n): the log-ratio is an exact quadratic in y."""
    rng = np.random.default_rng(0)
    x, var, t = _star(rng)
    xn, Wn = node_message(x, var, t)

    def logprod(y):
        T = var + t[:, None]
        return np.sum(-0.5 * (y - x) ** 2 / T, axis=0)

    for y in (xn, xn + 0.3, xn - 1.1):
        np.testing.assert_allclose(logprod(y) - logprod(xn), -0.5 * Wn * (y - xn) ** 2, rtol=1e-12, atol=1e-12)


def test_star_gradient_matches_central_differences():
    """Central FD at h and h/2: |FD(h/2) − g| ≤ |FD(h) − FD(h/2)| (Richardson estimate of the O(h²)
    truncation, conservative) + 4·eps64·Σ|ell terms|/h (rounding of the two ell evaluations)."""
    rng = np.random.default_rng(1)
    x, var, t = _star(rng)
    ell, g = star_loglik_grad(x, var, t)
    f = lambda tt: star_loglik_grad(x, var, tt)[0]
    scale = 3 * abs(ell) + x.size
    for c in range(len(t)):
        h = 1e-3 * max(t[c], 1e-2)
        f1, f2 = _fd(f, t, c, h), _fd(f, t, c, h / 2)
        assert abs(f2 - g[c]) <= abs(f1 - f2) + 4 * EPS64 * scale / h
    # log-t parametrisation (TH:4934-4944): d ell/d log t = t · d ell/dt
    s = np.log(np.maximum(t, 1e-3))
    tt = np.exp(s)
    fs = lambda ss: star_loglik_grad(x, var, np.exp(ss))[0]
    g_t = star_loglik_grad(x, var, tt)[1]
    for c in range(len(t)):
        f1, f2 = _fd(fs, s, c, 1e-3), _fd(fs, s, c, 5e-4)
        assert abs(f2 - tt[c] * g_t[c]) <= abs(f1 - f2) + 4 * EPS64 * scale / 5e-4


def test_star_gradient_equals_two_leaf_derivative():
    """dell/dt_c = der2LeafTree(t_c, var_c + 1/(W − w_c), (x_rest − x_c)²) (maths ref §3)."""
    rng = np.random.default_rng(2)
    x, var, t = _star(rng)
    _, g = star_loglik_grad(x, var, t)
    for c in range(len(t)):
        rest = np.arange(len(t)) != c
        xr, Wr = node_message(x[rest], var[rest], t[rest])
        want = two_leaf_dlogl(t[c], var[c] + 1.0 / Wr, (xr - x[c]) ** 2)
        # two different fp64 formulas for one quantity: each ≤ 8·eps64·Σ_g |its terms|
        xn, Wn = node_message(x, var, t)
        w = 1.0 / (var[c] + t[c])
        T = t[c] + var[c] + 1.0 / Wr
        mag = np.sum(w * (w * (xn - x[c]) ** 2 + 1 + w / Wn)) + np.sum(1 / T + (xr - x[c]) ** 2 / T**2)
        assert abs(g[c] - want) <= 8 * EPS64 * mag


def test_as_if_root_equals_message_after_rerooting():
    """AIR(n) computed by subtraction = node n's message when the tree is rooted at n (maths ref §2)."""
    rng = np.random.default_rng(3)
    parent, t = random_tree(15, rng)
    x, var = random_leaf_data(parent, 4, rng)
    ref = tree_loglik_grad(parent, t, x, var)
    for n in range(15, len(parent) - 1):  # internal, non-root (random_tree: leaves first, root last)
        xa, Wa = as_if_root(ref.x[n], ref.W[n], t[n], ref.x_air[parent[n]], ref.W_air[parent[n]])
        rr = tree_loglik_grad(*reroot(parent, t, n), x, var)
        # messages are convex combinations: fp64 differences ≤ 8·(1+2H)·eps64 relative
        tol = 8 * (1 + 2 * tree_height(parent)) * EPS64
        np.testing.assert_allclose(Wa, rr.W[n], rtol=tol)
        np.testing.assert_allclose(ref.W_air[n], rr.W[n], rtol=tol)
        np.testing.assert_allclose(xa, rr.x[n], rtol=0, atol=tol * np.abs(x).max())
        np.testing.assert_allclose(ref.x_air[n], rr.x[n], rtol=0, atol=tol * np.abs(x).max())


def test_as_if_root_guard_returns_own_message():
    """max_g W_{p\\n} < 1e-12: the parent is fully determined by n; AIR = own message (TH:5321)."""
    xn, Wn = np.array([0.5, -1.0]), np.array([2.0, 3.0])
    t = 0.25
    wbar = 1.0 / (t + 1.0 / Wn)
    xa, Wa = as_if_root(xn, Wn, t, xn, wbar)  # parent AIR made of n alone
    assert np.array_equal(xa, xn) and np.array_equal(Wa, Wn)


def test_star_loglik_grad_xn_wn_override():
    """``xn``/``Wn`` passed as the values ``t`` implies give the default result bitwise (TH:5077);
    passing only one of the pair (silently ignored before this fix) now raises."""
    rng = np.random.default_rng(6)
    x, var, t = _star(rng)
    xn, Wn = node_message(x, var, t)
    default = star_loglik_grad(x, var, t)
    overridden = star_loglik_grad(x, var, t, xn=xn, Wn=Wn)
    assert overridden[0] == default[0]
    np.testing.assert_array_equal(overridden[1], default[1])
    with pytest.raises(ValueError, match="xn and Wn"):
        star_loglik_grad(x, var, t, xn=xn + 5.0)
    with pytest.raises(ValueError, match="xn and Wn"):
        star_loglik_grad(x, var, t, Wn=Wn)


def test_star_functions_match_upstream_array_path():
    """node_message, star_loglik_grad, as_if_root, two_leaf_dlogl against upstream's array-path
    functions (findNodeLtqsGivenLeafs, getLoglikAndGradStarTree, getLtqsAsIfRoot, der2LeafTree) on
    the same random inputs: identical maths, different op order/grouping, so an fp64 rounding bound
    applies (8·eps64·Σ|the quantity's own terms|), not bitwise; rtol=0 throughout."""
    from bonsai._core.tree_helpers import (
        der2LeafTree,
        findNodeLtqsGivenLeafs,
        getLoglikAndGradStarTree,
        getLtqsAsIfRoot,
    )

    rng = np.random.default_rng(7)
    x, var, t = _star(rng)
    xn, Wn = node_message(x, var, t)
    xn_up, Wn_up = findNodeLtqsGivenLeafs(ltqs_gi=x.T, ltqsVars_gi=var.T, t_i=t)
    # numpy's assert_allclose needs a scalar atol (array atol breaks its message formatting);
    # take the elementwise bound's max, which is still a valid (looser) bound for every entry.
    np.testing.assert_allclose(xn, xn_up, rtol=0, atol=8 * EPS64 * (np.abs(xn).max() + 1))
    np.testing.assert_allclose(Wn, Wn_up, rtol=0, atol=8 * EPS64 * Wn.max())

    ell, grad = star_loglik_grad(x, var, t)
    ell_up, grad_up = getLoglikAndGradStarTree(ltqs_gi=x.T, ltqsVars_gi=var.T, t_i=t, returnGrad=True)
    assert abs(ell - ell_up) <= 8 * EPS64 * (abs(ell) + x.size)
    np.testing.assert_allclose(grad, grad_up, rtol=0, atol=8 * EPS64 * (np.abs(grad).max() + 1))

    rest = np.arange(len(t)) != 0
    xr, Wr = node_message(x[rest], var[rest], t[rest])
    xa, Wa = as_if_root(xn, Wn, t[0], xr, Wr)
    xa_up, Wa_up = getLtqsAsIfRoot(xn, Wn, t[0], xr, Wr)
    np.testing.assert_allclose(xa, xa_up, rtol=0, atol=8 * EPS64 * (np.abs(xa).max() + 1))
    np.testing.assert_allclose(Wa, Wa_up, rtol=0, atol=8 * EPS64 * Wa.max())

    S = var[0] + 1.0 / Wr
    d2 = (xr - x[0]) ** 2
    val = two_leaf_dlogl(t[0], S, d2)
    val_up = der2LeafTree(t[0], S, d2)
    assert abs(val - val_up) <= 8 * EPS64 * np.sum(1 / S + d2 / S**2)


def _fixed_unary_zero_tree(rng, n_genes=4):
    """A hand-built tree with two unary internal nodes (4, 5) and three t=0 edges (0, 2, 4)."""
    parent = np.array([4, 6, 7, 7, 5, 6, 7, -1])
    t = np.array([0.0, 0.5, 0.0, 0.1, 0.0, 0.2, 0.4, 0.0])
    x, var = random_leaf_data(parent, n_genes, rng)
    return parent, t, x, var


@pytest.mark.parametrize("topology", ["random2", "random4", "fixed"])
def test_tree_gradient_matches_fd_and_upstream(topology):
    from bonsai._ref.tree import tree_loglik

    from ._upstream import upstream_grad

    rng = np.random.default_rng(4)
    if topology == "fixed":
        parent, t, x, var = _fixed_unary_zero_tree(rng)
    else:
        max_children = 2 if topology == "random2" else 4
        parent, t = random_tree(9, rng, max_children=max_children)
        x, var = random_leaf_data(parent, 4, rng)
    ref = tree_loglik_grad(parent, t, x, var)
    assert ref.L == tree_loglik(parent, t, x, var)
    f = lambda tt: tree_loglik(parent, tt, x, var)
    scale = 3 * abs(ref.L) + x.size
    for c in np.flatnonzero(parent >= 0):
        h = 1e-3 * max(t[c], 1e-2)
        f1, f2 = _fd(f, t, c, h), _fd(f, t, c, h / 2)
        assert abs(f2 - ref.grad[c]) <= abs(f1 - f2) + 4 * EPS64 * scale / h
    L_up, g_up = upstream_grad(parent, t, x, var)
    tol_L, tol_g = tree_tols(parent, t, ref, EPS64)  # same formulas, different summation order
    assert abs(ref.L - L_up) <= tol_L
    assert np.all(np.abs(ref.grad - g_up) <= tol_g)


def test_tree_gradient_raises_on_unary_root():
    r"""A unary root's only child has W_{\c} = W_root − w_child = 0 exactly (the root's own message
    is its AIR, TH:3414-3430): the gradient is undefined. Upstream hits the same division; this is
    not a fidelity break, just documented instead of returned as NaN."""
    parent = np.array([1, -1])
    t = np.array([0.5, 0.0])
    x = np.array([[1.0, 2.0], [0.0, 0.0]])
    var = np.array([[0.2, 0.3], [1.0, 1.0]])
    with pytest.raises(ValueError, match="unary root"):
        tree_loglik_grad(parent, t, x, var)


def star_fixture(rng, C=12, G=40):
    """(C, G) children from ``random_leaf_data``; child 0 = the mean of the others with var 1, so its
    derivative at t = 0 is negative and the star optimum clamps it (both KKT branches occur)."""
    x, var = random_leaf_data(np.zeros(C), G, rng)
    x[0], var[0] = x[1:].mean(axis=0), 1.0
    return x, var


# A Jacobi stop at |Δt| = tol leaves |dell/dt_c| ~ tol·H_cc ~ tol·gscale_c, so 1e-6·gscale separates
# a tight fixed point from one stopped early (tol ≥ 1e-4). The FD error must sit well below it:
# truncation ~h²·|third derivative| ~ 1e-8·gscale (h = 1e-4·t_c), and rounding of the two ell
# evaluations ≤ 2·eps64·Σ|ell terms|/(2h) central, /h forward,
# asserted ≤ ½·1e-6·gscale per child (the other half covers truncation).
KKT_REL = 1e-6


def _kkt_violations(x, var, t):
    """Children violating the star KKT conditions (dell/dt_c = 0 if t_c > 0, ≤ 0 at t_c = 0), by FD."""
    bad = []
    for c in range(len(t)):
        fd, gs = fd_dell_dt(x, var, t, c)
        h = 1e-4 * max(t[c], 1e-3)
        assert 2 * EPS64 * ell_scale(x, var, t) / h <= 0.5 * KKT_REL * gs  # FD resolves the bound
        if (abs(fd) if t[c] > 0 else fd) > KKT_REL * gs:
            bad.append(c)
    return bad


def test_ref_star_em_kkt_by_finite_differences():
    x, var = star_fixture(np.random.default_rng(3))
    t, ell, ok = star_em(x, var, tol=1e-12)
    assert ok and np.any(t == 0) and np.any(t > 0)  # the fixture covers both KKT branches
    assert _kkt_violations(x, var, t) == []
    assert ell == star_loglik_grad(x, var, t)[0]
    assert ell >= star_loglik_grad(x, var, np.ones(len(t)))[0]  # not worse than the start t = 1


def test_ref_star_em_loose_tol_fails_kkt():
    """Non-vacuity: stopping the sweeps at |Δt| < 1e-2 leaves a child outside the KKT bound."""
    x, var = star_fixture(np.random.default_rng(3))
    t, _, ok = star_em(x, var, tol=1e-2)
    assert ok and _kkt_violations(x, var, t) != []


def test_ref_star_em_stops_on_absolute_change(monkeypatch):
    """Stop rule max_c |Δt_c| < tol, absolute (TH:5000-5004): the iterates fed to each sweep's parent
    message are recorded; the last sweep changed t by < tol, the one before by ≥ tol. Data ×10 puts the
    times near 50-100, where a relative rule would stop sweeps earlier."""
    x, var = star_fixture(np.random.default_rng(3))
    x = 10 * x
    seen = []

    def spy(xx, vv, tt):
        seen.append(tt.copy())
        return node_message(xx, vv, tt)

    monkeypatch.setattr(rstar, "node_message", spy)
    t, _, ok = star_em(x, var, tol=1e-4)
    assert ok and len(seen) >= 3 and t.max() > 10
    assert np.max(np.abs(t - seen[-1])) < 1e-4 <= np.max(np.abs(seen[-1] - seen[-2]))


def test_ref_star_em_not_worse_than_upstream():
    """Upstream (tol 1e-4 sweeps, brentq 1e-4) stops within its own gap of the tight optimum."""
    from bonsai._core.tree_helpers import optimiseTStar

    x, var = star_fixture(np.random.default_rng(3))
    t_up, ell_up, _, _ = optimiseTStar(x.T.copy(), var.T.copy())
    t, ell, ok = star_em(x, var, tol=1e-12)
    assert ok and ell_up > -1e9  # both completed
    # same ell formula up to op order: fp64 rounding ≤ 8·eps64·Σ|terms| ≈ 1e-13·|ell| here; 1e-12·|ell| bounds it
    assert ell >= ell_up - 1e-12 * abs(ell_up)


def test_ref_star_em_bfgs_fallback(monkeypatch):
    """Sweeps not converged (max_sweeps=1): BFGS on log t (TH:5009); BFGS failing → (ones, −1e9, False)."""
    x, var = star_fixture(np.random.default_rng(3))
    t_bfgs, ell_bfgs, ok = star_em(x, var, max_sweeps=1)
    assert not ok and np.all(np.isfinite(t_bfgs)) and ell_bfgs > star_loglik_grad(x, var, np.ones(len(x)))[0]

    class Failed:
        success = False

    monkeypatch.setattr(rstar, "minimize", lambda *a, **k: Failed())
    t, ell, ok = star_em(x, var, max_sweeps=1)
    assert np.array_equal(t, np.ones(len(x))) and (ell, ok) == (-1e9, False)
