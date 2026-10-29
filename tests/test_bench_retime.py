import os
from pathlib import Path

import numpy as np
import pytest
from bench import retime as rt
from bench.retime import free_set, gap_terms, line_gain, retime
from scipy.optimize import minimize_scalar

from bonsai._ref.tree import tree_loglik_grad

from ._trees import bm_tree
from ._upstream import upstream_opt_times


def _case(n, g):
    """The reviewer's probe setup: bm_tree(n, g, rng(7)), start times ×U(0.3, 3) from rng(1)."""
    parent, t, x, var = bm_tree(n, g, np.random.default_rng(7))[:4]
    return parent, t * np.random.default_rng(1).uniform(0.3, 3, len(t)), x, var


def _bound(n, g):
    return 1e-2 * 1e-6 * n * g  # A16: 1% of the A1 slack


def _ref_L(parent, t, x, var):
    return tree_loglik_grad(parent, t, x, var).L


def test_free_set_rule():
    s = np.array([rt.LOG_T_MIN, rt.LOG_T_MIN, rt.LOG_T_MIN, 0.0, rt.LOG_T_MAX, rt.LOG_T_MAX, rt.LOG_T_MAX])
    pg = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0])
    assert free_set(s, pg).tolist() == [True, False, False, True, True, False, False]


# --- hessian_gap on quadratics: L = −½(s−c)ᵀA(s−c), so pg = −A(s−c), H = A and the exact remaining gain
# --- is ½(s−c)ᵀA(s−c) = ½·pgᵀA⁻¹pg (the Newton gap is exact for a quadratic).


def _spd(n, log10_kappa, rng):
    """D·M·D with M unit-diagonal (Jacobi-scaled) SPD from log-spaced eigenvalues; D ~ U(0.5, 2)."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    M = (Q * np.logspace(0, log10_kappa, n)) @ Q.T
    d = 1 / np.sqrt(np.diag(M))
    D = rng.uniform(0.5, 2.0, n) * d
    A = D[:, None] * M * D[None, :]
    return 0.5 * (A + A.T)


def _never(*a):
    raise AssertionError("no flat or floor direction here")


def _terms(w, U, pg, tau, L=_never, Lg=_never, s=None):
    """gap_terms with every coordinate free, at s = 0 unless given."""
    s = np.zeros(len(pg)) if s is None else s
    return gap_terms(w, U, pg, tau, s, np.arange(len(pg)), L, Lg)


@pytest.mark.parametrize("n, log10_kappa", [(30, 1), (300, 5)])
def test_gap_terms_exact_on_quadratic(n, log10_kappa):
    """All eigenvalues > τ: the gap is the exact ½·pgᵀA⁻¹pg. Tolerance: eigh is backward stable, so each λ_i
    carries an absolute error ≲ n·eps·‖A‖ and the term c_i²/λ_i a relative error ≲ n·eps·κ ≈ 7e-9 at n = 300,
    κ = 1e5 (eps 2.2e-16); rel 1e-7 leaves a ×15 margin."""
    rng = np.random.default_rng(n)
    A = _spd(n, log10_kappa, rng)
    delta = rng.standard_normal(n)
    pg = -A @ delta
    w, U = np.linalg.eigh(A)
    g = _terms(w, U, pg, 1e-10 * float(np.max(np.abs(A) @ np.abs(delta))))
    assert (g["flat_sum"], g["flat_joint"], g["floor"], g["n_flat"], g["n_floor"]) == (0.0, 0.0, 0.0, 0, 0)
    assert g["newton"] == pytest.approx(0.5 * delta @ A @ delta, rel=1e-7, abs=0)


def _flat_quadratic():
    """L(z) = b·z − ½ zᵀ diag(1, 1e-13) z at z = 0 with b = (1e-3, 1e-6); τ = 1e-10.

    λ_2 = 1e-13 ≤ τ: its Newton term ½·b_2²/λ_2 = 5 is meaningless (the maximiser is at z_2 = 1e7, far outside
    the box), the measured gain along +e_2 stops at the box: 1e-6·15 − ½·1e-13·15². λ_1 = 1 gives ½·b_1².
    """
    b, h = np.array([1e-3, 1e-6]), np.array([1.0, 1e-13])

    def L(z):
        return float(b @ z - 0.5 * (h * z) @ z)

    exact = 0.5 * b[0] ** 2 + (b[1] * rt.LOG_T_MAX - 0.5 * h[1] * rt.LOG_T_MAX**2)
    return b, h, L, exact


def test_gap_terms_use_measured_gain_for_eigenvalues_at_or_below_tau():
    """Mutant "Newton gap over all eigenvalues including λ ≤ τ" gives ≈ 5. Tolerance: L is O(1e-5) and evaluated
    exactly up to rounding (1e-21); the box-limit grid point is evaluated directly, rel 1e-9."""
    b, h, L, exact = _flat_quadratic()
    w, U = np.linalg.eigh(np.diag(h))
    g = _terms(w, U, b, 1e-10, L)
    assert g["n_flat"] == 1 and g["flat_joint"] == 0.0  # one flat direction: no joint search
    assert g["newton"] == pytest.approx(0.5 * b[0] ** 2, rel=1e-12, abs=0)
    assert g["newton"] + g["flat_sum"] == pytest.approx(exact, rel=1e-9, abs=0)


def _bump(beta=1e-12):
    """L(z) = −½z_0² + h(z_1), h(α) = 1e-2·exp(−(α − 5)²) − 1e-9·α + ½·β·α², at z = 0 (box ±15).

    H = −∇²L = diag(1, −β − 1e-2·98·e⁻²⁵) ≈ diag(1, −1.5e-11): one negative eigenvalue. h first decreases
    (rounding-level: −1e-12 at α = 1e-3) and only rises near α = 5, where the gain is ≈ 1e-2.
    """

    def L(z):
        a = z[1]
        return float(-0.5 * z[0] ** 2 + 1e-2 * np.exp(-((a - 5.0) ** 2)) - 1e-9 * a + 0.5 * beta * a**2)

    best = -minimize_scalar(lambda a: -L(np.array([0.0, a])), bounds=(4.0, 6.0), method="bounded").fun - L(np.zeros(2))
    return L, best


def _doubling_gain(L, s, u):
    """A16b's search along +u: double α from 1e-3 while L improves, then Brent on [0, 2α] (for the control)."""
    a, fa = 1e-3, L(s + 1e-3 * u)
    while a < 15 and L(s + 2 * a * u) > fa:
        a, fa = 2 * a, L(s + 2 * a * u)
    r = minimize_scalar(lambda z: -L(s + z * u), bounds=(0.0, 2 * a), method="bounded", options={"xatol": 1e-12})
    return max(fa, -r.fun) - L(s)


def test_line_gain_finds_a_gain_only_at_large_steps():
    """A16f: the log grid reaches the bump at α ≈ 5 that the doubling search (control) misses at its first,
    rounding-level non-increase. Mutant "line gain stops at first non-increase" fails. Tolerance: Brent
    resolves the peak to xatol 1e-12, so the gain matches the bracketed optimum to rounding (rel 1e-9)."""
    L, best = _bump()
    s0, u = np.zeros(2), np.array([0.0, 1.0])
    assert best > 9e-3 and _doubling_gain(L, s0, u) < 1e-9  # control: the old rule misses it
    assert line_gain(L, s0, u) == pytest.approx(best, rel=1e-9, abs=0)


def test_gap_terms_count_the_negative_direction():
    """The negative eigenvalue's measured gain (≈ 1e-2) enters the gap; mutant "skip the flat or negative
    directions" leaves only the Newton part 0 (pg_0 = 0) and the mutant "Newton over all eigenvalues" gives
    ½·(1e-9)²/(−1.5e-11) < 0."""
    L, best = _bump()
    w, U = np.linalg.eigh(np.diag([1.0, -1.5e-11]))
    g = _terms(w, U, np.array([0.0, -1e-9]), 1e-10, L)
    assert g["n_flat"] == 1 and g["newton"] == 0.0
    assert g["flat_sum"] == pytest.approx(best, rel=1e-9, abs=0)


def _coupled(eps=1e-9, a=1e-6, b=1e-7):
    """L(y) = ε(y_0 + y_1) + a·y_0²y_1² − b(y_0⁴ + y_1⁴) at y = 0, box ±15: H = 0 (two flat directions).

    Along either axis alone the best gain is ¾·ε·(ε/4b)^⅓ ≈ 1e-10; jointly L grows along the diagonal
    (a > 2b) up to the box corner, L(15, 15) ≈ 4.05e-2, which only a joint search on the span finds.
    """

    def L(y):
        return float(eps * (y[0] + y[1]) + a * y[0] ** 2 * y[1] ** 2 - b * (y[0] ** 4 + y[1] ** 4))

    def Lg(y):
        g0 = eps + 2 * a * y[0] * y[1] ** 2 - 4 * b * y[0] ** 3
        g1 = eps + 2 * a * y[1] * y[0] ** 2 - 4 * b * y[1] ** 3
        return L(y), np.array([g0, g1])

    return L, Lg, L(np.array([15.0, 15.0]))


def test_gap_terms_joint_flat_search_beats_separate_gains():
    """A16f detail: with ≥ 2 flat directions the flat term is max(Σ separate gains, joint L-BFGS gain on their span).
    Mutant "flat: sum only" leaves ≈ 2e-10. The joint search ends at the box corner (both coordinates clipped,
    gradient zeroed), where L is evaluated directly: rel 1e-9."""
    L, Lg, corner = _coupled()
    w, U = np.linalg.eigh(np.diag([1e-14, 2e-14]))  # both ≤ τ; U = I
    g = _terms(w, U, np.array([1e-9, 1e-9]), 1e-10, L, Lg)
    assert g["n_flat"] == 2 and g["flat_sum"] < 1e-9
    assert g["flat_joint"] == pytest.approx(corner, rel=1e-9, abs=0)


def _floor_case(a=1.0, b=1e-3, s0=-10.0):
    """L(s) = −a·e^{s_0} − ½s_1² + b·s_1 at s = (s0, 0): near the floor L ≈ L∞ − a·eˢ.

    H = diag(a·e^{s0}, 1) (both > τ), pg = (−a·e^{s0}, b). The Newton term of edge 0 is ½·a·e^{s0}, but its true
    gain (down to the box floor −15) is a·(e^{s0} − e⁻¹⁵) ≈ 2× that.
    """

    def Lg(z):
        return float(-a * np.exp(z[0]) - 0.5 * z[1] ** 2 + b * z[1]), np.array([-a * np.exp(z[0]), b - z[1]])

    s = np.array([s0, 0.0])
    true0 = a * (np.exp(s0) - np.exp(rt.LOG_T_MIN))
    return s, Lg, np.diag([a * np.exp(s0), 1.0]), np.array([-a * np.exp(s0), b]), true0, 0.5 * b**2


def test_gap_terms_floor_term_catches_what_newton_understates():
    """A16f detail: the joint gain over free edges with s < −9 is added. Newton gives half the floor edge's true gain;
    the floor term recovers it, so the gap is at least the true remaining gain. Mutant "no floor term" leaves the gap
    below it. The floor search runs to the bound −15 (gradient −a·e⁻¹⁵ = −3e-7 never vanishes): rel 1e-9."""
    s, Lg, H, pg, true0, true1 = _floor_case()
    w, U = np.linalg.eigh(H)
    g = gap_terms(w, U, pg, 1e-20, s, np.arange(2), _never, Lg)
    assert g["n_floor"] == 1 and g["n_flat"] == 0
    assert g["newton"] == pytest.approx(0.5 * H[0, 0] + true1, rel=1e-9, abs=0)  # ½·a·e^{s0}: ≈ half of true0
    assert g["floor"] == pytest.approx(true0, rel=1e-9, abs=0)
    assert g["newton"] + g["floor"] >= true0 + true1
    g = gap_terms(np.ones(1), np.ones((1, 1)), pg[1:], 1e-20, s, np.array([1]), _never, Lg)  # edge 0 not free
    assert g["n_floor"] == 0 and g["floor"] == 0.0


def test_line_gain_projects_out_active_bounds():
    """A16c: the reviewer's case. Along +p the first component pushes the active lower bound outward; it
    is zeroed, so the step runs along e_1 and finds the maximum of z1 − ½z1² at z1 = 1: gain 0.5
    (without the projection the box caps the step at 0)."""
    gain = line_gain(lambda z: float(z[1] - 0.5 * z[1] ** 2), np.array([-15.0, 0.0]), np.array([-1e-3, 1.0]))
    assert gain == pytest.approx(0.5, rel=1e-12, abs=0)


def test_line_gain_respects_the_box():
    """The step stops at the box: L increases without limit along +e_0, so the gain is L at s_0 = 15."""
    s0 = np.array([14.0, 0.0])
    assert line_gain(lambda z: float(z[0]), s0, np.array([1.0, 0.0])) == pytest.approx(1.0, rel=1e-12, abs=0)
    assert line_gain(lambda z: float(-(z[0] ** 2)), np.zeros(2), np.array([1.0, 0.0])) == 0.0  # at a maximum


def _dense_fd_hessian(parent, t, x, var, free):
    """H = −∇²L in s on the free edges by central FD of the reference gradient (h = 1e-4)."""
    e = np.flatnonzero(parent >= 0)
    s = np.log(t[e])

    def g(z):
        tt = np.zeros(len(parent))
        tt[e] = np.exp(z)
        return (tree_loglik_grad(parent, tt, x, var).grad * tt)[e]

    f = np.flatnonzero(free)
    H = np.zeros((len(f), len(f)))
    for j, i in enumerate(f):
        sp, sm = s.copy(), s.copy()
        sp[i] += 1e-4
        sm[i] -= 1e-4
        H[:, j] = -(g(sp) - g(sm))[f] / 2e-4
    return H, g(s)[f]


@pytest.mark.x64
def test_free_hessian_matches_fd_of_the_reference_gradient(monkeypatch):
    """The batched-JVP Hessian of the engine gradient equals central FD of the independent ``_ref`` gradient, on a
    free subset (a dropped row/column mapping would mismatch). HESS_MAX_BATCH = 4 forces several batches with a
    padded last one (13 free columns). Tolerance: FD truncation h²/6·|∂³g| ≈ 2e-9·scale and rounding
    eps·|g|/h ≈ 2e-12·scale per entry; atol 1e-7·scale leaves ≥ ×50."""
    parent, t, x, var = _case(12, 20)
    e = np.flatnonzero(parent >= 0)
    s = np.log(t[e])
    free = np.zeros(len(e), bool)
    free[np.random.default_rng(0).choice(len(e), 13, replace=False)] = True
    monkeypatch.setattr(rt, "HESS_MAX_BATCH", 4)
    _, _, grad_s = rt._objective(parent, t, x, var)
    H = rt.free_hessian(grad_s, s, np.flatnonzero(free))
    H_fd, _ = _dense_fd_hessian(parent, t, x, var, free)
    scale = rt._grad_scale(parent, t, tree_loglik_grad(parent, t, x, var))
    np.testing.assert_allclose(H, H_fd, rtol=0, atol=1e-7 * scale)
    assert np.max(np.abs(H - H.T)) <= 1e-12 * np.max(np.abs(H))  # exact second derivatives: symmetric to rounding


# --- retime on trees


def test_retime_refuses_without_x64():
    parent, t0, x, var = _case(6, 5)
    with pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
        retime(parent, t0, x, var)


@pytest.mark.x64
def test_retime_meets_kkt_and_matches_reference_L():
    parent, t0, x, var = _case(30, 50)
    r = retime(parent, t0, x, var)
    assert r.status == "converged", r.reason
    assert r.L >= r.L_start
    assert r.gap <= _bound(30, 50) and r.extra_dL <= _bound(30, 50)  # A16 acceptance
    assert r.L == pytest.approx(_ref_L(parent, r.t, x, var), rel=1e-10, abs=0)


@pytest.mark.x64
def test_retime_at_least_upstream_opt_times():
    """Upstream's optTimes (tol 1e-6, ≤ 100 iterations, 2e-6 snap) never beats the re-timer by more than
    the A1 slack per entry (the gate's own scale)."""
    parent, t0, x, var = _case(30, 50)
    t_up, _ = upstream_opt_times(parent, t0, x, var)
    assert retime(parent, t0, x, var).L >= _ref_L(parent, t_up, x, var) - 1e-6 * 30 * 50


@pytest.mark.x64
def test_retime_zero_length_edges_start_inside_box():
    parent, t0, x, var = _case(30, 50)
    t0[[0, 5, 33, 40]] = 0.0  # two leaf edges, two internal edges
    r = retime(parent, t0, x, var)
    edge = parent >= 0
    assert np.all(r.t[edge] >= np.exp(rt.LOG_T_MIN)) and r.t[parent < 0] == 0.0
    assert r.status == "converged", r.reason


@pytest.mark.x64
def test_retime_unconverged_is_invalid(monkeypatch):
    """maxiter = 1 in every run: each stop (status 1) restarts (A16c), every restart still gains more than
    the bound, so the 10 restarts run out -> invalid."""
    monkeypatch.setattr(rt, "OPTIONS", rt.OPTIONS | {"maxiter": 1})
    parent, t0, x, var = _case(30, 50)
    r = retime(parent, t0, x, var)
    assert r.status == "invalid" and "10 restarts" in r.reason and r.restarts == 10


@pytest.mark.x64
def test_retime_other_scipy_status_is_invalid(monkeypatch):
    """Statuses other than 0, 1, 2 (e.g. a NaN stop) are INVALID without restarting (A16c)."""
    real = rt.minimize

    def fake(*a, **k):
        r = real(*a, **k)
        r.status = 3
        return r

    monkeypatch.setattr(rt, "minimize", fake)
    r = retime(*_case(30, 50))
    assert r.status == "invalid" and "scipy status 3" in r.reason and r.restarts == 0


def _early_stops(monkeypatch, n_early, status=0):
    """The first ``n_early`` L-BFGS-B runs stop after 30 iterations (scipy reports status 1) and report
    ``status`` instead unless it is None (a premature stop)."""
    real, calls = rt.minimize, []

    def fake(*a, **k):
        calls.append(1)
        if len(calls) > n_early:
            return real(*a, **k)
        r = real(*a, **(k | {"options": k["options"] | {"maxiter": 30}}))
        if status is not None:
            r.status = status
        return r

    monkeypatch.setattr(rt, "minimize", fake)


@pytest.mark.x64
@pytest.mark.parametrize("status", [0, None])  # None: scipy's own status 1 (maxiter), A16c
def test_retime_restarts_until_a_restart_gains_at_most_the_bound(monkeypatch, status):
    """A16b/c: after a premature stop (status 0, or 1 at maxiter) the optimiser restarts from the stopping
    point while a restart gains more than the bound; the reported point is the last run's end and passes."""
    parent, t0, x, var = _case(30, 50)
    _early_stops(monkeypatch, 2, status)
    r = retime(parent, t0, x, var)
    assert r.status == "converged", r.reason
    assert r.restarts >= 2 and r.extra_dL <= _bound(30, 50) and r.gap <= _bound(30, 50)


@pytest.mark.x64
def test_retime_restarts_exhausted_is_invalid(monkeypatch):
    parent, t0, x, var = _case(30, 50)
    _early_stops(monkeypatch, 2)
    monkeypatch.setattr(rt, "MAX_RESTARTS", 1)  # the one restart is itself premature and gains > bound
    r = retime(parent, t0, x, var)
    assert r.status == "invalid" and "1 restarts" in r.reason and r.extra_dL > _bound(30, 50)


@pytest.mark.x64
def test_retime_acceptance_needs_every_test(monkeypatch):
    """Floor alone is not enough: with the scipy floor met, a gap above the bound (in either part) or an
    engine/reference L mismatch each makes the result invalid; a flat or negative direction alone does not."""
    parent, t0, x, var = _case(30, 50)
    b = _bound(30, 50)
    ep = rt.lbfgs(parent, t0, x, var)

    def terms(newton, flat_sum=0.0, flat_joint=0.0, floor=0.0, n_flat=0, n_floor=0):
        d = {"newton": newton, "flat_sum": flat_sum, "flat_joint": flat_joint, "floor": floor}
        return lambda *a, **k: d | {"n_flat": n_flat, "n_floor": n_floor}

    monkeypatch.setattr(rt, "gap_terms", terms(0.5 * b))
    assert rt.accept(parent, x, var, ep).status == "converged"  # control: floor and restarts pass here
    monkeypatch.setattr(rt, "gap_terms", terms(0.2 * b, 0.1 * b, 0.3 * b, 0.1 * b, 3, 2))
    r = rt.accept(parent, x, var, ep)
    assert r.status == "converged" and r.gap == pytest.approx(0.6 * b, rel=1e-15, abs=0)  # newton + max + floor
    assert (r.gap_flat, r.gap_flat_sum, r.gap_flat_joint, r.gap_floor) == (0.3 * b, 0.1 * b, 0.3 * b, 0.1 * b)
    assert (r.n_flat, r.n_floor) == (3, 2)
    for t in [terms(2 * b), terms(0.5 * b, 0.6 * b), terms(0.5 * b, 0.0, 0.6 * b), terms(0.5 * b, floor=0.6 * b)]:
        monkeypatch.setattr(rt, "gap_terms", t)
        r = rt.accept(parent, x, var, ep)
        assert r.status == "invalid" and "gap" in r.reason
    monkeypatch.setattr(rt, "gap_terms", terms(0.5 * b))
    monkeypatch.setattr(rt, "L_MATCH_REL", -1.0)  # every |ΔL| exceeds it: checks the test is wired in
    assert "mismatch" in rt.accept(parent, x, var, ep).reason


@pytest.mark.x64
def test_accept_wiring_on_a_tree(monkeypatch):
    """accept() on bm_tree(30, 50) after L-BFGS hands ``gap_terms`` the right inputs: the eigenpairs of its own
    symmetrised H, pg from ``_ref`` on the free set, τ = 1e-10·scale, the endpoint s, the free indices, and L/Lg equal
    to the ``_ref`` value and t·∂L/∂t. The Hessian is checked against FD in
    test_free_hessian_matches_fd_of_the_reference_gradient; the terms on synthetic cases above. Same arithmetic on
    CPU, so the comparisons are exact or to rounding (rel 1e-12)."""
    parent, t0, x, var = _case(30, 50)
    ep = rt.lbfgs(parent, t0, x, var)
    seen = {}
    real_h, real_g = rt.free_hessian, rt.gap_terms

    def spy_h(grad_s, s, fi):
        seen["H"] = real_h(grad_s, s, fi)
        return seen["H"]

    def spy_g(w, U, pg, tau, s, fi, L, Lg):
        seen.update(w=w, U=U, pg=pg, tau=tau, s=s, fi=fi, L=L, Lg=Lg)
        return real_g(w, U, pg, tau, s, fi, L, Lg)

    monkeypatch.setattr(rt, "free_hessian", spy_h)
    monkeypatch.setattr(rt, "gap_terms", spy_g)
    r = rt.accept(parent, x, var, ep)
    e = np.flatnonzero(parent >= 0)
    R = tree_loglik_grad(parent, r.t, x, var)
    pg = (R.grad * r.t)[e]
    fi = np.flatnonzero(free_set(ep.s, pg))
    np.testing.assert_array_equal(seen["fi"], fi)
    np.testing.assert_array_equal(seen["s"], ep.s)
    np.testing.assert_array_equal(seen["pg"], pg[fi])
    w, U = np.linalg.eigh(0.5 * (seen["H"] + seen["H"].T))
    np.testing.assert_array_equal(seen["w"], w)
    assert seen["tau"] == 1e-10 * rt._grad_scale(parent, r.t, R)
    z = ep.s + 0.01 * np.random.default_rng(0).standard_normal(len(e))
    tz = rt.times_of(parent, z)
    Rz = tree_loglik_grad(parent, tz, x, var)
    assert seen["L"](z) == seen["Lg"](z)[0] == Rz.L
    np.testing.assert_allclose(seen["Lg"](z)[1], (Rz.grad * tz)[e], rtol=1e-12, atol=0)
    assert (r.n_free, r.lambda_min) == (len(fi), w[0]) and r.n_flat >= 1  # this endpoint has a flat eigenvalue
    assert r.gap == r.gap_newton + max(r.gap_flat_sum, r.gap_flat_joint) + r.gap_floor and r.status == "converged"
    assert 0 <= r.asym <= 1e-12 and r.hess_s > 0 and r.eigh_s >= 0


@pytest.mark.x64
def test_main_saves_the_endpoint_before_the_acceptance(tmp_path, monkeypatch):
    """A16f detail: a fresh endpoint is written to OUT.ep.npz before ``accept`` runs, so it survives a crash there."""
    parent, t0, x, var = _case(12, 20)
    np.savez(tmp_path / "in.npz", parent=parent, t=t0, x=x, var=var)
    ep = rt.lbfgs(parent, t0, x, var)
    monkeypatch.setattr(rt, "lbfgs", lambda *a: ep)

    def boom(*a):
        raise RuntimeError("acceptance failed")

    monkeypatch.setattr(rt, "accept", boom)
    with pytest.raises(RuntimeError, match="acceptance failed"):
        rt.main([str(tmp_path / "in.npz"), str(tmp_path / "out.npz")])
    saved = rt.load_endpoint(rt.endpoint_path(tmp_path / "out.npz"))
    assert rt.endpoint_path(tmp_path / "out.npz") == str(tmp_path / "out.ep.npz")
    np.testing.assert_array_equal(saved.s, ep.s)
    assert (saved.nit, saved.restarts, saved.message) == (ep.nit, ep.restarts, ep.message)
    assert not (tmp_path / "out.npz").exists()


@pytest.mark.x64
def test_endpoint_round_trip_and_accept_from_it(tmp_path):
    """The subprocess file carries the endpoint (``ep_`` fields); accepting the loaded endpoint reproduces the
    result (the acceptance is a deterministic function of the endpoint on CPU)."""
    parent, t0, x, var = _case(12, 20)
    ep = rt.lbfgs(parent, t0, x, var)
    r = rt.accept(parent, x, var, ep)
    rt.save(tmp_path / "o.npz", r, ep)
    r2, ep2 = rt.load(tmp_path / "o.npz")
    np.testing.assert_array_equal(ep2.s, ep.s)
    np.testing.assert_array_equal(ep2.history, ep.history)
    assert ep2.history.shape == (ep.restarts + 1, 3) and ep2.message == ep.message
    r3 = rt.accept(parent, x, var, ep2)
    assert (r3.L, r3.gap, r3.status) == (r.L, r.gap, r.status) == (r2.L, r2.gap, r2.status)


# --- A16f on the stored real endpoints (the eeccg/a16c diagnostics); skipped where the files are absent.

_BONSAI = Path(__file__).resolve().parents[2]
_FIXTURES = Path(  # env BONSAI_FIXTURES overrides (the directory holding the a16f endpoint files)
    os.environ.get(
        "BONSAI_FIXTURES", "/projects/site/pred/ihb-g-mlbe/USERS/fleckj/projects/bonsai/data/test_fixtures/a16f"
    )
)


def _real_case(which):
    from bench.gate import load_arm
    from bench.run import _leaf_arrays

    from bonsai._ref.newick import parse_newick

    if which == "eec":
        up_dir, ep_file = (
            _BONSAI / "data/eec_vasa_sub/bonsai_workdir/results",
            _FIXTURES / "lbfgs_ckpt_sequential.npz",
        )
        runs = _BONSAI / "data/m2_gate/runs/eec_vasa_sub_sequential_M256_m1"
    else:
        up_dir, ep_file = (
            _BONSAI / "bonsai-m2-baselines2/bench/results/upstream_redo_sim_1k_s1",
            _FIXTURES / "lbfgs_ckpt_1k_s1.npz",
        )
        runs = None
    if not (up_dir.exists() and ep_file.exists() and (runs is None or runs.exists())):
        pytest.skip(f"stored {which} endpoint or its arms not available")
    up = load_arm(up_dir if which == "eec" else next(p for p in up_dir.iterdir() if p.is_dir()))
    arm = up if runs is None else load_arm(next(p for p in runs.iterdir() if p.is_dir()), load_arrays=False)
    p, t, nm = parse_newick(arm["stage_newick"]["redo_starry"])
    X, V = _leaf_arrays(p, nm, up["cells"], up["x"], up["var"])
    with np.load(ep_file) as z:
        ep = rt.Endpoint(
            z["s"], 0.0, int(z["nit"]), int(z["restarts"]), float(z["extra_dL"]), True, "stored", np.zeros((0, 3))
        )
    return p, X, V, ep


@pytest.mark.x64
@pytest.mark.slow
def test_a16f_finds_the_negative_direction_gain_on_upstream_1k_seed1():
    """The A16f diagnostic: the exact free-set Hessian at the stored 1k seed-1 upstream endpoint has one negative
    eigenvalue, −3.19e-6, whose measured gain is 1.2285e-3 (the doubling search along CG's direction found 1e-8);
    still below the bound 1.851e-2, so the arm is accepted. Reference values: the independent dense build and NumPy
    eigh of the diagnostic (eeccg/hess_1k_s1.json). Tolerances: λ is resolved to n·eps·‖H‖ ≈ 5e-10 absolute (GPU
    scatter order included), rel 1e-3 on 3.19e-6 is ×6 above that; the gain is a difference of L ≈ −2.07e6 values
    with rounding ≈ 1e-9, rel 1e-3 = 1.2e-6 absolute."""
    p, X, V, ep = _real_case("1k")
    r = rt.accept(p, X, V, ep)
    assert r.n_free == 1817 and r.n_flat == 1
    assert r.lambda_min == pytest.approx(-3.1931e-6, rel=1e-3, abs=0)
    assert r.gap_flat == pytest.approx(1.22845e-3, rel=1e-3, abs=0)
    assert r.gap_newton == pytest.approx(1.62882e-7, rel=1e-3, abs=0)
    assert r.status == "converged", r.reason


@pytest.mark.x64
@pytest.mark.slow
def test_a16f_gap_on_eec_sequential_endpoint():
    """eec_vasa_sub jax sequential: the CG gap needed 2,082 iterations (INVALID at the 2,000 cap). The exact Newton gap is
    7.659e-5; λ_min = 4.7e-8 > 0 but ≤ τ, so one direction gets a (tiny) measured gain; 12 free edges with s < −9
    jointly gain 9.75e-5 (the reviewer's measurement; Newton gives about half of that near the floor), so the A16f
    gap is ≈ 1.74e-4, still 0.8% of the bound. Tolerance rel 1e-3: the gains are differences of L ≈ −2.9e6 with
    rounding ≈ 1e-9 (≤ 1e-5 relative here) and the optimiser's own stopping (gtol 1e-12)."""
    p, X, V, ep = _real_case("eec")
    r = rt.accept(p, X, V, ep)
    assert r.n_free == 799 and r.lambda_min > 0 and (r.n_flat, r.n_floor) == (1, 12)
    assert r.gap_newton == pytest.approx(7.659e-5, rel=1e-3, abs=0)
    assert r.gap_floor == pytest.approx(9.75e-5, rel=1e-3, abs=0)
    assert r.gap == r.gap_newton + r.gap_flat + r.gap_floor <= rt._bound(p, X)
