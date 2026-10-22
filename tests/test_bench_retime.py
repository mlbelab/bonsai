import numpy as np
import pytest
from bench import retime as rt
from bench.retime import free_set, line_gain, remaining_gap, retime

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


# --- remaining_gap on quadratics: L = −½(s−c)ᵀA(s−c), so pg = −A(s−c), H = A and the exact remaining gain
# --- is ½(s−c)ᵀA(s−c) = ½·pgᵀA⁻¹pg (the Newton gap is exact for a quadratic).


def _spd(n, log10_kappa, rng, neg=False):
    """D·M·D with M unit-diagonal (Jacobi-scaled) SPD from log-spaced eigenvalues; D ~ U(0.5, 2)."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.logspace(0, log10_kappa, n)
    if neg:
        lam[0] = -1.0
    M = (Q * lam) @ Q.T
    d = 1 / np.sqrt(np.abs(np.diag(M)))
    M = d[:, None] * M * d[None, :]
    D = rng.uniform(0.5, 2.0, n)
    A = D[:, None] * M * D[None, :]
    return 0.5 * (A + A.T), M


def _quad_case(n, log10_kappa, seed, neg=False):
    rng = np.random.default_rng(seed)
    A, M = _spd(n, log10_kappa, rng, neg)
    delta = rng.standard_normal(n)
    pg = -A @ delta
    scale = float(np.max(np.abs(A) @ np.abs(delta)))  # Σ_j |terms of ∂L/∂s_i|, max over i
    return A, M, pg, scale, 0.5 * delta @ A @ delta


@pytest.mark.x64
def test_gap_estimate_on_quadratic():
    # (1) n = 30, well conditioned: the residual rule fires (‖r‖ ≤ 1e-6·‖pg‖); the gap error ½rᵀA⁻¹r is then
    # ≤ 1e-12·κ(A) relative (κ ≲ 1e2 here), far inside rel 1e-6.
    # bound=0.0 throughout: disables the A16d plateau-acceptance alternative (0.1·bound = 0), so these
    # cases exercise only the residual/curvature stop rules, unaffected by A16d.
    A, _, pg, scale, exact = _quad_case(30, 1, seed=0)
    gap, it, ok, curv = remaining_gap(pg, np.ones(30, bool), lambda v: A @ v, scale, 0.0)
    assert ok and not curv and it < 2000
    assert gap == pytest.approx(exact, rel=1e-6, abs=0)
    # (2) n = 300, Jacobi-scaled κ ≈ 1e5: converged by the pre-declared rule, within 1e-3 of the dense solve
    A, M, pg, scale, exact = _quad_case(300, 5, seed=1)
    assert 3e4 <= np.linalg.cond(M) <= 3e5
    gap, it, ok, _ = remaining_gap(pg, np.ones(300, bool), lambda v: A @ v, scale, 0.0)
    assert ok and it < 2000
    assert gap == pytest.approx(0.5 * pg @ np.linalg.solve(A, pg), rel=1e-3, abs=0)
    assert gap == pytest.approx(exact, rel=1e-3, abs=0)
    # (3) the same with max_cg = 20: cap hit -> not converged
    gap, it, ok, _ = remaining_gap(pg, np.ones(300, bool), lambda v: A @ v, scale, 0.0, max_cg=20)
    assert not ok and it == 20
    # (4) one negative eigenvalue (pg has a component on it): curvature stop before the cap; without a
    # line search the estimate is not converged (the line-search path is test_negative_curvature_line_search)
    A, _, pg, scale, _ = _quad_case(30, 1, seed=2, neg=True)
    gap, it, ok, curv = remaining_gap(pg, np.ones(30, bool), lambda v: A @ v, scale, 0.0)
    assert curv and not ok and it < 2000


@pytest.mark.x64
def test_gap_free_set_projection():
    # (5) a bound coordinate with outward gradient is fixed: hvp never sees it and the gap equals the
    # reduced problem's ½·pg_fᵀA_ff⁻¹pg_f (A couples k to the others, so a missing projection changes it).
    A, _, pg, scale, _ = _quad_case(30, 1, seed=3)
    k = 4
    s = np.zeros(30)
    s[k] = rt.LOG_T_MIN
    pg[k] = -abs(pg[k])  # points out of the box at the lower bound
    free = free_set(s, pg)
    assert free.sum() == 29 and not free[k]
    seen = []

    def hvp(v):
        seen.append(v[k])
        return A @ v

    gap, _, ok, _ = remaining_gap(pg, free, hvp, scale, 0.0)  # bound=0.0: not exercising A16d here
    f = np.flatnonzero(free)
    assert ok and set(seen) == {0.0}
    assert gap == pytest.approx(0.5 * pg[f] @ np.linalg.solve(A[np.ix_(f, f)], pg[f]), rel=1e-6, abs=0)


def test_free_set_rule():
    s = np.array([rt.LOG_T_MIN, rt.LOG_T_MIN, rt.LOG_T_MIN, 0.0, rt.LOG_T_MAX, rt.LOG_T_MAX, rt.LOG_T_MAX])
    pg = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0])
    assert free_set(s, pg).tolist() == [True, False, False, True, True, False, False]


def _quartic_case(mu):
    """L(z) = Σ_j (b_j z_j − ½ z_j²) + z_k + ½ z_k² − ¼·mu·z_k⁴ at z = 0 (inside the box), k = last.

    pg = (b, 1) and H = −∇²L = diag(1, …, 1, −1): pgᵀH·pg = Σb² − 1 < 0, so CG meets negative
    curvature at its first direction p = pg (model gain so far 0) and the gap is the line-search gain.
    """
    b = np.full(5, 0.1)
    pg = np.r_[b, 1.0]
    h = np.r_[np.ones(5), -1.0]

    def L(z):
        return float(b @ z[:5] - 0.5 * z[:5] @ z[:5] + z[5] + 0.5 * z[5] ** 2 - 0.25 * mu * z[5] ** 4)

    u = pg / np.linalg.norm(pg)  # φ(α) = L(α·u) is a quartic: maximise it exactly over its stationary points
    c1, c2, c4 = pg @ u, -0.5 * (h * u) @ u, -0.25 * mu * u[5] ** 4
    roots = np.roots([4 * c4, 0, 2 * c2, c1])
    a = roots[np.abs(roots.imag) == 0].real
    assert np.all(np.abs(a) < rt.LOG_T_MAX)  # the maximiser lies inside the box
    best = float(np.max(c1 * a + c2 * a**2 + c4 * a**4))
    return pg, h, L, best


@pytest.mark.x64
@pytest.mark.parametrize("mu, valid", [(1e4, True), (1.0, False)])
def test_negative_curvature_line_search(mu, valid):
    """A16b: on non-positive curvature the gap = CG model gain so far + the measured exact line-search
    gain along ±p, the estimate counts as converged, and only gap ≤ bound decides validity. Here the
    bound is 0.1: the gain is ≈ 0.035 at mu = 1e4 (valid) and ≈ 1.5 at mu = 1 (INVALID).
    Tolerance: Brent locates α to 1e-12 (gain error ≤ ½|φ''|·1e-24) and L is O(1), so rel 1e-9 holds
    with fp64 rounding to spare."""
    pg, h, L, best = _quartic_case(mu)
    s0 = np.zeros(6)
    gap, it, ok, curv = remaining_gap(
        pg, np.ones(6, bool), lambda v: h * v, 1.0, 0.0, line_search=lambda p: line_gain(L, s0, p)
    )  # bound=0.0: the curvature stop fires on CG's first direction, before the A16d window clause is ever reached
    assert curv and ok and it == 1
    assert gap == pytest.approx(best, rel=1e-9, abs=0)
    assert (gap <= 0.1) == valid


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


def _ill_conditioned_cg_run():
    """Diagonal H: a compact cluster (50 eigenvalues in [1, 2], pg 1) holds almost all of the gap and
    is resolved early; 500 eigenvalues in [1e3, 1e5] (pg 1) hold almost none of it but dominate the
    residual, which decays slowly. Returns (lam, b, k, gaps, res): k is the first iteration where the
    bare window rule (gap change ≤ 1e-3 relative over the last 50 iterations) is met, gaps/res are
    1-indexed-by-iteration (gaps[0] = 0 before any step, res[i - 1] = ‖r‖ after iteration i)."""
    lam = np.r_[np.linspace(1, 2, 50), np.logspace(3, 5, 500)]
    b = np.ones(lam.size)
    d, r, p, res, gaps = np.zeros_like(b), b.copy(), b.copy(), [], [0.0]
    for _ in range(600):
        Hp = lam * p
        a = (r @ r) / (p @ Hp)
        d, r_new = d + a * p, r - a * Hp
        p, r = r_new + (r_new @ r_new) / (r @ r) * p, r_new
        res.append(np.linalg.norm(r))
        gaps.append(0.5 * b @ d)
    k = next(i for i in range(52, len(gaps)) if abs(gaps[i] - gaps[i - 50]) <= 1e-3 * gaps[i])
    return lam, b, k, gaps, res


def test_gap_window_clause_needs_small_residual():
    """The window rule also needs ‖r‖ ≤ 1e-3·‖pg‖ when the gap is not yet small vs. the bound. The
    gap-only window rule fires (at k) while ‖r‖ > 1e-3·‖pg‖; with bound = 100 the plateaued gap (≈17.4,
    see test_gap_plateau_converges_below_point_one_bound) stays above 0.1·bound = 10, so the A16d
    alternative cannot fire either and the stop must wait for the residual, exactly as before A16d."""
    lam, b, k, gaps, res = _ill_conditioned_cg_run()
    nb = np.linalg.norm(b)
    assert res[k - 1] > 1e-3 * nb  # the window clause decides here
    gap, it, ok, _ = remaining_gap(b, np.ones(lam.size, bool), lambda v: lam * v, 1.0, 100.0)
    assert ok and it > k and res[it - 1] <= 1e-3 * nb
    assert gap == pytest.approx(0.5 * np.sum(b**2 / lam), rel=1e-3, abs=0)


def test_gap_plateau_converges_below_point_one_bound():
    """A16d: with bound = 1000, 0.1·bound = 100 exceeds the plateaued gap (≈17.4) from the first window
    check (k), so CG converges there without waiting for the residual (which only shrinks enough at a
    much later iteration, see test_gap_window_clause_needs_small_residual). max_cg = k + 10 leaves no
    room to reach the residual-only stop, so dropping the 0.1·bound alternative would leave this
    uncoverged (mutant 1); asserting ``it`` stays near ``k`` and ``gap`` matches the converged value (not
    an early, unstabilised one) catches accepting the plateau without the 50-iteration window (mutant 2)."""
    lam, b, k, gaps, res = _ill_conditioned_cg_run()
    exact = 0.5 * np.sum(b**2 / lam)
    gap, it, ok, _ = remaining_gap(b, np.ones(lam.size, bool), lambda v: lam * v, 1.0, 1000.0, max_cg=k + 10)
    assert ok and k <= it <= k + 10
    assert gap == pytest.approx(exact, rel=1e-3, abs=0)


def test_gap_plateau_above_point_one_bound_is_invalid_at_cap():
    """A16d: with bound = 100, 0.1·bound = 10 is below the plateaued gap (≈17.4), so the alternative
    never fires; capping max_cg before the residual stop (reached only well after k, see
    test_gap_window_clause_needs_small_residual) leaves CG not converged."""
    lam, b, k, gaps, res = _ill_conditioned_cg_run()
    gap, it, ok, _ = remaining_gap(b, np.ones(lam.size, bool), lambda v: lam * v, 1.0, 100.0, max_cg=k + 14)
    assert not ok and it == k + 14


def test_remaining_gap_zero_gradient():
    assert remaining_gap(np.zeros(3), np.ones(3, bool), lambda v: v, 1.0, 0.0) == (0.0, 0, True, False)


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


@pytest.mark.x64
def test_cg_cap_scales_with_free_set(monkeypatch):
    """A16c: CG cap = max(MAX_CG, n_free); with MAX_CG lowered below n_free the cap is n_free.

    Also checks ``retime`` passes its own A16 acceptance bound through to ``remaining_gap`` (a dropped
    ``bound`` argument, e.g. a stray ``0.0``, would silently disable the A16d plateau alternative).
    """
    seen = {}
    real = rt.remaining_gap

    def spy(pg, free, hvp, scale, bound, max_cg=rt.MAX_CG, line_search=None):
        seen.update(max_cg=max_cg, n_free=int(free.sum()), bound=bound)
        return real(pg, free, hvp, scale, bound, max_cg, line_search)

    monkeypatch.setattr(rt, "remaining_gap", spy)
    monkeypatch.setattr(rt, "MAX_CG", 10)
    retime(*_case(30, 50))
    assert seen["n_free"] > 10 and seen["max_cg"] == seen["n_free"]
    assert seen["bound"] == _bound(30, 50)


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
    """Floor alone is not enough: with the scipy floor met, a gap above the bound, a CG that did not
    converge, or an engine/reference L mismatch each makes the result invalid."""
    parent, t0, x, var = _case(30, 50)
    b = _bound(30, 50)
    monkeypatch.setattr(rt, "remaining_gap", lambda *a, **k: (0.5 * b, 7, True, False))
    assert retime(parent, t0, x, var).status == "converged"  # control: floor and restarts pass here
    monkeypatch.setattr(rt, "remaining_gap", lambda *a, **k: (0.5 * b, 7, True, True))
    assert retime(parent, t0, x, var).status == "converged"  # A16b: a curvature stop alone is not INVALID
    monkeypatch.setattr(rt, "remaining_gap", lambda *a, **k: (2 * b, 7, True, False))
    r = retime(parent, t0, x, var)
    assert r.status == "invalid" and "gap" in r.reason
    monkeypatch.setattr(rt, "remaining_gap", lambda *a, **k: (0.5 * b, 2000, False, False))
    assert retime(parent, t0, x, var).status == "invalid"
    monkeypatch.setattr(rt, "remaining_gap", lambda *a, **k: (0.5 * b, 7, True, False))
    monkeypatch.setattr(rt, "L_MATCH_REL", -1.0)  # every |ΔL| exceeds it: checks the test is wired in
    assert "mismatch" in retime(parent, t0, x, var).reason


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
    return 0.5 * (H + H.T), g(s)[f]


@pytest.mark.x64
@pytest.mark.slow
def test_acceptance_on_bm_tree_100_300():
    """The reviewer's probe case: converged with gap ≤ 3.0e-4 = 1e-2·1e-6·N·G (A16b optimiser: measured gap
    1.26e-6, one restart, 372 CG iterations; the A16 ftol-1e-12 point left 7.9e-4 of L, gap 4.2e-4).

    Dense check at 1%: the pre-declared CG rule resolves the gap to ~1e-3 relative; the dense and CG
    Hessians differ only by FD truncation O(h²) ≈ 1e-8 relative and rounding ≈ eps64·scale/h.
    """
    parent, t0, x, var = _case(100, 300)
    r = retime(parent, t0, x, var)
    assert r.status == "converged", r.reason
    assert r.cg_converged and r.cg_iters < 2000
    assert r.gap <= _bound(100, 300)
    e = parent >= 0
    pg_all = (tree_loglik_grad(parent, r.t, x, var).grad * r.t)[e]
    H, pg_free = _dense_fd_hessian(parent, r.t, x, var, free_set(np.log(r.t[e]), pg_all))
    assert r.gap == pytest.approx(0.5 * pg_free @ np.linalg.solve(H, pg_free), rel=1e-2, abs=0)


@pytest.mark.x64
@pytest.mark.slow
def test_cg_gap_matches_dense_at_probe_point():
    """CG alone at the reviewer's probe point (/tmp/cg_probe2.py): bm_tree(100, 300) after four L-BFGS-B
    runs with the original A16 options on the reference gradient. Measured there: dense Newton gap 1.0087e-4,
    ~500 CG iterations. Checks the pre-declared stop rule against the dense solve (1%, as above)
    independently of the re-timer's floor."""
    from scipy.optimize import minimize

    parent, t0, x, var = _case(100, 300)
    e = np.flatnonzero(parent >= 0)

    def lg(z):
        tt = np.zeros(len(parent))
        tt[e] = np.exp(z)
        r = tree_loglik_grad(parent, tt, x, var)
        return r.L, (r.grad * tt)[e]

    s = np.log(np.clip(t0[e], np.exp(rt.LOG_T_MIN), np.exp(rt.LOG_T_MAX)))
    for _ in range(4):
        s = minimize(
            lambda z: tuple(-v for v in lg(z)),
            s,
            jac=True,
            method="L-BFGS-B",
            bounds=[(rt.LOG_T_MIN, rt.LOG_T_MAX)] * len(s),
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 15000, "maxfun": 30000},  # the probe's (A16)
        ).x
    t = np.zeros(len(parent))
    t[e] = np.exp(s)
    pg = lg(s)[1]
    free = free_set(s, pg)

    def hvp(v):
        nv = np.linalg.norm(v)
        return -nv * (lg(s + 1e-4 * v / nv)[1] - lg(s - 1e-4 * v / nv)[1]) / 2e-4

    scale = rt._grad_scale(parent, t, tree_loglik_grad(parent, t, x, var))
    gap, it, ok, curv = remaining_gap(pg, free, hvp, scale, 0.0)  # bound=0.0: checked against the dense solve, not A16d
    assert ok and not curv and it < 2000
    H, pg_free = _dense_fd_hessian(parent, t, x, var, free)
    assert gap == pytest.approx(0.5 * pg_free @ np.linalg.solve(H, pg_free), rel=1e-2, abs=0)
    assert gap <= _bound(100, 300)
