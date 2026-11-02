import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from bench import retime as rt
from bench.retime import free_set, gap_terms, line_gain, retime, settle
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


# --- A16g: the saddle escape


def _rr(s, L, gap_flat=0.0, gap_floor=0.0, gap_newton=0.0, bound=1e-6, extra=""):
    """A RetimeResult at ``s`` (other diagnostics neutral) as ``accept`` writes it: gap = the sum; above ``bound`` the reason
    is accept's gap clause (preceded by ``extra``, another failing test's clause), converged otherwise."""
    gap = gap_newton + gap_flat + gap_floor
    clause = f"gap {gap:.3e} > {bound:.3e} (Newton {gap_newton:.3e}, flat {gap_flat:.3e}, floor {gap_floor:.3e})"
    reason = "; ".join(filter(None, [extra, clause if gap > bound else ""]))
    f = dict.fromkeys(rt.RetimeResult.__dataclass_fields__, 0.0) | {"t": np.exp(s), "L": L, "gap": gap}
    f |= {"gap_flat": gap_flat, "gap_floor": gap_floor, "gap_newton": gap_newton, "reason": reason}
    return rt.RetimeResult(**f | {"status": "invalid" if reason else "converged"})


def _ep0(s, **kw):
    return replace(rt.Endpoint(np.asarray(s, float), -9.0, 3, 0, 0.0, True, "toy", np.zeros((1, 3))), **kw)


def test_needs_escape_rule():
    """Controller ruling: escape iff the gap is the only failure and flat + floor > bound (combined); never on the Newton
    term (mutant "escape counted on the Newton term" fails the first assert, "one term alone" the second)."""
    b = 1.0
    assert not rt.needs_escape(_rr([0.0], 0.0, gap_newton=5.0, gap_flat=0.4, gap_floor=0.4, bound=b), b)
    assert rt.needs_escape(_rr([0.0], 0.0, gap_flat=0.6, gap_floor=0.6, bound=b), b)
    assert rt.needs_escape(_rr([0.0], 0.0, gap_flat=1.1, bound=b), b)
    assert rt.needs_escape(_rr([0.0], 0.0, gap_floor=1.1, bound=b), b)
    assert not rt.needs_escape(_rr([0.0], 0.0, gap_flat=0.9, bound=b), b)  # converged: nothing to escape
    for other in (
        "scipy status 3 (NaN)",
        "10 restarts, the last raised L by 2.000e+00",
        "engine/reference L mismatch 1e-3",
    ):
        assert not rt.needs_escape(_rr([0.0], 0.0, gap_flat=1.1, bound=b, extra=other), b), other
    assert rt.escape_flags(_rr([0.0], 0.0, gap_flat=0.6, gap_floor=0.0)) == (True, False)
    assert rt.escape_flags(_rr([0.0], 0.0, gap_flat=0.0, gap_floor=1e-9)) == (False, True)


def _saddle(q=0.01, c=1e-3, d=1e-5):
    """L(y) = −½(y_0 − q·y_1²)² + c·y_1² − d·y_1⁴ (box ±15): a saddle at y = 0 with ∇L = 0 and H = diag(1, −2c).

    Along e_1 alone (y_0 = 0) the best gain is c²/(4(d + q²/2)) ≈ 4.17e-3 at y_1² = c/(2d + q²); with y_0 = q·y_1²
    re-optimised the maximum is c²/(4d) = 0.025 at y_1² = c/(2d) = 50, so only the resumed L-BFGS reaches it.
    """

    def Lg(y):
        r = y[0] - q * y[1] ** 2
        L = -0.5 * r**2 + c * y[1] ** 2 - d * y[1] ** 4
        return float(L), np.array([-r, 2 * q * y[1] * r + 2 * c * y[1] - 4 * d * y[1] ** 3])

    def H(y):  # −∇²L
        r = y[0] - q * y[1] ** 2
        h01 = 2 * q * y[1]
        return -np.array([[-1.0, h01], [h01, 2 * q * r - 4 * q**2 * y[1] ** 2 + 2 * c - 12 * d * y[1] ** 2]])

    return Lg, H, c**2 / (4 * d)


_TOY_PARENT = np.array([-1, 0, 0])  # two edges: s = y


def _toy(monkeypatch, Lg, H, tau=1e-10):
    """acc for ``settle`` on a toy L (the A16f gap terms with the exact H) and the toy L patched into ``escape_step``:
    ``escape_point`` reads the best point off the same A16f searches on the toy L, ``lbfgs`` is scipy L-BFGS-B with the
    re-timer's options in the box, started at log clip(t) like the real one."""
    from scipy.optimize import minimize

    def acc(ep):
        w, U = np.linalg.eigh(H(ep.s))
        L, pg = Lg(ep.s)
        g = gap_terms(w, U, pg, tau, ep.s, np.arange(len(ep.s)), lambda z: Lg(z)[0], Lg)
        return _rr(ep.s, L, max(g["flat_sum"], g["flat_joint"]), g["floor"], g["newton"])  # bound 1e-6

    def escape_point(parent, x, var, ep, flat, floor):
        seen = [(Lg(ep.s)[0], ep.s)]

        def L(z):
            seen.append((Lg(z)[0], np.array(z)))
            return seen[-1][0]

        w, U = np.linalg.eigh(H(ep.s))
        for k in np.flatnonzero(w <= tau):
            line_gain(L, ep.s, U[:, k])
        return max(seen, key=lambda c: c[0])[1]

    def lbfgs(parent, t, x, var):
        s0 = np.log(np.clip(t[parent >= 0], np.exp(rt.LOG_T_MIN), np.exp(rt.LOG_T_MAX)))
        r = minimize(lambda y: tuple(-v for v in Lg(y)), s0, jac=True, method="L-BFGS-B",
                     bounds=[(rt.LOG_T_MIN, rt.LOG_T_MAX)] * len(s0), options=rt.OPTIONS)  # fmt: skip
        hist = np.array([[r.nit, -r.fun, r.status]], float)
        return rt.Endpoint(np.asarray(r.x, float), Lg(s0)[0], int(r.nit), 0, 0.0, r.status in (0, 1, 2), "toy", hist)

    monkeypatch.setattr(rt, "escape_point", escape_point)
    monkeypatch.setattr(rt, "lbfgs", lbfgs)
    return acc, lambda e, r: rt.escape_step(_TOY_PARENT, None, None, e, *rt.escape_flags(r))


def test_settle_leaves_a_synthetic_saddle(monkeypatch):
    """L-BFGS started at the saddle stops there at once (∇L = 0); A16f measures the negative direction's gain 4.17e-3 >
    bound 1e-6 (INVALID). ``settle`` with the real ``escape_step`` moves along it, the resumed L-BFGS reaches the maximum
    0.025 and the acceptance passes. Mutants: "no escape" stays INVALID at L = 0; "escape but no resumed L-BFGS" stops at
    the line-search point, where ∂L/∂y_0 = q·y_1² ≈ 0.083 gives a Newton gap ≈ 3.5e-3 > bound. Tolerance: L-BFGS stops
    at gtol 1e-12 or at rounding; L ≈ 0.025 to a few eps, rel 1e-9."""
    Lg, H, top = _saddle()
    acc, step = _toy(monkeypatch, Lg, H)
    ep0 = rt.lbfgs(_TOY_PARENT, np.ones(3), None, None)
    assert ep0.nit == 0 and np.array_equal(ep0.s, np.zeros(2))  # L-BFGS stops at the saddle
    r0 = acc(ep0)
    assert r0.status == "invalid" and r0.gap_flat == pytest.approx(4.1667e-3, rel=1e-4, abs=0)
    saved = []
    r, (ep, n, rows, L0) = settle(acc, step, ep0, 1e-6, save=saved.append)
    assert r.status == "converged", r.reason
    assert (n, L0) == (1, 0.0) and r.L == pytest.approx(top, rel=1e-9, abs=0) and r.L > r0.L
    assert ep.nit == ep0.nit + ep.history[-1, 0] and ep.L_start == ep0.L_start and ep.history.shape == (2, 3)
    np.testing.assert_array_equal(rows, [[r.L, r.gap, r.gap_newton, r.gap_flat, r.gap_floor, 1.0]])
    assert len(saved) == 1 and saved[0][1] == 1 and saved[0][2].shape == (0, 6)  # saved before its acceptance
    np.testing.assert_array_equal(saved[0][0].s, ep.s)


def test_settle_cap_of_five_escapes():
    """An acceptance that always asks for an escape: exactly MAX_ESCAPES = 5 escapes, then INVALID with the reason; each
    escape's acceptance is recorded. A stored state continues: from n = 4 with the last row still missing, the row is
    added and one more escape is allowed. A known first acceptance ``r`` is not recomputed."""
    calls, accs = [], []

    def acc(ep):
        accs.append(ep)
        return _rr(ep.s, float(ep.s[0]), gap_flat=1.0)

    def step(ep, r):
        calls.append(ep)
        return replace(ep, s=ep.s + 1.0)

    r, (ep, n, rows, L0) = settle(acc, step, _ep0([0.0, 0.0]), 1e-6)
    assert (n, len(calls), len(accs), r.status) == (5, 5, 6, "invalid") and "5 escapes" in r.reason
    assert rows.shape == (5, 6) and rows[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0] and (r.L, L0) == (5.0, 0.0)
    calls.clear(), accs.clear()
    r, (_, n, rows, _) = settle(acc, step, _ep0([4.0, 4.0]), 1e-6, None, 4, rows[:3], 0.0)
    assert (n, len(calls), rows.shape) == (5, 1, (5, 6)) and rows[3, 0] == 4.0 and r.L == 5.0
    accs.clear()
    r, (_, n, _, _) = settle(acc, step, _ep0([0.0]), 1e-6, _rr([0.0], 0.0))  # r given and passing: nothing runs
    assert (n, accs, r.status) == (0, [], "converged")


def test_settle_stops_when_an_escape_makes_no_progress_or_lowers_L(monkeypatch):
    """An escape whose point is the endpoint (no search gained) stops at once: INVALID "escape made no progress", no
    escape counted (``escape_step`` returns the endpoint without running L-BFGS). An escaped endpoint with L lower by
    more than 1e-10·|L| stops too: INVALID with the drop."""
    ep0 = _ep0([0.0, 0.0])
    r, (_, n, rows, _) = settle(lambda e: _rr(e.s, 0.0, gap_flat=1.0), lambda e, r: e, ep0, 1e-6)
    assert (r.status, n, rows.shape) == ("invalid", 0, (0, 6)) and r.reason.endswith("escape made no progress")
    monkeypatch.setattr(rt, "escape_point", lambda parent, x, var, ep, flat, floor: ep.s.copy())
    monkeypatch.setattr(rt, "lbfgs", lambda *a: pytest.fail("L-BFGS ran without progress"))
    assert rt.escape_step(_TOY_PARENT, None, None, ep0, True, False) is ep0
    r, (_, n, rows, _) = settle(
        lambda e: _rr(e.s, -float(e.s[0]), gap_flat=1.0), lambda e, r: replace(e, s=e.s + 1), ep0, 1e-6
    )
    assert (r.status, n, rows.shape) == ("invalid", 1, (1, 6)) and r.reason.endswith("escape lowered L by 1.000e+00")
    r, _ = settle(lambda e: _rr(e.s, -1e6 - 1e-5 * e.s[0], gap_flat=1.0), lambda e, r: replace(e, s=e.s + 1), ep0, 1e-6)
    assert "lowered" not in r.reason and "5 escapes" in r.reason  # a drop of 1e-5 ≤ 1e-10·|L| = 1e-4 is rounding
    # resuming a stored state whose escaped endpoint (pending, n = 1 > len(rows) = 0) is below L0 = 0: INVALID too, not
    # converged with a negative L_escape_gain (fix round 2)
    r, (_, n, rows, L0) = settle(
        lambda e: _rr(e.s, -1.0), lambda e, r: pytest.fail("escaped"), ep0, 1e-6, None, 1, (), 0.0
    )
    assert (r.status, n, rows.shape, L0) == ("invalid", 1, (1, 6), 0.0) and r.reason == "escape lowered L by 1.000e+00"
    rows1 = [(0.5, 0.0, 0.0, 0.0, 0.0, 1.0)]  # with rows the reference is the last recorded L (0.5), not L0
    assert (
        settle(lambda e: _rr(e.s, 0.2), None, ep0, 1e-6, None, 2, rows1, 0.0)[0].reason
        == "escape lowered L by 3.000e-01"
    )
    assert settle(lambda e: _rr(e.s, 0.7), None, ep0, 1e-6, None, 2, rows1, 0.0)[0].status == "converged"


def test_main_runs_one_escape_step(tmp_path, monkeypatch):
    """``esc_flat``/``esc_floor`` in the input: ``main`` runs ``escape_step`` with those flags and writes the endpoint."""
    ep = _ep0([0.5, -1.0])
    np.savez(tmp_path / "in.npz", parent=_TOY_PARENT, t=np.ones(3), x=np.zeros(1), var=np.ones(1), esc_flat=True,
             esc_floor=False, **{f"ep_{k}": v for k, v in rt.asdict(ep).items()})  # fmt: skip
    seen = []
    monkeypatch.setattr(rt, "escape_step", lambda p, x, v, e, fl, fo: seen.append((fl, fo)) or replace(e, nit=99))
    rt.main([str(tmp_path / "in.npz"), str(tmp_path / "out.npz")])
    out = rt.load_endpoint(tmp_path / "out.npz")
    np.testing.assert_array_equal(out.s, ep.s)
    assert out.nit == 99 and seen == [(True, False)]


@pytest.mark.x64
def test_escape_point_reads_the_a16f_search_points(monkeypatch):
    """bm_tree(30, 50): the point ``escape_point`` returns gains exactly the A16f flat term over the endpoint (same
    deterministic searches on CPU; this endpoint has one flat eigenvalue, so the term is that line search). Without any
    flag it returns the endpoint and builds no Hessian."""
    parent, t0, x, var = _case(30, 50)
    ep = rt.lbfgs(parent, t0, x, var)
    r = rt.accept(parent, x, var, ep)
    assert r.n_flat == 1 and r.status == "converged"
    z = rt.escape_point(parent, x, var, ep, True, False)
    L = rt.ref_tree.tree_loglik  # the evaluation the A16f searches use
    gain = L(parent, rt.times_of(parent, z), x, var) - L(parent, rt.times_of(parent, ep.s), x, var)
    assert gain == r.gap_flat and gain > 0
    monkeypatch.setattr(rt, "free_hessian", lambda *a: pytest.fail("Hessian built"))
    np.testing.assert_array_equal(rt.escape_point(parent, x, var, ep, False, False), ep.s)


@pytest.mark.x64
def test_escape_point_floor_branch():
    """bm_tree(30, 50) endpoint with four leaf edges pushed to s = −10 (< FLOOR_S): the A16f floor term is the joint gain
    of moving them back, and the point ``escape_point(floor=True)`` returns gains exactly that term (same deterministic
    search on CPU). Reviewer mutant "skip the floor search" returns the endpoint (gain 0)."""
    parent, t0, x, var = _case(30, 50)
    ep = rt.lbfgs(parent, t0, x, var)
    leaves = np.flatnonzero(~np.isin(np.arange(len(parent)), parent))[:4]
    s = ep.s.copy()
    s[np.searchsorted(np.flatnonzero(parent >= 0), leaves)] = -10.0
    ep = replace(ep, s=s)
    r = rt.accept(parent, x, var, ep)
    assert r.n_floor >= 4 and r.gap_floor > _bound(30, 50) and rt.needs_escape(r, rt._bound(parent, x))
    assert rt.escape_flags(r)[1]
    z = rt.escape_point(parent, x, var, ep, False, True)
    L = rt.ref_tree.tree_loglik_grad  # the evaluation the floor search uses
    gain = L(parent, rt.times_of(parent, z), x, var).L - L(parent, rt.times_of(parent, ep.s), x, var).L
    assert gain == r.gap_floor > 0


@pytest.mark.x64
def test_settle_without_excess_does_no_escape(monkeypatch):
    """bm_tree(30, 50) passes A16f, so ``retime`` returns the plain acceptance and never escapes."""
    parent, t0, x, var = _case(30, 50)
    ep = rt.lbfgs(parent, t0, x, var)
    plain = rt.accept(parent, x, var, ep)
    monkeypatch.setattr(rt, "lbfgs", lambda *a: ep)
    monkeypatch.setattr(rt, "escape_step", lambda *a: pytest.fail("escaped"))
    r = retime(parent, t0, x, var)
    assert (r.L, r.gap, r.status) == (plain.L, plain.gap, "converged")


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
# env BONSAI_FIXTURES overrides (the directory holding the a16f endpoint files)
_FIXTURES = Path(os.environ.get("BONSAI_FIXTURES", _BONSAI / "data/test_fixtures/a16f"))


def _real_case(which):
    from bench.gate import load_arm
    from bench.run import _leaf_arrays

    from bonsai._ref.newick import parse_newick

    if which == "eec":
        up_dir, ep_file = (
            _BONSAI / "data/eec_vasa_sub/bonsai_workdir/results",
            _FIXTURES / "lbfgs_ckpt_sequential.npz",
        )
        runs = _FIXTURES / "eec_vasa_sub_sequential_M256_m1"  # its results.json, frozen from data/m2_gate/runs
    else:
        up_dir, ep_file = (
            _BONSAI / "data/m2_baselines/results_baselines2/upstream_redo_sim_1k_s1",
            _FIXTURES / "lbfgs_ckpt_1k_s1.npz",
        )
        runs = None
    if not (up_dir.exists() and ep_file.exists() and (runs is None or runs.exists())):
        pytest.skip(f"stored {which} endpoint or its arms not available")
    up = load_arm(up_dir if which == "eec" else next(p for p in up_dir.iterdir() if p.is_dir()))
    arm = up if runs is None else load_arm(runs, load_arrays=False)
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


@pytest.mark.x64
@pytest.mark.slow
def test_a16g_escapes_the_saddle_of_perturb_ctrl_sub_upstream():
    """The A16f re-check: the perturb_ctrl_sub upstream redo-starry endpoint (cached in data/m2_gate, copied to the
    a16g fixtures) sits at a saddle: two eigenvalues ≤ τ (λ_min ≈ −1.09e-6), joint gain ≈ 1.894e-3 > bound 1.680e-3,
    Newton part 2e-7. The escape moves to the best measured point and resumes L-BFGS; the acceptance then passes within
    the cap. L-BFGS-B never lowers L, so L rises by at least the measured joint gain, up to rounding: L ≈ −1.8e5, engine
    and `_ref` sums agree to ~eps·|L|·log(N·G) ≈ 1e-10, and 1e-6 (0.06% of the bound) is far above it. Reference
    values from the A16f recheck (its memo entry is frozen in the a16g fixtures): rel 1e-3 as in the a16f tests."""
    from bench.gate import load_arm
    from bench.run import _leaf_arrays

    from bonsai._ref.newick import parse_newick

    up_dir = _BONSAI / "data/perturb_ctrl_sub/bonsai_workdir/results"
    ep_file = _FIXTURES.parent / "a16g" / "perturb_ctrl_sub_upstream_endpoint.npz"
    if not (up_dir.exists() and ep_file.exists()):
        pytest.skip("perturb_ctrl_sub upstream arm or its stored endpoint not available")
    up = load_arm(up_dir)
    p, t, nm = parse_newick(up["stage_newick"]["redo_starry"])
    X, V = _leaf_arrays(p, nm, up["cells"], up["x"], up["var"])
    ep = rt.load_endpoint(ep_file)
    bound = rt._bound(p, X)
    r0 = rt.accept(p, X, V, ep)
    assert r0.status == "invalid" and r0.n_flat == 2 and r0.lambda_min == pytest.approx(-1.0911e-6, rel=1e-3, abs=0)
    assert r0.gap_flat == pytest.approx(1.8942e-3, rel=1e-3, abs=0)
    assert r0.gap_flat > bound > r0.gap_newton + r0.gap_floor and rt.needs_escape(r0, bound)
    r, (ep_fin, n, rows, L0) = settle(
        lambda e: rt.accept(p, X, V, e),
        lambda e, r: rt.escape_step(p, X, V, e, *rt.escape_flags(r)),
        ep, bound, r0,
    )  # fmt: skip
    print(f"A16g perturb_ctrl_sub upstream: L {r0.L!r} -> {r.L!r} (+{r.L - L0:.4e}), {n} escapes, gap {r.gap:.3e}")
    assert r.status == "converged", r.reason
    assert 1 <= n <= rt.MAX_ESCAPES and rows.shape == (n, 6) and rows[-1, -1] == 1.0
    assert L0 == r0.L and r.L - r0.L >= r0.gap_flat_joint - 1e-6
    assert r.gap <= bound and r.extra_dL <= bound and ep_fin.L_start == ep.L_start
