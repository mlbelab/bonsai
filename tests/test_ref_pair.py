import numpy as np
import pytest

from bonsai._core import tree_helpers as th
from bonsai._core.helpers import calcTInit
from bonsai._ref import pair as ref_pair
from bonsai._ref.pair import (
    KKT_RTOL,
    LOG_T_MAX,
    LOG_T_MIN,
    init_times,
    joint_objective,
    pair_score,
    seq_objective,
    two_leaf_time,
    ub_gradients,
    ub_slopes,
)
from bonsai._ref.star import node_message
from bonsai._ref.tree import tree_loglik

from ._bounds import EPS64


def _parent(rng, C=12, G=40):
    """A parent star: C children (leaf data), times from upstream's own star EM (TH:4983-5027)."""
    x = rng.normal(0.0, 1.0, (C, G))
    var = rng.uniform(0.05, 0.6, (C, G))
    t = th.optimiseTStar(x.T.copy(), var.T.copy())[0]
    xr, Wr = node_message(x, var, t)
    return x, var, t, xr, Wr


def test_two_leaf_time_closed_form_and_clamp():
    """S_g ≡ S ⇒ t* = max(0, mean_g d² − S) (maths ref §5). Root solved to 4·eps64 relative (brentq rtol)."""
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=50), rng.normal(size=50)
    d2 = (x1 - x2) ** 2
    for S in (0.1, 0.3):
        t, ok = two_leaf_time(x1, np.full(50, S / 2), x2, np.full(50, S / 2))
        # brentq stops within rtol = 4·eps64 of t; the closed form sums 50 terms (≤ 50·eps64 each)
        assert ok and abs(t - max(0.0, d2.mean() - S)) <= 4 * EPS64 * t + 50 * EPS64 * (d2.mean() + S)
    t, ok = two_leaf_time(x1, np.full(50, 10.0), x2, np.full(50, 10.0))  # f(0) ≤ 0
    assert ok and t == 0.0  # exact zero clamp (TH:4899-4901)
    assert two_leaf_time(np.zeros(3), np.full(3, 1e-9), np.full(3, 1e5), np.full(3, 1e-9)) == (None, False)  # lb > 1e6


@pytest.mark.parametrize("mode", ["sequential", "joint"])
def test_dlogl_is_the_exact_change_of_the_tree_loglik(mode):
    """dLogL = L_tree(ancestor a of 1, 2 inserted below the parent) − L_tree(star) (maths ref §6).

    The parent is the root of a star, so its AIR message is its own message. Two fp64 evaluations
    of L_tree (N·G terms each): ≤ 2·8·(1+H)·eps64·Σ|terms|, bounded here by 16·3·eps64·(|L|+N·G).
    """
    rng = np.random.default_rng(1)
    x, var, t, xr, Wr = _parent(rng)
    C, G = x.shape
    s = pair_score(x[0], var[0], t[0], x[1], var[1], t[1], xr, Wr, mode=mode)
    assert s.converged
    before_p = np.full(C + 1, C)
    before_p[C] = -1
    X, V = np.vstack([x, np.zeros((1, G))]), np.vstack([var, np.ones((1, G))])
    L0 = tree_loglik(before_p, np.r_[t, 0.0], X, V)
    after_p = np.r_[np.full(C, C), -1, C]  # node C+1 = a; children 0, 1 move under a
    after_p[[0, 1]] = C + 1
    tt = np.r_[t, 0.0, s.t[2]]
    tt[[0, 1]] = s.t[:2]
    X1, V1 = np.vstack([X, np.zeros((1, G))]), np.vstack([V, np.ones((1, G))])
    L1 = tree_loglik(after_p, tt, X1, V1)
    assert abs(s.dlogl - (L1 - L0)) <= 48 * EPS64 * (abs(L0) + abs(L1) + 2 * C * G)


@pytest.mark.parametrize("mode", ["sequential", "joint"])
def test_reference_optimum_satisfies_kkt_by_finite_differences(mode):
    """At the returned times no feasible coordinate move increases ell_new: central FD of ell_new
    in each free coordinate is ≤ the FD error bound (truncation by Richardson + rounding)."""
    rng = np.random.default_rng(2)
    x, var, t, xr, Wr = _parent(rng)
    s = pair_score(x[2], var[2], t[2], x[3], var[3], t[3], xr, Wr, mode=mode)
    from bonsai._ref.star import star_loglik_grad

    ell = lambda tt: star_loglik_grad(s.xs, s.vs, tt)[0]
    if mode == "sequential":
        t12 = s.t[0] + s.t[1]
        # no silent fallback to the joint solve: t_1a + t_2a is the 2-leaf time (one rounding in t12 − t_1a)
        assert abs(t12 - two_leaf_time(x[2], var[2], x[3], var[3])[0]) <= 2 * EPS64 * t12
        u = np.array([s.t[0], np.log(s.t[2])])
        f = lambda uu: ell(np.array([uu[0], t12 - uu[0], np.exp(uu[1])]))
        lo, hi = np.array([0.0, LOG_T_MIN]), np.array([t12, LOG_T_MAX])
    else:
        u = np.log(s.t)
        f = lambda uu: ell(np.exp(uu))
        lo, hi = np.full(3, LOG_T_MIN), np.full(3, LOG_T_MAX)
    scale = 3 * abs(f(u)) + s.xs.size
    for i in range(len(u)):
        if u[i] <= lo[i] or u[i] >= hi[i]:
            continue  # active bound: checked by the sign of the gradient in the engine tests

        def fd(h, i=i):
            up, um = u.copy(), u.copy()
            up[i] += h
            um[i] -= h
            return (f(up) - f(um)) / (2 * h)

        h = 1e-4 * max(abs(u[i]), 1e-2)
        assert abs(fd(h / 2)) <= abs(fd(h) - fd(h / 2)) + 4 * EPS64 * scale / h + 1e-8 * (1 + abs(f(u)))  # + KKT_RTOL


def test_sequential_matches_upstream_calc_single_dlogl():
    """Upstream's calcSingleDLogL(tol=1e-9) (TH:5362-5408) stops early (scipy ftol/gtol 1e-9,
    brentq rtol 1e-9), at times t_up near ours. Both scores share ell_old, so the difference is
    ell_new(t_ref) − ell_new(t_up), bounded by Taylor at t_up: |g_up·Δt| + ½·‖g_up − g_ref‖·‖Δt‖
    (secant curvature) + the rounding of two fp64 evaluations, 16·eps64·Σ|terms|."""
    from bonsai._ref.star import star_loglik_grad

    rng = np.random.default_rng(3)
    x, var, t, xr, Wr = _parent(rng)
    for i, j in [(0, 1), (2, 5), (7, 3)]:
        s = pair_score(x[i], var[i], t[i], x[j], var[j], t[j], xr, Wr)
        w1, w2 = 1 / (t[i] + var[i]), 1 / (t[j] + var[j])
        up_dl, up_t = th.calcSingleDLogL(
            xr,
            Wr,
            x[i],
            var[i],
            w1,
            t[i],
            Wr - w1,
            xr * Wr - w1 * x[i],
            x[j],
            var[j],
            w2,
            t[j],
            sequential=True,
            tol=1e-9,
        )
        assert s.converged and np.isfinite(up_dl)  # both sides converged (upstream returns −inf otherwise)
        t_up = np.asarray(up_t, dtype=float)
        dt = s.t - t_up
        g_up, g_ref = star_loglik_grad(s.xs, s.vs, t_up)[1], star_loglik_grad(s.xs, s.vs, s.t)[1]
        w = 1 / (s.vs + s.t[:, None])
        xa, Wa = node_message(s.xs, s.vs, s.t)
        scale = np.sum(1 + np.abs(np.log(w)) + w * (xa - s.xs) ** 2) + np.sum(np.abs(np.log(Wa)))
        bound = abs(g_up @ dt) + 0.5 * np.linalg.norm(g_up - g_ref) * np.linalg.norm(dt) + 16 * EPS64 * scale
        assert abs(s.dlogl - up_dl) <= bound
        assert s.dlogl >= up_dl - 16 * EPS64 * scale  # the tight reference is never worse


def test_ub_slopes_match_upstream_and_finite_differences():
    """(a, b) equal upstream's estimateDerBasedDLogLUB (same formula: 8·eps64 relative per term);
    the per-gene derivatives equal central FD of dLogL in x_r and W_r with t frozen (envelope)."""
    rng = np.random.default_rng(4)
    x, var, t, xr, Wr = _parent(rng, G=8)
    s = pair_score(x[0], var[0], t[0], x[1], var[1], t[1], xr, Wr)
    assert s.converged
    a, b = ub_slopes(s, xr, Wr)
    _, ua, ub = th.estimateDerBasedDLogLUB(
        {"pos": xr, "prec": Wr, "nChild": 12}, s.t, ltqs_gi=s.xs.T, ltqsVars_gi=s.vs.T, wir_gi=s.w_old.T
    )
    tol = (8 + len(xr)) * EPS64  # ≤ 8 rounded ops per gene term, then a G-term norm
    assert abs(a - ua) <= tol * a and abs(b - ub) <= tol * b
    from bonsai._ref.pair import rest_of_parent
    from bonsai._ref.star import star_loglik_grad

    w1, w2 = s.w_old[0], s.w_old[1]

    def dlogl(xr_, Wr_):  # frozen times; R moves with the parent (TH:5372-5374)
        xR, WR = rest_of_parent(xr_, Wr_, s.xs[0], w1, s.xs[1], w2)
        xs, vs = np.stack([s.xs[0], s.xs[1], xR]), np.stack([s.vs[0], s.vs[1], 1 / WR])
        new = star_loglik_grad(xs, vs, s.t)[0]
        old = star_loglik_grad(xs, vs, None, xn=xr_, Wn=Wr_, w=np.stack([w1, w2, WR]))[0]
        return new - old

    dx, dW = ub_gradients(s, xr, Wr)
    base = abs(dlogl(xr, Wr)) + x.size
    for g in range(len(xr)):
        e = np.zeros_like(xr)
        e[g] = 1.0
        moves = (
            (dx, lambda h, e=e: dlogl(xr + h * e, Wr)),
            (dW, lambda h, e=e, g=g: dlogl(xr, Wr + h * e * Wr[g]) / Wr[g]),
        )
        for vec, f in moves:
            h = 1e-4
            fd1, fd2 = (f(h) - f(-h)) / (2 * h), (f(h / 2) - f(-h / 2)) / h
            assert abs(fd2 - vec[g]) <= abs(fd1 - fd2) + 4 * EPS64 * base / h


def test_unknown_mode_raises():
    rng = np.random.default_rng(5)
    x, var, t, xr, Wr = _parent(rng)
    with pytest.raises(ValueError, match="mode"):
        pair_score(x[0], var[0], t[0], x[1], var[1], t[1], xr, Wr, mode="greedy")


def test_init_times_match_upstream_calc_t_init():
    """init_times is calcTInit (HE:788-794) for both modes, exactly (same operations)."""
    for t1, t2 in [(0.3, 0.7), (1.2, 0.05), (0.0, 0.4)]:
        seq0, joint0 = init_times(t1, t2)
        assert seq0 == calcTInit(t1, t2, True) and joint0 == calcTInit(t1, t2, False)


def test_sequential_falls_back_to_joint_when_t12_bracket_fails():
    """No 2-leaf root below 1e6 (lb > 1e6) ⇒ sequential returns the joint solve (TH:4822-4831, 5381-5385)."""
    rng = np.random.default_rng(6)
    x, var, t, xr, Wr = _parent(rng, G=4)
    x2 = x[0] + 1e5  # d² = 1e10 per gene: the 2-leaf root is ~1e10
    assert two_leaf_time(x[0], var[0], x2, var[1]) == (None, False)
    seq = pair_score(x[0], var[0], t[0], x2, var[1], t[1], xr, Wr)
    joint = pair_score(x[0], var[0], t[0], x2, var[1], t[1], xr, Wr, mode="joint")
    assert seq.converged and joint.converged
    np.testing.assert_array_equal(seq.t, joint.t)  # same deterministic solve
    assert seq.dlogl == joint.dlogl


def test_sequential_active_bounds_match_upstream():
    """Child 2 sits at R (x_2 = x_R, W_R large) ⇒ a collapses onto 2: t_1a hits its upper bound t_12 (t_2a = 0
    exactly) and t_aR its lower bound. KKT signs by the analytic gradient (FD-checked in Task 8): ∂ell/∂t_1a =
    g_1 − g_2 ≥ 0 at the upper bound, ∂ell/∂log t_aR ≤ 0 at the lower bound. Upstream (tol=1e-9) stops at the
    same vertex, so the values agree to the rounding of two evaluations, 16·eps64·Σ|terms|."""
    from bonsai._ref.star import star_loglik_grad

    rng = np.random.default_rng(7)
    G = 20
    x1, v1, t1 = rng.normal(size=G) + 3.0, np.full(G, 0.2), 0.5
    x2, v2, t2 = rng.normal(size=G), np.full(G, 0.01), 0.3
    WR = np.full(G, 100.0)
    w1, w2 = 1 / (t1 + v1), 1 / (t2 + v2)
    Wr = WR + w1 + w2
    xr = (WR * x2 + w1 * x1 + w2 * x2) / Wr
    s = pair_score(x1, v1, t1, x2, v2, t2, xr, Wr)
    assert s.converged
    assert s.t[1] == 0.0 and np.log(s.t[2]) == pytest.approx(LOG_T_MIN, abs=EPS64 * abs(LOG_T_MIN))
    g = star_loglik_grad(s.xs, s.vs, s.t)[1]
    assert g[0] - g[1] >= 0 and s.t[2] * g[2] <= 0
    up_dl, up_t = th.calcSingleDLogL(xr, Wr, x1, v1, w1, t1, Wr - w1, xr * Wr - w1 * x1, x2, v2, w2, t2, tol=1e-9)
    assert np.isfinite(up_dl) and up_t[1] == 0.0
    w = 1 / (s.vs + s.t[:, None])
    xa, Wa = node_message(s.xs, s.vs, s.t)
    scale = np.sum(1 + np.abs(np.log(w)) + w * (xa - s.xs) ** 2) + np.sum(np.abs(np.log(Wa)))
    assert abs(s.dlogl - up_dl) <= 16 * EPS64 * scale


def _fd_grad(f, u, rel=1e-4):
    """Central FD of a scalar f at u, per coordinate at steps h and h/2 (h = rel·max(|u_i|, 1e-2))."""
    fd1, fd2, hs = np.zeros(len(u)), np.zeros(len(u)), np.zeros(len(u))
    for i in range(len(u)):
        h = rel * max(abs(u[i]), 1e-2)
        e = np.zeros(len(u))
        e[i] = 1.0
        fd1[i] = (f(u + h * e) - f(u - h * e)) / (2 * h)
        fd2[i] = (f(u + h / 2 * e) - f(u - h / 2 * e)) / h
        hs[i] = h
    return fd1, fd2, hs


@pytest.mark.parametrize("which", ["sequential", "joint"])
def test_objective_gradients_match_finite_differences(which):
    """The chain rules of the optimised variables (the JAX port mirrors them): sequential
    d/dt_1a = g_1 − g_2 and d/dlog t_aR = t_aR·g_R; joint d/dlog t = t·g. Central FD at h/2 has
    truncation error ≈ |fd(h) − fd(h/2)|/3 (Richardson; bounded here by the full difference) plus
    rounding 4·eps64·Σ|terms|/h (two fp64 evaluations of −ell, each ≤ 2·eps64·Σ|terms|, over 2·(h/2))."""
    rng = np.random.default_rng(8)
    x, var, t, xr, Wr = _parent(rng, G=10)
    s = pair_score(x[0], var[0], t[0], x[1], var[1], t[1], xr, Wr)
    t12 = 1.7  # any interior point works; away from the optimum the gradient is O(1)
    points = [np.array([0.3, -1.0]), np.array([1.2, 0.5])] if which == "sequential" else [np.array([-1.0, 0.2, -2.0])]
    for u in points:
        if which == "sequential":
            f = lambda uu: seq_objective(uu, t12, s.xs, s.vs)
            tt = np.array([u[0], t12 - u[0], np.exp(u[1])])
        else:
            f = lambda uu: joint_objective(uu, s.xs, s.vs)
            tt = np.exp(u)
        fv, g = f(u)
        w = 1 / (s.vs + tt[:, None])
        xa, Wa = node_message(s.xs, s.vs, tt)
        scale = np.sum(1 + np.abs(np.log(w)) + w * (xa - s.xs) ** 2) + np.sum(1 + np.abs(np.log(Wa)))
        fd1, fd2, h = _fd_grad(lambda uu, f=f: f(uu)[0], u)
        assert np.all(np.abs(g) > 1e-3)  # a non-vacuous point: every component is far from zero
        assert np.all(np.abs(fd2 - g) <= np.abs(fd1 - fd2) + 4 * EPS64 * scale / h)


def test_all_pairs_converge_with_kkt_signs_and_joint_dominates_interior():
    """Every pair of three simulated parents (3 × 66 pairs) converges in both modes. At every
    coordinate on a bound, the analytic objective gradient (FD-checked above) has the KKT sign within
    the stopping rule, KKT_RTOL·(1 + |f|). On pairs whose times are all inside the joint box in both
    modes, dLogL(joint) ≥ dLogL(seq) − tol. Joint is not a strict superset of sequential (sequential
    allows t_2a = 0 and fixes t_12; joint keeps t ≥ 1e-7 and has its own upper box), hence the
    interior restriction. tol: the joint optimum is stationary within its stopping rule; first-order
    Taylor at s_joint with secant curvature, |g_j·Δs| + ½·‖g_j − g_s‖·‖Δs‖ (s = log t), plus the
    rounding of two fp64 evaluations, 16·eps64·Σ|terms|."""
    n_interior = 0
    for seed in (1, 2, 3):
        x, var, t, xr, Wr = _parent(np.random.default_rng(seed))
        C = len(t)
        for i in range(C):
            for j in range(i + 1, C):
                args = (x[i], var[i], t[i], x[j], var[j], t[j], xr, Wr)
                sq, jt = pair_score(*args), pair_score(*args, mode="joint")
                assert sq.converged and jt.converged, (seed, i, j)
                # joint: s = log t in [LOG_T_MIN, LOG_T_MAX]^3
                sj = np.log(jt.t)
                fj, gj = joint_objective(sj, jt.xs, jt.vs)
                kkt = KKT_RTOL * (1 + abs(fj))
                assert np.all(gj[sj <= LOG_T_MIN] >= -kkt) and np.all(gj[sj >= LOG_T_MAX] <= kkt)
                # sequential: u = (t_1a in [0, t12], log t_aR in the box)
                t12 = sq.t[0] + sq.t[1]
                u = np.array([sq.t[0], np.log(sq.t[2])])
                fs, gs = seq_objective(u, t12, sq.xs, sq.vs)
                kkt = KKT_RTOL * (1 + abs(fs))
                lo, hi = np.array([0.0, LOG_T_MIN]), np.array([t12, LOG_T_MAX])
                assert np.all(gs[u <= lo] >= -kkt) and np.all(gs[u >= hi] <= kkt)
                ss = np.log(sq.t)
                inside = lambda z: np.all((z > LOG_T_MIN) & (z < LOG_T_MAX))
                if inside(ss) and inside(sj):
                    n_interior += 1
                    ds = ss - sj
                    gss = joint_objective(ss, jt.xs, jt.vs)[1]
                    w = 1 / (jt.vs + jt.t[:, None])
                    xa, Wa = node_message(jt.xs, jt.vs, jt.t)
                    scale = np.sum(1 + np.abs(np.log(w)) + w * (xa - jt.xs) ** 2) + np.sum(np.abs(np.log(Wa)))
                    tol = abs(gj @ ds) + 0.5 * np.linalg.norm(gj - gss) * np.linalg.norm(ds) + 16 * EPS64 * scale
                    assert jt.dlogl >= sq.dlogl - tol, (seed, i, j)
    assert n_interior >= 10  # the dominance check ran on a fair share of pairs


def test_both_optimisers_failing_gives_minus_inf(monkeypatch):
    """Both solves fail ⇒ dLogL = −∞, t = joint t⁰ = calcTInit(…, False), converged False (TH:5390-5394)."""
    monkeypatch.setattr(ref_pair, "_minimise_box", lambda f, u0, box: (np.asarray(u0, dtype=float), False))
    x, var, t, xr, Wr = _parent(np.random.default_rng(9))
    s = pair_score(x[0], var[0], t[0], x[1], var[1], t[1], xr, Wr)
    assert s.dlogl == -np.inf and not s.converged
    np.testing.assert_array_equal(s.t, calcTInit(t[0], t[1], False))


def test_two_child_parent_raises():
    """A parent with only children 1 and 2 has no rest: W_R = W_r − w_1 − w_2 is rounding noise."""
    x, var, t, xr, Wr = _parent(np.random.default_rng(10), C=2)
    with pytest.raises(ValueError, match="3 children"):
        pair_score(x[0], var[0], t[0], x[1], var[1], t[1], xr, Wr)
