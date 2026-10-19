"""JAX pair kernels and host accumulators vs the fp64 references (maths ref §5-§7, A8).

float32 cases run in the default (x64-off) lane; float64 cases (``x64`` marker) check the formulas.
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bonsai._engine import pair as kernel
from bonsai._engine.host_acc import ParentAcc
from bonsai._engine.pair import LM_RISE, PAIR_LOG_T_MAX, PAIR_LOG_T_MIN, PG_EPS_FACTOR, pair_score, two_leaf_time
from bonsai._engine.star import child_weights
from bonsai._ref import pair as rp
from bonsai._ref.star import node_message as ref_message
from bonsai._ref.star import star_loglik_grad

from ._bounds import EPS64, rounded

F64 = pytest.param(jnp.float64, marks=pytest.mark.x64, id="float64")
F32 = pytest.param(jnp.float32, id="float32")


def _family(rng, C=12, G=40, shift=0.0):
    x = rng.normal(shift, 1.0, (C, G))
    var = rng.uniform(0.05, 0.6, (C, G))
    t = rng.uniform(0.05, 0.8, C)
    return x, var, t


P_PAD = 5  # every default-lane call scores 5 pairs of 40 genes: one compiled solver per (mode, dtype)
ALL_PAIRS = list(itertools.combinations(range(12), 2))  # a 12-child parent; its batch is the second shape


def _centred(x):
    """Centring contract of ``pair_score``: the per-gene mean of the leaves subtracted in float64,
    as ``ArrayTree`` does (the M2 engine centres once per dataset, before any cast)."""
    return x - x.mean(axis=0)


def _ref_parent(x, var, t, dtype):
    """Reference inputs = what the kernel sees: centred children rounded to dtype; parent in fp64."""
    xr_, vr_, tr_ = (rounded(a, np.dtype(dtype)) for a in (_centred(x), var, t))
    return xr_, vr_, tr_, *ref_message(xr_, vr_, tr_)


def _kernel(x, var, t, i, j, dtype, mode):
    """Kernel call as the engine makes it (A8): centred children, device child weights, float64 host
    accumulator, R = parent − pair formed on the host from those same weights, then cast."""
    c = lambda a: jnp.asarray(a, dtype)
    xd = c(_centred(x))
    w = child_weights(c(var), c(t))
    acc = ParentAcc.from_children(w, xd)
    xr, Wr = acc.message()
    xR, WR = acc.rest(w[i], xd[i], w[j], xd[j])
    return pair_score(xd[i], c(var[i]), c(t[i]), xd[j], c(var[j]), c(t[j]), c(xR), c(1.0 / WR), c(xr), c(Wr), mode=mode)


def _objective(s: rp.PairScore, mode, eps=EPS64):
    """Reference -ell_new in the kernel's variables, its gradient, the box and the per-gene
    gradient-term scale (Σ_g |terms of dell/dt_c|, chain rule to u).

    u = log t of times stored at precision eps: a solver clipped onto a log-t bound b returns
    t = fl(exp(fl(b))), so log t is b to within (1 + |b|)·eps and is snapped onto it."""
    xs, vs = s.xs, s.vs

    def snapped(f, u, lo, hi):
        tol = (1 + np.abs(lo)) * eps, (1 + np.abs(hi)) * eps
        u = np.where(np.abs(u - lo) <= tol[0], lo, np.where(np.abs(u - hi) <= tol[1], hi, u))
        return f, u, lo, hi

    def scale(tt):
        w = 1 / (vs + tt[:, None])
        xn, Wn = ref_message(xs, vs, tt)
        return np.sum(w * (w * (xn - xs) ** 2 + 1 + w / Wn), axis=1)

    if mode == "sequential":
        t12 = s.t[0] + s.t[1]

        def f_seq(u):
            tt = np.array([u[0], t12 - u[0], np.exp(u[1])])
            ell, g = star_loglik_grad(xs, vs, tt)
            sc = scale(tt)
            return -ell, -np.array([g[0] - g[1], tt[2] * g[2]]), np.array([sc[0] + sc[1], tt[2] * sc[2]])

        u = np.array([s.t[0], np.log(s.t[2])])
        return snapped(f_seq, u, np.array([0.0, PAIR_LOG_T_MIN]), np.array([t12, PAIR_LOG_T_MAX]))

    def f(u):
        tt = np.exp(u)
        ell, g = star_loglik_grad(xs, vs, tt)
        return -ell, -tt * g, tt * scale(tt)

    return snapped(f, np.log(s.t), np.full(3, PAIR_LOG_T_MIN), np.full(3, PAIR_LOG_T_MAX))


def _value_scale(s: rp.PairScore, xr, Wr):
    """Σ_g of |terms| of ell_new and ell_old (TH:5093-5105, 5400-5402), each residual term with its
    cross term: r = x_a − x_i is formed from numbers of size |x|, so its rounding δr ≤ eps·(|x_a| +
    |x_i|) moves w·r² by 2·w·|r|·δr; budgeted as w·(r² + 2|r|(|x_a| + |x_i|)) (small once centred)."""
    wa = 1 / (s.vs + s.t[:, None])
    xa, Wa = ref_message(s.xs, s.vs, s.t)
    r_new, r_old = np.abs(xa - s.xs), np.abs(xr - s.xs)
    new = np.sum(1 + np.abs(np.log(wa)) + wa * (r_new**2 + 2 * r_new * (np.abs(xa) + np.abs(s.xs))))
    new += np.sum(np.abs(np.log(Wa)))
    old = np.sum(np.abs(np.log(s.w_old)) + s.w_old * (r_old**2 + 2 * r_old * (np.abs(xr) + np.abs(s.xs))))
    return new + old + np.sum(np.abs(np.log(Wr)))


def _lambda_min(f, u, lo, hi):
    """Smallest eigenvalue of the reference Hessian in the free coordinates, by central differences
    of the analytic gradient (h = 1e-5·max(1, |u|); its O(h²) error is far below λ itself here)."""
    idx = np.flatnonzero(~((u <= lo) | (u >= hi)))
    if len(idx) == 0:
        return np.inf
    H = np.zeros((len(idx), len(idx)))
    for a, i in enumerate(idx):
        h = 1e-5 * max(1.0, abs(u[i]))
        up, um = u.copy(), u.copy()
        up[i] += h
        um[i] -= h
        H[:, a] = (f(up)[1][idx] - f(um)[1][idx]) / (2 * h)
    return float(np.linalg.eigvalsh(0.5 * (H + H.T))[0])


def _check_pairs(dtype, mode, x, var, t, pairs, *, fallback=False, max_flagged=0, batch=None):
    """Kernel vs reference on ``pairs``. A pair is flagged if not converged or if its fallback flag
    differs from ``fallback`` (joint times where sequential ones were expected); at most
    ``max_flagged`` may be, and those must still carry their finite last iterate (A11). Every other
    pair is checked in full (status, KKT, value, active set, UB slopes). ``batch``: the pairs scored
    in one kernel call (default: ``pairs`` padded to P_PAD); vmapped float32 rounding depends on it.
    Returns (out restricted to ``pairs``, n_flagged)."""
    eps = float(jnp.finfo(dtype).eps)
    G = x.shape[1]
    batch = batch or (pairs * P_PAD)[: max(P_PAD, len(pairs))]  # pad to the shared shape
    i, j = np.array(batch).T
    out = _kernel(x, var, t, i, j, dtype, mode)
    out = jax.tree.map(lambda a: a[np.array([batch.index(p) for p in pairs])], out)
    assert out.dlogl.dtype == out.t.dtype == out.a.dtype == dtype
    n = len(pairs)
    flagged = ~np.asarray(out.converged)[:n] | (np.asarray(out.fallback)[:n] != fallback)
    assert flagged.sum() <= max_flagged
    assert np.isfinite(np.asarray(out.dlogl)[:n]).all() and np.isfinite(np.asarray(out.t)[:n]).all()
    omode = "joint" if fallback else mode  # the objective the returned times optimise
    xr_, vr_, tr_, xr, Wr = _ref_parent(x, var, t, dtype)
    for k, (a, b) in enumerate(pairs):  # padding repeats pairs; only the given ones are checked
        if flagged[k]:
            continue
        ref = rp.pair_score(xr_[a], vr_[a], tr_[a], xr_[b], vr_[b], tr_[b], xr, Wr, mode=mode)
        assert ref.converged
        # 1. first-order optimality of the kernel's times for the reference objective: the kernel
        #    stops at |pg| ≤ (PG_EPS_FACTOR + G)·eps·scale of its own objective, whose gradient
        #    differs from the reference one by its rounding, ≤ (8 + G)·eps·scale (plain gene sum)
        at_kernel = rp.PairScore(ref.dlogl, np.asarray(out.t[k], dtype=np.float64), True, ref.xs, ref.vs, ref.w_old)
        f, u, lo, hi = _objective(at_kernel, omode, eps)
        _, g, sc = f(u)
        pg = np.where(((u <= lo) & (g > 0)) | ((u >= hi) & (g < 0)), 0.0, g)
        pg_bound = (PG_EPS_FACTOR + 2 * G + 8) * eps * sc
        assert np.all(np.abs(pg) <= pg_bound)
        # 2. optimal value: per-gene terms ≤ 8 rounded ops each, +1 for the compensated gene sum
        #    (_value_scale), plus the solver gap of the quadratic model, ‖pg‖²/(2·λ_min)
        lam = _lambda_min(f, u, lo, hi)
        assert lam > 0  # a strict local optimum of the reference objective
        gap = float(np.sum(pg_bound**2)) / (2 * lam)
        assert abs(float(out.dlogl[k]) - ref.dlogl) <= 9 * eps * _value_scale(ref, xr, Wr) + gap
        # 3. the same active bounds as the reference optimum
        _, u_ref, lo_ref, hi_ref = _objective(ref, omode)
        assert np.array_equal(u == lo, u_ref == lo_ref) and np.array_equal(u == hi, u_ref == hi_ref)
        # 4. upper-bound slopes, evaluated by the reference formula at the kernel's times
        a_ref, b_ref = rp.ub_slopes(at_kernel, xr, Wr)
        tol_a, tol_b = _ub_tols(at_kernel, xr, Wr, eps)
        assert abs(float(out.a[k]) - a_ref) <= tol_a and abs(float(out.b[k]) - b_ref) <= tol_b
    return out, int(flagged.sum())


def _ub_tols(s, xr, Wr, eps):
    """|error| of a = ‖dx/√W_r‖, b = ‖dW·W_r‖ ≤ 16·eps·‖|terms|‖: the formula (TH:5166-5211) with
    every difference replaced by a sum of absolute values; ≤ 16 rounded operations per term."""
    w1, w2, WR = s.w_old
    wa = 1 / (s.vs + s.t[:, None])
    w1a, w2a, wRa = wa
    ra, rr = wRa / wa.sum(axis=0), WR / Wr
    x1, x2, xR = s.xs
    v12, v1R, v2R = (x1 - x2) ** 2, (x1 - xR) ** 2, (x2 - xR) ** 2
    da = w1a * w2a * v12 + w1a * wRa * v1R + w2a * wRa * v2R
    dr = w1 * w2 * v12 + w1 * WR * v1R + w2 * WR * v2R
    a1, a2 = np.abs(x1 - xR), np.abs(x2 - xR)
    mW = (
        wRa * (1 + ra) + WR * (1 + rr) + ra * (w1a * w2a * v12 + (1 + ra) * da) + rr * (w1 * w2 * v12 + (1 + rr) * dr)
    ) / WR**2
    mW += (2 * np.abs(xr - xR) / WR) * (ra * (w1a * a1 + w2a * a2) + rr * (w1 * a1 + w2 * a2))
    mx = 2 * (Wr / WR) * (a1 * (w1a * ra + w1 * rr) + a2 * (w2a * ra + w2 * rr))
    return 16 * eps * np.linalg.norm(mx / np.sqrt(Wr)), 16 * eps * np.linalg.norm(mW * Wr)


PAIRS = [(0, 1), (2, 5), (7, 3), (4, 11), (9, 10)]


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("mode", ["sequential", "joint"])
@pytest.mark.parametrize("shift", [0.0, 30.0, 3e4])
def test_pair_score_matches_reference(dtype, mode, shift):
    """shift = 30: uncentred input like upstream's x (μ_g added), centred as the contract says.
    shift = 3e4 gives the centring contract test power: fed uncentred, the float32 inputs are
    rounded by ~eps32·3e4/2 ≈ 2e-3 and every checked quantity exceeds its bound (12-55× on these
    pairs); at shift 30 the same error stays within the worst-case bounds (≤ 0.08×)."""
    x, var, t = _family(np.random.default_rng(0), shift=shift)
    _check_pairs(dtype, mode, x, var, t, PAIRS)


def test_reference_dlogl_is_invariant_under_a_gene_shift():
    """Why centring is allowed: dLogL depends on x only through differences (flat root prior). Two
    fp64 evaluations at |x| ≈ 30: ≤ 2·9·eps64·Σ|terms| of the uncentred one (with cross terms)."""
    x, var, t = _family(np.random.default_rng(0), shift=30.0)
    for a, b in PAIRS:
        res = []
        for xx in (x, _centred(x)):
            xr, Wr = ref_message(xx, var, t)
            res.append(rp.pair_score(xx[a], var[a], t[a], xx[b], var[b], t[b], xr, Wr))
        xr, Wr = ref_message(x, var, t)
        assert abs(res[0].dlogl - res[1].dlogl) <= 18 * EPS64 * _value_scale(res[0], xr, Wr)


@pytest.mark.parametrize("dtype", [F32, F64])
def test_two_leaf_time_matches_reference_and_branches(dtype):
    """Root error ≤ |f(t)|/|f'(t)| + eps·t with |f| ≤ 8·eps·Σ_g |(d² − T)/T²| (per-gene terms,
    compensated gene sum)."""
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=(2, 30))
    v1, v2 = rng.uniform(0.01, 0.1, (2, 30))
    eps = float(jnp.finfo(dtype).eps)
    a, b, c, d = (rounded(z, np.dtype(dtype)) for z in (x1, v1, x2, v2))
    t_ref, ok = rp.two_leaf_time(a, b, c, d)
    t, conv = two_leaf_time(jnp.asarray(b + d, dtype), jnp.asarray((a - c) ** 2, dtype))
    S, d2 = b + d, (a - c) ** 2
    T = t_ref + S
    fprime = abs(np.sum((T - 2 * d2) / T**3))
    assert ok and bool(conv) and t.dtype == dtype
    assert abs(float(t) - t_ref) <= 8 * eps * np.sum(np.abs((d2 - T) / T**2)) / fprime + eps * t_ref
    f = lambda v: jnp.full(4, v, dtype)
    t, conv = two_leaf_time(f(5.0), f(0.1))  # f(0) ≤ 0: exact zero clamp
    assert float(t) == 0.0 and bool(conv)
    t, conv = two_leaf_time(f(1e-9), f(1e10))  # bracket exhausted (TH:4915)
    assert not bool(conv)


def test_identical_cells_take_the_zero_branches():
    """Two identical children: t_12 = 0 exactly, so t_1a = t_2a = 0 (box [0, 0]) in both codes."""
    x, var, t = _family(np.random.default_rng(2))
    x[1], var[1] = x[0], var[0]
    out, *_ = _check_pairs(jnp.float32, "sequential", x, var, t, [(0, 1)])
    assert float(out.t[0, 0]) == 0.0 and float(out.t[0, 1]) == 0.0


def test_root_coincides_with_ancestor_hits_the_lower_log_t_bound():
    """R next to the pair's ancestor drives t_aR to its lower bound 1e-7 (log −16.1180) in both codes."""
    x, var, t = _family(np.random.default_rng(3), C=4)
    x[2:] = 0.5 * (x[0] + x[1])  # the rest of the parent sits between children 0 and 1
    var[2:] = 1e-3
    out, *_ = _check_pairs(jnp.float32, "joint", x, var, t, [(0, 1)])
    assert np.log(float(out.t[0, 2])) == pytest.approx(PAIR_LOG_T_MIN, abs=1e-6)


def _heavy_family(rng, C=202, G=40):
    """Two heavy children (t = 0, sigma² ~ 1e-6: w ~ 1e6) among 200 light ones (w ≈ 0.2)."""
    x, var, t = rng.normal(size=(C, G)), rng.uniform(4.0, 6.0, (C, G)), rng.uniform(4.0, 6.0, C)
    var[:2], t[:2] = rng.uniform(1e-6, 3e-6, (2, G)), 0.0
    return x, var, t


def test_same_weight_rule_keeps_the_rest_of_the_parent_exact():
    """A8 same-weight rule: removing the float32 weights that were added (cast to float64 on the
    host) leaves the float64 sum over the other children, up to float64 rounding. NumPy's pairwise
    sums of C terms err by ≤ ⌈log2 C⌉·eps64·Σ|terms|, the two subtractions add 2 roundings and the
    division 1: |W_R − W_exact| ≤ (⌈log2 C⌉ + 3)·eps64·W and |x_R − x_exact| ≤ (b_WX + |x|·b_W)/W_R
    + eps64·|x| with b_WX the same bound on Σ|w·x|. Recomputing the weights in float64 instead leaves
    their float32 rounding (~eps32·1e6) in W_R, beyond that bound by a factor > 1e6."""
    x, var, t = _heavy_family(np.random.default_rng(4))
    xd = jnp.asarray(x, jnp.float32)
    w = child_weights(jnp.asarray(var, jnp.float32), jnp.asarray(t, jnp.float32))
    acc = ParentAcc.from_children(w, xd)
    xR, WR = acc.rest(w[:1], xd[:1], w[1:2], xd[1:2])
    xe, We = ParentAcc.from_children(w[2:], xd[2:]).message()
    w64, x64 = np.asarray(w, dtype=np.float64), np.asarray(xd, dtype=np.float64)
    k = int(np.ceil(np.log2(len(x)))) + 3
    b_W, b_WX = k * EPS64 * acc.W, k * EPS64 * np.abs(w64 * x64).sum(axis=0)
    assert np.all(np.abs(WR[0] - We) <= b_W)
    assert np.all(np.abs(xR[0] - xe) <= (b_WX + np.abs(xe) * b_W) / We + EPS64 * np.abs(xe))
    w_recomputed = 1.0 / (var[:2] + t[:2, None])  # float64, not the weights that were added
    W_bad = (acc.W - w_recomputed[0]) - w_recomputed[1]
    assert np.any(np.abs(W_bad - We) > 1e6 * b_W)


def test_pair_score_with_two_heavy_children():
    """Maths ref §6 hazard (W_R = W_r − w_1 − w_2, the worst cancellation): with R from the host
    accumulator the float32 kernel meets the ordinary bounds."""
    x, var, t = _heavy_family(np.random.default_rng(5))
    _check_pairs(jnp.float32, "sequential", x, var, t, [(0, 1)])


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("mode", ["sequential", "joint"])
def test_pair_score_on_all_pairs_of_a_parent(dtype, mode):
    """Every pair of a 12-child parent (66 pairs), not hand-picked ones: status,
    value, projected-gradient KKT (check 1 of ``_check_pairs``: at an active bound only an outward
    gradient may exceed the threshold), active set and UB slopes against the reference."""
    x, var, t = _family(np.random.default_rng(8))
    pairs = ALL_PAIRS
    for k in range(0, len(pairs), P_PAD):
        _check_pairs(dtype, mode, x, var, t, pairs[k : k + P_PAD])


# Pairs (rng seed of ``_family``, mode) that missed the convergence test, scored with all 66 pairs
# of their family in one batch, in earlier solver versions or variants (A11): the fixed Gershgorin
# shift; LM drafts whose acceptance compared raw |pg| or needed a strict decrease of f (stalls on
# the tail of a log t heading for PAIR_LOG_T_MIN); LM without Marquardt's scaling. Rounding under vmap
# depends on the batch, so they are rescored in that batch.
REGRESSION = {
    (100, "joint"): [(2, 3)],
    (101, "joint"): [(3, 8), (6, 11), (8, 11)],
    (104, "joint"): [(6, 11)],
    (107, "joint"): [(0, 10), (1, 8), (2, 10), (6, 10)],
    (108, "joint"): [(8, 10), (10, 11)],
    (110, "joint"): [(2, 8)],
    (116, "joint"): [(6, 8)],
    (202, "joint"): [(4, 10)],
    (204, "joint"): [(2, 3)],
    (224, "joint"): [(1, 8)],
    (239, "joint"): [(1, 10)],
    (102, "sequential"): [(0, 6)],
    (104, "sequential"): [(0, 6)],
    (105, "sequential"): [(3, 11)],
    (106, "sequential"): [(2, 11)],
    (111, "sequential"): [(3, 8)],
    (114, "sequential"): [(2, 4)],
    (118, "sequential"): [(1, 8), (1, 10), (10, 11)],
    (123, "joint"): [(3, 11)],  # an early LM variant with an uncapped r branch climbed 1.7e4·fnoise here
    (206, "sequential"): [(1, 4)],
    (226, "sequential"): [(1, 2)],
    (227, "sequential"): [(6, 11)],
    (230, "sequential"): [(0, 11)],
    (234, "sequential"): [(7, 9)],
}


@pytest.mark.parametrize(("seed", "mode"), list(REGRESSION))
def test_previously_unconverged_pairs(seed, mode):
    """A11 regression cases: converged, no fallback, and the full reference checks."""
    x, var, t = _family(np.random.default_rng(seed))
    _check_pairs(jnp.float32, mode, x, var, t, REGRESSION[seed, mode], batch=ALL_PAIRS)


MAX_FLAGGED = 3  # of 1,320 pairs per mode (0.25%), declared before the test seeds were run with LM


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["sequential", "joint"])
def test_convergence_rate_on_regression_family_seeds(mode):
    """A11: all 66 pairs of 20 12-child families already used while tuning the solver (seeds
    100-119; the iteration budget was chosen on seeds 200-239), each family in one batch as the
    budget was measured (vmapped float32 rounding depends on the batch shape). Flagged pairs (not
    converged, or a sequential fallback) are at most MAX_FLAGGED and keep a finite last iterate;
    every other pair meets the reference checks. Held-out evidence (seeds never used while tuning
    the solver) is ``test_convergence_rate_on_held_out_seeds`` below."""
    pairs = ALL_PAIRS
    flagged = 0
    for seed in range(100, 120):
        x, var, t = _family(np.random.default_rng(seed))
        kw = {"max_flagged": len(pairs), "batch": pairs}
        flagged += _check_pairs(jnp.float32, mode, x, var, t, pairs, **kw)[1]
    assert flagged <= MAX_FLAGGED


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["sequential", "joint"])
def test_convergence_rate_on_held_out_seeds(mode):
    """A11 held-out check: all 66 pairs of 20 12-child families on seeds 300-319, never used while
    tuning the solver (``N_NEWTON``, ``LM_RISE``) or choosing ``MAX_FLAGGED``. Same family generator
    and pre-declared bound as ``test_convergence_rate_on_regression_family_seeds``; run once."""
    pairs = ALL_PAIRS
    flagged = 0
    for seed in range(300, 320):
        x, var, t = _family(np.random.default_rng(seed))
        kw = {"max_flagged": len(pairs), "batch": pairs}
        flagged += _check_pairs(jnp.float32, mode, x, var, t, pairs, **kw)[1]
    assert flagged <= MAX_FLAGGED


@pytest.mark.parametrize("mode", ["sequential", "joint"])
def test_no_accepted_step_raises_f_beyond_the_cap(mode):
    """A step accepted for a smaller relative projected gradient may raise f by at most
    LM_RISE·fnoise (without the cap an earlier LM variant jumped uphill to a worse all-bound KKT
    point, by up to 1.7e4·fnoise; this family had such jumps in both modes). Whole family, one batch."""
    x, var, t = _family(np.random.default_rng(102))
    c = lambda a: jnp.asarray(a, jnp.float32)
    xd = c(_centred(x))
    w = child_weights(c(var), c(t))
    acc = ParentAcc.from_children(w, xd)
    xr, Wr = acc.message()
    i, j = np.array(ALL_PAIRS).T
    xR, WR = acc.rest(w[i], xd[i], w[j], xd[j])
    rises = jax.jit(jax.vmap(functools.partial(kernel._one, mode=mode, diag=True), in_axes=(0,) * 8 + (None,) * 2))(
        xd[i], c(var[i]), c(t[i]), xd[j], c(var[j]), c(t[j]), c(xR), c(1.0 / WR), c(xr), c(Wr)
    )
    assert max(float(jnp.max(r)) for r in rises) <= LM_RISE


@pytest.mark.parametrize(("mode", "budget"), [("sequential", 8), ("joint", 16)])
def test_budget_limited_stall_is_flagged(monkeypatch, mode, budget):
    """A11 status with finite input: with a cut iteration budget some solves stop short. Every pair
    reported converged must pass the reference checks (independent KKT first), and at least one is
    flagged. Optimal times are below 1 here (t_aR ~ 0.05), where a log-t convergence scale missing
    its t factor would be too loose; the budgets leave both converged and unconverged pairs in this
    batch. A batch shape no other test uses and cleared caches make the patched budget traced and
    keep it from leaking."""
    x, var, t = _family(np.random.default_rng(11))
    pairs = ALL_PAIRS[:6]
    monkeypatch.setattr(kernel, "N_NEWTON", budget)
    jax.clear_caches()
    try:
        kw = {"max_flagged": len(pairs) - 1, "batch": pairs}
        assert _check_pairs(jnp.float32, mode, x, var, t, pairs, **kw)[1] >= 1
    finally:
        monkeypatch.undo()
        jax.clear_caches()


def test_sequential_falls_back_to_joint_without_a_t12_bracket():
    """Σ_g d² = 40·(2e5)² > 1e11: the ×10 bracket ends before f(ub) < 0 (TH:4915), as in the
    reference, which then solves jointly; the kernel returns the joint times and flags them."""
    x, var, t = _family(np.random.default_rng(9))
    x[1] = x[0] + 2e5
    out, *_ = _check_pairs(jnp.float32, "sequential", x, var, t, [(0, 1)], fallback=True)
    assert np.asarray(out.converged).all() and np.asarray(out.fallback).all()


def test_unconverged_pair_is_flagged_not_dropped():
    """A11 status: a pair whose solve cannot pass its test (a NaN mean) comes back converged=False
    (never as a silent −inf, never silently switched to joint); the other pairs of the batch are
    unaffected, bit for bit."""
    x, var, t = _family(np.random.default_rng(10))
    c = lambda a: jnp.asarray(a, jnp.float32)
    xd = c(_centred(x))
    w = child_weights(c(var), c(t))
    acc = ParentAcc.from_children(w, xd)
    xr, Wr = acc.message()
    i, j = np.array(PAIRS).T
    xR, WR = acc.rest(w[i], xd[i], w[j], xd[j])
    args = [xd[i], c(var[i]), c(t[i]), xd[j], c(var[j]), c(t[j]), c(xR), c(1.0 / WR), c(xr), c(Wr)]
    for mode in ("sequential", "joint"):
        clean = pair_score(*args, mode=mode)
        bad = pair_score(args[0].at[0, 3].set(jnp.nan), *args[1:], mode=mode)
        assert np.asarray(clean.converged).all() and not bool(bad.converged[0])
        assert not np.isneginf(np.asarray(bad.dlogl)).any()
        for field in ("dlogl", "t", "a", "b", "converged", "fallback"):
            np.testing.assert_array_equal(np.asarray(getattr(bad, field))[1:], np.asarray(getattr(clean, field))[1:])
    # NaN in R only: t_12 (children 1, 2 alone) has its bracket, both 3-leaf solves fail; sequential
    # mode returns its own iterate, flagged not converged, and does not switch to joint
    args[6] = args[6].at[0, 3].set(jnp.nan)
    bad = pair_score(*args, mode="sequential")
    assert not bool(bad.converged[0]) and not bool(bad.fallback[0])


def test_two_child_parent_is_rejected_through_the_accumulator():
    """For a 2-child parent W − w_1 − w_2 from the float32 weights is rounding
    noise, which a threshold on W cannot certify; ``rest`` raises on the child count, which add and
    remove keep."""
    x, var, t = _family(np.random.default_rng(7), C=3)
    xd = jnp.asarray(_centred(x), jnp.float32)
    w = child_weights(jnp.asarray(var, jnp.float32), jnp.asarray(t, jnp.float32))
    acc = ParentAcc.from_children(w, xd)
    acc.remove(w[2], xd[2])
    assert acc.n == 2
    with pytest.raises(ValueError, match="3 children"):
        acc.rest(w[:1], xd[:1], w[1:2], xd[1:2])
    acc.add(w[2], xd[2])
    xR, WR = acc.rest(w[:1], xd[:1], w[1:2], xd[1:2])
    assert np.all(WR > 0) and np.all(np.isfinite(xR))


def test_unknown_mode_raises():
    x, var, t = _family(np.random.default_rng(5))
    with pytest.raises(ValueError, match="mode"):
        _kernel(x, var, t, np.array([0]), np.array([1]), jnp.float32, "greedy")


@pytest.mark.gpu
def test_pair_score_on_gpu(gpu_device):
    x, var, t = _family(np.random.default_rng(6), G=256)
    with jax.default_device(gpu_device):
        out, *_ = _check_pairs(jnp.float32, "sequential", x, var, t, PAIRS)
    assert out.dlogl.devices() == {gpu_device}
