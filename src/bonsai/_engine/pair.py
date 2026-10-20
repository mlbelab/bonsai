"""Batched 3-leaf pair score (maths ref §5-§7; amendment A2: sequential and joint modes).

Per pair: children 1, 2 of a parent with AIR message (xr, Wr), and R = the parent without them.
Everything runs in the compute dtype (A8). R = parent − child 1 − child 2 is the worst cancellation
in the code (maths ref §6), so the kernel does not form it: the caller passes (x_R, var_R) computed
in float64 on the host by ``host_acc.ParentAcc.rest`` from the same float32 weights. Sums over
genes use the compensated ``dsum``. Solvers run a fixed number of iterations so pairs batch.
Box solver: projected Newton with adaptive Levenberg–Marquardt damping (amendment A11).
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_solve
from jaxtyping import Array, Bool, Float

from bonsai._ref.pair import PAIR_LOG_T_MAX, PAIR_LOG_T_MIN

from .numerics import dsum

N_BRACKET = 8  # upper brackets 1, 10, ..., 1e7: upstream gives up once lb > 1e6 (TH:4915)
N_ROOT = 64  # safeguarded Newton/bisection steps: 1e7·2^-64 < eps32·1e-5
# LM iterations (A11): on 40 12-child families from seeds 200-239 (2,640 pairs per mode, disjoint
# from the test seeds) 40 iterations left none unconverged in either mode; 60 keeps a 1.5× margin
N_NEWTON, N_HALVINGS = 60, 40
LM_LAMBDA0 = 1e-3  # initial damping, relative to each coordinate's curvature |H_ii| (A11)
LM_UP, LM_DOWN = 10.0, 3.0  # damping × LM_UP on a rejected step, / LM_DOWN on an accepted one
# A step accepted only for lowering the relative projected gradient r may raise f by at most
# LM_RISE·fnoise. Two evaluations of f differ by up to 2·fnoise from rounding alone, and fnoise is a
# first-order worst case (the observed rounding is 10-100× smaller), so 10·fnoise still lets the
# solver cross ulp-level noise near the optimum and on a log t tail; a real uphill move, e.g. a jump
# to a worse point where every coordinate is on a bound (r = 0 there), is accepted only if its rise
# is ≤ LM_RISE·fnoise, otherwise rejected. Without the cap such rises reached 1.7e4·fnoise. With 10: 0 unconverged,
# 0 fallbacks on rng 100-119 and 200-239.
LM_RISE = 10.0
# converged iff |projected grad| ≤ (PG_EPS_FACTOR + G)·eps·Σ_g|grad terms|: 8 rounded ops per term
# × 8 slack, plus (G − 1)·eps for the plain (autodiff) float32 sum over genes in the gradient
PG_EPS_FACTOR = 64


class PairScore(NamedTuple):
    dlogl: Float[Array, " P"]  # ell_new − ell_old at the returned times (also when not converged)
    t: Float[Array, "P 3"]  # (t_1a, t_2a, t_aR)
    a: Float[Array, " P"]  # upper-bound slope for eps_x (TH:5207-5211)
    b: Float[Array, " P"]  # upper-bound slope for eps_W (TH:5192-5200)
    converged: Bool[Array, " P"]  # the returned times passed their solve's convergence test
    fallback: Bool[Array, " P"]  # sequential mode: joint times returned (no t_12 bracket, or sequential failed)


def _per_gene_ell(times, xs, vs):
    """Ell of a 3-star per gene, compute dtype (TH:5093-5105), and per-gene |dell/dt_c| scale."""
    w = 1.0 / (vs + times[:, None])
    W = w.sum(axis=0)  # three positive terms
    xn = (w * xs).sum(axis=0) / W
    r = xn - xs
    ell_g = jnp.log(w).sum(axis=0) - jnp.log(W) - (w * r * r).sum(axis=0)
    scale = w * (w * r * r + 1.0 + w / W)  # |terms| of dell/dt_c (TH:5107)
    return ell_g, scale


def _neg_ell(times, xs, vs):
    return -dsum(_per_gene_ell(times, xs, vs)[0])


def _ell_noise(times, xs, vs):
    """Rounding bound of ``_neg_ell``: 9·eps·Σ_g |terms of ell| (≤ 8 rounded ops per term, +1 dsum)."""
    w = 1.0 / (vs + times[:, None])
    W = w.sum(axis=0)
    r = (w * xs).sum(axis=0) / W - xs
    terms = jnp.abs(jnp.log(w)).sum(axis=0) + jnp.abs(jnp.log(W)) + (w * r * r).sum(axis=0)
    return 9 * jnp.finfo(xs.dtype).eps * dsum(terms)


def two_leaf_time(S, d2):
    """argmax_t −Σ_g [log(t+S) + d²/(t+S)] with upstream's bracket and zero clamp (TH:4893-4923)."""
    dt = S.dtype

    def f(t):  # Σ_g (d² − T)/T², per gene in the compute dtype, compensated sum over genes
        T = t + S
        return dsum(((d2 - T) / T) / T)

    def fp(t):
        T = t + S
        return dsum(((T - 2.0 * d2) / T) / (T * T))

    ubs = (10.0 ** jnp.arange(N_BRACKET)).astype(dt)
    neg = jax.vmap(f)(ubs) < 0
    k = jnp.argmax(neg)
    lb0, ub0 = jnp.where(k == 0, 0.0, ubs[k - 1]).astype(dt), ubs[k]

    def body(_, st):
        lo, hi, t = st
        ft, dft = f(t), fp(t)
        lo, hi = jnp.where(ft > 0, t, lo), jnp.where(ft > 0, hi, t)
        tn = t - ft / dft
        return lo, hi, jnp.where((dft < 0) & (tn > lo) & (tn < hi), tn, 0.5 * (lo + hi))

    t = lax.fori_loop(0, N_ROOT, body, (lb0, ub0, 0.5 * (lb0 + ub0)))[2]
    at_zero = f(jnp.zeros((), dt)) <= 0
    return jnp.where(at_zero, 0.0, t).astype(dt), at_zero | jnp.any(neg)


def _pg(u, g, lo, hi):
    """Projected gradient: zero where a bound is active and the gradient points outward."""
    return jnp.where(((u <= lo) & (g > 0)) | ((u >= hi) & (g < 0)), 0.0, g)


def _lm_scale(H):
    """Marquardt scaling D = |diag H|, floored at eps·max|diag H| (and tiny) so that D > 0."""
    d = jnp.abs(jnp.diag(H))
    return jnp.maximum(d, jnp.finfo(H.dtype).eps * jnp.max(d) + jnp.finfo(H.dtype).tiny)


def _box_newton(fun, u0, lo, hi, gscale, fnoise):
    """Projected Newton on a box (u has 2 or 3 entries) with Levenberg–Marquardt damping (A11).

    Step: Cholesky solve with the free Hessian + λD, D = |diag H| (Marquardt's scaling: a log t
    heading for its lower bound has curvature ∝ t, which an unscaled λI would swamp). A step is
    accepted if fun does not rise beyond its rounding bound fnoise(u) (LM's acceptance; on the tail
    of a log t heading for its bound, and near the optimum, true decreases are below the rounding of
    fun, which then moves by single ulps either way), or if it lowers r(u) = max_i |pg_i|/gscale_i,
    the quantity the convergence test bounds (relative, so that a t-scaled coordinate reaching its
    bound counts against the gradient noise of the others). A non-positive-definite factor or a
    rejected step multiplies λ by LM_UP, an accepted one divides it by LM_DOWN. λ stays in
    [eps, cap], cap the scaled Gershgorin bound max_i (Σ_j≠i |H_ij| − H_ii)/D_i, at which H + λD is
    positive definite; there a rejected step is backtracked by halvings instead. A step accepted on
    r alone may raise fun by at most LM_RISE·fnoise. Returns (u, r(u), the largest accepted rise of
    fun in units of fnoise: a diagnostic, ≤ LM_RISE).
    """
    grad, hess = jax.grad(fun), jax.hessian(fun)
    d = u0.shape[0]
    eye = jnp.eye(d, dtype=u0.dtype)
    eps = jnp.finfo(u0.dtype).eps

    def free(u):
        g, H = grad(u), hess(u)
        fixed = ((u <= lo) & (g > 0)) | ((u >= hi) & (g < 0))
        return g, jnp.where(fixed, 0.0, g), jnp.where(~fixed[:, None] & ~fixed[None, :], H, eye)

    def r(u, g):
        return jnp.max(jnp.abs(_pg(u, g, lo, hi)) / gscale(u))

    def body(_, st):
        u, lam, rise = st
        g, gf, Hf = free(u)
        D = _lm_scale(Hf)
        # Gershgorin: Hf + λD is diagonally dominant, so positive definite, once λ·D_i > Σ_j≠i |H_ij| −
        # H_ii for all i; the 8·eps·max|H| margin covers the rounding of the factorisation
        radius = jnp.sum(jnp.abs(Hf), axis=1) - 2.0 * jnp.diag(Hf) + 8 * eps * jnp.max(jnp.abs(Hf))
        cap = jnp.maximum(jnp.max(radius / D), eps)
        lam = jnp.clip(lam, eps, cap)
        L = jnp.linalg.cholesky(Hf + lam * jnp.diag(D))
        pd = jnp.all(jnp.isfinite(L))
        step = -cho_solve((jnp.where(pd, L, eye), True), gf)
        step = jnp.where(pd & jnp.all(jnp.isfinite(step)), step, 0.0)
        f0, r0, fn = fun(u), r(u, g), fnoise(u)

        def trial(a):
            un = jnp.clip(u + a * step, lo, hi)
            fu = fun(un)
            return un, fu, (fu <= f0 + fn) | ((r(un, grad(un)) < r0) & (fu <= f0 + LM_RISE * fn))

        def accept(a):
            return trial(a)[2]

        n_max = jnp.where(lam >= cap, N_HALVINGS, 0)  # backtrack only once damping cannot grow
        a = lax.while_loop(
            lambda st: ~accept(st[0]) & (st[1] < n_max),
            lambda st: (0.5 * st[0], st[1] + 1),
            (jnp.ones((), u.dtype), 0),
        )[0]
        un, fu, acc = trial(a)
        ok = pd & acc
        rise = jnp.where(ok, jnp.maximum(rise, (fu - f0) / (fn + jnp.finfo(u0.dtype).tiny)), rise)
        return jnp.where(ok, un, u), jnp.where(ok, lam / LM_DOWN, lam * LM_UP), rise

    init = (u0, jnp.asarray(LM_LAMBDA0, u0.dtype), jnp.asarray(-jnp.inf, u0.dtype))
    u, _, rise = lax.fori_loop(0, N_NEWTON, body, init)
    return u, r(u, grad(u)), rise


def _one(x1, v1, t1, x2, v2, t2, xR, vR, xr, Wr, mode, diag=False):
    """Score one pair (vmapped by ``pair_score``).

    ``diag`` (static, tests only): return the solvers' largest accepted rises of f in units of
    fnoise, (joint, sequential), instead of the score.
    """
    dt = x1.dtype
    eps = jnp.finfo(dt).eps
    G = x1.shape[0]
    pg_eps = (PG_EPS_FACTOR + G) * eps
    w1, w2 = 1.0 / (v1 + t1), 1.0 / (v2 + t2)
    xs = jnp.stack([x1, x2, xR])
    vs = jnp.stack([v1, v2, vR])
    tau = jnp.minimum(t1, t2) / 2
    joint0 = jnp.stack([t1 - tau, t2 - tau, tau])

    # joint (A2) / sequential fallback (TH:4822-4831): log t of all three in [PAIR_LOG_T_MIN, PAIR_LOG_T_MAX]
    lo3, hi3 = jnp.full(3, PAIR_LOG_T_MIN, dt), jnp.full(3, PAIR_LOG_T_MAX, dt)
    tiny = jnp.finfo(dt).tiny

    def joint_scale(s):  # Σ_g |terms of dell/dlog t_c|
        return jnp.exp(s) * _per_gene_ell(jnp.exp(s), xs, vs)[1].sum(axis=1) + tiny

    s, r_j, rise_j = _box_newton(
        lambda s: _neg_ell(jnp.exp(s), xs, vs),
        jnp.log(jnp.maximum(joint0, 1e-4)),
        lo3,
        hi3,
        joint_scale,
        lambda s: _ell_noise(jnp.exp(s), xs, vs),
    )
    tj = jnp.exp(s)
    ok_j = r_j <= pg_eps
    times, ok, fallback, rise_s = tj, ok_j, jnp.zeros((), bool), jnp.asarray(-jnp.inf, dt)
    if mode == "sequential":  # TH:4853-4867
        t12, ok12 = two_leaf_time(v1 + v2, (x1 - x2) ** 2)

        def seq_times(u):
            return jnp.stack([u[0], t12 - u[0], jnp.exp(u[1])])

        lo2, hi2 = (
            jnp.stack([jnp.zeros((), dt), jnp.asarray(PAIR_LOG_T_MIN, dt)]),
            jnp.stack([t12, jnp.asarray(PAIR_LOG_T_MAX, dt)]),
        )
        u0 = jnp.stack([0.5 * t12, jnp.log(jnp.maximum(tau, 1e-4))])

        def seq_scale(u):  # chain rule of the |terms| to (t_1a, log t_aR)
            sc = _per_gene_ell(seq_times(u), xs, vs)[1].sum(axis=1)
            return jnp.stack([sc[0] + sc[1], jnp.exp(u[1]) * sc[2]]) + tiny

        u, r_s, rise_s = _box_newton(
            lambda u: _neg_ell(seq_times(u), xs, vs),
            u0,
            lo2,
            hi2,
            seq_scale,
            lambda u: _ell_noise(seq_times(u), xs, vs),
        )
        ts = seq_times(u)
        ok_s = r_s <= pg_eps
        # joint fallback where the reference falls back (no t_12 bracket, or the sequential solve
        # failed while the joint one converged), flagged; otherwise the sequential iterate and status
        fallback = ~ok12 | (~ok_s & ok_j)
        times, ok = jnp.where(fallback, tj, ts), jnp.where(fallback, ok_j, ok_s)

    if diag:
        return rise_j, rise_s
    ell_new, _ = _per_gene_ell(times, xs, vs)
    # old term: the parent with children (1, 2, R) at weights (w_1, w_2, W_R) and the given (xr, Wr) (TH:5400-5402)
    WR_ = 1.0 / vR
    ell_old = (
        jnp.log(w1)
        + jnp.log(w2)
        + jnp.log(WR_)
        - jnp.log(Wr)
        - w1 * (xr - x1) ** 2
        - w2 * (xr - x2) ** 2
        - WR_ * (xr - xs[2]) ** 2
    )
    dlogl = dsum(ell_new - ell_old)  # per-gene difference first

    # upper-bound slopes at the optimum (estimateDerBasedDLogLUB, TH:5166-5211)
    wa = 1.0 / (vs + times[:, None])
    w1a, w2a, wRa = wa
    rho_a = wRa / wa.sum(axis=0)
    rho_r = WR_ / Wr
    xx1, xx2, xxR = xs
    v12, v1R, v2R = (xx1 - xx2) ** 2, (xx1 - xxR) ** 2, (xx2 - xxR) ** 2
    da = w1a * w2a * v12 + w1a * wRa * v1R + w2a * wRa * v2R
    dr = w1 * w2 * v12 + w1 * WR_ * v1R + w2 * WR_ * v2R
    dW = (
        wRa * (1 - rho_a)
        - WR_ * (1 - rho_r)
        + rho_a * (w1a * w2a * v12 - (1 - rho_a) * da)
        - rho_r * (w1 * w2 * v12 - (1 - rho_r) * dr)
    ) / WR_**2 + (2 * (xr - xxR) / WR_) * (
        rho_a * (w1a * (xx1 - xxR) + w2a * (xx2 - xxR)) - rho_r * (w1 * (xx1 - xxR) + w2 * (xx2 - xxR))
    )
    dx = 2 * (Wr / WR_) * ((xx1 - xxR) * (w1a * rho_a - w1 * rho_r) + (xx2 - xxR) * (w2a * rho_a - w2 * rho_r))
    a = jnp.sqrt(dsum(dx * dx / Wr))
    b = jnp.sqrt(dsum((dW * Wr) ** 2))
    return PairScore(dlogl, times, a, b, ok, fallback)


@eqx.filter_jit
def pair_score(x1, v1, t1, x2, v2, t2, xR, vR, xr, Wr, mode: str = "sequential") -> PairScore:
    """Score P candidate pairs, all inputs in one compute dtype.

    x*, v*: (P, G) children 1 and 2 (v = 1/W for internal nodes); t*: (P,); xR, vR: (P, G) the
    parent without the pair, from ``ParentAcc.rest`` (float64 on the host, then cast); xr, Wr: (G,)
    the parent's AIR message. Centring contract: all means (x1, x2, xR, xr) carry the same per-gene
    shift, the mean of the leaves subtracted in float64 before the cast (as ``ArrayTree`` does; the
    M2 engine centres once per dataset). dLogL is shift-invariant, but residuals of uncentred float32
    means (|x| ~ 30 upstream) would lose ~5 digits. Sequential mode always runs the joint solve too
    (both are evaluated for every pair under ``vmap``, so a ``cond`` would not save work) and falls
    back to it where the reference does: t_12 has no bracket, or the sequential solve misses its own
    convergence test (projected gradient; upstream uses scipy's ``success`` flag) while the joint one
    passes. ``mode`` (static): "sequential" reproduces upstream (t_12 from the 2-leaf problem, then
    (t_1a, log t_aR) on [0, t_12] × [PAIR_LOG_T_MIN, PAIR_LOG_T_MAX], joint fallback); "joint" optimises
    (log t_1a, log t_2a, log t_aR) together (A2).

    Status (A11): ``converged`` says whether the returned times passed their solve's convergence
    test; a pair that did not is returned with its last iterate and the dLogL there (never −inf).
    ``fallback`` marks joint times returned in sequential mode. The M2 engine re-scores every pair
    with ``~converged`` or ``fallback`` on the host with the fp64 reference (``bonsai._ref.pair``),
    so no merge decision rests on an unconverged device solve. Results depend deterministically on
    the batch shape (XLA compiles per shape); the M2 engine scores in a fixed padded batch size.
    """
    if mode not in ("sequential", "joint"):
        raise ValueError(f"mode must be 'sequential' or 'joint', got {mode!r}")
    axes = (0, 0, 0, 0, 0, 0, 0, 0, None, None, None)
    return jax.vmap(_one, in_axes=axes)(x1, v1, t1, x2, v2, t2, xR, vR, xr, Wr, mode)
