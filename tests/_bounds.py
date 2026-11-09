"""Derived rounding-error bounds for comparing JAX kernels with the fp64 references.

Each bound is first order: 8 (an upper bound on the rounded operations feeding one per-gene term)
× eps × the term's absolute size including its conditioning. float64 message errors grow at most
additively per tree level (messages are convex combinations), hence the (1 + height) factors.
"""

import numpy as np

from bonsai._engine.greedy import pair_round_resolution
from bonsai._ref.pair import _scales
from bonsai._ref.star import node_message, star_loglik_grad
from bonsai._ref.tree import postorder

EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)


def heights(parent: np.ndarray) -> np.ndarray:
    """Per node: edges on the longest path down to a leaf (0 at leaves)."""
    order, kids = postorder(parent)
    height = np.zeros(len(parent), dtype=np.int64)
    for v in order:
        if kids[v]:
            height[v] = 1 + max(height[c] for c in kids[v])
    return height


def tree_height(parent: np.ndarray) -> int:
    """Edges on the longest root-to-leaf path."""
    return int(heights(parent).max())


def loglik_scale(parent: np.ndarray, t: np.ndarray, X: np.ndarray, W: np.ndarray) -> float:
    """A = Σ_edges,g (1 + |log w_c| + w_c (x_p − x_c)²) + Σ_parents,g (1 + |log W_p|).

    The absolute size of every per-gene term of ell_n (TH:5093-5105), from the node messages
    (X, W) of the fp64 reference (leaf rows: data and 1/sigma²).
    """
    c = np.flatnonzero(parent >= 0)
    p = parent[c]
    w = 1.0 / (1.0 / W[c] + t[c][:, None])
    parents = np.unique(p)
    return float(np.sum(1 + np.abs(np.log(w)) + w * (X[p] - X[c]) ** 2) + np.sum(1 + np.abs(np.log(W[parents]))))


def rounded(a, dtype=np.float32) -> np.ndarray:
    """``a`` rounded to ``dtype`` and widened back to float64: the inputs a float32 kernel sees."""
    return np.asarray(a, dtype=np.float64).astype(dtype).astype(np.float64)


def tree_tols(parent: np.ndarray, t: np.ndarray, ref, eps: float) -> tuple[float, np.ndarray]:
    """(tol_L, tol_grad per node) for a tree pass that runs entirely at precision ``eps`` (A8).

    ``ref`` is the fp64 ``TreeGrad`` on the same (already rounded) inputs. Messages are sums of
    positive weights (and of w·x) without subtraction, so per level they gain at most C + 3
    roundings (C = most children of a node; var→1/W, +t, 1/·, C−1 additions): relative error of
    w, W below height h is ρ(h) = (h + 1)(C + 3)·eps; message means are convex combinations, so
    their absolute error is ρ(h)·X with X = max|x_leaf|. Per gene term (≤ 8 own roundings, +1 for
    the compensated gene sum):
      ell:  ρ + 9·eps·(1 + |log w|) + w·(4|r|·ρX + r²(ρ + 9·eps)), plus ρ + 9·eps·(1 + |log W|)
            per parent;
      grad: the pre-order "outside" messages add up to H more levels, ρ_g = (2H + 1)(C + 3)·eps:
            (1/T + 2d²/T²)·ρ_g + 2|d|·2ρ_g·X/T² + 9·eps·(1/T + d²/T²) per gene.
    """
    height = heights(parent)
    H = int(height.max())
    kids = postorder(parent)[1]
    C = max(len(k) for k in kids)
    leaves = np.array([not k for k in kids])
    X = np.abs(ref.x[leaves]).max()
    c = np.flatnonzero(parent >= 0)
    p = parent[c]
    rho = ((height[p] + 1) * (C + 3) * eps)[:, None]
    w = 1.0 / (1.0 / ref.W[c] + t[c][:, None])
    r = ref.x[p] - ref.x[c]
    parents = np.unique(p)
    rho_p = ((height[parents] + 1) * (C + 3) * eps)[:, None]
    logW = np.abs(np.log(ref.W[parents]))
    tol_L = np.sum(rho + 9 * eps * (1 + np.abs(np.log(w))) + w * (4 * np.abs(r) * rho * X + r**2 * (rho + 9 * eps)))
    tol_L += np.sum(rho_p + 9 * eps * (1 + logW))
    rho_g = (2 * H + 1) * (C + 3) * eps
    T = t[c][:, None] + 1.0 / ref.W[c] + 1.0 / ref.W_ex[c]
    d = ref.x_ex[c] - ref.x[c]
    tol_g = np.zeros(len(parent))
    tol_g[c] = np.sum(
        (1 / T + 2 * d**2 / T**2) * rho_g + 4 * np.abs(d) * rho_g * X / T**2 + 9 * eps * (1 / T + d**2 / T**2), axis=1
    )
    return float(tol_L), tol_g


def input_rounding(parent: np.ndarray, t: np.ndarray, var: np.ndarray, ref, eps: float) -> float:
    """First-order change of L_tree when every input (x, sigma², t) is rounded to precision ``eps``.

    From the 2-leaf decomposition at a leaf c (maths ref §4): L = const − Σ_g [log T + d²/T],
    T = sigma²_c + t_c + 1/W_{\\c}, d = x_c − x_{\\c}, so |dL/dx_c| = 2|d|/T, |dL/dsigma²_c| =
    |(−1 + d²/T)/T| per gene; dL/dt_e is the gradient. Uses the (raw, uncentred) leaf x of ``ref``.
    """
    leaves = np.flatnonzero([not k for k in postorder(parent)[1]])
    T = t[leaves][:, None] + var[leaves] + 1.0 / ref.W_ex[leaves]
    d = ref.x[leaves] - ref.x_ex[leaves]
    dx = np.sum(2 * np.abs(d) / T * np.abs(ref.x[leaves]))
    dv = np.sum(np.abs((-1 + d**2 / T) / T) * var[leaves])
    return float(eps * (dx + dv + np.sum(np.abs(ref.grad * t))))


def gscale(x: np.ndarray, var: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per child: Σ_g w_c(w_c r_c² + 1 + w_c/W), the absolute size of dell/dt_c's terms (TH:5107)."""
    w = 1.0 / (var + t[:, None])
    xn, W = node_message(x, var, t)
    return np.sum(w * (w * (xn - x) ** 2 + 1.0 + w / W), axis=1)


def ell_scale(x: np.ndarray, var: np.ndarray, t: np.ndarray) -> float:
    """Σ|terms| of a star's ell incl. the cross term of forming r = x̄ − x_c (``_ref.pair._scales``)."""
    return _scales(t, x, var)[0]


def fd_dell_dt(x: np.ndarray, var: np.ndarray, t: np.ndarray, c: int) -> tuple[float, float]:
    """(FD of dell/dt_c from ``star_loglik_grad``'s ell, gscale_c), h = 1e-4·max(t_c, 1e-3).

    Central for t_c > 0 (truncation ~h²·|third derivative|), forward at the bound t_c = 0 (~h·|second
    derivative|); rounding ~eps64·Σ|ell terms|/h.
    """
    h = 1e-4 * max(t[c], 1e-3)
    tp, tm = t.copy(), t.copy()
    tp[c] += h
    f = lambda tt: star_loglik_grad(x, var, tt)[0]
    if t[c] > 0:
        tm[c] -= h
        fd = (f(tp) - f(tm)) / (2 * h)
    else:
        fd = (f(tp) - f(t)) / h
    return fd, float(gscale(x, var, t)[c])


# --- near-tie rule of the fp32 greedy sequence test (plan M2 "Near-tie rule", Task 6a; A21b) -------------


def _score_with_rest(uh, mode: str):
    """``_ref.pair.pair_score`` (fp64) at the engine's inputs û, including its own rest (x_R, v_R).

    ``uh`` = (x1, v1, t1, x2, v2, t2, xr, Wr, xR, vR): the values the device kernel saw, widened to
    float64. The kernel takes R from the host (``ParentAcc.rest``, cast; recorded in the trace), so the
    reference's own ``rest_of_parent`` is replaced by those values: f(û) then sees exactly the kernel's
    inputs. xR None: a host re-scored pair, f(û) is the reference at the host's fp64 inputs, unpatched.
    """
    from unittest import mock

    from bonsai._ref import pair as rp

    x1, v1, t1, x2, v2, t2, xr, Wr, xR, vR = uh
    if xR is None:
        return rp.pair_score(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
    with mock.patch.object(rp, "rest_of_parent", lambda *_: (xR, 1.0 / vR)):
        return rp.pair_score(x1, v1, t1, x2, v2, t2, xr, Wr, mode)


def _rest(uh, eps: float):
    """(x_R, v_R, eps) of û: as recorded for a device row; for a host re-scored row (xR None) the reference's own
    ``rest_of_parent`` at the host's fp64 inputs, with eps64 (the host computes in fp64)."""
    from bonsai._ref import pair as rp

    x1, v1, t1, x2, v2, t2, xr, Wr, xR, vR = uh
    if xR is not None:
        return xR, vR, eps
    xR, WR = rp.rest_of_parent(xr, Wr, x1, 1.0 / (v1 + t1), x2, 1.0 / (v2 + t2))
    return xR, 1.0 / WR, EPS64


def pair_resolution(d_u: float, uh, mode: str, eps: float = EPS32):
    """(r(p), f(û)) of the near-tie rule: r = r_in + r_round (plan M2 "Near-tie rule").

    ``d_u`` = f(u), the reference's dLogL at its own fp64 inputs; ``uh`` as ``_score_with_rest``.
    r_in = |f(û) − f(u)| (exact effect of the engine's inputs); r_round = eps·[Σρ(new) + Σρ(old) +
    Σ(T(new) + T(old)) + 9·ΣT(new)] at û, "new" = the optimal times at û (x̄ = x_a), "old" = the
    weights (w1, w2, W_R) with x̄ = x_r. A failed reference solve at û gives r = inf. A host re-scored pair
    (xR None) stored exactly f(û), computed in fp64: its r_round uses eps64.
    """

    s = _score_with_rest(uh, mode)
    if not np.isfinite(s.dlogl):
        return np.inf, s
    x1, v1, t1, x2, v2, t2, xr, Wr, _, _ = uh
    xR, vR, eps = _rest(uh, eps)
    r_round = pair_round_resolution(x1, v1, t1, x2, v2, t2, xR, vR, xr, Wr, s.t, eps)  # the engine's A24 estimate
    return abs(s.dlogl - d_u) + r_round, s


def _snap_resolution(uh, s, c: int, mode: str, eps: float) -> float:
    """δ_c of rule (e): (64 + G)·eps·gscale/H of t_c's solver coordinate at û, H by central FD (fp64).

    Sequential solve (t_1a + t_2a = t_12): the linear coordinate u_0 = t_1a on [0, t_12] (t_2a = t_12 − u_0,
    same δ). Joint or fallback: s_c = log t_c on [PAIR_LOG_T_MIN, PAIR_LOG_T_MAX], δ = t_c·δs. A11's test
    bounds the projected gradient, so a coordinate the solver pins to a bound has δ = 0 as long as its
    outward gradient exceeds that resolution (it cannot leave the bound under rounding); likewise a
    zero-width box (t_12 = 0 by the 2-leaf zero clamp) whose clamp test f(0) < 0 exceeds its resolution.
    Otherwise, and for H ≤ 0 (unresolved curvature): the formula, or inf.
    """
    from bonsai._ref import pair as rp

    x1, v1, t1, x2, v2, t2, xr, Wr, _, _ = uh
    xR, vR, eps = _rest(uh, eps)
    xs, vs = np.stack([x1, x2, xR]), np.stack([v1, v2, vR])
    G = xs.shape[1]
    t12 = rp.two_leaf_time(x1, v1, x2, v2)[0] if mode == "sequential" else None
    if t12 is not None and abs(s.t[0] + s.t[1] - t12) <= 1e-12 * max(t12, 1e-300):
        if t12 == 0:
            S, d2 = v1 + v2, (x1 - x2) ** 2
            f0 = rp.two_leaf_dlogl(0.0, S, d2)
            return 0.0 if -f0 > (64 + G) * eps * float(np.sum(1 / S + d2 / S**2)) else np.inf
        u, k, lo, hi = np.array([s.t[0], np.log(s.t[2])]), 0, 0.0, t12
        f = lambda uu: rp.seq_objective(uu, t12, xs, vs, True)
        lin = True
    else:
        u, k, lo, hi = np.log(s.t), c, rp.PAIR_LOG_T_MIN, rp.PAIR_LOG_T_MAX
        f = lambda uu: rp.joint_objective(uu, xs, vs, True)
        lin = False
    _, g, _, gs = f(u)
    res = (64 + G) * eps * gs[k]
    if (u[k] <= lo and g[k] > res) or (u[k] >= hi and g[k] < -res):
        return 0.0  # pinned to its bound
    h = 1e-4 * max(abs(u[k]), 1e-8) if lin else 1e-4
    e = np.zeros_like(u)
    e[k] = h
    H = (f(u + e)[1][k] - f(u - e)[1][k]) / (2 * h)
    if not H > 0:
        return np.inf
    return res / H if lin else s.t[c] * res / H


def _engine_inputs(eng_entry, eng_x, eng_v, i: int, j: int, dtype):
    """û of pair (i, j) at one engine trace round, exactly as the engine scored it.

    Device row: its stored rows, its float64 host t, x_r, W_r cast to ``dtype``, and the rest (x_R, v_R) as
    recorded in the trace (as sent). Host re-scored row: the host's fp64 inputs (t, x_r, W_r uncast), xR None.
    """
    scores, tk, xr, Wr, _ = eng_entry
    _, _, host, xR, vR = scores[i, j]
    x1, v1, x2, v2 = (np.asarray(a, np.float64) for a in (eng_x[i], eng_v[i], eng_x[j], eng_v[j]))
    if host:
        return (x1, v1, tk[i], x2, v2, tk[j], xr, Wr, None, None)
    rd = lambda a: np.asarray(a, np.float64).astype(dtype).astype(np.float64)
    w64 = lambda a: np.asarray(a, np.float64)
    return (x1, v1, rd(tk[i]), x2, v2, rd(tk[j]), rd(xr), rd(Wr), w64(xR), w64(vR))


def _same_inputs(a, b) -> bool:
    return all(
        (u is None and v is None) or (u is not None and v is not None and np.array_equal(u, v))
        for u, v in zip(a, b, strict=True)
    )


def merge_resolution_violations(ref_trace, eng_trace, eng_table, mode: str) -> list:
    """Rounds (k, p1, |d_eng − d_ref|, r) where the engine's score of the reference's chosen pair p1 misses
    |dLogL_eng(p1) − d_ref(p1)| ≤ r(p1) (``pair_resolution``); traces cut to rounds 0..d by the caller."""
    from bonsai._ref.greedy import decide

    eps = float(np.finfo(eng_table.dtype).eps)
    out = []
    for k, (ref_e, eng_e) in enumerate(zip(ref_trace, eng_trace, strict=True)):
        p1 = decide((p, v[0]) for p, v in ref_e[0].items())
        if p1 is None:
            continue
        r = pair_resolution(
            ref_e[0][p1][0], _engine_inputs(eng_e, eng_table.x, eng_table.v, *p1, eng_table.dtype), mode, eps
        )[0]
        err = abs(eng_e[0][p1][0] - ref_e[0][p1][0])
        if not err <= r:
            out.append((k, p1, err, r))
    return out


def first_near_tie(ref_trace, eng_trace, ref_tree, eng_table, mode: str) -> int:
    """First round k of the traces (rounds 0..d, cut by the caller) that is a near-tie; len(trace) if none.

    Near-tie (plan M2 "Near-tie rule", A21b: no zero-root branch, so rules (b) and (d) are void), with
    d(p) = the reference's dLogL (its trace), r(p) from ``pair_resolution``, n* = the best pair
    (d desc, (i, j) asc), p1 = the reference's chosen pair (n* when it stops):
    (a) some p ≠ p1 with d(p1) − d(p) ≤ r(p1) + r(p), except exact duplicates of p1 (bitwise equal
        engine inputs: the tie rule, not rounding, decides);
    (c) |d(n*) − 1e-6| ≤ r(n*);
    (e) for a merge, each A18 snap candidate (t_1a if child 1 is internal; t_2a if child 2 is internal
        and child 1's snap does not fire at u): |t_c(û) − 1e-6| ≤ δ_c + |t_c(û) − t_c(u)|.
    Re-optimisation rounds (no scores) are skipped: their decision is the counter's.
    """
    from bonsai._ref.greedy import ACCEPT_DLOGL, ADD_AS_CHILD_T, decide

    dtype = eng_table.dtype
    eps = float(np.finfo(dtype).eps)
    for k, (ref_e, eng_e) in enumerate(zip(ref_trace, eng_trace, strict=True)):
        scores = ref_e[0]
        if not scores:
            continue
        assert set(scores) == set(eng_e[0]), f"round {k}: the two arms score different pairs"
        res, uhat = {}, {}
        for i, j in scores:
            uhat[i, j] = _engine_inputs(eng_e, eng_table.x, eng_table.v, i, j, dtype)
            res[i, j] = pair_resolution(scores[i, j][0], uhat[i, j], mode, eps)
        d = {p: scores[p][0] for p in scores}
        n_star = sorted(scores, key=lambda p: (-d[p] if np.isfinite(d[p]) else np.inf, p))[0]
        p1 = decide(d.items())
        top = n_star if p1 is None else p1
        r1 = res[top][0]
        for p in scores:
            if p == top or _same_inputs(uhat[p], uhat[top]):
                continue
            if d[top] - d[p] <= r1 + res[p][0]:
                return k
        if abs(d[n_star] - ACCEPT_DLOGL) <= res[n_star][0]:
            return k
        if p1 is not None:
            i, j = p1
            t_u, s_hat = scores[p1][1], res[p1][1]
            cand = [0] if not ref_tree.is_cell[i] else []
            if not ref_tree.is_cell[j] and not (cand and t_u[0] < ADD_AS_CHILD_T):
                cand.append(1)
            for c in cand:
                delta = _snap_resolution(uhat[p1], s_hat, c, mode, eps)
                if abs(s_hat.t[c] - ADD_AS_CHILD_T) <= delta + abs(s_hat.t[c] - t_u[c]):
                    return k
    return len(ref_trace)
