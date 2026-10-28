"""Engine greedy core over one parent, exhaustive rounds (``bonsai._engine.greedy``; plan M2 Task 6a, A8, A11, A18,
A20, A21b). The fp64 brute-force reference ``bonsai._ref.greedy`` is the oracle."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from bonsai._engine import greedy as eg
from bonsai._engine.greedy import GreedyConfig, NodeTable, _set_rows, greedy_star, select_candidate
from bonsai._ref import greedy as rg
from bonsai._ref import pair as rp
from bonsai._ref.pair import _scales
from bonsai._ref.star import node_message, star_em

from ._bounds import EPS64, first_near_tie, merge_resolution_violations
from ._trees import bm_tree

G = 60
N = 24
EXACT = GreedyConfig(knn=False, lazy=False, star_tol=1e-10)
FP32 = GreedyConfig(knn=False, lazy=False, star_tol=1e-4)  # upstream's star tolerance
# First near-tie round k* per fixture: measured by the supervisor 2026-10-22 (Task 9 Step 2a), regression pin.
# Remeasure with (prints "fixture: first near-tie k*, first divergence d, rounds n"):
#   JAX_PLATFORMS=cpu uv run --locked pytest -q -s tests/test_engine_greedy.py -k "up_to_first_near_tie or 48_leaves"
MIN_K_STAR: dict[str, int] = {"seed11": 22, "seed12": 22, "dup": 22, "nonroot": 14, "bm48": 49}
FIXTURES = ["seed11", "seed12", "dup", "nonroot"]


def _fixture(name: str):
    """(x, var, outside) of a family with centred fp64 leaf children; outside None at the root.

    seed11/seed12: bm_tree(24, 60) leaves at the root. dup: seed 13 with leaves 16..23 copies of leaves
    0, 0, 1, 1, 2, 2, 3, 3 (triplicates: exact top ties for the tie rule). nonroot: seed 14, leaves 0..15
    as the children of a non-root node, outside = the star of leaves 16..23 seen through an edge of 0.5.
    bm48: bm_tree(48, 60) seed 21 at the root.
    """
    n, seed = (48, 21) if name == "bm48" else (N, {"seed11": 11, "seed12": 12, "dup": 13, "nonroot": 14}[name])
    _, _, x, var = bm_tree(n, G, np.random.default_rng(seed))
    x, var = x[:n].copy(), var[:n].copy()
    if name == "dup":
        src = [0, 0, 1, 1, 2, 2, 3, 3]
        x[16:], var[16:] = x[src], var[src]
    if name != "nonroot":
        return x - x.mean(axis=0), var, None
    xc, vc = x[:16], var[:16]
    centre = xc.mean(axis=0)
    xo, vo = x[16:] - centre, var[16:]
    x_out, W_out = node_message(xo, vo, star_em(xo, vo)[0])
    return xc - centre, vc, (x_out, 1.0 / (0.5 + 1.0 / W_out))


def _ref_tree(x, var, t) -> rg.Tree:
    n = len(x)
    return rg.Tree(
        np.r_[np.full(n, n), -1],
        np.r_[t, 0.0],
        np.vstack([x, np.zeros(G)]),
        np.vstack([var, np.ones(G)]),
        np.r_[np.ones(n, bool), False],
    )


def _run(name: str, cfg: GreedyConfig, dtype):
    """Engine and reference on one fixture from the same fp64 star times: (stats, eng trace, table, ref events,
    ref trace, ref tree)."""
    x, var, outside = _fixture(name)
    n = len(x)
    t0 = star_em(x, var, tol=cfg.star_tol)[0]
    table = NodeTable(x, var, dtype=dtype)
    table.t[:n] = t0
    eng_trace: list = []
    stats = greedy_star(table, n, outside, outside is None, cfg, trace=eng_trace)
    ref_trace: list = []
    ref_tree, ref_events = rg.greedy_star(_ref_tree(x, var, t0), n, outside, cfg.mode, cfg.star_tol, trace=ref_trace)
    return stats, eng_trace, table, ref_events, ref_trace, ref_tree


_run_cached = functools.cache(_run)


def _key(e):
    return (e.kind, e.i, e.j, e.new)


def test_constants_match_reference():
    names = [
        "ACCEPT_DLOGL",
        "ADD_AS_CHILD_T",
        "REDO_TIME_FRAC",
        "REDO_TIME_MIN",
        "REDO_NN_FRAC",
        "REDO_NN_MIN",
        "NN_MIN_CHILDREN",
        "LAZY_MIN_CHILDREN",
        "STAR_TOL",
        "STAR_MAX_SWEEPS",
        "STAR_T_MAX",
        "ZERO_EDGE_PASSES",
        "K_NN",
    ]
    assert {k: getattr(eg, k) for k in names} == {k: getattr(rg, k) for k in names}
    assert not hasattr(eg, "ZERO_ROOT_T") and not hasattr(eg, "DLOGL_FLOOR")  # A21b


def test_select_candidate_rules():
    ij = np.array([[0, 1], [0, 2], [1, 2]])

    def sel(d, ub, fr, M=2):
        k, idx = select_candidate(np.array(d, float), np.array(ub, float), np.array(fr), ij, M)
        return k, list(idx)

    assert sel([5, 4, 1], [5, 4.5, 1.2], [True, False, False]) == ("accept", [0])  # 5 >= 4.5
    assert sel([5, 4, 1], [5, 5.5, 6], [True, False, False]) == ("rescore", [2, 1])  # by ub desc
    assert sel([5, 4, 1], [5, 5.0, 1], [True, False, False])[0] == "accept"  # equality accepts (A20)
    assert sel([5, 5, 1], [5, 5, 1], [True, True, True]) == ("accept", [0])  # tie -> lower ij
    assert sel([1, 5, 5], [1, 5, 5], [True, True, True]) == ("accept", [1])
    assert sel([0, 0, 0], [np.inf] * 3, [False] * 3) == ("rescore", [0, 1])  # all stale (+inf): ij asc, M rows
    assert sel([0, 0, 0], [np.inf, 1, np.inf], [False] * 3, M=1) == ("rescore", [0])
    assert sel([-np.inf, 2, 0], [0, 3, 0], [True, False, False]) == ("rescore", [1, 2])  # -inf never eligible
    assert sel([-np.inf, 0, 0], [0, 0, 0], [True, True, True]) == ("accept", [1])
    assert sel([-np.inf] * 3, [0] * 3, [True] * 3) == ("none", [])
    assert sel([5, 4, 1], [5, np.nan, 1], [True, False, False]) == ("rescore", [1, 2])  # NaN bound = +inf


def _pair_scale(ref_tree, entry, i, j, t3) -> float:
    """Σ|terms| of a pair's new and old ell at the reference's inputs (``_scales`` and the old term)."""
    _, tk, xr, Wr, _ = entry
    x, v = ref_tree.x, ref_tree.var
    w1, w2 = 1.0 / (v[i] + tk[i]), 1.0 / (v[j] + tk[j])
    xR, WR = rp.rest_of_parent(xr, Wr, x[i], w1, x[j], w2)
    xs, vs, w_old = np.stack([x[i], x[j], xR]), np.stack([v[i], v[j], 1.0 / WR]), np.stack([w1, w2, WR])
    old = np.sum(np.abs(np.log(w_old)) + w_old * (xr - xs) ** 2) + np.sum(np.abs(np.log(Wr)))
    return _scales(np.asarray(t3), xs, vs)[0] + float(old)


@pytest.mark.x64
@pytest.mark.parametrize("name", FIXTURES)
def test_engine_matches_reference_x64(name):
    stats, _, _, ref_events, ref_trace, ref_tree = _run(name, EXACT, jnp.float64)
    assert [_key(e) for e in stats.events] == [_key(e) for e in ref_events]
    assert any(e.kind == "reopt" for e in ref_events) == (name != "nonroot")  # the periodic reopt is exercised
    # dLogL: both solvers stop at their fp64 KKT resolution, (G + 64)·eps64·Σ|terms| per pair
    for k, (a, b) in enumerate(zip(stats.events, ref_events, strict=True)):
        if b.kind == "merge":
            scale = _pair_scale(ref_tree, ref_trace[k], b.i, b.j, ref_trace[k][0][b.i, b.j][1])
            assert abs(a.dlogl - b.dlogl) <= (G + 64) * EPS64 * scale, (k, a.dlogl, b.dlogl)
    assert stats.n_flagged_used_unrescored == 0


def _first_divergence(ref_events, eng_events) -> int | None:
    for k, (a, b) in enumerate(zip(ref_events, eng_events, strict=False)):
        if _key(a) != _key(b):
            return k
    return None if len(ref_events) == len(eng_events) else min(len(ref_events), len(eng_events))


def _assert_reopt_schedule(trace, events, n_children: int, is_root: bool) -> None:
    """A21: the counter is the number of merges since the last re-optimisation (or the start), replayed from the
    events; a root round re-optimises iff it is > max(⌊0.05·C⌋, 10), C = the children at the last one (or start)."""
    since, redo_at = 0, max(int(0.05 * n_children), 10)
    for entry, ev in zip(trace, events, strict=False):  # the stop round has no event
        assert entry[4] == since
        assert (ev.kind == "reopt") == (is_root and since > redo_at)
        since, redo_at = (0, max(int(0.05 * len(entry[1])), 10)) if ev.kind == "reopt" else (since + 1, redo_at)


def _assert_parent_state(trace, table, outside) -> None:
    """The engine's fp64 AIR (x_r, W_r) at every round equals a from-scratch sum over its children (host fp64
    weights from the stored rows and the round's t, plus the outside message), as W and W·x_r.

    Bound: by round k the accumulator has had ≤ C0 + 1 + 3k rounded updates (build, outside, two removals and one
    addition per round) and the recomputation ≤ C + 1 more, each ≤ eps64 times a partial sum of |terms|; removals
    only shrink partial sums, so every one is ≤ S = the largest round total of |terms| so far. x_r·W_r adds 2
    roundings (division in ``message``, product here) of size ≤ S.
    """
    C0 = len(trace[0][1])
    S_W = S_X = 0.0
    for k, (_, tk, xr, Wr, _) in enumerate(trace):
        ids = list(tk)
        w = 1.0 / (table.v[ids].astype(np.float64) + np.array([tk[c] for c in ids])[:, None])
        wx = w * table.x[ids].astype(np.float64)
        if outside is not None:
            w, wx = np.vstack([w, outside[1]]), np.vstack([wx, outside[1] * outside[0]])
        W, WX = w.sum(axis=0), wx.sum(axis=0)
        S_W, S_X = np.maximum(S_W, W), np.maximum(S_X, np.abs(wx).sum(axis=0))
        n_ops = C0 + 1 + 3 * k + len(ids) + 1
        assert np.all(np.abs(Wr - W) <= n_ops * EPS64 * S_W), k
        assert np.all(np.abs(xr * Wr - WX) <= (n_ops + 2) * EPS64 * S_X), k


def _check_fp32(name: str):
    stats, eng_trace, table, ref_events, ref_trace, ref_tree = _run_cached(name, FP32, jnp.float32)
    assert stats.n_flagged_used_unrescored == 0
    # exhaustive rounds (bounds off): every pair of the current children is freshly scored before each decision
    sizes = [(len(e[0]), len(e[1])) for e in eng_trace if e[0]]
    assert all(m == c * (c - 1) // 2 for m, c in sizes) and stats.n_pair_scores == sum(m for m, _ in sizes)
    _assert_reopt_schedule(eng_trace, stats.events, len(eng_trace[0][1]), name != "nonroot")
    _assert_parent_state(eng_trace, table, _fixture(name)[2])
    d = _first_divergence(ref_events, stats.events)
    last = len(ref_trace) - 1 if d is None else d
    # the engine's score of each chosen pair is within its derived resolution of the reference's
    assert merge_resolution_violations(ref_trace[: last + 1], eng_trace[: last + 1], table, FP32.mode) == []
    k_star = first_near_tie(ref_trace[: last + 1], eng_trace[: last + 1], ref_tree, table, FP32.mode)
    print(f"{name}: first near-tie {k_star}, first divergence {d}, rounds {len(ref_trace)}")
    assert d is None or k_star <= d  # a divergence needs a near-tie at or before it
    if name in MIN_K_STAR:
        assert k_star >= MIN_K_STAR[name]


@pytest.mark.parametrize("name", FIXTURES)
def test_fp32_matches_reference_up_to_first_near_tie(name):
    _check_fp32(name)


@pytest.mark.slow
def test_fp32_matches_reference_48_leaves():
    _check_fp32("bm48")


def test_rounds_are_exhaustive_with_small_batches():
    """A20 with bounds off: every candidate is stale (UB = +inf) each round, so with M = 8 < C(C−1)/2 every round
    rescores all pairs in ⌈C(C−1)/2 / 8⌉ batches before deciding (the 24-leaf fixtures fit one M = 256 batch)."""
    x, var, _ = _fixture("seed12")
    x, var = x[:12], var[:12]
    table = NodeTable(x, var)
    table.t[:12] = star_em(x, var)[0]
    trace: list = []
    stats = greedy_star(table, 12, None, True, GreedyConfig(knn=False, lazy=False, rescore_top=8), trace=trace)
    sizes = [(len(e[0]), len(e[1])) for e in trace]
    assert len(stats.events) >= 5 and all(m == c * (c - 1) // 2 for m, c in sizes)
    assert stats.n_rescore_batches == sum(-(-m // 8) for m, _ in sizes)


@pytest.mark.parametrize("kind", ["unconverged", "fallback"])
def test_flagged_pairs_rescored_on_host(monkeypatch, kind):
    """A11: a flagged device row (``~converged | fallback``: converged False, or a converged sequential→joint
    fallback; dLogL 1e9) is stored with the fp64 reference's score; a reference −inf is never chosen and the run
    completes."""
    x, var, _ = _fixture("seed11")
    x, var = x[:12], var[:12]
    t0 = star_em(x, var)[0]
    cfg = FP32

    def run():
        table = NodeTable(x, var)
        table.t[:12] = t0
        trace: list = []
        return greedy_star(table, 12, None, True, cfg, trace=trace), trace, table

    base, base_trace, table = run()
    assert base.n_host_rescored == 0  # precondition: nothing flagged naturally, so the counts below are ours
    i, j = rg.decide((p, s[0]) for p, s in base_trace[0][0].items())  # round 0's best pair
    real = eg.pair_score
    calls = {"n": 0, "once": True}

    def flag(*args, **kw):  # flags the real rows of pair (i, j): the first one only, or all ("once" False)
        s = real(*args, **kw)
        hit = np.flatnonzero(
            np.all(np.asarray(args[0]) == table.x[i], axis=1) & np.all(np.asarray(args[3]) == table.x[j], axis=1)
        )
        if (calls["once"] and calls["n"]) or not len(hit):
            return s
        calls["n"] += 1
        r = hit[0]  # padding repeats row 0, so later hits of the same pair are padded copies
        if kind == "fallback":
            return s._replace(
                dlogl=s.dlogl.at[r].set(1e9), converged=s.converged.at[r].set(True), fallback=s.fallback.at[r].set(True)
            )
        return s._replace(dlogl=s.dlogl.at[r].set(1e9), converged=s.converged.at[r].set(False))

    monkeypatch.setattr(eg, "pair_score", flag)
    stats, trace, table2 = run()
    assert calls["n"] == 1 and stats.n_host_rescored == 1 and stats.n_flagged_used_unrescored == 0
    _, tk, xr, Wr, _ = trace[0]
    x64 = table2.x[: 12 + 1].astype(np.float64)
    v64 = table2.v[: 12 + 1].astype(np.float64)
    ref = rp.pair_score(x64[i], v64[i], tk[i], x64[j], v64[j], tk[j], xr, Wr, cfg.mode)
    assert trace[0][0][i, j][0] == ref.dlogl  # the stored score is the fp64 reference's, bitwise
    np.testing.assert_array_equal(trace[0][0][i, j][1], ref.t)
    assert trace[0][0][i, j][2] and not any(v[2] for p, v in trace[0][0].items() if p != (i, j))  # host flag
    assert [_key(e) for e in stats.events] == [_key(e) for e in base.events]

    # second case: the reference fails (-inf) on the flagged pair -> never chosen, the run still completes
    calls.update(n=0, once=False)
    real_ref = rp.pair_score

    def failing(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        s = real_ref(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
        return s._replace(dlogl=-np.inf, converged=False)

    monkeypatch.setattr(eg.ref_pair, "pair_score", failing)
    stats, trace, table3 = run()
    alive = [entry[0][i, j][0] for entry in trace if (i, j) in entry[0]]
    assert stats.n_host_rescored >= len(alive) >= 1 and all(d == -np.inf for d in alive)
    assert (i, j) not in [(e.i, e.j) for e in stats.events]
    # completed: down to 3 root children, or a final stop round with no pair > 1e-6
    assert len(table3.children(12)) == 3 or max(v[0] for v in trace[-1][0].values()) <= eg.ACCEPT_DLOGL


def test_duplicate_cells_are_deterministic():
    """Review focus 1: 8 duplicated leaves among 24 (exact ties): two runs give identical events, no NaN stored."""
    first = _run_cached("dup", FP32, jnp.float32)
    x, var, _ = _fixture("dup")
    table = NodeTable(x, var)
    table.t[:N] = star_em(x, var, tol=FP32.star_tol)[0]
    trace: list = []
    stats = greedy_star(table, N, None, True, FP32, trace=trace)
    assert [e[:5] for e in stats.events] == [e[:5] for e in first[0].events]
    assert all(not np.isnan(s[0]) for entry in trace for s in entry[0].values())
    ties = [e for e in trace[0][0] if trace[0][0][e][0] == trace[0][0][0, 16][0]]
    assert {(0, 16), (0, 17), (16, 17)} <= set(ties)  # the triplicate's pairs tie exactly (non-vacuity)
    assert any(_key(e)[1:3] == (0, 16) for e in stats.events)  # and the tie rule's pair is the one merged


def test_device_row_writes_donate():
    buf = jnp.zeros((8, 4), jnp.float32)
    out = _set_rows(buf, jnp.asarray([3]), jnp.ones((1, 4), jnp.float32))
    assert buf.is_deleted()  # donated: the write reused the buffer instead of copying it
    expected = np.zeros((8, 4), np.float32)
    expected[3] = 1.0
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_near_tie_rule_on_host_rescored_rows():
    """``first_near_tie`` / rule (e) on a round whose pairs were all host re-scored (xR None in û), with an
    internal child 1 in the chosen pair: û is the host's fp64 input, so f(û) = f(u), r_in = 0 and the snap
    resolution uses the reference's own rest at eps64 (equal to passing that rest explicitly)."""
    from types import SimpleNamespace

    from ._bounds import _snap_resolution

    x, var, _ = _fixture("seed11")
    x, var = x[:5], var[:5]
    t = star_em(x, var)[0]
    kids = list(range(5))
    xr, Wr = node_message(x, var, t)
    scores = {
        (i, j): rp.pair_score(x[i], var[i], t[i], x[j], var[j], t[j], xr, Wr) for i in kids for j in kids if i < j
    }
    tk = {c: float(t[c]) for c in kids}
    p1 = rg.decide((p, s.dlogl) for p, s in scores.items())
    ref_tree = SimpleNamespace(is_cell=np.array([p1[0] != c for c in range(5)]))  # child 1 of p1 internal
    ref_trace = [({p: (s.dlogl, s.t) for p, s in scores.items()}, tk, xr, Wr, 0)]
    eng_trace = [({p: (s.dlogl, s.t, True, None, None) for p, s in scores.items()}, tk, xr, Wr, 0)]
    table = SimpleNamespace(x=x, v=var, dtype=np.dtype(np.float32))
    assert first_near_tie(ref_trace, eng_trace, ref_tree, table, "sequential") in (0, 1)  # runs, no crash
    i, j = p1
    w1, w2 = 1.0 / (var[i] + t[i]), 1.0 / (var[j] + t[j])
    xR, WR = rp.rest_of_parent(xr, Wr, x[i], w1, x[j], w2)
    host = (x[i], var[i], t[i], x[j], var[j], t[j], xr, Wr, None, None)
    explicit = (*host[:8], xR, 1.0 / WR)
    s = scores[p1]
    assert _snap_resolution(host, s, 0, "sequential", 1e-7) == _snap_resolution(explicit, s, 0, "sequential", EPS64)


def test_node_table_grows_when_ids_exceed_capacity():
    """Ids are never reused, so redo starry can run past 2N+1: a table started at N+2 rows grows (bucketed) and the
    run is identical to one with ample capacity: same ids, events, parents, times and device mirror."""
    x, var, _ = _fixture("seed12")
    x, var = x[:12], var[:12]
    cfg = GreedyConfig(knn=False, lazy=False)
    tables, events = [], []
    for cap in (None, 14):
        table = NodeTable(x, var, cap=cap)
        table.t[:12] = star_em(x, var)[0]
        events.append([_key(e) for e in greedy_star(table, 12, None, True, cfg).events])
        tables.append(table)
    full, small = tables
    assert len(full.parent) == 32 and len(small.parent) == 32 and small.n == 22  # grown 14 -> 16 -> 32
    assert events[0] == events[1] and len(events[0]) == 9
    np.testing.assert_array_equal(small.parent, full.parent)
    np.testing.assert_array_equal(small.t, full.t)
    np.testing.assert_array_equal(small.x, full.x)
    np.testing.assert_array_equal(small.v, full.v)
    np.testing.assert_array_equal(small.is_cell, full.is_cell)
    np.testing.assert_array_equal(np.asarray(small.dev_x), small.x)
    np.testing.assert_array_equal(np.asarray(small.dev_v), small.v)


@pytest.mark.slow
def test_node_table_growth_with_knn_candidate_keys():
    """52 leaves with kNN on (C > 50): ``set_candidates(all_pairs())`` runs at C = 50, where the next id (54) equals
    the starting capacity N+2. Pair keys i*cap+j must use the live capacity, or pairs collide after the growth."""
    _, _, x, var = bm_tree(52, G, np.random.default_rng(21))
    x, var = x[:52].copy(), var[:52].copy()
    x = x - x.mean(axis=0)
    cfg = GreedyConfig()
    out = []
    for cap in (None, 54):
        table = NodeTable(x, var, cap=cap)
        table.t[:52] = star_em(x, var, tol=cfg.star_tol)[0]
        out.append((table, [_key(e) for e in greedy_star(table, 52, None, True, cfg).events]))
    (full, ev_full), (small, ev_small) = out
    assert len(small.parent) > 54 and small.n > 54  # grew
    assert ev_small == ev_full and len(ev_full) > 0
    np.testing.assert_array_equal(small.parent[: small.n], full.parent[: full.n])
