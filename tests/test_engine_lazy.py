"""Lazy bounds, kNN candidates and merges_per_round of the greedy engine (``bonsai._engine.greedy``; plan M2 Task 6b,
A3, A19, A20). The fp64 brute-force reference ``bonsai._ref.greedy`` is the oracle for the merge sequence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bench.run import count_compilations

from bonsai._engine import greedy as eg
from bonsai._engine import knn as knn_mod
from bonsai._engine.array_tree import _bucket
from bonsai._engine.greedy import GreedyConfig, NodeTable, greedy_star, select_candidate
from bonsai._ref import greedy as rg
from bonsai._ref.star import star_em

from ._bounds import EPS64
from ._trees import bm_tree
from .test_engine_greedy import _fixture, _ref_tree

G = 60
LAZY = GreedyConfig(knn=False)  # bounds on (C > 10), all pairs, m = 1


def _leaves(n: int, seed: int):
    _, _, x, var = bm_tree(n, G, np.random.default_rng(seed))
    x, var = x[:n], var[:n]
    return x - x.mean(axis=0), var


def _run(x, var, cfg, mp=None, outside=None, dtype=jnp.float32):
    """Engine from fp64 star times. With a ``MonkeyPatch`` ``mp``, every ``select_candidate`` call of the root's own
    rounds is logged with a snapshot of the candidate store and the current children, and every kNN build with its
    valid-row count. The returned trace holds the root's own rounds (counter ≥ 0): special-child greedies (A18b) have
    their own store, run exhaustively (bounds and kNN off) and count as part of the merge that triggered them."""
    n = len(x)
    table = NodeTable(x, var, dtype=dtype)
    table.t[:n] = star_em(x, var, tol=cfg.star_tol)[0]
    trace: list = []
    log: dict = {"stores": [], "calls": [], "knn": []}
    if mp is not None:

        class Store(eg._Store):
            def __init__(self):
                super().__init__()
                log["stores"].append(self)

        real_sel, real_knn = eg.select_candidate, eg.knn

        def sel(d, ub, fresh, ij, M):
            st = log["stores"][0]  # the root's store; certify passes its d array itself
            if d is st.d:
                snap = {k: getattr(st, k).copy() for k in ("a", "b", "epoch", "Dx", "DW")}
                snap |= {"d": d.copy(), "ub": ub.copy(), "fresh": fresh.copy(), "ij": ij.copy()}
                log["calls"].append(snap | {"kids": table.children(n), "round": sum(e[4] >= 0 for e in trace)})
            return real_sel(d, ub, fresh, ij, M)

        def knn(points, k, valid):
            log["knn"].append(int(np.sum(valid)))
            return real_knn(points, k, valid)

        mp.setattr(eg, "_Store", Store)
        mp.setattr(eg, "select_candidate", sel)
        mp.setattr(eg, "knn", knn)
    stats = greedy_star(table, n, outside, outside is None, cfg, trace=trace)
    return stats, [e for e in trace if e[4] >= 0], table, log


def _key(e):
    return (e.kind, e.i, e.j, e.new, e.depth)


@pytest.fixture(scope="module")
def lazy24():
    """bm_tree(24, 60) seed 11 at the root, bounds on, all pairs, m = 1, instrumented."""
    with pytest.MonkeyPatch.context() as mp:
        return _run(*_leaves(24, 11), LAZY, mp)


@pytest.fixture(scope="module")
def knn120():
    """bm_tree(120, 60) seed 31 at the root, default config (kNN while C > 50, bounds, m = 1), instrumented."""
    with pytest.MonkeyPatch.context() as mp:
        return _run(*_leaves(120, 31), GreedyConfig(), mp)


def test_select_candidate_rescore_order_is_exact():
    """The O(n) partition (ub, then the combined (i, j) key among ties at the M-th bound) returns exactly the first M
    stale candidates of the full (ub desc, ij asc) sort, also when every stale bound is +inf (all tied)."""
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(1, 60))
        ij = np.sort(rng.choice(40, size=(n, 2), replace=True), axis=1)
        ij[:, 1] += ij[:, 0] == ij[:, 1]
        ij = np.unique(ij, axis=0)
        rng.shuffle(ij)
        n = len(ij)
        ub = rng.choice([np.inf, 1.0, 2.0, 3.0], size=n) if trial % 2 else np.full(n, np.inf)
        fresh = rng.random(n) < 0.2
        d = np.where(fresh, 10.0, 0.0)  # fresh d above every finite bound but below +inf: never accepts with inf
        ub[~np.isinf(ub)] += 20.0  # finite stale bounds above d too, so the call always rescores
        M = int(rng.integers(1, 12))
        kind, idx = select_candidate(d, ub, fresh, ij, M)
        stale = np.flatnonzero(~fresh)
        if not len(stale):
            continue
        full = stale[np.lexsort((ij[stale, 1], ij[stale, 0], -ub[stale]))][:M]
        assert kind == "rescore" and list(idx) == list(full), trial


def _hand_movement(trace):
    """(Dx, DW) after e moves, e = 0..len(trace) − 1, recomputed from the trace's fp64 AIRs (one move per round
    between consecutive entries at m = 1 without kNN): ‖(x₁ − x₀)√W₀‖₂ and ‖(W₁ − W₀)/W₀‖₂ (A20)."""
    Dx, DW = [0.0], [0.0]
    for (_, _, x0, W0, _), (_, _, x1, W1, _) in zip(trace, trace[1:], strict=False):
        Dx.append(Dx[-1] + float(np.sqrt(np.sum((x1 - x0) ** 2 * W0))))
        DW.append(DW[-1] + float(np.sqrt(np.sum(((W1 - W0) / W0) ** 2))))
    return np.array(Dx), np.array(DW)


def _check_bounds(stats, trace, log) -> int:
    """A3/A20 at every selection of an m = 1 run (see ``test_bounds_follow_parent_movement``); returns the number of
    stale finite bounds checked.

    Tolerance: each hand and engine movement term is a G-term sum of squares, a square root and products (relative
    error ≤ (G + 3)·eps64 each), summed over ≤ R moves (≤ R·eps64 more), so |D_hand − D_eng| ≤ (2G + 6 + 2R)·eps64·D;
    the bound itself is 3 roundings of terms of size ≤ |d| + |a|·D + |b|·D.
    """
    # precondition of the epoch mapping: every round but the last moves the parent once (a merge or a reopt)
    assert all(not e[0] or len(trace[k + 1][1]) == len(e[1]) - 1 for k, e in enumerate(trace[:-1]))
    Dx, DW = _hand_movement(trace)
    R = len(trace)
    rel = (2 * G + 6 + 2 * R) * EPS64
    reopt_rounds = {k for k, e in enumerate(trace) if not e[0]}
    n_bounded = 0
    seen_rounds: set = set()
    for c in log["calls"]:
        cur = c["round"]  # moves so far = rounds completed before this one
        first_of_round = cur not in seen_rounds
        seen_rounds.add(cur)
        if first_of_round and cur - 1 in reopt_rounds:  # A20: every child time changed; only new pairs are fresh
            assert np.all(c["fresh"] | np.isinf(c["ub"]))
        scored = c["epoch"] >= 0
        assert np.array_equal(c["fresh"], c["epoch"] == cur)
        e = c["epoch"][scored]
        # the bound was refreshed when the pair was (re)scored: its stored totals are those of its epoch
        np.testing.assert_allclose(c["Dx"][scored], Dx[e], rtol=rel, atol=0)
        np.testing.assert_allclose(c["DW"][scored], DW[e], rtol=rel, atol=0)
        st = scored & ~c["fresh"]
        if len(trace[cur][1]) <= eg.LAZY_MIN_CHILDREN:
            continue
        d, a, b, e = c["d"][st], c["a"][st], c["b"][st], c["epoch"][st]
        ub = d + a * (Dx[cur] - Dx[e]) + b * (DW[cur] - DW[e])
        tol = np.abs(a) * rel * (Dx[cur] + Dx[e]) + np.abs(b) * rel * (DW[cur] + DW[e])
        tol += 3 * EPS64 * (np.abs(d) + np.abs(a) * Dx[cur] + np.abs(b) * DW[cur])
        assert np.all(np.abs(c["ub"][st] - ub) <= tol)
        n_bounded += int(st.sum())
    # the bounds saved work: fewer device rows than the exhaustive Σ C(C−1)/2 over the merge rounds
    assert stats.n_pair_scores < sum(len(e[1]) * (len(e[1]) - 1) // 2 for e in trace if e[0])
    return n_bounded


def test_bounds_follow_parent_movement(lazy24):
    """A3/A20: at every selection, a stored score is fresh iff scored at the current epoch; a stale one carries
    UB = d + a·(Dx − Dx_e) + b·(DW − DW_e), with Dx_e, DW_e the totals when it was last (re)scored, the totals
    recomputed by hand from the parent's AIR; after the re-optimisation every non-fresh bound is +inf."""
    stats, trace, _, log = lazy24
    reopt = [e for e in trace if not e[0]]
    assert len(reopt) == 1 and len(reopt[0][1]) > eg.LAZY_MIN_CHILDREN  # a reopt while bounds are on
    assert _check_bounds(stats, trace, log) > 0


def test_bounds_follow_parent_movement_knn(knn120):
    """The same bookkeeping in a kNN run (candidate rebuilds, new-node queries, reopts while C > 50)."""
    assert _check_bounds(*(knn120[i] for i in (0, 1, 3))) > 0


def test_lazy_off_below_eleven_children(lazy24):
    """A20: at C ≤ 10 bounds are off, so every round freshly scores all C(C−1)/2 pairs before deciding; above, the
    bounds let some rounds decide with fewer fresh scores."""
    _, trace, _, _ = lazy24
    sizes = [(len(e[0]), len(e[1])) for e in trace if e[0]]
    assert all(m == c * (c - 1) // 2 for m, c in sizes if c <= eg.LAZY_MIN_CHILDREN)
    assert any(c <= eg.LAZY_MIN_CHILDREN for _, c in sizes)
    assert any(m < c * (c - 1) // 2 for m, c in sizes if c > eg.LAZY_MIN_CHILDREN)


@pytest.mark.x64
def test_lazy_matches_exhaustive_reference_x64():
    """With bounds on, kNN off and m = 1 the fp64 engine reproduces the reference's exhaustive merge sequence
    (bm_tree(40, 60) seed 31: C = 40 > 10, several rescore batches of M = 256 and one periodic re-optimisation inside
    the bounded phase), while scoring fewer pairs than the exhaustive rounds. A regression pin on this fixture, not a
    guarantee: the A3 bound is heuristic, so lazy selection can differ from the exhaustive one elsewhere (it replaces
    the plan's forced-stale test, ruling 2026-10-22)."""
    x, var = _leaves(40, 31)
    cfg = GreedyConfig(knn=False, star_tol=1e-10)
    stats, trace, _, _ = _run(x, var, cfg, dtype=jnp.float64)
    t0 = star_em(x, var, tol=cfg.star_tol)[0]
    _, ref_events = rg.greedy_star(_ref_tree(x, var, t0), 40, None, cfg.mode, cfg.star_tol)
    assert [_key(e) for e in stats.events] == [_key(e) for e in ref_events]
    assert any(e.kind == "reopt" for e in ref_events) and stats.n_flagged_used_unrescored == 0
    assert stats.n_rescore_batches > 0
    assert stats.n_pair_scores < sum(len(e[1]) * (len(e[1]) - 1) // 2 for e in trace if e[0])


def test_knn_rebuild_schedule(knn120):
    """A19, replayed from the trace: a kNN build at the start (C > 50), after each re-optimisation, when merges since
    the last build > max(⌊0.1·C⌋, 10) (C at that build), and once before stopping; none at C ≤ 50."""
    stats, trace, table, log = knn120
    C_end = len(table.children(120))
    C0 = len(trace[0][1])
    expected, since, at, active = [C0], 0, max(int(0.1 * C0), 10), C0 > eg.NN_MIN_CHILDREN

    def build(C):
        nonlocal since, at
        expected.append(C)
        since, at = 0, max(int(0.1 * C), 10)

    for k, entry in enumerate(trace):
        C = len(entry[1])
        C_next = len(trace[k + 1][1]) if k + 1 < len(trace) else C_end
        if not entry[0]:  # periodic re-optimisation round
            if active:
                build(C)
        elif C_next == C - 1:  # merge round
            since += 1
            if active and C_next <= eg.NN_MIN_CHILDREN:
                active = False
            elif active and since > at:
                build(C_next)
        elif active and since > 0:  # a stop round: one rebuild, then the next round
            build(C)
    assert log["knn"] == expected and stats.n_rebuilds == len(expected)
    n_reopt_builds = sum(1 for e in trace if not e[0] and len(e[1]) > eg.NN_MIN_CHILDREN)
    assert n_reopt_builds >= 1 and len(expected) - 1 - n_reopt_builds >= 1  # both triggers exercised


def test_switch_to_all_pairs_at_50(knn120):
    """A19: while C > 50 the candidates are a proper subset of all pairs (kNN); at C ≤ 50 the candidate set is exactly all pairs of the current children."""
    _, _, _, log = knn120
    n_knn = n_all = 0
    for c in log["calls"]:
        kids = c["kids"]
        have = {tuple(p) for p in c["ij"].tolist()}
        allp = {(a, b) for k, a in enumerate(kids) for b in kids[k + 1 :]}
        assert have <= allp
        if len(kids) <= eg.NN_MIN_CHILDREN:
            assert have == allp
            n_all += 1
        else:
            assert len(have) < len(allp)
            n_knn += 1
    assert n_knn and n_all


def test_rebuild_before_stopping(monkeypatch):
    """A19: a kNN run whose current candidates have none above 1e-6 rebuilds once, then stops.

    60 children: 2 pairs of identical cells (var 0.1) and 56 cells with var 1e6. The engine merges the first pair
    (d ≈ 258), finds no pair above 1e-6 at C = 59 > 50 (the near-threshold ones decided on the fp64 host score,
    A24), rebuilds, finds none again and stops. Only the outcome of that genuine fp32 run is pinned here.
    """
    rng = np.random.default_rng(5)
    x = np.vstack([np.repeat(rng.normal(0, 2, (2, G)), 2, axis=0), rng.normal(0, 1, (56, G))])
    var = np.vstack([np.full((4, G), 0.1), np.full((56, G), 1e6)])
    stats, trace, table, log = _run(x - x.mean(axis=0), var, GreedyConfig(), monkeypatch)
    assert [(e.kind, e.i, e.j) for e in stats.events] == [("merge", 0, 1)]
    assert log["knn"] == [60, 59] and stats.n_rebuilds == 2 and len(table.children(60)) == 59
    assert len(trace) == 3 and all(max(v[0] for v in e[0].values()) <= eg.ACCEPT_DLOGL for e in trace[1:])
    assert stats.n_threshold_rescored > 0


def _noisy60():
    """5 pairs of identical cells (var 0.1) and 50 cells with var 1e6, centred: fp32 rounding of a noisy pair's
    dLogL (≈ eps32·Σ|terms|, |log w| ≈ 14 per gene and child) is far above the 1e-6 threshold."""
    rng = np.random.default_rng(5)
    x = np.vstack([np.repeat(rng.normal(0, 2, (5, G)), 2, axis=0), rng.normal(0, 1, (50, G))])
    var = np.vstack([np.full((10, G), 0.1), np.full((50, G), 1e6)])
    return x - x.mean(axis=0), var


def test_threshold_decisions_follow_fp64_on_noisy_cells():
    """A24: on the noisy fixture the fp32 engine (bounds on, all pairs) makes the fp64 reference's merge decisions;
    without the host re-score it kept merging noisy cells at d ≈ 2e-6 to 1e-5 (fp32 quanta of the ell sums).
    Regression pin on this fixture: the reference's merges have d ≥ 1.8e-4 and its stop is a host-decided round."""
    x, var = _noisy60()
    stats, *_ = _run(x, var, LAZY)
    t0 = star_em(x, var, tol=LAZY.star_tol)[0]
    _, ref_events = rg.greedy_star(_ref_tree(x, var, t0), 60, None, LAZY.mode, LAZY.star_tol)
    assert [_key(e) for e in stats.events] == [_key(e) for e in ref_events]
    assert stats.n_threshold_rescored > 0 and stats.n_flagged_used_unrescored == 0


@pytest.mark.parametrize("host_d", [5e-7, 2e-6])
def test_near_threshold_merge_decided_on_host(monkeypatch, host_d):
    """A24: a device d = 1e-6 + 1e-9 (within any pair's resolution of 1e-6) is re-scored on the host and the
    decision uses that score: 5e-7 → not merged, the run stops; 2e-6 → merged with the host score."""
    x, var = _leaves(12, 11)
    table0 = NodeTable(x, var)
    target = (3, 7)
    real, real_ref = eg.pair_score, eg.ref_pair.pair_score

    def device(*args, **kw):  # only the target pair is above the threshold, barely
        s = real(*args, **kw)
        hit = np.all(np.asarray(args[0]) == table0.x[3], axis=1) & np.all(np.asarray(args[3]) == table0.x[7], axis=1)
        return s._replace(dlogl=jnp.where(jnp.asarray(hit), 1e-6 + 1e-9, -1.0))

    def host(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        return real_ref(x1, v1, t1, x2, v2, t2, xr, Wr, mode)._replace(dlogl=host_d)

    monkeypatch.setattr(eg, "pair_score", device)
    monkeypatch.setattr(eg.ref_pair, "pair_score", host)
    stats, trace, _, _ = _run(x, var, GreedyConfig(knn=False, lazy=False))
    assert stats.n_threshold_rescored >= 1 and stats.n_flagged_used_unrescored == 0
    d, _, is_host, *_ = trace[0][0][target]
    assert d == host_d and is_host  # the stored score is the host's
    if host_d < eg.ACCEPT_DLOGL:
        assert stats.events == [] and len(trace) == 1
    else:
        assert _key(stats.events[0])[1:3] == target and stats.events[0].dlogl == host_d


@pytest.mark.parametrize("m", [4])
def test_merges_per_round_disjoint(lazy24, m):
    """A20: with m > 1 at the root each round merges the certified best plus up to m − 1 further fresh, disjoint
    pairs with dLogL > 1e-6, in (d desc, ij asc) order, with their stored times; the first merge equals the m = 1
    run's; the root still stops at 3 children. At a non-root node m is ignored (redo starry merges one per round)."""
    x, var = _leaves(24, 11)
    stats, trace, table, _ = _run(x, var, GreedyConfig(knn=False, merges_per_round=m))
    assert _key(stats.events[0]) == _key(lazy24[0].events[0])
    assert len(table.children(24)) == 3
    ev = iter([e for e in stats.events if e.depth == 0])  # the root's own merges (A18b)
    multi = 0
    for k, (scores, tk, *_) in enumerate(trace):
        C_next = len(trace[k + 1][1]) if k + 1 < len(trace) else 3
        if not scores:
            assert next(ev).kind == "reopt"
            continue
        group = [next(ev) for _ in range(len(tk) - C_next)]
        expect, used = [], set()  # greedy disjoint picks over this round's fresh scores
        for ij in sorted(scores, key=lambda p: (-scores[p][0], p)):
            if len(expect) < min(m, len(tk) - 3) and scores[ij][0] > eg.ACCEPT_DLOGL and not used & set(ij):
                expect.append(ij)
                used.update(ij)
        assert [(e.i, e.j) for e in group] == expect
        for e in group:
            t3 = scores[e.i, e.j][1]
            assert e.dlogl == scores[e.i, e.j][0] and e.t[2] == t3[2]
        multi += len(group) > 1
    assert multi > 0
    nx, nvar, outside = _fixture("nonroot")
    one = _run(nx, nvar, GreedyConfig(knn=False), outside=outside)[0]
    four = _run(nx, nvar, GreedyConfig(knn=False, merges_per_round=m), outside=outside)[0]
    assert [_key(e) for e in four.events] == [_key(e) for e in one.events]


def test_near_threshold_extra_pair_decided_on_host(monkeypatch):
    """A24 on the extra pairs of an m > 1 round: pair (3, 7) at device d = 1 is merged; the extra pair (4, 9) at
    device d = 1e-6 + 1e-9 is re-scored on the host (5e-7) and not merged, in that round or later."""
    x, var = _leaves(12, 11)
    table0 = NodeTable(x, var)
    real, real_ref = eg.pair_score, eg.ref_pair.pair_score

    def rows(args, i, j):
        return np.all(np.asarray(args[0]) == table0.x[i], axis=1) & np.all(np.asarray(args[3]) == table0.x[j], axis=1)

    def device(*args, **kw):
        s = real(*args, **kw)
        d = np.where(rows(args, 3, 7), 1.0, np.where(rows(args, 4, 9), 1e-6 + 1e-9, -1.0))
        return s._replace(dlogl=jnp.asarray(d, s.dlogl.dtype))

    def host(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        s = real_ref(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
        hit = np.array_equal(x1, table0.x[4].astype(np.float64)) and np.array_equal(x2, table0.x[9].astype(np.float64))
        return s._replace(dlogl=5e-7 if hit else s.dlogl)

    monkeypatch.setattr(eg, "pair_score", device)
    monkeypatch.setattr(eg.ref_pair, "pair_score", host)
    stats, trace, _, _ = _run(x, var, GreedyConfig(knn=False, lazy=False, merges_per_round=4))
    assert [_key(e)[1:3] for e in stats.events] == [(3, 7)]
    assert stats.n_threshold_rescored >= 2 and stats.n_flagged_used_unrescored == 0  # round 0 extra, round 1 best


def test_compiles_bounded_across_buckets(monkeypatch):
    """Review focus 3: one kNN run from C = 70 (kNN buffer buckets 128 → 64, star re-optimisations down through
    smaller buckets) compiles each kernel at most twice per shape bucket it sees; an identical second run compiles
    nothing and gives identical events and counters (determinism)."""
    x, var = _leaves(70, 41)
    jax.clear_caches()
    stats, trace, _, log = _run(x, var, GreedyConfig(), monkeypatch)
    knn_buckets = {_bucket(C) for C in log["knn"]}
    star_buckets = {_bucket(len(e[1])) for e in trace if not e[0]}
    assert len(knn_buckets) >= 2 and len(star_buckets) >= 2  # buckets crossed
    caches = {  # kernel: (jitted function, shape buckets it sees)
        # rescore_top and new_batch rows, plus SPECIAL_BATCH rows when a special-child greedy ran (A18b)
        "pair_score": (eg.pair_score, 2 + any(e.depth for e in stats.events)),
        "star_em": (eg.star_em, len(star_buckets)),
        "knn": (knn_mod._knn, len(knn_buckets)),
        "knn_query": (knn_mod._knn_query, len(knn_buckets)),
        "set_rows": (eg._set_rows, 1 + len(knn_buckets)),  # the node-table mirror plus one kNN buffer per bucket
    }
    for name, (fn, n_shapes) in caches.items():
        assert 0 < fn._cached._cache_size() <= 2 * n_shapes, name
    with count_compilations() as c:
        again = _run(x, var, GreedyConfig())[0]
    assert c.count == 0
    assert [_key(e) for e in again.events] == [_key(e) for e in stats.events] and again[1:] == stats[1:]
