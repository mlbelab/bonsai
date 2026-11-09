"""fp64 brute-force greedy reference (``bonsai._ref.greedy``; plan M2 Task 5, maths ref §6, §8, §11, A18, A21, A21b)."""

import math

import numpy as np
import pytest

from bonsai._ref import greedy as rg
from bonsai._ref.greedy import (
    Tree,
    children,
    collapse_zero_edges,
    compact,
    decide,
    greedy_star,
    greedy_tree,
    redo_starry,
)
from bonsai._ref.pair import PAIR_LOG_T_MIN
from bonsai._ref.star import star_em
from bonsai._ref.tree import tree_loglik, tree_loglik_grad

from ._bounds import EPS64, loglik_scale, tree_height
from ._trees import bm_tree, random_tree
from ._upstream import upstream_merge_children

G = 30


def _leaves(C: int, seed: int):
    _, _, x, var = bm_tree(C, G, np.random.default_rng(seed))
    return x[:C], var[:C]


def _star(x, var, t=None) -> Tree:
    """Root N with the N leaves as children, star times from ``star_em`` unless given."""
    N = len(x)
    tree = Tree(
        np.r_[np.full(N, N), -1],
        np.zeros(N + 1),
        np.vstack([x, np.zeros(G)]),
        np.vstack([var, np.ones(G)]),
        np.r_[np.ones(N, bool), False],
    )
    tree.t[:N] = star_em(x, var)[0] if t is None else t
    return tree


def _nonroot_fixture(C: int, seed: int):
    """C leaf children of a non-root node p = C, and an outside message (x_out, w̃ = 1 per gene)."""
    rng = np.random.default_rng(seed)
    _, _, x, var = bm_tree(C, G, rng)
    return _star(x[:C], var[:C]), (rng.normal(size=G), np.ones(G))


def _from_tree(parent, t, x, var) -> Tree:
    """Tree from node-indexed arrays; leaves are cells, internal rows get their messages."""
    tree = Tree(parent.copy(), t.copy(), x.copy(), var.copy(), ~np.isin(np.arange(len(parent)), parent))
    g = tree_loglik_grad(parent, t, x, var)
    internal = ~tree.is_cell
    tree.x[internal], tree.var[internal] = g.x[internal], 1.0 / g.W[internal]
    return tree


def _bm_on(parent, t, rng):
    """Brownian leaf data on a given tree (as ``bm_tree``): z from 0 at the root, x = z + N(0, var)."""
    var = rng.uniform(0.05, 0.3, size=(len(parent), G))
    z = np.zeros((len(parent), G))
    for c in range(len(parent) - 2, -1, -1):  # random_tree: parents have larger ids, root last
        z[c] = z[parent[c]] + np.sqrt(t[c]) * rng.normal(size=G)
    return z + np.sqrt(var) * rng.normal(size=z.shape), var


def _rounding_tol(parent, t, x, var, n_scores: int) -> float:
    """Bound on |ΔL_tree − Σ dLogL| from fp64 rounding (first order, ``_bounds`` conventions).

    One ``tree_loglik``: ≤ 8 roundings per per-gene term, ⌈log2 T⌉ for numpy's pairwise sum of the T
    terms of a node, + 1 per internal node for the running total; two evaluations. One pair score:
    its two star ells (8 + ⌈log2 3G⌉ each), plus its inputs, the parent AIR, carrying the relative
    rounding of the pre-order messages, ≤ (2H + 1)(C + 3)·eps (``tree_tols``' ρ_g). All terms are
    bounded by the tree's Σ|terms| (``loglik_scale``).
    """
    g = tree_loglik_grad(parent, t, x, var)
    scale = loglik_scale(parent, t, g.x, g.W)
    kids = np.bincount(parent[parent >= 0])
    n_int, C, H = int(np.count_nonzero(kids)), int(kids.max()), tree_height(parent)
    one_L = 8 + math.ceil(math.log2((C + 1) * G)) + n_int
    pair = 2 * (8 + math.ceil(math.log2(3 * G))) + (2 * H + 1) * (C + 3)
    return (2 * one_L + n_scores * pair) * EPS64 * scale


def _snap_candidate(tree: Tree, e) -> bool:
    """A merge whose applied t_1a / t_2a < 1e-6 sits above an internal child (A18 snap applies)."""
    return e.kind == "merge" and (
        (e.t[0] < rg.ADD_AS_CHILD_T and not tree.is_cell[e.i]) or (e.t[1] < rg.ADD_AS_CHILD_T and not tree.is_cell[e.j])
    )


# --- decision rule ------------------------------------------------------------------------------


def test_decide_rules():
    """Synthetic rounds (TH:1214-1215, 2428; A21b): best pair by strict > in (i, j) order, merged iff d > 1e-6."""
    # tie rule: equal d, lower (i, j) wins, regardless of input order
    assert decide([((2, 3), 5.0), ((0, 4), 5.0), ((1, 2), 4.0)]) == (0, 4)
    # accept threshold: strict > 1e-6
    assert decide([((0, 1), 1e-6)]) is None
    assert decide([((0, 1), 1.0000001e-6)]) == (0, 1)
    assert decide([((0, 1), 5e-7), ((1, 2), -1.0)]) is None
    # -inf / nan scores are never chosen
    assert decide([((0, 1), -np.inf), ((0, 2), np.nan), ((1, 2), 3.0)]) == (1, 2)


# --- greedy_star --------------------------------------------------------------------------------


def test_each_merge_score_is_the_exact_tree_change():
    """Maths ref §6: dLogL = L_tree(after) − L_tree(before), all other edges fixed (independent brute force)."""
    x, var = _leaves(12, 11)
    tree0 = _star(x, var)
    trace: list = []
    final, events = greedy_star(tree0, 12, None, trace=trace)
    assert all(e.depth == 0 for e in events)  # precondition: root rounds only, trace[k] is event k's round
    N, root = 12, 12
    checked = 0
    for k, ev in enumerate(events):
        if ev.kind != "merge" or _snap_candidate(final, ev):
            continue
        a = ev.new
        t_kids = trace[k][1]  # every root child's time before this round's decision
        before_p = np.where(final.parent[:a] >= a, root, final.parent[:a])
        before_t = final.t[:a].copy()
        for c, tc in t_kids.items():
            before_t[c] = tc
        after_p = np.where(final.parent[: a + 1] > a, root, final.parent[: a + 1])
        after_t = np.r_[before_t, ev.t[2]]
        after_t[[ev.i, ev.j]] = ev.t[:2]
        xs, vs = final.x[: a + 1], final.var[: a + 1]  # only leaf rows are read
        L0 = tree_loglik(before_p, before_t, xs[:a], vs[:a])
        L1 = tree_loglik(after_p, after_t, xs, vs)
        tol = max(_rounding_tol(before_p, before_t, xs[:a], vs[:a], 1), _rounding_tol(after_p, after_t, xs, vs, 1))
        assert abs((L1 - L0) - ev.dlogl) <= tol, (k, L1 - L0, ev.dlogl, tol)
        checked += 1
    assert checked >= N // 2  # non-vacuous: most of the 9 merges are checked


def test_special_child_merges_are_exact_tree_changes():
    """A18b, maths ref §6 (independent brute force): over a root greedy with nested special-child greedies, L_tree(end)
    − L_tree(star) = Σ dLogL of every merge, special-child ones included. This checks each special-child score
    against the true tree change, so it checks the outside message the receiving node sees from its parent.

    C = 10, seed 8 (the "nested_snap" fixture); precondition (asserted): every snap was scored at exactly t = 0, so
    snapping moves no time and no reopt runs (C ≤ 13)."""
    x, var = _leaves(10, 8)
    tree0 = _star(x, var)
    trace: list = []
    final, events = greedy_star(tree0, 10, None, trace=trace)
    assert 2 in [e.depth for e in events]
    merges = iter(events)
    for scores, *_ in trace:
        ij = decide((p, d) for p, (d, _) in scores.items())
        if ij is None:
            continue
        e = next(merges)
        assert e.kind == "merge" and (e.i, e.j) == ij
        for c, n in ((0, e.i), (1, e.j)):
            assert e.t[c] != 0.0 or final.is_cell[n] or scores[ij][1][c] == 0.0
    p1, t1, x1, v1, _ = compact(final, 10)
    p0, t0 = tree0.parent, tree0.t
    gain = tree_loglik(p1, t1, x1, v1) - tree_loglik(p0, t0, tree0.x, tree0.var)
    tol = _rounding_tol(p0, t0, tree0.x, tree0.var, 0) + _rounding_tol(p1, t1, x1, v1, len(events))
    assert abs(gain - sum(e.dlogl for e in events)) <= tol


def test_tie_rule_prefers_lower_ids(monkeypatch):
    """Exact duplicates score bitwise equal (canonical child order, A20), and on equal top scores the lower
    (i, j) is merged (TH:1214-1215, strict > in scan order)."""
    x, var = _leaves(12, 3)
    x[4], var[4] = x[2], var[2]  # leaf 4 is a copy of leaf 2
    trace: list = []
    greedy_star(_star(x, var), 12, None, trace=trace)
    scores = trace[0][0]
    for k in set(range(12)) - {2, 4}:  # (k, 2) and (k, 4) have identical inputs
        assert scores[tuple(sorted((k, 2)))][0] == scores[tuple(sorted((k, 4)))][0]
    # equal top scores, forced: a natural top tie needs two pairs with identical inputs, i.e. copied
    # leaves, and then the copies' own pair scores higher. Pairs (3, 5) and (1, 8) at 1e3, same times
    real = rg.pair_score

    def fake(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        s = real(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
        forced = any(np.array_equal(x1, x[i]) and np.array_equal(x2, x[j]) for i, j in ((3, 5), (1, 8)))
        return s._replace(dlogl=1e3, t=np.array([0.3, 0.4, 0.5])) if forced else s

    monkeypatch.setattr(rg, "pair_score", fake)
    _, events = greedy_star(_star(x, var), 12, None)
    assert events[0][:4] == ("merge", 1, 8, 13) and events[1][:3] == ("merge", 3, 5)


def test_root_stops_at_three_nonroot_at_two():
    x, var = _leaves(12, 11)
    final, events = greedy_star(_star(x, var), 12, None)
    assert len(children(final, 12)) == 3
    assert [e.new for e in events if e.kind == "merge"] == list(range(13, 22))  # ids count up from N + 1
    tree, outside = _nonroot_fixture(8, 0)
    final, events = greedy_star(tree, 8, outside)
    assert len(children(final, 8)) == 2
    assert all(e.kind == "merge" for e in events)


def test_no_positive_pair_terminates():
    """var = 1e6: every dLogL <= 1e-6, no event, children and times unchanged."""
    x, _ = _leaves(10, 0)
    tree = _star(x, np.full_like(x, 1e6))
    trace: list = []
    final, events = greedy_star(tree, 10, None, trace=trace)
    assert events == []
    assert len(trace) == 1 and max(d for d, _ in trace[0][0].values()) <= rg.ACCEPT_DLOGL
    np.testing.assert_array_equal(final.parent, tree.parent)
    np.testing.assert_array_equal(final.t, tree.t)


def test_pair_at_the_t_aR_bound_is_merged():
    """A21b: a best pair whose t_aR sits at its lower bound (≈ 1e-7, PAIR_LOG_T_MIN) is merged when d > 1e-6,
    with that t_aR applied (no zero-root re-optimisation). Root, C = 8, seed 6 (A21's divergence fixture)."""
    trace: list = []
    _, events = greedy_star(_star(*_leaves(8, 6)), 8, None, trace=trace)
    t_min = np.exp(PAIR_LOG_T_MIN)
    seen = 0
    for r, (scores, _, _, _, _) in enumerate(trace):
        ij, (d, t3) = max(sorted(scores.items()), key=lambda kv: kv[1][0])
        if d > rg.ACCEPT_DLOGL and t3[2] <= 1e-6:
            assert t3[2] == pytest.approx(t_min, rel=1e-9)  # at the bound
            e = events[r]
            assert (e.kind, e.i, e.j, e.t[2]) == ("merge", *ij, t3[2])
            seen += 1
    assert seen >= 1
    assert all(e.kind == "merge" for e in events)


@pytest.mark.parametrize(
    ("pair", "t3", "applied"),
    [
        ((1, 7), (5e-7, 4e-7, 0.5), (5e-7, 0.0, 0.5)),  # cell child 1 keeps t_1a; internal child 2 snapped
        ((7, 8), (5e-7, 3e-7, 0.5), (0.0, 3e-7, 0.5)),  # both internal: child 1 first, child 2 untouched
        ((1, 2), (5e-7, 5e-7, 0.5), (5e-7, 5e-7, 0.5)),  # cells: never snapped
        ((7, 8), (2e-6, 5e-7, 0.5), (2e-6, 0.0, 0.5)),  # t_1a >= 1e-6: child 2 snapped
    ],
)
def test_add_as_child_snap(monkeypatch, pair, t3, applied):
    """A18: t_1a < 1e-6 and child 1 internal → t_1a := 0 exactly, else the same for child 2; cells keep theirs.

    Root 9: cells 1, 2, 3, internal 7 (cells 4, 5) and 8 (cells 0, 6). ``pair_score`` is wrapped to give
    the chosen pair a dominant score with the given times, so it is the first merge.
    """
    parent = np.array([8, 9, 9, 9, 7, 7, 8, 9, 9, -1])
    rng = np.random.default_rng(5)
    x, var = rng.normal(size=(10, G)), rng.uniform(0.05, 0.3, (10, G))
    tree = _from_tree(parent, np.r_[rng.uniform(0.2, 1.0, 9), 0.0], x, var)
    real = rg.pair_score

    def fake(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        s = real(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
        if np.array_equal(x1, tree.x[pair[0]]) and np.array_equal(x2, tree.x[pair[1]]):
            return s._replace(dlogl=1e3, t=np.array(t3))
        return s

    monkeypatch.setattr(rg, "pair_score", fake)
    trace: list = []
    final, events = greedy_star(tree, 9, None, trace=trace)
    e = events[0]
    assert (e.kind, e.i, e.j) == ("merge", *pair) and e.t == applied
    recv = [c for c, tc in zip(pair, applied[:2], strict=True) if tc == 0.0 and not tree.is_cell[c]]
    if not recv:
        assert (final.t[e.i], final.t[e.j]) == applied[:2]
        assert not any(entry[4] < 0 for entry in trace)  # no special-child greedy
        return
    # A18b: the receiving child r is collapsed into the new node at once, and the next round is the special-child
    # greedy there: its star is r's children plus the moved child m, scoring only the pairs (m, c) (TH:1631)
    r, m = recv[0], pair[1] if recv[0] == pair[0] else pair[0]
    assert final.parent[r] == -2 and final.t[r] == 0.0
    sub = trace[1]
    assert sub[4] == -1 and set(sub[1]) == set(children(tree, r)) | {m}
    assert set(sub[0]) == {(min(m, c), max(m, c)) for c in children(tree, r)}


@pytest.mark.parametrize("C", [24, pytest.param(40, marks=pytest.mark.slow)])
def test_periodic_reopt_schedule(monkeypatch, C):
    """Root reopt exactly at the rounds where counter > max(⌊0.05·C⌋, 10), C and the threshold recomputed at
    each reopt (TH:1154, 1926-1927, 2354; A21: on any round; A21b: no other reopt).

    ``star_em`` (called only by a reopt) raises past the expected reopt count + 1, so a counter that is not
    reset after a reopt (a reopt every round, forever) fails fast instead of hanging."""
    tree0 = _star(*_leaves(C, 0))
    expected = {24: 1, 40: 3}[C]  # 11 merges per reopt cycle from C down to 3
    real, calls = rg.star_em, []

    def counted(*args, **kwargs):
        calls.append(1)
        if len(calls) > expected + 1:
            raise RuntimeError("star_em called more often than the reopt schedule allows")
        return real(*args, **kwargs)

    monkeypatch.setattr(rg, "star_em", counted)
    trace: list = []
    _, events = greedy_star(tree0, C, None, trace=trace)
    assert any(e.depth for e in events)  # special-child rounds (A18b) are interleaved; the root's own are checked
    trace, events = [e for e in trace if e[4] >= 0], [e for e in events if e.depth == 0]
    n_child, counter = C, 0
    redo = max(int(0.05 * n_child), 10)
    for r, e in enumerate(events):
        assert (trace[r][4], len(trace[r][1])) == (counter, n_child)
        assert (e.kind == "reopt") == (counter > redo), (r, counter, redo)
        assert (e.kind == "reopt") == (trace[r][0] == {})  # a periodic round scores nothing
        if e.kind == "reopt":
            counter, redo = 0, max(int(0.05 * n_child), 10)
        else:
            counter, n_child = counter + 1, n_child - 1
    assert n_child == 3
    assert sum(e.kind == "reopt" for e in events) == len(calls) == expected


# --- collapse, redo starry, greedy_tree ---------------------------------------------------------


def test_collapse_zero_edges_hand_tree():
    """Nested zero edges collapse recursively (children first); a cell with t = 0 stays (TH:789-807)."""
    #        9 (root)
    #   8 (t)   4 (cell, t=0)   5
    #  7 (0)  3
    # 6 (0) 2
    # 0 1
    parent = np.array([6, 6, 7, 8, 9, 9, 7, 8, 9, -1])
    rng = np.random.default_rng(0)
    x, var = rng.normal(size=(10, G)), rng.uniform(0.05, 0.3, (10, G))
    t = np.array([0.3, 0.4, 0.5, 0.6, 0.0, 0.7, 0.0, 0.0, 0.2, 0.0])
    tree = _from_tree(parent, t, x, var)
    out, gained = collapse_zero_edges(tree, 9)
    assert gained == [8]
    assert children(out, 8) == [0, 1, 2, 3] and children(out, 9) == [4, 5, 8]
    assert out.parent[6] == out.parent[7] == -2
    p, tt, xx, vv, _ = compact(out, 9)
    assert tree_loglik(p, tt, xx, vv) == pytest.approx(tree_loglik(parent, t, x, var), rel=1e-12)
    t[8] = 0.0  # now the root edge too: everything below 8 lands at the root
    out, gained = collapse_zero_edges(_from_tree(parent, t, x, var), 9)
    assert gained == [9] and children(out, 9) == [0, 1, 2, 3, 4, 5]
    assert out.parent[8] == -2


def test_redo_starry_gain_equals_sum_of_scores():
    """random_tree(14, max_children=5) + BM data: L(after) − L(before) = Σ merge dLogL (no reopts or snaps,
    asserted; maths ref §6, §11)."""
    rng = np.random.default_rng(0)  # the first seed meeting the precondition
    parent, t = random_tree(14, rng, max_children=5)
    x, var = _bm_on(parent, t, rng)
    tree = _from_tree(parent, t, x, var)
    root = len(parent) - 1
    out, events, passes = redo_starry(tree, root)
    assert passes == 1 and len(events) >= 2
    assert all(e.kind == "merge" and not _snap_candidate(out, e) for e in events)
    p1, t1, x1, v1, _ = compact(out, root)
    gain = tree_loglik(p1, t1, x1, v1) - tree_loglik(parent, t, x, var)
    tol = _rounding_tol(parent, t, x, var, 0) + _rounding_tol(p1, t1, x1, v1, len(events))
    assert abs(gain - sum(e.dlogl for e in events)) <= tol


def test_redo_starry_resolves_snaps_inside_the_merge(monkeypatch):
    """A snap in redo starry is resolved inside the merge (A18b; TH:2637-2641, ``mergeChildrenRecursive`` calls
    ``mergeChildrenUB`` with mergeDownstream=True, TH:1004-1007): the receiving child is collapsed at once and the
    special-child greedy runs at the new node, so one pass leaves no zero edge.

    Root 7: cells 4, 5, 6 and internal 9 (internal 8 (cells 0, 1), cells 2, 3). ``pair_score`` is wrapped
    to force the first merge at 9: (2, 8) with t_2a = 5e-7 (snapped to 0)."""
    parent = np.array([8, 8, 9, 9, 7, 7, 7, -1, 9, 7])
    rng = np.random.default_rng(2)
    x, var = rng.normal(size=(10, G)), rng.uniform(0.05, 0.3, (10, G))
    tree = _from_tree(parent, np.r_[rng.uniform(0.2, 1.0, 7), 0.0, rng.uniform(0.2, 1.0, 2)], x, var)
    real = rg.pair_score
    forced = []

    def fake(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        s = real(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
        if not forced and np.array_equal(x2, tree.x[8]) and np.array_equal(x1, tree.x[2]):
            forced.append(True)
            return s._replace(dlogl=1e3, t=np.array([0.3, 5e-7, 0.5]))
        return s

    monkeypatch.setattr(rg, "pair_score", fake)
    out, events, passes = redo_starry(tree, 7)
    # post-order: at 9 the forced merge snaps t(8) := 0, 8 collapses into 10 (cells 0, 1, 2) and the special-child
    # greedy (special 2) merges (1, 2) under 10; at the root (4, 9) snaps t(9) := 0 (its optimum t_2a < 1e-6), 9
    # collapses into 12 (cells 3, 4, node 10) and the special-child greedy (special 4) merges (4, 10) (pins)
    assert forced and [(e.i, e.j, e.new, e.depth, e.t[1]) for e in events] == [
        (2, 8, 10, 0, 0.0),
        (1, 2, 11, 1, events[1].t[1]),
        (4, 9, 12, 0, 0.0),
        (4, 10, 13, 1, events[3].t[1]),
    ]
    assert passes == 1 and out.parent[8] == out.parent[9] == -2
    assert children(out, 10) == [0, 11] and children(out, 12) == [3, 13] and children(out, 13) == [4, 10]
    assert not rg._zero_edge(out, 7)
    # natural case (random_tree(14, max_children=5), seed 3: before A18b the first seed needing a second pass): no
    # zero edge above an internal non-cell node remains
    monkeypatch.undo()
    rng = np.random.default_rng(3)
    parent, t = random_tree(14, rng, max_children=5)
    x, var = _bm_on(parent, t, rng)
    out, events, passes = redo_starry(_from_tree(parent, t, x, var), len(parent) - 1)
    assert any(e.depth for e in events)  # a snap with its special-child greedy
    assert not rg._zero_edge(out, len(parent) - 1)


@pytest.mark.parametrize("n_forced", [1, 6, 7])
def test_redo_starry_later_passes_run_only_at_touched_nodes(monkeypatch, n_forced):
    """Passes after the first run the greedy only at the nodes the collapse gave children (A18; Task 7), and passes
    repeat while a zero edge remains, at most N − 1 = 7 (8 cells; A18b), then ``RuntimeWarning``.

    Root 8: cell 7, internal 10 (internal 9 (cells 0, 1), cells 2, 3) and 11 (cells 4, 5, 6 with var 1e10:
    no pair qualifies, so 11 keeps three children). The merge at 10 is forced, ``n_forced`` times, to (2, 9) with
    t_aR = 0 exactly (a zero edge above the new internal node, no snap); the next pass collapses that node into 10
    (children 2, 3, 9 again) and runs the greedy there, not again at 11. So n_forced + 1 passes are needed.
    """
    parent = np.array([9, 9, 10, 10, 11, 11, 11, 8, -1, 10, 8, 8])
    rng = np.random.default_rng(4)
    x, var = rng.normal(size=(12, G)), rng.uniform(0.05, 0.3, (12, G))
    var[4:7] = 1e10  # pair dLogL ∝ 1/var: ≈ 5e-9 here, 200× below the 1e-6 accept threshold
    tree = _from_tree(parent, np.r_[rng.uniform(0.2, 1.0, 8), 0.0, rng.uniform(0.2, 1.0, 3)], x, var)
    real_score, real_star, calls, forced = rg.pair_score, rg.greedy_star, [], []

    def fake(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        s = real_score(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
        if len(forced) < n_forced and np.array_equal(x1, tree.x[2]) and np.array_equal(x2, tree.x[9]):
            forced.append(True)
            return s._replace(dlogl=1e3, t=np.array([0.3, 0.4, 0.0]))
        return s

    def record(tr, p, *args, **kwargs):
        calls.append(p)
        return real_star(tr, p, *args, **kwargs)

    monkeypatch.setattr(rg, "pair_score", fake)
    monkeypatch.setattr(rg, "greedy_star", record)
    assert rg.max_passes(8) == 7
    if n_forced + 1 > 7:
        with pytest.warns(RuntimeWarning, match="zero edges remain after 7 passes"):
            out, events, passes = redo_starry(tree, 8)
        assert rg._zero_edge(out, 8)  # returned as is
    else:
        out, events, passes = redo_starry(tree, 8)
        assert not rg._zero_edge(out, 8)
    assert events[0][:4] == ("merge", 2, 9, 12) and events[0].t[2] == 0.0 and out.parent[12] == -2
    assert passes == min(n_forced + 1, 7) and calls == [10, 11] + [10] * (passes - 1)
    assert children(out, 11) == [4, 5, 6]


def test_redo_starry_repeats_after_a_reopt_zero_edge(monkeypatch):
    """A root star reopt that sets an internal child's t to 0 also triggers a further pass (zero edges are
    found on the tree, not only on merge events).

    Root 15 with 15 cells: 11 merges, then the periodic reopt at C = 4 (counter 11 > 10). ``star_em`` is
    wrapped to set the newest child's (an internal node's) t to 0, and that node's pairs score −inf so it
    stays at the root. Pass 2 collapses it into the root and runs the root greedy again.
    """
    tree = _star(*_leaves(15, 2))  # seed 2: the first without a snap before the reopt
    real_em, real_score, zeroed = rg.star_em, rg.pair_score, []

    def em(x, var, *args, **kwargs):
        if zeroed:  # exactly one reopt is expected; fail fast instead of looping (counter not reset)
            raise RuntimeError("second star reopt")
        t, ell, conv = real_em(x, var, *args, **kwargs)
        t[-1] = 0.0  # kids list: the last child is the newest merged node
        zeroed.append(x[-1].copy())
        return t, ell, conv

    def score(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        s = real_score(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
        if zeroed and (np.array_equal(x1, zeroed[0]) or np.array_equal(x2, zeroed[0])):
            return s._replace(dlogl=-np.inf)
        return s

    monkeypatch.setattr(rg, "star_em", em)
    monkeypatch.setattr(rg, "pair_score", score)
    out, events, passes = redo_starry(tree, 15)
    k = [e.kind for e in events].index("reopt")
    assert k == 11 and len(zeroed) == 1
    assert not any(_snap_candidate(out, e) for e in events[:k])  # no snap: only the reopt makes a zero edge
    z = int(np.flatnonzero(np.all(out.x == zeroed[0], axis=1))[0])
    assert out.parent[z] == -2 and passes == 2  # collapsed in pass 2
    assert events[-1].kind == "merge" and out.parent[events[-1].new] == 15  # pass 2: root greedy


def test_greedy_tree_pipeline():
    """Star EM, root greedy (= greedy_star from the star times), collapsed snapshot, redo starry."""
    x, var = _leaves(12, 11)
    tree, events, info = greedy_tree(x, var)
    star_t, _, conv = star_em(x, var)
    assert info["star_converged"] == conv
    _, root_events = greedy_star(_star(x, var, star_t), 12, None)
    assert events[: info["n_greedy_events"]] == root_events
    snap = info["greedy"]
    p, t, *_ = compact(snap, 12)
    internal = np.isin(np.arange(len(p)), p)
    assert not np.any((t == 0) & internal & (p >= 0))  # cleanup_tree: no zero internal edge (DP:354-355)
    assert info["passes"] >= 1 and len(children(tree, 12)) >= 3


# --- upstream semantics -------------------------------------------------------------------------

# Pre-declared margin (plan Task 5): every decision-relevant comparison of a reference round must be
# separated by > 1e-3 relative (≫ upstream's tol-1e-4 scoring); the first round failing one ends the
# comparison. Upstream re-solves the merged pair at tol 1e-9 (TH:2445-2447), so the snap test on the
# pair's scored t_1a / t_2a is resolved by the same margin.
MARGIN = 1e-3


def _sep(a: float, b: float) -> bool:
    return abs(a - b) > MARGIN * max(abs(a), abs(b))


def _round_ok(scores, event, is_cell) -> str | None:
    """None if the reference's round is well separated, else the comparison that is not (A21b: no class split)."""
    ds = sorted((d for d, _ in scores.values()), reverse=True)
    if not _sep(ds[0], rg.ACCEPT_DLOGL):
        return "best vs 1e-6"
    if event is not None and event.kind == "merge":
        if len(ds) > 1 and not _sep(ds[0], ds[1]):
            return "top-2 gap"
        scored = scores[(event.i, event.j)][1]
        if any(not is_cell[c] and not _sep(scored[k], rg.ADD_AS_CHILD_T) for k, c in ((0, event.i), (1, event.j))):
            return "snap time vs 1e-6"
    return None


def _unordered(decisions):
    return [(kind, frozenset((a, b))) for kind, a, b in decisions]


def _ref_decisions(x, var, t):
    """(decisions as upstream_merge_children reports them, compared events, end reason or None, final tree).

    Walks the trace in execution order, special-child rounds included (A18b): a scored round consumes the next event
    iff ``decide`` merges, a re-optimisation round its reopt event. Leaf sets come from the events: a node's leaf set
    is fixed when it is created (restructuring below it keeps it), so it is the one upstream sees at the decision.
    """
    trace: list = []
    final, events = greedy_star(_star(x, var, t), len(x), None, trace=trace)
    leaves = {c: frozenset([c]) for c in range(len(x))}
    out: list = []
    k = 0
    for scores, _, _, _, _ in trace:
        merges = not scores or decide((ij, d) for ij, (d, _) in scores.items()) is not None
        e = events[k] if merges else None
        reason = _round_ok(scores, e, final.is_cell) if scores else None
        if reason is not None:
            return out, events[:k], reason, final
        if e is None:
            continue
        k += 1
        if e.kind == "reopt":
            out.append(("reopt", frozenset(), frozenset()))
        else:
            leaves[e.new] = leaves[e.i] | leaves[e.j]
            out.append((e.kind, leaves[e.i], leaves[e.j]))
    assert k == len(events)
    return out, events, None, final


# (C, seed): seed = the first in 0, 1, ... whose reference run meets the case's precondition (asserted below), chosen
# from the reference alone. "zero_root_merge" (seed 6) is where A21's zero-root reopt diverged from upstream.
# "snap_then_more": a special-child merge (A18b) followed by further root merges; "nested_snap": a special-child
# merge two snaps deep; both compared over the whole run.
UPSTREAM_CASES = {
    "plain": (8, 0),
    "add_as_child": (8, 2),
    "zero_root_merge": (8, 6),
    "snap_then_more": (8, 10),
    "nested_snap": (10, 8),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(UPSTREAM_CASES))
def test_reference_matches_upstream_merge_sequence(case):
    """C bm_tree leaves, G = 30, star times from ``star_em``; upstream ``mergeChildrenUB`` (all pairs, C <= 10) with
    ``mergeDownstream=True``, as its root greedy (MA:177-180).

    The reference's decisions, as (kind, {leaf set 1, leaf set 2}), equal upstream's over the compared prefix, in call
    order: an upstream add-as-child is the reference's snapped merge (same leaf sets), followed by the merges of its
    recursive special-child greedy (A18b).
    """
    C, seed = UPSTREAM_CASES[case]
    x, var = _leaves(C, seed)
    t = star_em(x, var)[0]
    ref, compared, reason, final = _ref_decisions(x, var, t)
    if case == "plain":
        assert reason is None and {e.kind for e in compared} == {"merge"}  # the whole run is compared
    elif case == "add_as_child":
        assert any(_snap_candidate(final, e) for e in compared), reason
    elif case == "zero_root_merge":  # a merged pair with t_aR <= 1e-6 in the compared prefix
        assert any(e.kind == "merge" and e.t[2] <= 1e-6 for e in compared), reason
    elif case == "snap_then_more":
        depth = [e.depth for e in compared]
        assert reason is None and 1 in depth and 0 in depth[depth.index(1) + 1 :], (reason, depth)
    else:
        assert reason is None and 2 in [e.depth for e in compared], reason
    up = upstream_merge_children(x, var, t)
    # unordered: upstream's add-as-child keeps child 1's nodeInd, so the child order can differ after a snap
    up, ref = _unordered(up), _unordered(ref)
    assert up[: len(ref)] == ref
    if reason is None:
        assert up == ref
