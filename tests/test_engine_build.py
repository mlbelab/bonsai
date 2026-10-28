"""Greedy tree builder and redo starry (``bonsai._engine.build``; plan M2 Task 7, A18, A21b, A24). The fp64
brute-force reference ``bonsai._ref.greedy`` is the oracle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bench.run import count_compilations

from bonsai._engine import build
from bonsai._engine import greedy as eg
from bonsai._engine.build import collapse_zero_edges, greedy_tree, redo_starry
from bonsai._engine.greedy import ACCEPT_DLOGL, GreedyConfig, NodeTable
from bonsai._ref import greedy as rg
from bonsai._ref.newick import children_lists, parse_newick
from bonsai._ref.star import node_message
from bonsai._ref.tree import postorder, tree_loglik, tree_loglik_grad

from ._bounds import EPS32, EPS64, first_near_tie, tree_height
from ._trees import bm_tree, random_tree
from .test_engine_greedy import FP32, _first_divergence, _run
from .test_ref_greedy import _bm_on, _from_tree

G = 60
N = 24
EXACT = GreedyConfig(knn=False, lazy=False, star_tol=1e-10)
# Times: engine and reference solve the same pair/star problems to their fp64 KKT resolution (≲ 1e-12 here); 1e-8 is
# far above that and far below any branch length that decides a topology (plan Task 7).
T_TOL = 1e-8


def _leaves(n: int, seed: int):
    _, _, x, var = bm_tree(n, G, np.random.default_rng(seed))
    return x[:n].copy(), var[:n].copy()


def _names(n: int) -> list[str]:
    return [f"cell{i}" for i in range(n)]


def _key(e):
    return (e.kind, e.i, e.j, e.new, e.depth)


def _clades(parent, t, n_leaves: int) -> dict:
    """{leaf set below a non-root node: its edge t}, leaves = nodes 0..n_leaves−1 (id-free topology + times)."""
    order, kids = postorder(np.asarray(parent))
    below: dict[int, frozenset] = {}
    for v in order:
        below[v] = frozenset([v]) if v < n_leaves else frozenset().union(*(below[c] for c in kids[v]))
    return {below[v]: float(t[v]) for v in order if parent[v] >= 0}


def _assert_same_tree(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    assert max(abs(a[k] - b[k]) for k in a) <= T_TOL


def _newick_clades(s: str, names: list[str]) -> dict:
    """``_clades`` of a newick string, leaves renumbered to their index in ``names``."""
    p, t, nm = parse_newick(s)
    leaf = [nm.index(c) for c in names]
    rest = [i for i in range(len(nm)) if i not in set(leaf)]
    perm = np.array(leaf + rest)  # new index -> parsed index
    pos = np.empty(len(nm), np.int64)
    pos[perm] = np.arange(len(nm))
    return _clades(np.where(p[perm] >= 0, pos[p[perm]], -1), t[perm], len(names))


def _ref_clades(tree: rg.Tree, root: int, n_leaves: int) -> dict:
    parent, t, *_ = rg.compact(tree, root)  # ids sorted: leaves keep 0..n_leaves−1
    return _clades(parent, t, n_leaves)


def _scorable(s: str, x, var, names) -> float:
    p, t, nm = parse_newick(s)
    X, V = np.zeros((len(nm), x.shape[1])), np.ones((len(nm), x.shape[1]))
    for k, c in enumerate(nm):
        if c in names:
            X[k], V[k] = x[names.index(c)], var[names.index(c)]
    return tree_loglik(p, t, X, V)


@pytest.fixture(scope="module")
def built11():
    """bm_tree(24, 60) seed 11, fp32, default config (bounds on; kNN off at C ≤ 50); also the NodeTable after redo."""
    x, var = _leaves(N, 11)
    real, seen = build.redo_starry, []

    def spy(table, root, cfg):
        seen.append(table)
        return real(table, root, cfg)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build, "redo_starry", spy)
        b = greedy_tree(x, var, _names(N))
    return x, var, b, seen[0]


# --- x64: engine == reference --------------------------------------------------------------------------------------


@pytest.mark.x64
@pytest.mark.parametrize("seed", [11, 13])
def test_greedy_tree_matches_reference_x64(seed):
    x, var = _leaves(N, seed)
    b = greedy_tree(x, var, _names(N), EXACT, dtype=jnp.float64)
    ref, events, info = rg.greedy_tree(x, var, star_tol=EXACT.star_tol)
    assert [_key(e) for e in b.stats["events"]] == [_key(e) for e in events]
    assert b.stats["passes"] == info["passes"] and b.stats["n_flagged_used_unrescored"] == 0
    _assert_same_tree(_clades(b.parent, b.t, N), _ref_clades(ref, N, N))
    _assert_same_tree(_newick_clades(b.stage_newick["greedy"], _names(N)), _ref_clades(info["greedy"], N, N))


def _zero_edge_fixture():
    """random_tree(20, max_children=5) seed 0 with BM data, the two lowest-id internal non-root edges set to 0."""
    rng = np.random.default_rng(0)
    parent, t = random_tree(20, rng, max_children=5)
    x, var = _bm_on(parent, t, rng)
    x -= x[:20].mean(axis=0)  # centred, as NodeTable centres the leaves: both arms see the same rows
    internal = np.flatnonzero(np.isin(np.arange(len(parent)), parent) & (parent >= 0))
    t[internal[:2]] = 0.0
    return _from_tree(parent, t, x, var), len(parent) - 1


def _table(tree: rg.Tree, dtype) -> NodeTable:
    """NodeTable holding ``tree`` (same ids, rows and messages)."""
    n_leaves, n = int(tree.is_cell.sum()), len(tree.parent)
    table = NodeTable(tree.x[:n_leaves], tree.var[:n_leaves], cap=4 * n, dtype=dtype)
    table.parent[:n], table.t[:n], table.is_cell[:n] = tree.parent, tree.t, tree.is_cell
    table.x[:n], table.v[:n] = tree.x, tree.var
    table.n = n
    table.dev_x, table.dev_v = jnp.asarray(table.x), jnp.asarray(table.v)
    return table


@pytest.mark.x64
def test_redo_starry_matches_reference_x64():
    tree, root = _zero_edge_fixture()
    table = _table(tree, jnp.float64)
    stats = redo_starry(table, root, EXACT)
    ref, events, passes = rg.redo_starry(tree, root, star_tol=EXACT.star_tol)
    assert stats["zero_edges_collapsed"] >= 2  # the fixture's zero edges (plus any made on the way)
    assert len(events) >= 2 and [_key(e) for e in stats["events"]] == [_key(e) for e in events]
    assert stats["passes"] == passes
    live = np.flatnonzero(table.parent[: table.n] != -2)
    p, t = table.parent[live], table.t[live]
    pos = np.full(table.n, -1)
    pos[live] = np.arange(len(live))
    _assert_same_tree(_clades(np.where(p >= 0, pos[p], -1), t, 20), _ref_clades(ref, root, 20))


# --- default lane --------------------------------------------------------------------------------------------------


def test_collapse_zero_edges_hand_tree():
    """Nested zero edges collapse children first; a cell with t = 0 stays (TH:789-807; the ``_ref`` hand tree)."""
    #        9 (root)
    #   8 (t)   4 (cell, t=0)   5
    #  7 (0)  3
    # 6 (0) 2
    # 0 1
    parent = np.array([6, 6, 7, 8, 9, 9, 7, 8, 9, -1])
    rng = np.random.default_rng(0)
    x, var = rng.normal(size=(10, G)), rng.uniform(0.05, 0.3, (10, G))
    t = np.array([0.3, 0.4, 0.5, 0.6, 0.0, 0.7, 0.0, 0.0, 0.2, 0.0])
    table = _table(_from_tree(parent, t, x, var), jnp.float32)
    assert collapse_zero_edges(table, 9) == [8]
    assert table.children(8) == [0, 1, 2, 3] and table.children(9) == [4, 5, 8]
    assert table.parent[6] == table.parent[7] == -2 and table.parent[4] == 9


def test_built_tree_scorable_and_newick_round_trip(built11):
    x, var, b, _ = built11
    names = _names(N)
    for key in ("greedy", "redo_starry"):
        assert np.isfinite(_scorable(b.stage_newick[key], x, var, names))
    p, t, nm = parse_newick(b.stage_newick["redo_starry"])
    assert sorted(n for n in nm if not n.startswith(("internal_", "root"))) == sorted(names)
    order = [nm.index(n) for n in b.names]  # BuiltTree index -> parsed index (names are unique)
    assert np.array_equal(t[order], b.t)  # repr round trip is exact
    assert [b.names.index(nm[q]) if q >= 0 else -1 for q in p[order]] == list(b.parent)
    assert b.names[:N] == names and b.names[-1] == "root" and b.parent[-1] == -1
    assert list(b.stats["stage_s"]) == ["star_em", "greedy", "redo_starry"]


def test_no_internal_zero_edges_after_redo(built11):
    """On seed 11 (the plan's fixture) the greedy snaps internal children (A18), each resolved inside its merge by a
    special-child greedy (A18b); no non-root internal edge has t == 0 afterwards, nor in the greedy snapshot."""
    _, _, b, _ = built11
    ev = b.stats["events"]
    snaps = [e for e in ev if e.kind == "merge" and ((e.t[0] == 0 and e.i >= N) or (e.t[1] == 0 and e.j >= N))]
    assert snaps and any(e.depth for e in ev)  # preconditions
    internal = np.isin(np.arange(len(b.parent)), b.parent)
    assert not np.any(internal & (b.parent >= 0) & (b.t == 0))
    p, t, _ = parse_newick(b.stage_newick["greedy"])
    assert not np.any(np.isin(np.arange(len(p)), p) & (p >= 0) & (t == 0))
    kids = children_lists(b.parent)
    assert all(len(k) >= 2 for k in kids[N:])  # no unary node


def test_noisy_data_stops_early_and_builds_valid_tree():
    """Review focus 2: var = 1e6 everywhere, so no pair's fp64 dLogL exceeds 1e-6 (the reference merges nothing) and
    the root keeps all N children; the star is a valid, scorable tree. (Here every star time is 0 and every fp32
    device d ≤ 0, so the A24 host re-score is not needed; mixed noisy/informative cells are Task 6b's A24 test.)"""
    x, _ = _leaves(N, 11)
    var = np.full_like(x, 1e6)
    b = greedy_tree(x, var, _names(N))
    _, ref_events, _ = rg.greedy_tree(x, var)
    assert ref_events == [] and b.stats["events"] == []  # the fp64 reference merges nothing either
    assert len(b.parent) == N + 1 and np.all(b.parent[:N] == N)
    assert b.stats["n_pair_scores"] > 0  # the greedy did score pairs (C = 24 > 10: bounds on, new pairs scored)
    assert np.isfinite(_scorable(b.stage_newick["redo_starry"], x, var, _names(N)))


@pytest.mark.parametrize("n", [3, 4])
def test_tiny_inputs(n):
    """Review focus 5: N = 3 returns the star (no merge); N = 4 at most one merge; both scorable."""
    x, var = _leaves(n, 11)
    b = greedy_tree(x, var, _names(n))
    merges = [e for e in b.stats["events"] if e.kind == "merge"]
    assert len(merges) <= n - 3 and len(b.parent) == n + 1 + len(merges)
    for key in ("greedy", "redo_starry"):
        assert np.isfinite(_scorable(b.stage_newick[key], x, var, _names(n)))


def test_names_length_checked():
    x, var = _leaves(4, 11)
    with pytest.raises(ValueError, match="names"):
        greedy_tree(x, var, _names(3))


@pytest.mark.gpu
def test_greedy_tree_on_gpu(gpu_device):
    """bm_tree(48, 60): the GPU root greedy equals the fp64 reference's events up to the first near-tie (Task 6a's
    rule, fp32); ``greedy_tree`` on the GPU builds a scorable tree, and a second run compiles nothing and gives the
    same newick (compile count printed)."""
    with jax.default_device(gpu_device):
        stats, eng_trace, table, ref_events, ref_trace, ref_tree = _run("bm48", FP32, jnp.float32)
        d = _first_divergence(ref_events, stats.events)
        last = len(ref_trace) - 1 if d is None else d
        k_star = first_near_tie(ref_trace[: last + 1], eng_trace[: last + 1], ref_tree, table, FP32.mode)
        assert d is None or k_star <= d
        x, var = _leaves(48, 21)
        with count_compilations() as c0:
            b = greedy_tree(x, var, _names(48))
        with count_compilations() as c1:
            b2 = greedy_tree(x, var, _names(48))
    print(f"gpu bm48: first near-tie {k_star}, divergence {d}; compiles cold {c0.count}, warm {c1.count}")
    assert c1.count == 0 and b2.stage_newick == b.stage_newick
    assert np.isfinite(_scorable(b.stage_newick["redo_starry"], x, var, _names(48)))
    assert all(np.isfinite(e.dlogl) and e.dlogl > ACCEPT_DLOGL for e in b.stats["events"] if e.kind == "merge")


# --- message bookkeeping (fix round 1) -----------------------------------------------------------------------------


def _live(table: NodeTable, root: int):
    """(compact parent, t, ids) of the live tree, ids increasing (leaves first, as ``rg.compact``)."""
    ids = np.r_[np.flatnonzero(table.parent[: table.n] >= 0), root]
    ids = np.sort(ids)
    pos = np.full(table.n, -1)
    pos[ids] = np.arange(len(ids))
    return np.where(ids == root, -1, pos[table.parent[ids]]), table.t[ids], ids


def _fp32_tols(table: NodeTable, root: int) -> tuple[float, float]:
    """(ρ, e): relative error of any stored precision W (or weight) and absolute error of any stored mean x, against
    the fp64 recursion from the stored leaf rows.

    Each stored message is one float32 rounding (≤ eps32 relative, generously) of an fp64 combination of stored child
    rows; fp64 noise is far below. Precisions are sums of positive weights w = 1/(v + t) (t ≥ 0 shrinks a relative
    error), so per level the relative error grows by ≤ eps32: ρ = (H + 1)·eps32 for height H. A mean is a convex
    combination: per level its error grows by the children's error, plus ≤ 2ρ·max|x_c − x_n| ≤ 4ρX from the weights
    and eps32·X from the rounding, X = max|x_leaf|: e = (H + 1)(1 + 4(H + 1))·eps32·X.
    """
    parent, _, ids = _live(table, root)
    H = tree_height(parent)
    X = float(np.abs(table.x64(ids[table.is_cell[ids]])).max())
    return (H + 1) * EPS32, (H + 1) * (1 + 4 * (H + 1)) * EPS32 * X


def _assert_messages_consistent(table: NodeTable, root: int) -> None:
    """Every live internal node's stored (x, v) equals ``node_message`` of its stored child rows within ``_fp32_tols``
    (a stale message, e.g. an ancestor not updated after its subtree changed, is off by O(1)). The bounds are those of
    the full leaf-to-root recursion, so for this one-level check they are conservative (observed ~1e-3 of them)."""
    rho, e = _fp32_tols(table, root)
    _, _, ids = _live(table, root)
    for n in ids[~table.is_cell[ids]]:
        kids = table.children(int(n))
        xn, W = node_message(table.x64(kids), table.v64(kids), table.t[kids])
        assert np.all(np.abs(1.0 / table.v64(n) - W) <= rho * W), n
        assert np.all(np.abs(table.x64(n) - xn) <= e), n


def test_messages_consistent_after_redo_fixture():
    tree, root = _zero_edge_fixture()
    table = _table(tree, jnp.float32)
    stats = redo_starry(table, root, GreedyConfig())
    assert stats["events"]  # non-vacuity: redo starry restructured the tree
    _assert_messages_consistent(table, root)


def test_messages_consistent_after_greedy_tree(built11):
    *_, table = built11
    _assert_messages_consistent(table, N)


def test_redo_pass_order_and_outside_messages(monkeypatch):
    """Pass 1 calls ``greedy_star`` at the nodes qualifying at pass start (> 3 children at the root, > 2 elsewhere) in
    post-order with children by increasing id, and each ``outside`` equals the fp64 recursion over the current table
    at call time (``tree_loglik_grad``'s x_ex, W_ex through the node's edge) within ``_fp32_tols``.

    Preconditions (asserted): two qualifying nodes in disjoint subtrees (their order is then decided by the child
    order), the earlier one merges (so the later one's outside contains a contribution changed during the pass), and
    a qualifying node below another (pre- vs post-order).
    """
    tree, root = _zero_edge_fixture()
    table = _table(tree, jnp.float32)
    collapsed = rg.collapse_zero_edges(tree, root)[0]  # independent of build's helpers
    start = collapsed.parent

    def post(v):  # children by increasing id, then v
        return [u for c in rg.children(collapsed, v) for u in post(c)] + [v]

    expected = [v for v in post(root) if len(rg.children(collapsed, v)) > (3 if v == root else 2)]
    real, calls = build.greedy_star, []

    def spy(tab, p, outside, is_root, cfg):
        parent, t, ids = _live(tab, root)
        k = int(np.flatnonzero(ids == p)[0])
        leaf = tab.is_cell[ids]
        X, V = np.zeros((len(ids), tab.x.shape[1])), np.ones((len(ids), tab.x.shape[1]))
        X[leaf], V[leaf] = tab.x64(ids[leaf]), tab.v64(ids[leaf])
        rho, e = _fp32_tols(tab, root)
        if outside is None:
            assert p == root
        else:
            g = tree_loglik_grad(parent, t, X, V)
            w_ref = 1.0 / (t[k] + 1.0 / g.W_ex[k])
            amp = g.W_air[parent[k]] / g.W_ex[k]  # fp64 cancellation in W_ex = W − w_n, x_ex = (WX − w_n x_n)/W_ex
            assert np.all(np.abs(outside[1] - w_ref) <= (rho + 16 * EPS64 * amp) * w_ref), p
            assert np.all(np.abs(outside[0] - g.x_ex[k]) <= e + 4 * rho * np.abs(X).max() * (1 + amp)), p
        s = real(tab, p, outside, is_root, cfg)
        calls.append((p, len(s.events)))
        return s

    monkeypatch.setattr(build, "greedy_star", spy)
    redo_starry(table, root, GreedyConfig())
    assert [p for p, _ in calls[: len(expected)]] == expected

    def anc(v):
        out = set()
        while v >= 0:
            v = int(start[v])
            out.add(v)
        return out

    merged = {p for p, n_ev in calls[: len(expected)] if n_ev}
    assert any(
        a in merged and a not in anc(b) and b not in anc(a) for i, a in enumerate(expected) for b in expected[i + 1 :]
    )
    assert any(a in anc(b) for a in expected for b in expected)


@pytest.mark.parametrize("n_zero", [1, 9])
def test_pass_cap_warns_and_returns_collapsed_tree(monkeypatch, n_zero):
    """Passes repeat while a zero edge remains, up to N − 1 = 23 (A18b): each of the first ``n_zero`` redo passes is
    wrapped to leave an exact-zero edge above a live internal node (as an exact-zero optimum would), so n_zero + 1
    passes are needed (9 → 10 > the withdrawn cap 8). With the cap at 1 (n_zero = 1) redo starry warns, and the
    returned tree and its newick are still collapsed (no internal zero edge, as upstream's cleanup_tree)."""
    real, zeroed = build._redo_pass, []

    def pass_then_zero(table, root, touched, cfg):
        out = real(table, root, touched, cfg)
        if len(zeroed) < n_zero:
            p = table.parent[: table.n]
            internal = np.isin(np.arange(table.n), p) & (p >= 0) & ~table.is_cell[: table.n]
            zeroed.append(int(np.flatnonzero(internal)[0]))
            table.t[zeroed[-1]] = 0.0
        return out

    monkeypatch.setattr(build, "_redo_pass", pass_then_zero)
    assert build.max_passes(N) == N - 1
    x, var = _leaves(N, 11)
    if n_zero == 1:
        monkeypatch.setattr(build, "max_passes", lambda n: 1)
        with pytest.warns(RuntimeWarning, match="zero edges remain after 1 passes"):
            b = greedy_tree(x, var, _names(N))
        assert b.stats["passes"] == 1
    else:
        b = greedy_tree(x, var, _names(N))
        assert b.stats["passes"] == n_zero + 1 and len(zeroed) == n_zero
    internal = np.isin(np.arange(len(b.parent)), b.parent)
    assert not np.any(internal & (b.parent >= 0) & (b.t == 0))
    p, t, _ = parse_newick(b.stage_newick["redo_starry"])
    assert not np.any(np.isin(np.arange(len(p)), p) & (p >= 0) & (t == 0))


def test_special_child_greedy_merges_at_most_once(monkeypatch):
    """A18b, both arms: the special-child greedy scores only the moved child's pairs and merges at most one pair
    (expNChild = max(C − 1, 2), TH:2301, 1596), then the root greedy continues.

    Root 10: cells 0, 1, 2, internal 8 (cells 3, 4, 5) and 9 (cells 6, 7). Forced scores (1e3, above every other
    pair): the root merges (1, 8) with t_2a = 5e-7, a snap, so 8 collapses into 11 (star 1, 3, 4, 5, special 1); there
    (1, 3) and (1, 4) tie at 1e3, (1, 3) is merged (lower ids) and the greedy stops: the next round is the root's.
    """
    parent = np.array([10, 10, 10, 8, 8, 8, 9, 9, 10, 10, -1])
    rng = np.random.default_rng(7)
    x, var = rng.normal(size=(11, G)), rng.uniform(0.05, 0.3, (11, G))
    tree = _from_tree(parent, np.r_[rng.uniform(0.2, 1.0, 10), 0.0], x, var)
    table = _table(tree, jnp.float32)
    forced = {(1, 8): (0.3, 5e-7, 0.5), (1, 3): (0.3, 0.4, 0.2), (1, 4): (0.3, 0.4, 0.2)}
    real_ref, real_dev = rg.pair_score, eg.pair_score

    def ref_score(x1, v1, t1, x2, v2, t2, xr, Wr, mode="sequential"):
        s = real_ref(x1, v1, t1, x2, v2, t2, xr, Wr, mode)
        for (i, j), t3 in forced.items():
            if np.array_equal(x1, tree.x[i]) and np.array_equal(x2, tree.x[j]):
                return s._replace(dlogl=1e3, t=np.array(t3))
        return s

    def dev_score(*args, **kw):
        s = real_dev(*args, **kw)
        d, t = np.array(s.dlogl), np.array(s.t)
        for (i, j), t3 in forced.items():
            hit = np.all(np.asarray(args[0]) == table.x[i], axis=1) & np.all(np.asarray(args[3]) == table.x[j], axis=1)
            d[hit], t[hit] = 1e3, t3
        return s._replace(dlogl=jnp.asarray(d), t=jnp.asarray(t))

    monkeypatch.setattr(rg, "pair_score", ref_score)
    monkeypatch.setattr(eg, "pair_score", dev_score)
    ref_trace: list = []
    _, ref_events = rg.greedy_star(tree, 10, None, trace=ref_trace)
    eng_trace: list = []
    stats = eg.greedy_star(table, 10, None, True, GreedyConfig(knn=False, lazy=False), trace=eng_trace)
    assert [_key(e) for e in stats.events] == [_key(e) for e in ref_events]
    for events, trace in ((ref_events, ref_trace), (stats.events, eng_trace)):
        assert [_key(e) for e in events[:2]] == [("merge", 1, 8, 11, 0), ("merge", 1, 3, 12, 1)] and events[0].t[1] == 0
        assert events[2].depth == 0  # back at the root (its later rounds may snap again)
        sub = trace[1]  # the one special-child round; the next round is the root's
        assert sub[4] == -1 and trace[2][4] >= 0
        assert set(sub[0]) == {(1, 3), (1, 4), (1, 5)} and set(sub[1]) == {1, 3, 4, 5}
