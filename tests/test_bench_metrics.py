import collections
from itertools import combinations, permutations

import numpy as np
import pytest
from bench import metrics as _metrics
from bench.metrics import quartet_accuracy, rf_distance

from bonsai._ref.newick import parse_newick
from bonsai._ref.tree import postorder

from ._trees import random_tree, reroot


def _tree(s):
    parent, _, names = parse_newick(s)
    return parent, names


def test_four_leaf_examples():
    a, b = _tree("((a,b),(c,d));"), _tree("((a,c),(b,d));")
    assert quartet_accuracy(a, a) == (1.0, 1) and rf_distance(a, a) == 0.0
    assert quartet_accuracy(a, b) == (0.0, 1)  # ab|cd vs ac|bd
    assert rf_distance(a, b) == 1.0
    assert quartet_accuracy(a, _tree("(a,(b,(c,d)));")) == (1.0, 1)  # same unrooted tree, other root


def test_unresolved_quartets_count_only_if_unresolved_in_both():
    star, resolved = _tree("(a,b,c,d);"), _tree("((a,b),c,d);")
    assert quartet_accuracy(star, star) == (1.0, 1)
    assert quartet_accuracy(star, resolved) == (0.0, 1)


def _named(parent):
    n_leaves = (len(parent) + 1) // 2
    return [f"c{i}" if i < n_leaves else f"i{i}" for i in range(len(parent))]


def test_invariances():
    """Both metrics ignore node numbering and the root position (amendment A7)."""
    rng = np.random.default_rng(0)
    parent, t = random_tree(30, rng)
    names = _named(parent)
    perm = rng.permutation(len(parent))
    inv = np.argsort(perm)
    p2 = np.where(parent[perm] >= 0, inv[np.maximum(parent[perm], 0)], -1)
    assert quartet_accuracy((parent, names), (p2, [names[k] for k in perm]), n_samples=5000)[0] == 1.0
    assert rf_distance((parent, names), (p2, [names[k] for k in perm])) == 0.0
    p3, _ = reroot(parent, t, 45)  # rerooting leaves a unary node behind
    assert quartet_accuracy((parent, names), (p3, names), n_samples=5000)[0] == 1.0
    assert rf_distance((parent, names), (p3, names)) == 0.0


def test_sampled_quartets_estimate_the_exact_value():
    """Sampled accuracy is a binomial mean: within 4·sqrt(p(1−p)/n) of the exact enumeration."""
    rng = np.random.default_rng(1)
    a, _ = random_tree(20, rng)
    b, _ = random_tree(20, rng)
    names = _named(a)
    exact, n_all = quartet_accuracy((a, names), (b, names))
    assert n_all == 20 * 19 * 18 * 17 // 24
    est, n = quartet_accuracy((a, names), (b, names), n_samples=2000, seed=3)
    assert abs(est - exact) <= 4 * np.sqrt(exact * (1 - exact) / n)


def test_leaf_sets_must_match():
    with pytest.raises(ValueError, match="same leaf names"):
        rf_distance(_tree("((a,b),(c,e));"), _tree("((a,b),(c,d));"))


def test_duplicate_leaf_names_rejected():
    with pytest.raises(ValueError, match="duplicate leaf names"):
        rf_distance(_tree("((a,a),(c,d));"), _tree("((a,a),(c,d));"))


def test_too_few_or_no_quartets_raises():
    with pytest.raises(ValueError, match="no quartets"):
        quartet_accuracy(_tree("(a,b,c);"), _tree("(a,b,c);"))  # N < 4
    four = _tree("((a,b),(c,d));")
    with pytest.raises(ValueError, match="no quartets"):
        quartet_accuracy(four, four, n_samples=0)  # every sampled draw rejected (none drawn)


def test_rf_fractional_value():
    """Hand-computed: both trees split {a,b}|{c,d,e} vs {a,c}|{b,d,e} (unique to each) but share
    {d,e}|{a,b,c}, so |AΔB| = 2 and |A|+|B| = 4 -> RF = 0.5."""
    a, b = _tree("((a,b),(c,(d,e)));"), _tree("((a,c),(b,(d,e)));")
    assert rf_distance(a, b) == 0.5


def test_rf_with_leaf_child_of_root():
    """A root with a leaf child (no bifurcating pair of internal subtrees) exercises the
    non-trivial-split size bound ``size <= n - 2``: the sibling of the root's leaf child has size
    n - 1, which must be excluded (it is the complement of the leaf's own trivial split), or the
    extra, trivial split silently appears only on one side and biases RF.

    ``(a,(b,(c,(d,e))))`` is, read as an unrooted tree, the caterpillar cherry(a,b)-c-cherry(d,e)
    (node1 = (b, (c,(d,e))) is adjacent to a, b and node2, so {a, b} already form a cherry there),
    which is exactly ``((a,b),(c,(d,e)))`` rooted elsewhere: both trees have unrooted splits
    {a,b}|{c,d,e} and {d,e}|{a,b,c}, so the hand-computed RF is 0.
    """
    rooted_leaf_child = _tree("(a,(b,(c,(d,e))));")
    balanced = _tree("((a,b),(c,(d,e)));")
    assert rf_distance(rooted_leaf_child, balanced) == 0.0


def _brute_splits(parent: np.ndarray, names: list[str], ref_leaf: str) -> set[frozenset]:
    """Reference splits: leaf-name sets below each edge, no hashing, canonicalised to the side
    without ``ref_leaf``."""
    order, kids = postorder(parent)
    leaf_set: dict[int, frozenset] = {}
    for v in order:  # children before parents
        leaf_set[v] = frozenset([names[v]]) if not kids[v] else frozenset().union(*(leaf_set[c] for c in kids[v]))
    all_leaves = leaf_set[order[-1]]
    n = len(all_leaves)
    out = set()
    for v in range(len(parent)):
        if parent[v] < 0:
            continue
        s = leaf_set[v]
        if 2 <= len(s) <= n - 2:
            out.add(s if ref_leaf not in s else all_leaves - s)
    return out


def _brute_rf(tree_a, tree_b) -> float:
    (pa, na), (pb, nb) = tree_a, tree_b
    leaves = sorted(n for n, k in zip(na, postorder(pa)[1], strict=True) if not k)
    a, b = _brute_splits(pa, na, leaves[0]), _brute_splits(pb, nb, leaves[0])
    return len(a ^ b) / (len(a) + len(b)) if (a or b) else 0.0


def _brute_quartet_split(parent: np.ndarray, names: list[str], quad: tuple[str, str, str, str]) -> int:
    """Reference 4-point rule: BFS shortest paths on the tree's undirected edges (no LCA/depth)."""
    adj: dict[int, list[int]] = collections.defaultdict(list)
    for v, p in enumerate(parent):
        if p >= 0:
            adj[v].append(p)
            adj[p].append(v)

    def dist(u, v):
        seen, q = {u}, collections.deque([(u, 0)])
        while q:
            x, d = q.popleft()
            if x == v:
                return d
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append((y, d + 1))
        raise AssertionError("disconnected tree")

    idx = {n: i for i, n in enumerate(names) if n}
    rows = [idx[x] for x in quad]
    s = [
        dist(rows[0], rows[1]) + dist(rows[2], rows[3]),
        dist(rows[0], rows[2]) + dist(rows[1], rows[3]),
        dist(rows[0], rows[3]) + dist(rows[1], rows[2]),
    ]
    m = min(s)
    return -1 if s.count(m) > 1 else s.index(m)


def test_brute_force_cross_check():
    """Independent reference (leaf-set splits; BFS-path quartets, no hashing/LCA) against the real
    implementation, over a few dozen small random trees, some rerooted. Under 1s."""
    rng = np.random.default_rng(7)
    for trial in range(30):
        n_leaves = int(rng.integers(4, 9))
        parent, t = random_tree(n_leaves, rng, max_children=int(rng.integers(2, 5)))
        names = [f"c{i}" if i < n_leaves else f"i{i}" for i in range(len(parent))]
        if trial % 2 == 0 and len(parent) > n_leaves:
            parent, t = reroot(parent, t, int(rng.integers(n_leaves, len(parent))))  # an internal node
        parent2, t2 = random_tree(n_leaves, rng, max_children=int(rng.integers(2, 5)))
        names2 = [f"c{i}" if i < n_leaves else f"j{i}" for i in range(len(parent2))]
        tree_a, tree_b = (parent, names), (parent2, names2)

        assert rf_distance(tree_a, tree_b) == pytest.approx(_brute_rf(tree_a, tree_b))

        leaves = sorted(names[:n_leaves])
        rows_a = _metrics._leaf_rows(parent, names, leaves)
        for quad in combinations(leaves, 4):
            rows = rows_a[[leaves.index(x) for x in quad]].reshape(1, 4)
            got = int(_metrics._quartet_split(parent, rows)[0])
            assert got == _brute_quartet_split(parent, names, quad)


def test_random_relabelling_quartet_accuracy_is_exactly_one_third():
    """Averaged over every leaf-label permutation, quartet accuracy against a fixed binary tree is
    exactly 1/3 (amendment A7's "about 1/3" requirement, made exact instead of sampled).

    Fixed binary 6-leaf tree, so every quartet is resolved (no ties) in both trees and all
    C(6, 4) = 15 quartets are scored exactly (``quartet_accuracy``'s own exact-enumeration path,
    since 15 <= the default ``n_samples``). For a fixed quartet of leaves, as the label
    permutation ranges uniformly over all 6! = 720 relabellings, the three possible leaf pairings
    of that quartet occur equally often (exchangeability of labels under a uniform permutation),
    so exactly 1/3 of the 720 permutations keep that quartet's topology unchanged. Averaging that
    exact 1/3 over all 15 quartets, and then over all 720 permutations, is still exactly 1/3 - this
    uses the real per-quartet topology function (``quartet_accuracy``), not a re-implementation.

    Checked as an exact integer count rather than a float tolerance: each permutation's accuracy is
    (matching quartets)/15, so the total number of matches over all 720 permutations must be exactly
    720 * 15 / 3 = 3600.
    """
    parent, _, names = parse_newick("(((a,b),c),((d,e),f));")
    leaf_pos = [i for i, n in enumerate(names) if n]
    leaf_names = [names[i] for i in leaf_pos]
    total_matches = 0
    for perm in permutations(leaf_names):
        relabelled = names.copy()
        for pos, label in zip(leaf_pos, perm, strict=True):
            relabelled[pos] = label
        acc, n = quartet_accuracy((parent, names), (parent, relabelled))
        assert n == 15
        total_matches += round(acc * 15)
    assert total_matches == 3600
