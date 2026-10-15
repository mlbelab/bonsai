import numpy as np
import pytest

from bonsai._ref.newick import children_lists, parse_newick, to_newick

from ._trees import random_tree
from .golden import GOLDEN


def test_parse_small_tree():
    parent, t, names = parse_newick("((a:1,b:2.5)x:3,c:4)root;")
    assert parent.tolist() == [-1, 0, 1, 1, 0]
    assert t.tolist() == [0.0, 3.0, 1.0, 2.5, 4.0]
    assert names == ["root", "x", "a", "b", "c"]
    assert children_lists(parent) == [[1, 4], [2, 3], [], [], []]


def test_golden_newick_round_trips_byte_for_byte():
    s = (GOLDEN / "tree.nwk").read_text().strip()
    parent, t, names = parse_newick(s)
    assert to_newick(parent, t, names) == s
    assert int((parent < 0).sum()) == 1 and len(parent) == 127  # 64 cells, binary, rooted


def test_random_trees_round_trip_exactly():
    rng = np.random.default_rng(0)
    for max_children in (2, 4):
        parent, t = random_tree(30, rng, max_children=max_children)
        names = [f"n{i}" for i in range(len(parent))]
        p2, t2, n2 = parse_newick(to_newick(parent, t, names))
        # node numbering differs (order of appearance); compare through the names
        idx = np.array([int(v[1:]) for v in n2])
        assert np.array_equal(np.where(p2 >= 0, idx[np.maximum(p2, 0)], -1), parent[idx])
        assert np.array_equal(t2, t[idx])  # repr round-trips float64 exactly


@pytest.mark.parametrize("bad", ["((a,b);", "(a,b);(c,d);"])
def test_malformed_newick_raises(bad):
    with pytest.raises(ValueError):
        parse_newick(bad)
