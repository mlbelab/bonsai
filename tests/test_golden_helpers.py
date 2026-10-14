from .golden import GOLDEN, _parse_newick, load_tree


def test_parse_newick_small():
    clades, lengths = _parse_newick("((a:1,b:2)n1:3,c:4)root;")
    ab, abc = frozenset("ab"), frozenset("abc")
    assert clades == {ab, abc}
    assert lengths == {frozenset("a"): 1.0, frozenset("b"): 2.0, ab: 3.0, frozenset("c"): 4.0}


def test_golden_tree_has_64_leaves_and_binary_internal_nodes():
    t = load_tree(GOLDEN)
    root = max(t.clades, key=len)
    assert len(root) == 64
    assert all(n.startswith("Cell") for n in root)  # internal labels never become leaves
    assert t.loglik == -65745.68334074633
