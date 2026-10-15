"""Newick <-> parent-array conversion (rooted trees, full-precision branch lengths)."""

import re

import numpy as np

_TOKEN = re.compile(r"[(),;]|[^(),;]+")


def parse_newick(s: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse a rooted newick string.

    Returns ``(parent, t, names)``: ``parent[i]`` is the index of node i's parent (-1 at the root),
    ``t[i]`` the branch length above node i (0.0 where absent, always 0.0 at the root) and
    ``names[i]`` its label ("" if unnamed). Nodes are numbered in order of appearance.
    """
    parent: list[int] = []
    t: list[float] = []
    names: list[str] = []
    open_nodes: list[int] = []  # internal nodes whose ')' has not been read yet
    last = -1  # node that the next label/length belongs to
    for tok in _TOKEN.findall(s.strip()):
        if tok == "(":
            parent.append(open_nodes[-1] if open_nodes else -1)
            t.append(0.0)
            names.append("")
            open_nodes.append(len(parent) - 1)
            last = -1
        elif tok == ")":
            if not open_nodes:
                raise ValueError("unbalanced parentheses in newick string: extra ')'")
            last = open_nodes.pop()
        elif tok in ",;":
            last = -1
        else:
            if last < 0:  # a leaf
                parent.append(open_nodes[-1] if open_nodes else -1)
                t.append(0.0)
                names.append("")
                last = len(parent) - 1
            name, _, length = tok.strip().partition(":")
            names[last] = name
            if length:
                t[last] = float(length)
    if open_nodes:
        raise ValueError("unbalanced parentheses in newick string")
    p = np.array(parent, dtype=np.int64)
    if np.count_nonzero(p < 0) != 1:
        raise ValueError("newick string must describe exactly one rooted tree")
    tt = np.array(t, dtype=np.float64)
    tt[p < 0] = 0.0
    return p, tt, names


def children_lists(parent: np.ndarray) -> list[list[int]]:
    """Children of every node, in increasing node index."""
    kids: list[list[int]] = [[] for _ in range(len(parent))]
    for c, p in enumerate(parent):
        if p >= 0:
            kids[p].append(c)
    return kids


def to_newick(parent: np.ndarray, t: np.ndarray, names: list[str]) -> str:
    """Write a rooted tree as newick, lengths with ``repr`` (round-trips float64 exactly)."""
    kids = children_lists(parent)
    root = int(np.flatnonzero(np.asarray(parent) < 0)[0])
    out: list[str] = []
    stack: list[tuple[int, int]] = [(root, 0)]  # (node, next child position)
    while stack:
        n, k = stack.pop()
        if k == 0 and kids[n]:
            out.append("(")
        if k < len(kids[n]):
            if k > 0:
                out.append(",")
            stack.append((n, k + 1))
            stack.append((kids[n][k], 0))
            continue
        if kids[n]:
            out.append(")")
        out.append(names[n] if n == root else f"{names[n]}:{float(t[n])!r}")
    return "".join(out) + ";"
