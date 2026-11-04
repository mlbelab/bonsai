"""Tree-accuracy metrics against a ground-truth tree (amendments A7, A15).

Unrooted quartet accuracy and normalised Robinson–Foulds distance; neither depends on the rooting.
``contract_zero_edges`` turns zero-length internal edges into polytomies before scoring (A15).
"""

from itertools import combinations

import numpy as np

from bonsai._ref.tree import postorder


def _leaf_rows(parent: np.ndarray, names: list[str], leaves: list[str]) -> np.ndarray:
    """Node index of every name in ``leaves`` (must be exactly the tree's leaf set, no duplicates)."""
    _, kids = postorder(parent)
    leaf_idx = [i for i in range(len(parent)) if not kids[i]]
    leaf_names = [names[i] for i in leaf_idx]
    if len(set(leaf_names)) != len(leaf_names):
        raise ValueError("duplicate leaf names in a tree")
    tree_leaves = dict(zip(leaf_names, leaf_idx, strict=True))
    if set(tree_leaves) != set(leaves):
        raise ValueError("the two trees do not have the same leaf names")
    return np.array([tree_leaves[n] for n in leaves])


def _lca(parent: np.ndarray):
    """Vectorised LCA by binary lifting; returns (depth, lca(a, b))."""
    order, _ = postorder(parent)
    root = order[-1]
    depth = np.zeros(len(parent), dtype=np.int64)
    for v in order[::-1][1:]:
        depth[v] = depth[parent[v]] + 1
    levels = max(1, int(depth.max()).bit_length())
    up = [np.where(parent < 0, root, parent)]
    for _ in range(levels):
        up.append(up[-1][up[-1]])

    def lca(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = a.copy(), b.copy()
        swap = depth[a] < depth[b]
        a[swap], b[swap] = b[swap], a[swap]
        diff = depth[a] - depth[b]
        for k in range(levels + 1):
            m = (diff >> k) & 1 == 1
            a[m] = up[k][a[m]]
        for k in reversed(range(levels + 1)):
            m = up[k][a] != up[k][b]
            a[m], b[m] = up[k][a[m]], up[k][b[m]]
        return np.where(a == b, a, up[0][a])

    return depth, lca


def _quartet_split(parent: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per quartet (a, b, c, d) of node rows: 0 if ab|cd, 1 if ac|bd, 2 if ad|bc, -1 unresolved.

    Four-point condition on edge-count path lengths d(u, v) = depth u + depth v − 2·depth LCA: the
    split pairs the leaves whose two distances sum to the strict minimum. A root (or unary node)
    subdivides an edge, which adds equally to all competing sums or only to the losing ones, so
    the result does not depend on the rooting.
    """
    depth, lca = _lca(parent)

    def dist(i, j):
        u, v = rows[:, i], rows[:, j]
        return depth[u] + depth[v] - 2 * depth[lca(u, v)]

    s = np.stack([dist(0, 1) + dist(2, 3), dist(0, 2) + dist(1, 3), dist(0, 3) + dist(1, 2)], axis=1)
    unique = (s == s.min(axis=1)[:, None]).sum(axis=1) == 1
    return np.where(unique, s.argmin(axis=1), -1)


def quartet_accuracy(tree_a, tree_b, n_samples: int = 100_000, seed: int = 0) -> tuple[float, int]:
    """Mean quartet score of ``tree_b`` against ``tree_a`` (amendment A15).

    Per quartet: 1 if both trees give the same unrooted split or both leave it unresolved, 1/3 if
    exactly one leaves it unresolved, 0 otherwise. ``tree_*`` = (parent, names). All C(N, 4)
    quartets if that is ≤ ``n_samples``, else ``n_samples`` uniformly random draws, keeping those
    of four distinct leaves. Returns (accuracy, n_quartets).
    """
    leaves = sorted(n for n, k in zip(tree_a[1], postorder(tree_a[0])[1], strict=True) if not k)
    ra = _leaf_rows(tree_a[0], tree_a[1], leaves)
    rb = _leaf_rows(tree_b[0], tree_b[1], leaves)
    n = len(leaves)
    if n * (n - 1) * (n - 2) * (n - 3) // 24 <= n_samples:
        quads = np.array(list(combinations(range(n), 4)), dtype=np.int64).reshape(-1, 4)
    else:
        quads = np.random.default_rng(seed).integers(0, n, size=(n_samples, 4))
        quads = quads[np.array([len(set(q)) == 4 for q in quads.tolist()], dtype=bool)]
    if len(quads) == 0:
        raise ValueError("no quartets: fewer than 4 leaves, or every sampled draw was rejected")
    sa, sb = _quartet_split(tree_a[0], ra[quads]), _quartet_split(tree_b[0], rb[quads])
    return float(np.where(sa == sb, 1.0, np.where((sa < 0) ^ (sb < 0), 1 / 3, 0.0)).mean()), len(quads)


def contract_zero_edges(
    parent: np.ndarray, t: np.ndarray, names: list[str], snap: float = 0.0
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Remove every non-root internal node with ``t < snap`` or ``t == 0`` and compact (amendment A15).

    The removed node's children are re-attached to its nearest kept ancestor with their own t.
    Leaves (cells) and the root are never removed. Returns the compacted (parent, t, names); kept
    nodes keep their relative order.
    """
    parent, t = np.asarray(parent, dtype=np.int64), np.asarray(t, dtype=np.float64)
    _, kids = postorder(parent)
    drop = np.array([p >= 0 and bool(k) and (ti < snap or ti == 0) for p, k, ti in zip(parent, kids, t, strict=True)])
    up = parent.copy()  # nearest kept ancestor; ancestors precede their descendants in pre-order
    for v in postorder(parent)[0][::-1]:
        if up[v] >= 0 and drop[up[v]]:
            up[v] = up[up[v]]
    keep = np.flatnonzero(~drop)
    new = np.full(len(parent), -1, dtype=np.int64)
    new[keep] = np.arange(len(keep))
    pk = up[keep]
    return np.where(pk >= 0, new[np.maximum(pk, 0)], -1), t[keep], [names[i] for i in keep]


def _splits(parent: np.ndarray, rows: np.ndarray) -> set[int]:
    """64-bit hashes of the non-trivial unrooted splits (canonical side: without leaf ``rows[0]``)."""
    order, _ = postorder(parent)
    h = np.zeros(len(parent), dtype=np.uint64)
    size = np.zeros(len(parent), dtype=np.int64)
    has0 = np.zeros(len(parent), dtype=bool)
    h[rows] = np.random.default_rng(12345).integers(1, 2**63, size=len(rows), dtype=np.uint64)
    size[rows] = 1
    has0[rows[0]] = True
    for v in order[:-1]:  # children before parents; the root is last
        p = parent[v]
        h[p] ^= h[v]
        size[p] += size[v]
        has0[p] |= has0[v]
    total, n = h[order[-1]], len(rows)
    # The root always has size == n, already excluded by size <= n - 2, so no separate root check.
    nontrivial = (size >= 2) & (size <= n - 2)
    return {int(np.where(has0[v], h[v] ^ total, h[v])) for v in np.flatnonzero(nontrivial)}


def rf_distance(tree_a, tree_b) -> float:
    """Normalised Robinson–Foulds distance of the unrooted trees: |A Δ B| / (|A| + |B|)."""
    leaves = sorted(n for n, k in zip(tree_a[1], postorder(tree_a[0])[1], strict=True) if not k)
    a = _splits(tree_a[0], _leaf_rows(tree_a[0], tree_a[1], leaves))
    b = _splits(tree_b[0], _leaf_rows(tree_b[0], tree_b[1], leaves))
    return len(a ^ b) / (len(a) + len(b)) if (a or b) else 0.0
