"""Small random trees and helpers for reference-kernel tests (not a benchmark simulator)."""

import numpy as np


def random_tree(n_leaves: int, rng: np.random.Generator, max_children: int = 2, t_scale: float = 0.5):
    """Random rooted tree: repeatedly join 2..max_children active nodes under a new parent.

    Leaves are 0..n_leaves-1; every parent index is larger than its children's; the last node is
    the root. Returns ``(parent, t)`` with exponential branch lengths (t = 0 at the root).
    """
    parent = [-1] * n_leaves
    active = list(range(n_leaves))
    while len(active) > 1:
        k = int(rng.integers(2, min(max_children, len(active)) + 1))
        pick = sorted(rng.choice(len(active), size=k, replace=False), reverse=True)
        new = len(parent)
        parent.append(-1)
        for i in pick:
            parent[active.pop(i)] = new
        active.append(new)
    p = np.array(parent)
    t = rng.exponential(t_scale, size=len(p))
    t[p < 0] = 0.0
    return p, t


def random_leaf_data(parent: np.ndarray, n_genes: int, rng: np.random.Generator):
    """Node-indexed (x, var) with random leaf rows (internal rows are ignored by the kernels)."""
    n = len(parent)
    x = rng.normal(0.0, 1.0, size=(n, n_genes))
    var = rng.uniform(0.05, 1.0, size=(n, n_genes))
    return x, var


def bm_tree(n_leaves: int, n_genes: int, rng: np.random.Generator):
    """Random binary tree with Brownian leaf data (the simulator's model, small and unscaled).

    Returns node-indexed ``(parent, t, x, var)``: z diffuses with N(parent, t·I) from 0 at the root;
    leaves observe x = z + N(0, var), var ~ U(0.05, 0.3).
    """
    parent, t = random_tree(n_leaves, rng)
    var = rng.uniform(0.05, 0.3, size=(len(parent), n_genes))
    z = np.zeros((len(parent), n_genes))
    for c in range(len(parent) - 2, -1, -1):  # parents have larger indices; the root is last
        z[c] = z[parent[c]] + np.sqrt(t[c]) * rng.normal(size=n_genes)
    return parent, t, z + np.sqrt(var) * rng.normal(size=z.shape), var


def reroot(parent: np.ndarray, t: np.ndarray, new_root: int):
    """Same unrooted tree rooted at ``new_root``: edges on the path to the old root are reversed."""
    p, tt = parent.copy(), t.copy()
    child, up, t_up = new_root, parent[new_root], t[new_root]
    p[new_root], tt[new_root] = -1, 0.0
    while up >= 0:
        nxt, t_nxt = p[up], tt[up]
        p[up], tt[up] = child, t_up
        child, up, t_up = up, nxt, t_nxt
    return p, tt


def leaf_lca_cov(parent: np.ndarray, t: np.ndarray, leaves: np.ndarray) -> np.ndarray:
    """C_ij = summed branch length from the root to LCA(i, j) (Brownian covariance, flat root)."""

    def path(i):
        out = {}
        acc = []
        while i >= 0:
            acc.append(i)
            i = parent[i]
        dist = 0.0
        for node in reversed(acc):  # root first
            dist += t[node] if parent[node] >= 0 else 0.0
            out[node] = dist
        return out

    paths = [path(i) for i in leaves]
    C = np.zeros((len(leaves), len(leaves)))
    for a, pa in enumerate(paths):
        for b, pb in enumerate(paths):
            C[a, b] = max(pa[k] for k in pa.keys() & pb.keys())
    return C
