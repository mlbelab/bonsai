"""Derived rounding-error bounds for comparing JAX kernels with the fp64 references.

Each bound is first order: 8 (an upper bound on the rounded operations feeding one per-gene term)
× eps × the term's absolute size including its conditioning. float64 message errors grow at most
additively per tree level (messages are convex combinations), hence the (1 + height) factors.
"""

import numpy as np

from bonsai._ref.tree import postorder

EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)


def tree_height(parent: np.ndarray) -> int:
    """Edges on the longest root-to-leaf path."""
    order, kids = postorder(parent)
    depth = np.zeros(len(parent), dtype=np.int64)
    for v in order[::-1]:
        for c in kids[v]:
            depth[c] = depth[v] + 1
    return int(depth.max())


def loglik_scale(parent: np.ndarray, t: np.ndarray, X: np.ndarray, W: np.ndarray) -> float:
    """A = Σ_edges,g (1 + |log w_c| + w_c (x_p − x_c)²) + Σ_parents,g (1 + |log W_p|).

    The absolute size of every per-gene term of ell_n (TH:5093-5105), from the node messages
    (X, W) of the fp64 reference (leaf rows: data and 1/sigma²).
    """
    c = np.flatnonzero(parent >= 0)
    p = parent[c]
    w = 1.0 / (1.0 / W[c] + t[c][:, None])
    parents = np.unique(p)
    return float(np.sum(1 + np.abs(np.log(w)) + w * (X[p] - X[c]) ** 2) + np.sum(1 + np.abs(np.log(W[parents]))))


def rounded(a, dtype=np.float32) -> np.ndarray:
    """``a`` rounded to ``dtype`` and widened back to float64: the inputs a float32 kernel sees."""
    return np.asarray(a, dtype=np.float64).astype(dtype).astype(np.float64)
