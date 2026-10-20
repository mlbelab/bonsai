"""Derived rounding-error bounds for comparing JAX kernels with the fp64 references.

Each bound is first order: 8 (an upper bound on the rounded operations feeding one per-gene term)
× eps × the term's absolute size including its conditioning. float64 message errors grow at most
additively per tree level (messages are convex combinations), hence the (1 + height) factors.
"""

import numpy as np

from bonsai._ref.tree import postorder

EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)


def heights(parent: np.ndarray) -> np.ndarray:
    """Per node: edges on the longest path down to a leaf (0 at leaves)."""
    order, kids = postorder(parent)
    height = np.zeros(len(parent), dtype=np.int64)
    for v in order:
        if kids[v]:
            height[v] = 1 + max(height[c] for c in kids[v])
    return height


def tree_height(parent: np.ndarray) -> int:
    """Edges on the longest root-to-leaf path."""
    return int(heights(parent).max())


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


def tree_tols(parent: np.ndarray, t: np.ndarray, ref, eps: float) -> tuple[float, np.ndarray]:
    """(tol_L, tol_grad per node) for a tree pass that runs entirely at precision ``eps`` (A8).

    ``ref`` is the fp64 ``TreeGrad`` on the same (already rounded) inputs. Messages are sums of
    positive weights (and of w·x) without subtraction, so per level they gain at most C + 3
    roundings (C = most children of a node; var→1/W, +t, 1/·, C−1 additions): relative error of
    w, W below height h is ρ(h) = (h + 1)(C + 3)·eps; message means are convex combinations, so
    their absolute error is ρ(h)·X with X = max|x_leaf|. Per gene term (≤ 8 own roundings, +1 for
    the compensated gene sum):
      ell:  ρ + 9·eps·(1 + |log w|) + w·(4|r|·ρX + r²(ρ + 9·eps)), plus ρ + 9·eps·(1 + |log W|)
            per parent;
      grad: the pre-order "outside" messages add up to H more levels, ρ_g = (2H + 1)(C + 3)·eps:
            (1/T + 2d²/T²)·ρ_g + 2|d|·2ρ_g·X/T² + 9·eps·(1/T + d²/T²) per gene.
    """
    height = heights(parent)
    H = int(height.max())
    kids = postorder(parent)[1]
    C = max(len(k) for k in kids)
    leaves = np.array([not k for k in kids])
    X = np.abs(ref.x[leaves]).max()
    c = np.flatnonzero(parent >= 0)
    p = parent[c]
    rho = ((height[p] + 1) * (C + 3) * eps)[:, None]
    w = 1.0 / (1.0 / ref.W[c] + t[c][:, None])
    r = ref.x[p] - ref.x[c]
    parents = np.unique(p)
    rho_p = ((height[parents] + 1) * (C + 3) * eps)[:, None]
    logW = np.abs(np.log(ref.W[parents]))
    tol_L = np.sum(rho + 9 * eps * (1 + np.abs(np.log(w))) + w * (4 * np.abs(r) * rho * X + r**2 * (rho + 9 * eps)))
    tol_L += np.sum(rho_p + 9 * eps * (1 + logW))
    rho_g = (2 * H + 1) * (C + 3) * eps
    T = t[c][:, None] + 1.0 / ref.W[c] + 1.0 / ref.W_ex[c]
    d = ref.x_ex[c] - ref.x[c]
    tol_g = np.zeros(len(parent))
    tol_g[c] = np.sum(
        (1 / T + 2 * d**2 / T**2) * rho_g + 4 * np.abs(d) * rho_g * X / T**2 + 9 * eps * (1 / T + d**2 / T**2), axis=1
    )
    return float(tol_L), tol_g


def input_rounding(parent: np.ndarray, t: np.ndarray, var: np.ndarray, ref, eps: float) -> float:
    """First-order change of L_tree when every input (x, sigma², t) is rounded to precision ``eps``.

    From the 2-leaf decomposition at a leaf c (maths ref §4): L = const − Σ_g [log T + d²/T],
    T = sigma²_c + t_c + 1/W_{\\c}, d = x_c − x_{\\c}, so |dL/dx_c| = 2|d|/T, |dL/dsigma²_c| =
    |(−1 + d²/T)/T| per gene; dL/dt_e is the gradient. Uses the (raw, uncentred) leaf x of ``ref``.
    """
    leaves = np.flatnonzero([not k for k in postorder(parent)[1]])
    T = t[leaves][:, None] + var[leaves] + 1.0 / ref.W_ex[leaves]
    d = ref.x[leaves] - ref.x_ex[leaves]
    dx = np.sum(2 * np.abs(d) / T * np.abs(ref.x[leaves]))
    dv = np.sum(np.abs((-1 + d**2 / T) / T) * var[leaves])
    return float(eps * (dx + dv + np.sum(np.abs(ref.grad * t))))
