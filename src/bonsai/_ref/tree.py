"""fp64 NumPy reference: Bonsai tree log-likelihood (maths ref §2, §4; the gradient arrives in Task 8).

Convention (amendment A1): upstream's ``finalLoglik`` units, i.e. the sum over internal nodes of
ell_n (TH:3383-3392) = 2 x Gaussian log marginal likelihood + (N-1)·G·log(2π), with no
``loglikVarCorr``. All arrays are node-indexed: ``x``, ``var`` have shape (n_nodes, G) and only
their leaf rows are read; ``t[i]`` is the length of the edge above node i (ignored at the root).
"""

from typing import NamedTuple

import numpy as np

from .newick import children_lists


def postorder(parent: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Node order with every child before its parent (reversed BFS from the root), and children."""
    parent = np.asarray(parent)
    roots = np.flatnonzero(parent < 0)
    if len(roots) != 1:
        raise ValueError(f"tree must have exactly one root, found {len(roots)}")
    kids = children_lists(parent)
    order = [int(roots[0])]
    i = 0
    while i < len(order):
        order.extend(kids[order[i]])
        i += 1
    if len(order) != len(parent):
        raise ValueError("parent array contains a cycle or nodes unreachable from the root")
    return np.array(order[::-1]), kids


def _upward(parent, t, x, var):
    """Post-order pass (TH:629-653, 5257-5294): node messages (x_n, W_n) and L_tree.

    ponytail: allocates and copies full (2N-1, G) arrays even though only internal rows are
    written; `tests/_bounds.py:loglik_scale` indexes the returned X/W by both leaf and internal
    node id, so shrinking this to internal-only storage means changing that contract too. At
    100k cells and ~2k genes used (the M0 target, A-3), the caller's own node-indexed x/var plus
    this function's X/V copies are four float64 (2N-1, G) ~= (2e5, 2e3) arrays at ~3.2 GB each,
    about 13 GB total. Upgrade path (M4): a node-message type (leaf view + internal array) once
    that caller is revisited for scale.
    """
    order, kids = postorder(parent)
    t = np.asarray(t, dtype=np.float64)
    X = np.array(x, dtype=np.float64)
    V = np.array(var, dtype=np.float64)  # node variance 1/W (sigma^2 at leaves)
    L = 0.0
    for n in order:
        ch = kids[n]
        if not ch:
            continue
        w = 1.0 / (V[ch] + t[ch][:, None])  # w_c = 1/(sigma^2_c + t_c), (C, G)
        Wn = w.sum(axis=0)
        xn = (w * X[ch]).sum(axis=0) / Wn
        # ell_n = Σ_g [Σ_c log w_c − log W − Σ_c w_c (x − x_c)^2]  (TH:5093-5105)
        L += float(np.sum(np.log(w).sum(axis=0) - np.log(Wn) - (w * (xn - X[ch]) ** 2).sum(axis=0)))
        X[n], V[n] = xn, 1.0 / Wn
    return L, X, V


def tree_loglik(parent: np.ndarray, t: np.ndarray, x: np.ndarray, var: np.ndarray) -> float:
    """L_tree in the A1 convention (fp64)."""
    return _upward(parent, t, x, var)[0]


class TreeGrad(NamedTuple):
    L: float  # L_tree, A1 convention
    grad: np.ndarray  # (n,) dL/dt per node's parent edge; 0 at the root
    x: np.ndarray  # (n, G) downward messages (leaf rows = data)
    W: np.ndarray  # (n, G) downward precisions (leaf rows = 1/sigma^2)
    x_ex: np.ndarray  # (n, G) parent's AIR message without this child: x_{\c}; root row = 0
    W_ex: np.ndarray  # (n, G) W_{\c}; root row = 0
    x_air: np.ndarray  # (n, G) as-if-root message of every node
    W_air: np.ndarray  # (n, G)


def tree_loglik_grad(parent: np.ndarray, t: np.ndarray, x: np.ndarray, var: np.ndarray) -> TreeGrad:
    r"""L_tree and dL/dt by the pre-order pass of ``getDerivativesDownstream`` (TH:594-618).

    A unary root raises: the root's own message is its AIR (TH:3414-3430), so its one child's
    W_{\c} = W_root − w_child is exactly 0 and the gradient is undefined (divide by zero; not a
    fidelity break, upstream hits the same division).
    """
    L, X, V = _upward(parent, t, x, var)
    order, kids = postorder(parent)
    root = int(order[-1])
    if len(kids[root]) == 1:
        raise ValueError("tree_loglik_grad: a unary root makes the gradient undefined (W_{\\c} = 0)")
    t = np.asarray(t, dtype=np.float64)
    W = 1.0 / V
    n, G = X.shape
    grad = np.zeros(n)
    x_ex, W_ex = np.zeros((n, G)), np.zeros((n, G))
    x_air, W_air = X.copy(), W.copy()  # root AIR = its own message (TH:3414-3430)
    for p in order[::-1]:  # parents before children
        for c in kids[p]:
            w = 1.0 / (t[c] + V[c])
            Wx = W_air[p] - w  # W_{\c} = W_A − w_c
            xx = (x_air[p] * W_air[p] - w * X[c]) / Wx
            T = t[c] + V[c] + 1.0 / Wx
            grad[c] = np.sum((-1.0 + (xx - X[c]) ** 2 / T) / T)  # der2LeafTree (TH:4927-4930)
            wt = 1.0 / (t[c] + 1.0 / Wx)
            W_air[c] = W[c] + wt
            x_air[c] = (X[c] * W[c] + wt * xx) / W_air[c]
            x_ex[c], W_ex[c] = xx, Wx
    return TreeGrad(L, grad, X, W, x_ex, W_ex, x_air, W_air)
