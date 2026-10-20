"""Run upstream (src/bonsai/_core, b16ef6f) code paths on node-indexed arrays, for parity tests."""

from pathlib import Path

import numpy as np

from bonsai._core import globals as bs_glob
from bonsai._core.dataprocessing import read_and_filter
from bonsai._core.mpi_wrapper import get_mpi_info
from bonsai._core.tree_helpers import Tree, TreeNode


def upstream_leaves(
    data_folder: Path,
    means: str,
    stds: str | None,
    sanity: bool,
    cutoff: float = 1.0,
    rescale_by_var: bool = True,
    all_genes: bool = False,
):
    """Upstream's filtered, prior-removed, rescaled leaf arrays (read_and_filter + SCData.__init__ rescale).

    Returns (x (N, G), var (N, G), gene_var (G,), kept gene indices (G,)), cells in file order.
    """
    x, var, gv, _, _, keep, _, _ = read_and_filter(
        str(data_folder),
        means,
        stds,
        sanity,
        -1 if all_genes else cutoff,
        get_mpi_info(singleProcess=True),
        all_genes=all_genes,
    )
    if rescale_by_var:  # DP:205-214
        x /= np.sqrt(gv[:, None])
        var /= gv[:, None]
    else:  # DP:221-230
        s = np.mean(gv)
        x /= np.sqrt(s)
        var /= s
    return x.T.copy(), var.T.copy(), gv, np.asarray(keep)


def _tree(parent, t, x, var) -> Tree:
    n = len(parent)
    nodes = [TreeNode(nodeInd=i, nodeId=str(i)) for i in range(n)]
    tree = Tree()
    for i, p in enumerate(parent):
        if p < 0:
            nodes[i].isRoot = True  # type: ignore[assignment]  # upstream TreeNode fields are typed None
            tree.root = nodes[i]  # type: ignore[assignment]  # upstream Tree.root is typed None
        else:
            nodes[i].tParent = float(t[i])  # type: ignore[assignment]  # upstream TreeNode fields are typed None
            nodes[i].parentNode = nodes[p]
            nodes[p].childNodes.append(nodes[i])
    for node in nodes:
        node.isLeaf = not node.childNodes  # type: ignore[assignment]  # upstream TreeNode fields are typed None
        if node.isLeaf:
            node.ltqs = np.array(x[node.nodeInd], dtype=np.float64)
            node.setLtqsVarsOrW(ltqsVars=np.array(var[node.nodeInd], dtype=np.float64))
    bs_glob.nGenes = x.shape[1]
    return tree


def upstream_loglik(parent, t, x, var) -> float:
    """Tree.calcLogLComplete(loglikVarCorr=None): upstream's finalLoglik quantity (TH:3383-3392)."""
    try:
        return float(_tree(parent, t, x, var).calcLogLComplete(mem_friendly=False))
    finally:
        bs_glob.reset()


def upstream_opt_times(parent, t, x, var, tol: float = 1e-6, maxiter: int = 100) -> tuple[np.ndarray, int]:
    """Tree.optTimes (TH:3444-3484, incl. the 2e-6 zero snap) on a fixed topology.

    Returns (node-indexed t, scipy iterations); iterations < maxiter means scipy stopped on its own
    ftol/pgtol rule, not on the cap (optTimes exposes no other status).
    """
    try:
        tree = _tree(parent, t, x, var)
        opt, nit = tree.optTimes(
            singleProcess=True, mem_friendly=False, tol=tol, maxiter=maxiter, return_diagnostics=True
        )
        out = np.zeros(len(parent))
        for i, ti in opt.items():
            out[i] = ti
        return out, int(nit)
    finally:
        bs_glob.reset()


def upstream_grad(parent, t, x, var) -> tuple[float, np.ndarray]:
    """Tree.getLikelihoodGradGivenTimes (TH:3414-3430): (L_tree, dL/dt node-indexed, 0 at the root)."""
    try:
        tree = _tree(parent, t, x, var)
        inds = [i for i in range(len(parent)) if parent[i] >= 0]
        L, g = tree.getLikelihoodGradGivenTimes(np.asarray(t)[inds], inds, mem_friendly=False)
        out = np.zeros(len(parent))
        out[inds] = g
        return float(L), out
    finally:
        bs_glob.reset()
