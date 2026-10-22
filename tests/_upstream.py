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


def upstream_merge_children(x, var, t, mode: str = "sequential") -> list[tuple[str, frozenset, frozenset]]:
    """Upstream's greedy decisions at a root with C leaf children (``mergeChildrenUB``, TH:1039-1620).

    ``x``, ``var``: (C, G) leaves (cells), ``t``: (C,) their times. Returns, per decision,
    ("merge" | "reopt" | "reset", leaf ids under child 1, under child 2); empty sets for "reopt".
    Recorded by wrapping ``TreeNode.mergeNodes``, ``optimiseConnected`` and ``resetChildTimes`` (test
    side; ``_core`` untouched). With C ≤ 10 upstream uses neither bounds nor kNN (``nChildUB``,
    ``nChildNN``), so every round scores all pairs.
    """
    C, G = x.shape
    decisions: list[tuple[str, frozenset, frozenset]] = []
    orig = {name: getattr(TreeNode, name) for name in ("mergeNodes", "optimiseConnected", "resetChildTimes")}

    def leaves(node) -> frozenset:
        return frozenset([node.nodeInd]) if node.isLeaf else frozenset().union(*map(leaves, node.childNodes))

    def pair(kind):
        def wrapped(self, xr, Wr, i1, i2, *args, **kwargs):
            kids = {c.nodeInd: c for c in self.childNodes}
            decisions.append((kind, leaves(kids[i1]), leaves(kids[i2])))
            return orig["mergeNodes" if kind == "merge" else "resetChildTimes"](self, xr, Wr, i1, i2, *args, **kwargs)

        return wrapped

    def reopt(self, *args, **kwargs):
        decisions.append(("reopt", frozenset(), frozenset()))
        return orig["optimiseConnected"](self, *args, **kwargs)

    try:
        TreeNode.mergeNodes, TreeNode.resetChildTimes = pair("merge"), pair("reset")  # type: ignore[method-assign]
        TreeNode.optimiseConnected = reopt  # type: ignore[method-assign]
        root = _tree(
            np.r_[np.full(C, C), -1], np.r_[t, 0.0], np.vstack([x, np.zeros(G)]), np.vstack([var, np.ones(G)])
        ).root
        assert root is not None
        for child in root.childNodes:
            child.isCell = True  # type: ignore[assignment]  # leaves are cells (never add-as-child ancestors)
        root.getLtqsComplete(mem_friendly=False)
        bs_glob.max_node_ind = C
        root.mergeChildrenUB(
            root.ltqs,
            root.getW(),
            sequential=mode == "sequential",
            ellipsoidSize=1.0,
            nChildNN=50,
            kNN=10,
            mergeDownstream=False,
            singleProcess=True,
        )
    finally:
        for name, fn in orig.items():
            setattr(TreeNode, name, fn)
        bs_glob.reset()
    return decisions
