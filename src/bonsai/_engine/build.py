"""Greedy tree builder: star EM, root greedy, redo starry (maths ref §8, §9, §11; plan M2 Task 7; A18, A21b).

Private entry point ``greedy_tree`` of the M2 engine. It reproduces the semantics of the fp64 reference
``bonsai._ref.greedy.greedy_tree`` (itself re-implementing upstream b16ef6f's ``mergeChildrenRecursive``,
TH:968-1012, and ``mergeZeroTimeChilds``, TH:789-807, as the M2 engine defines them) with the incremental
message updates of ``greedy.NodeTable`` instead of from-scratch recomputation. Units: 2x log-likelihoods (A1).
"""

import dataclasses
import time
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bonsai._ref import star as ref_star
from bonsai._ref.newick import to_newick

from .array_tree import _bucket
from .greedy import STAR_MAX_SWEEPS, GreedyConfig, GreedyStats, NodeTable, greedy_star, max_passes
from .host_acc import ParentAcc
from .star import star_em

__all__ = ["BuiltTree", "collapse_zero_edges", "greedy_tree", "redo_starry"]


class BuiltTree(NamedTuple):
    parent: np.ndarray  # compact: leaves 0..N−1 in input order, internal nodes (increasing engine id), root last
    t: np.ndarray  # float64, the edge above each node (0 at the root)
    names: list[str]  # cells, "internal_<i>", "root"
    stage_newick: dict[str, str]  # "greedy", "redo_starry" (``to_newick``, full precision)
    stats: dict  # GreedyStats totals and "events", "passes", "zero_edges_collapsed", "stage_s"


def _children(parent: np.ndarray, n: int) -> list[list[int]]:
    """Children of the first ``n`` ids, increasing id (one O(n) pass)."""
    kids: list[list[int]] = [[] for _ in range(n)]
    for c in np.flatnonzero(parent[:n] >= 0):
        kids[parent[c]].append(int(c))
    return kids


def _postorder(kids: list[list[int]], root: int) -> list[int]:
    """Every node below ``root`` before its parent (iterative; trees can be 1e5 deep)."""
    order, stack = [], [root]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(kids[n])
    return order[::-1]


def _collapse(parent: np.ndarray, t: np.ndarray, is_cell: np.ndarray, root: int, n: int) -> list[int]:
    """``mergeZeroTimeChilds`` (TH:789-807) on ``parent`` in place; returns the live nodes that gained children.

    Children first, an internal non-cell node with t == 0 is removed (parent −2) and its children move to its
    parent. Messages stay valid: a zero edge passes the child's message unchanged (w = W_c).
    """
    kids = _children(parent, n)
    gained: set[int] = set()
    for c in _postorder(kids, root):
        if c != root and kids[c] and t[c] == 0 and not is_cell[c]:
            p = int(parent[c])
            parent[kids[c]] = p
            kids[p].remove(c)  # ponytail: O(C) per collapse, O(C²) for a wide parent losing many children; M4 item
            kids[p] += kids[c]
            kids[c] = []
            parent[c] = -2
            gained.discard(c)
            gained.add(p)
    return sorted(gained)


def collapse_zero_edges(table: NodeTable, root: int) -> list[int]:
    """Collapse every zero edge above a live internal non-cell node (in place); returns the nodes that gained children."""
    return _collapse(table.parent, table.t, table.is_cell, root, table.n)


def _zero_edge(table: NodeTable) -> bool:
    """Whether a live internal non-cell node below the root has t == 0 (A18 snap, exact-zero optimum or reopt)."""
    p = table.parent[: table.n]
    internal = np.zeros(table.n, bool)
    internal[p[p >= 0]] = True
    return bool(np.any((p >= 0) & internal & (table.t[: table.n] == 0) & ~table.is_cell[: table.n]))


def _set_message(table: NodeTable, n: int, kids: list[int]) -> None:
    xn, W = ref_star.node_message(table.x64(kids), table.v64(kids), table.t[kids])
    table.x[n], table.v[n] = xn, 1.0 / W
    table.sync(n)


def _redo_pass(table: NodeTable, root: int, touched: set[int] | None, cfg: GreedyConfig) -> list[GreedyStats]:
    r"""One redo-starry pass: post-order, ``greedy_star`` at nodes with > 3 children (root) / > 2 (other).

    Iterative post-order, children by increasing id; ``touched`` None: every node, else only those nodes.

    Each node's ``outside`` = (x_{p\n}, w̃) comes from its parent's float64 accumulator (the parent's children plus
    its own outside, TH:5313-5331) at descent time; it does not depend on n's subtree, so it stays valid while that
    subtree changes. Once a child is done its contribution in the parent's accumulator is replaced (TH:5353-5358),
    so later siblings see it, and a node whose subtree changed has its message recomputed from its children.
    """
    kids = _children(table.parent, table.n)
    out: list[GreedyStats] = []

    def frame(n: int, outside, old) -> dict:
        acc = ParentAcc.from_children(table.weights64(kids[n]), table.x64(kids[n]))
        if outside is not None:
            acc.add(outside[1], outside[0])
        return {"n": n, "outside": outside, "acc": acc, "k": 0, "old": old, "changed": False}

    stack = [frame(root, None, None)]
    while stack:
        f = stack[-1]
        n = f["n"]
        if f["k"] < len(kids[n]):
            c = kids[n][f["k"]]
            f["k"] += 1
            if kids[c]:  # cells and childless nodes are never visited
                w, xc = table.weights64(c), table.x64(c)
                W_ex = f["acc"].W - w  # > 0: n has another child or an outside message
                x_ex = (f["acc"].WX - w * xc) / W_ex
                stack.append(frame(c, (x_ex, 1.0 / (table.t[c] + 1.0 / W_ex)), (w, xc)))
            continue
        stack.pop()
        if (touched is None or n in touched) and len(kids[n]) > (3 if n == root else 2):
            out.append(greedy_star(table, n, f["outside"], n == root, cfg))  # sets n's message
            f["changed"] = True
        elif f["changed"]:
            _set_message(table, n, kids[n])
        if stack and f["changed"]:  # replace n's contribution in its parent's accumulator
            up = stack[-1]
            up["acc"].remove(*f["old"])
            up["acc"].add(table.weights64(n), table.x64(n))
            up["changed"] = True
    return out


def _totals(stats: list[GreedyStats]) -> dict:
    """GreedyStats summed over calls, ``events`` concatenated."""
    tot = {k: sum(getattr(s, k) for s in stats) for k in GreedyStats._fields if k != "events"}
    return tot | {"events": [e for s in stats for e in s.events]}


def redo_starry(table: NodeTable, root: int, cfg: GreedyConfig) -> dict:
    """Redo starry (MA:225-237; ``mergeChildrenRecursive``, TH:968-1012; A18) on ``table`` in place, m = 1.

    A pass: ``collapse_zero_edges``, then ``_redo_pass`` (the first at every node, later ones only at the nodes the
    collapse gave children). Passes repeat while a live internal non-cell node has t == 0 (snaps are resolved inside
    ``greedy_star``, A18b, so these come from exact-zero optima or star re-optimisations), at most
    ``max_passes(N)`` = N − 1; reaching the cap warns (``RuntimeWarning``) and returns the tree as is (valid, with those
    zero edges). Returns the GreedyStats totals plus "passes" and "zero_edges_collapsed".
    """
    cfg = dataclasses.replace(cfg, merges_per_round=1)
    stats: list[GreedyStats] = []
    removed0 = int(np.sum(table.parent[: table.n] == -2))
    cap = max_passes(int(table.is_cell[: table.n].sum()))
    for n_pass in range(1, cap + 1):
        gained = collapse_zero_edges(table, root)
        stats += _redo_pass(table, root, None if n_pass == 1 else set(gained), cfg)
        if not _zero_edge(table):
            break
    else:
        warnings.warn(f"redo_starry: zero edges remain after {cap} passes", RuntimeWarning, stacklevel=2)
    removed = int(np.sum(table.parent[: table.n] == -2)) - removed0
    return _totals(stats) | {"passes": n_pass, "zero_edges_collapsed": removed}


def _compact(parent: np.ndarray, t: np.ndarray, N: int, root: int, names: list[str]):
    """(parent, t, names) of the live tree: leaves 0..N−1, internal nodes by increasing id, the root last."""
    live = np.flatnonzero(parent >= 0)
    ids = np.r_[np.arange(N), live[live >= N], root]
    pos = np.full(len(parent), -1)
    pos[ids] = np.arange(len(ids))
    p = np.r_[pos[parent[ids[:-1]]], -1]
    tt = np.r_[t[ids[:-1]], 0.0]
    n_int = len(ids) - N - 1
    return p, tt, list(names) + [f"internal_{k}" for k in range(n_int)] + ["root"]


def _leaf_star_times(table: NodeTable, N: int, cfg: GreedyConfig) -> np.ndarray:
    """Star EM on the N leaves (device, padded to bucket(N)); upstream's BFGS fallback in the fp64 reference."""
    Nb, G = _bucket(N), table.x.shape[1]
    xs, vs = np.zeros((Nb, G), table.dtype), np.ones((Nb, G), table.dtype)
    xs[:N], vs[:N] = table.x[:N], table.v[:N]
    res = star_em(jnp.asarray(xs), jnp.asarray(vs), jnp.asarray(np.arange(Nb) < N), cfg.star_tol, STAR_MAX_SWEEPS)
    if bool(res.converged):
        return np.asarray(res.t, np.float64)[:N]
    return ref_star.star_em(table.x64(np.arange(N)), table.v64(np.arange(N)), cfg.star_tol, STAR_MAX_SWEEPS)[0]


def greedy_tree(x, var, names, cfg: GreedyConfig = GreedyConfig(), dtype=jnp.float32) -> BuiltTree:
    """Build a tree from leaf data: star EM, greedy at the root, redo starry (``_ref.greedy.greedy_tree``).

    ``x``, ``var`` (N, G): leaf means and variances (float64 on the host; centred per gene in float64 and cast to
    ``dtype`` by ``NodeTable``); ``names``: N cell names. The greedy-stage newick is written from a copy with zero
    internal edges collapsed (upstream's ``cleanup_tree``, DP:354-355), and so are the returned tree and its newick
    (a no-op unless redo starry hit its pass cap). Stage seconds (``stats["stage_s"]``) are
    wall times with ``jax.block_until_ready`` at the boundaries.
    """
    x, var = np.asarray(x, np.float64), np.asarray(var, np.float64)
    N = x.shape[0]
    if len(names) != N:
        raise ValueError(f"greedy_tree: {len(names)} names for {N} cells")
    stage_s: dict[str, float] = {}
    table = NodeTable(x, var, dtype=dtype)

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready((table.dev_x, table.dev_v))
        stage_s[name] = time.perf_counter() - t0
        return out

    def leaf_star():
        table.t[:N] = _leaf_star_times(table, N, cfg)

    timed("star_em", leaf_star)
    root_stats = timed("greedy", lambda: greedy_star(table, N, None, True, cfg))
    snap = table.parent.copy()
    _collapse(snap, table.t, table.is_cell, N, table.n)
    stage = {"greedy": to_newick(*_compact(snap[: table.n], table.t, N, N, names))}
    redo = timed("redo_starry", lambda: redo_starry(table, N, cfg))
    final = table.parent.copy()  # collapsed copy: zero edges remain only at the pass cap (cleanup_tree, DP:354-355)
    _collapse(final, table.t, table.is_cell, N, table.n)
    parent, t, all_names = _compact(final[: table.n], table.t, N, N, names)
    stage["redo_starry"] = to_newick(parent, t, all_names)
    tot = _totals([root_stats])
    stats = {k: tot[k] + redo[k] for k in tot} | {
        "passes": redo["passes"],
        "zero_edges_collapsed": redo["zero_edges_collapsed"],
        "stage_s": stage_s,
    }
    return BuiltTree(parent, t, all_names, stage, stats)
