"""fp64 brute-force reference of the greedy merge engine and redo starry (maths ref §8, §11; A18, A21, A21b).

Re-implements the semantics of upstream b16ef6f's greedy loop (``mergeChildrenUB``, TH:1039-1620;
``getOptimalDLogLPairInfoNew``, TH:2381-2456; ``mergeNodes``, TH:2459-2646; ``mergeZeroTimeChilds``, TH:789-807; ``mergeChildrenRecursive``, TH:968-1012), CC BY-NC
4.0, as the M2 engine defines them: every round scores **all** pairs (no bounds, no kNN), the
add-as-child move is a zero edge (A18), there is no zero-root branch (A21b: upstream's
reopt/reset on a best pair with t_aR <= 1e-6 sees the t_aR of its tol-1e-4 solve and practically never fires), and ``mergeDownstreamChildrenSingle`` is replaced by the
deferred collapse of redo starry. Node ids: leaves 0..N−1, root N, new nodes count up in merge order
and are never reused. Units: 2x log-likelihoods (A1).

Messages are recomputed from scratch (no incremental updates): a parent's AIR from its current
children plus the outside message, and before each redo-starry node the whole tree by
``tree_loglik_grad``. That is the engine's incremental scheme without its rounding (and without
upstream's 1e-12 AIR guard, TH:5321, unreachable for a node with a sibling or a parent).
"""

import itertools
import warnings
from typing import NamedTuple

import numpy as np

from .pair import pair_score
from .star import node_message, star_em
from .tree import tree_loglik_grad

# Own copies of the engine's constants (plan "Constants"; Task 6a asserts both sets are equal).
ACCEPT_DLOGL = 1e-6  # merge iff the best normal dLogL > this (TH:2428, A4)
ADD_AS_CHILD_T = 1e-6  # t_1a < this with child 1 internal: t_1a := 0 (TH:2467-2476, A18)
REDO_TIME_FRAC, REDO_TIME_MIN = 0.05, 10  # redoTimeAt = max(⌊0.05·C⌋, 10) (TH:1154, 2354)
REDO_NN_FRAC, REDO_NN_MIN = 0.1, 10  # kNN rebuild schedule (TH:1951, A19); unused here (all pairs)
NN_MIN_CHILDREN = 50  # kNN only when C > 50 (MA:176, TH:1888); unused here
LAZY_MIN_CHILDREN = 10  # bounds only when C > 10 (TH:1040); unused here
STAR_TOL, STAR_MAX_SWEEPS, STAR_T_MAX = 1e-4, 1000, 1e8  # star EM (TH:4992-5004)
ZERO_EDGE_PASSES = 8  # cap on collapse + redo passes (A18)
K_NN = 10  # use_knn (config.py:75); unused here


class Event(NamedTuple):
    kind: str  # "merge" | "reopt"
    i: int  # child ids (i < j); -1 for "reopt"
    j: int
    new: int  # id of the created node; -1 for "reopt"
    dlogl: float  # the pair's score; nan for "reopt"
    t: tuple[float, float, float]  # (t_1a, t_2a, t_aR) as applied (after the A18 snap); nan for "reopt"


class Tree(NamedTuple):
    """Node table, node-indexed; the arrays grow by one row per merge.

    ``parent[i]``: −1 at the root, −2 for a node removed by a collapse; ``t[i]``: the edge above i;
    ``x``/``var`` (n, G): data at cells, the downward message (x_n, 1/W_n) at internal nodes. Rows of
    removed nodes stay (ids are never reused).
    """

    parent: np.ndarray
    t: np.ndarray
    x: np.ndarray
    var: np.ndarray
    is_cell: np.ndarray


def children(tree: Tree, p: int) -> list[int]:
    """Children of ``p``, increasing id."""
    return [int(c) for c in np.flatnonzero(tree.parent == p)]


def compact(tree: Tree, root: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(parent, t, x, var, ids) of the nodes reachable from ``root``, in increasing id; ``ids`` maps back."""
    live, i = [root], 0
    while i < len(live):
        live.extend(children(tree, live[i]))
        i += 1
    ids = np.sort(np.array(live))
    pos = np.full(len(tree.parent), -1)
    pos[ids] = np.arange(len(ids))
    parent = np.where(ids == root, -1, pos[tree.parent[ids]])
    return parent, tree.t[ids].copy(), tree.x[ids].copy(), tree.var[ids].copy(), ids


def _copy(tree: Tree) -> Tree:
    return Tree(*(a.copy() for a in tree))


def _air(tree: Tree, kids: list[int], outside) -> tuple[np.ndarray, np.ndarray]:
    """Parent AIR (x_r, W_r): its children plus the outside pseudo-child (x_out, w̃) (TH:5313-5331)."""
    w = 1.0 / (tree.var[kids] + tree.t[kids][:, None])
    W, s = w.sum(axis=0), (w * tree.x[kids]).sum(axis=0)
    if outside is not None:
        xo, wo = outside
        W, s = W + wo, s + wo * xo
    return s / W, W


def decide(cands) -> tuple[int, int] | None:
    """The pair a round merges (TH:1214-1215, 2381-2456; A21b), or None (stop).

    ``cands``: iterable of ((i, j), dLogL). Scanned in (i, j) order with a running maximum updated on
    strict >, so on equal dLogL the lower (i, j) wins; the best pair is merged iff its dLogL > 1e-6
    (TH:2428), whatever its t_aR.
    """
    best, best_d = None, ACCEPT_DLOGL
    for ij, d in sorted(cands, key=lambda c: c[0]):
        if d > best_d:
            best, best_d = ij, d
    return best


def _set_message(tree: Tree, p: int) -> None:
    kids = children(tree, p)
    xn, W = node_message(tree.x[kids], tree.var[kids], tree.t[kids])
    tree.x[p], tree.var[p] = xn, 1.0 / W


def greedy_star(
    tree: Tree,
    p: int,
    outside: tuple[np.ndarray, np.ndarray] | None,
    mode: str = "sequential",
    star_tol: float = STAR_TOL,
    trace: list | None = None,
) -> tuple[Tree, list[Event]]:
    r"""Greedily merge the children of ``p`` (maths ref §8), scoring all pairs every round.

    ``outside`` = (x_{pp\p}, w̃): the message of everything outside p as seen from p (None at the
    root). Per round: at the root, periodic star re-optimisation when counter > redoTimeAt =
    max(⌊0.05·C⌋, 10) (recomputed at each re-optimisation); else score every pair of current children
    (child 1 = lower id) at p's current AIR and merge ``decide``'s pair into a new node with the A18 snap (t_1a < 1e-6 and child 1 internal → t_1a := 0, else the
    same for child 2), counter += 1. Stops at ≤ 3 children (root) / ≤ 2 (other) or when no pair has dLogL > 1e-6.
    ``trace``: one entry per round, appended before the decision, (scores: {(i, j): (dLogL, t3)},
    {child: t}, x_r, W_r, counter); a periodic re-optimisation round has no scores. Returns the new
    tree (input untouched; p's row holds its updated message) and the events.
    """
    tree = _copy(tree)
    is_root = outside is None
    kids = children(tree, p)
    redo_at = max(int(REDO_TIME_FRAC * len(kids)), REDO_TIME_MIN)
    counter, events = 0, []
    nan3 = (np.nan, np.nan, np.nan)

    def reopt():
        nonlocal tree, redo_at, counter
        tree.t[kids] = star_em(tree.x[kids], tree.var[kids], tol=star_tol, max_sweeps=STAR_MAX_SWEEPS)[0]
        redo_at, counter = max(int(REDO_TIME_FRAC * len(kids)), REDO_TIME_MIN), 0
        events.append(Event("reopt", -1, -1, -1, np.nan, nan3))

    while len(kids) > (3 if is_root else 2):
        if is_root and counter > redo_at:  # TH:1926-1927 (A21: on any round)
            if trace is not None:
                trace.append(({}, {c: float(tree.t[c]) for c in kids}, *_air(tree, kids, None), counter))
            reopt()
            continue
        xr, Wr = _air(tree, kids, outside)
        x, v, t = tree.x, tree.var, tree.t
        scores = {
            (i, j): pair_score(x[i], v[i], t[i], x[j], v[j], t[j], xr, Wr, mode)
            for i, j in itertools.combinations(kids, 2)
        }
        if trace is not None:
            trace.append(
                (
                    {ij: (s.dlogl, s.t.copy()) for ij, s in scores.items()},
                    {c: float(t[c]) for c in kids},
                    xr,
                    Wr,
                    counter,
                )
            )
        ij = decide((ij, s.dlogl) for ij, s in scores.items())
        if ij is None:
            break
        i, j = ij
        s = scores[ij]
        t1a, t2a, t_ar = (float(u) for u in s.t)
        if t1a < ADD_AS_CHILD_T and not tree.is_cell[i]:  # child 1 first (TH:2467-2476)
            t1a = 0.0
        elif t2a < ADD_AS_CHILD_T and not tree.is_cell[j]:
            t2a = 0.0
        a = len(tree.parent)
        xa, Wa = node_message(x[[i, j]], v[[i, j]], np.array([t1a, t2a]))
        tree = Tree(
            np.append(tree.parent, p),
            np.append(tree.t, t_ar),
            np.vstack([x, xa]),
            np.vstack([v, 1.0 / Wa]),
            np.append(tree.is_cell, False),
        )
        tree.parent[[i, j]] = a
        tree.t[i], tree.t[j] = t1a, t2a
        kids = [c for c in kids if c not in (i, j)] + [a]
        counter += 1
        events.append(Event("merge", i, j, a, s.dlogl, (t1a, t2a, t_ar)))
    _set_message(tree, p)
    return tree, events


def collapse_zero_edges(tree: Tree, root: int) -> tuple[Tree, list[int]]:
    """``mergeZeroTimeChilds`` (TH:789-807). Returns (tree, live nodes that gained children).

    Recursively (children first), an internal non-cell child with t == 0 is removed and its children
    move to its parent. Messages stay valid: a zero edge passes the child's message unchanged (w = W_c).
    """
    tree = _copy(tree)
    gained: set[int] = set()

    def visit(n: int) -> None:
        for c in children(tree, n):
            if children(tree, c):
                visit(c)
                if tree.t[c] == 0 and not tree.is_cell[c]:
                    tree.parent[tree.parent == c] = n
                    tree.parent[c] = -2
                    gained.discard(c)
                    gained.add(n)

    visit(root)
    return tree, sorted(gained)


def _refresh(tree: Tree, root: int):
    """Recompute every internal message from the leaves; return (tree, outside messages by id)."""
    parent, t, x, var, ids = compact(tree, root)
    g = tree_loglik_grad(parent, t, x, var)
    tree = _copy(tree)
    internal = np.unique(parent[parent >= 0])
    tree.x[ids[internal]], tree.var[ids[internal]] = g.x[internal], 1.0 / g.W[internal]
    outside = {
        int(ids[k]): (g.x_ex[k], 1.0 / (t[k] + 1.0 / g.W_ex[k])) for k in range(len(ids)) if parent[k] >= 0
    }  # w̃ = 1/(t + 1/W_{p\n}) (TH:5313-5331)
    return tree, outside


def _zero_edge(tree: Tree, root: int) -> bool:
    """Whether a live internal non-cell node has t == 0 (an A18 snap, an exact-zero optimum or a star reopt)."""
    parent, t, _, _, ids = compact(tree, root)
    internal = np.isin(np.arange(len(ids)), parent)
    return bool(np.any((t == 0) & internal & (parent >= 0) & ~tree.is_cell[ids]))


def _redo_pass(tree: Tree, root: int, touched: set[int] | None, mode: str, star_tol: float):
    """One post-order pass; ``touched`` None: greedy at every node, else only at those nodes."""
    events: list[Event] = []

    def visit(n: int) -> None:
        nonlocal tree
        for c in children(tree, n):
            if children(tree, c):
                visit(c)
        if (touched is None or n in touched) and len(children(tree, n)) > (3 if n == root else 2):
            tree, outside = _refresh(tree, root)
            tree, ev = greedy_star(tree, n, None if n == root else outside[n], mode, star_tol)
            events.extend(ev)

    visit(root)
    return tree, events


def redo_starry(
    tree: Tree, root: int, mode: str = "sequential", star_tol: float = STAR_TOL
) -> tuple[Tree, list[Event], int]:
    """Redo starry (MA:225-237; ``mergeChildrenRecursive``, TH:968-1012; A18). Returns (tree, events, passes).

    A pass: ``collapse_zero_edges``, then post-order (children by increasing id) ``greedy_star`` at
    every node with > 3 children (root) / > 2 (other), with its AIR from the current tree. Passes
    repeat while the tree has a zero edge above a live internal non-cell node; passes after the first run
    the greedy only at the nodes the collapse gave children. More than ``ZERO_EDGE_PASSES`` passes
    needed: ``RuntimeWarning``, the tree is returned as is (valid, with those zero edges).
    """
    events: list[Event] = []
    for n_pass in range(1, ZERO_EDGE_PASSES + 1):
        tree, gained = collapse_zero_edges(tree, root)
        tree, pass_events = _redo_pass(tree, root, None if n_pass == 1 else set(gained), mode, star_tol)
        events += pass_events
        if not _zero_edge(tree, root):
            return tree, events, n_pass
    warnings.warn(f"redo_starry: zero edges remain after {ZERO_EDGE_PASSES} passes", RuntimeWarning, stacklevel=2)
    return tree, events, ZERO_EDGE_PASSES


def greedy_tree(x: np.ndarray, var: np.ndarray, mode: str = "sequential", star_tol: float = STAR_TOL):
    """Star EM on the leaves (root id N), greedy at the root, collapsed greedy-stage snapshot, redo starry.

    ``x``, ``var``: (N, G) leaf data (fp64). Returns (tree, events, info) with info = {"greedy": the
    collapsed greedy-stage tree (upstream's ``cleanup_tree``, DP:354-355), "n_greedy_events",
    "passes", "star_converged"}.
    """
    N, G = x.shape
    tree = Tree(
        np.r_[np.full(N, N), -1],
        np.zeros(N + 1),
        np.vstack([x, np.zeros(G)]).astype(np.float64),
        np.vstack([var, np.ones(G)]).astype(np.float64),
        np.r_[np.ones(N, bool), False],
    )
    tree.t[:N], _, converged = star_em(tree.x[:N], tree.var[:N], tol=star_tol, max_sweeps=STAR_MAX_SWEEPS)
    _set_message(tree, N)
    tree, events = greedy_star(tree, N, None, mode, star_tol)
    snapshot = collapse_zero_edges(tree, N)[0]
    tree, redo_events, passes = redo_starry(tree, N, mode, star_tol)
    info = {"greedy": snapshot, "n_greedy_events": len(events), "passes": passes, "star_converged": converged}
    return tree, events + redo_events, info
