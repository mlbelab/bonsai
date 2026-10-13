"""Compare a Bonsai result folder with the golden upstream reference."""

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

GOLDEN = Path(__file__).parent / "data" / "golden_b16ef6f"
FIXTURE = Path(__file__).parent / "data" / "simulated_binary_6_gens_samplingNoise"


class Tree(NamedTuple):
    clades: frozenset  # frozenset of frozensets of leaf names (rooted clades)
    lengths: dict  # clade -> branch length above it (full precision, from tree.nwk)
    loglik: float


def _parse_newick(s: str) -> tuple[frozenset, dict]:
    """Rooted clades and branch lengths from a newick string with named leaves."""
    clades, lengths = set(), {}
    stack: list[list[frozenset]] = []  # one list of child leaf-sets per open '('
    prev = None  # node just read or closed, waiting for its label/length and parent
    for tok in re.findall(r"[(),;]|[^(),;]+", s.strip()):
        if tok == "(":
            stack.append([])
        elif tok in ",;":
            if prev is not None and stack:
                stack[-1].append(prev)
            prev = None
        elif tok == ")":
            if prev is not None:
                stack[-1].append(prev)
            prev = frozenset().union(*stack.pop())
            clades.add(prev)
        else:  # "label[:length]": a leaf if nothing is pending, else the label of a closed clade
            name, _, length = tok.partition(":")
            if prev is None:
                prev = frozenset([name])
            if length:
                lengths[prev] = float(length)
    return frozenset(clades), lengths


def load_tree(folder: Path) -> Tree:
    clades, lengths = _parse_newick((folder / "tree.nwk").read_text())
    meta = pd.read_csv(folder / "metadata_bonsai.txt", index_col=0)
    return Tree(clades, lengths, float(meta.loc["metadata", "finalLoglik"]))  # type: ignore[arg-type]


def assert_trees_match(got: Path, golden: Path = GOLDEN) -> None:
    a, b = load_tree(got), load_tree(golden)
    assert a.clades == b.clades, "topology differs from upstream golden tree"
    if (got / "tree.nwk").read_bytes() == (golden / "tree.nwk").read_bytes() and a.loglik == b.loglik:
        return  # bitwise identical: expected on the reference platform
    # Cross-platform (other BLAS/libm): the final stage is Tree.optTimes (tree_helpers.py:3463-3465),
    # scipy L-BFGS-B with tol=1e-6, i.e. ftol=1e-6: it stops once the relative decrease of -loglik per
    # step is <= ftol. Two platforms can stop at points whose loglik differs by a few such steps -> 10*ftol.
    # Near a smooth optimum df ~ (1/2) H dt^2, so branch lengths are fixed only to ~sqrt(ftol)
    # relative -> 10*sqrt(ftol).
    # The final call (main.py:430) passes maxiter=100, so the ftol bound above only holds if that call
    # converges before hitting maxiter (as it does for this fixture); a maxiter-capped run could differ
    # by more than these tolerances. `Tree.test_for_zero_times` (main.py:433) runs right after it and
    # may snap near-zero optimised times to exactly 0, which does not reopen the ftol argument above.
    FTOL = 1e-6
    LOGLIK_RTOL = 10 * FTOL
    T_RTOL = 10 * FTOL**0.5
    np.testing.assert_allclose(a.loglik, b.loglik, rtol=LOGLIK_RTOL, atol=0)
    keys = sorted(b.lengths, key=sorted)
    np.testing.assert_allclose([a.lengths[k] for k in keys], [b.lengths[k] for k in keys], rtol=T_RTOL, atol=0)
