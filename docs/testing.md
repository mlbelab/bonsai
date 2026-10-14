# Testing *Bonsai*'s numerical core

## Running the tests

```bash
uv sync                                        # installs test deps
uv run pytest                                  # run everything
uv run pytest -v                               # verbose, one line per test
uv run pytest tests/test_pairwise_merge_dlogl.py -v   # run a single file
uv run pytest tests/test_pairwise_merge_dlogl.py::test_dlogl_prefers_similar_pairs_over_dissimilar_pairs # run a single test
```

## What's covered

| Test file | Functions under test | What it checks |
|---|---|---|
| `test_two_leaf_branch_length.py` | `der2LeafTree`, `getOptTime2LeafTree`, `getLoglik2LeafTree` | Optimal diffusion time between two leaves: analytic closed-form cases, the zero-clamp boundary, and the analytic derivative vs. a finite-difference check. |
| `test_star_node_position.py` | `findNodeLtqsGivenLeafs`, `findNodeLtqsGivenLeafsSingleGene` | The core "combine children into a parent" step: weighted-average identities, precision-is-sum-of-weights, single-gene/vectorised agreement. |
| `test_star_tree_component_optimisers.py` | `findOptimalTSingleCell`, `derLogLikStar`, `optimiseT3LeafStar`, `optimiseT3LeafStarSequential` | The EM single-step time optimiser (against an analytic solution) and the two specialised 3-leaf optimisers (cross-checked against each other and against `optimiseTStar`). |
| `test_star_tree_time_optimisation.py` | `optimiseTStar`, `logLGradStarTreeLogT`, `getLoglikAndGradStarTree` | The general N-child star-tree time optimiser: first-order optimality at convergence, and a direct finite-difference check of the shared log-likelihood/gradient function. |
| `test_pairwise_merge_dlogl.py` | `calcSingleDLogL` | The pairwise merge-scoring function that drives which nodes Bonsai merges next: the "merging never decreases likelihood" invariant, and that similar pairs score higher than dissimilar ones. |
| `test_reroot_as_if_root.py` | `getLtqsAsIfRoot`, `getLtqsAsIfRoot_vectorized` | The "reroot" primitive, verified against an independently computed reference position/precision, plus the root-fully-dominates degenerate case. |
| `test_incremental_updates.py` | `subtract_contrib_ltqs`, `add_contrib_ltqs`, `getLtqsAfterChildUpdate` | Incrementally adding/removing/swapping one child's contribution to a parent, cross-checked against recomputing the parent from scratch. |
| `test_gene_variance_estimation.py` | `neg_loglik_grad_using_measurements_and_errors`, `loglik_given_true_var_log`, `infer_true_var_log` | Per-gene variance estimation used in `read_and_filter`; gradient check; cross-check against the (unused) reference implementation. |
| `test_pdist_index.py` | `get_pdist_inds` | Condensed-distance-matrix indexing, checked directly against `scipy.spatial.distance.pdist`'s actual ordering. |
| `test_branch_length_init.py` | `calcTInit` | Initial branch-length guesses fed to the optimisers above. |
| `test_nn_recall.py` | `getFracCorrectNNs` | NN-search accuracy metric used in `_core/approx_nn.py`'s standalone benchmarking script. |

257 tests total (`uv run pytest --collect-only -q`).

## Golden regression tests

Three files compare a run against the frozen upstream reference `tests/data/golden_b16ef6f`
(provenance in `docs/provenance.md`), each marked `slow` (run with `uv run pytest`, skipped by
`uv run pytest -m "not slow"`; they take minutes on this node):

| Test file | Entry point exercised | What it checks |
|---|---|---|
| `test_golden_core.py` | `bonsai._core.main.run_from_config` | The upstream config/run path directly: the result matches golden (`test_core_matches_golden`), and two in-process runs on the same config are bitwise-identical newicks (`test_second_in_process_run_is_identical`). |
| `test_tl.py` | `bonsai.pp.read_sanity` + `bonsai.tl.reconstruct` | The public API path end-to-end against golden, plus that `adata.uns["bonsai"]` carries exactly what upstream wrote (loglik, newick) and that the AnnData slots (`obs`/`var`/`uns`) are populated as documented. |
| `test_cli.py` | `bonsai.cli` (`bonsai run`) | The CLI subprocess path against golden; also (non-slow) that `bonsai config` writes a YAML and that an unrecognized CLI option is rejected rather than silently dropped. |

`tests/golden.py` holds the comparison logic (`assert_trees_match`): topology (the set of rooted
clades) must match exactly. For the loglik and branch lengths, if `tree.nwk` is byte-identical and
the recorded `finalLoglik` matches exactly, that's asserted directly (expected on the reference
platform used to record the fixture). Otherwise it falls back to a derived tolerance: the final
stage (`Tree.optTimes`, `tree_helpers.py`) is `scipy.optimize.minimize` with L-BFGS-B at
`ftol=1e-6`, i.e. it stops once the relative decrease in `-loglik` per step is `<= ftol`; two
platforms (different BLAS/libm) can therefore stop a few such steps apart, giving `loglik` a
`rtol` of `10 * ftol`. Near a smooth optimum `df ~ (1/2) H dt^2`, so branch lengths are fixed only
to `~sqrt(ftol)` relative, giving branch-length `rtol = 10 * sqrt(ftol)`. Both tolerances use
`atol=0` (pure `rtol`, derived from the optimizer's own stopping rule, not tuned to pass).

## Call graph of the tested core

This traces how the tree-merge scoring path actually calls into the tested
functions (solid arrows = direct calls; the `Tree`/`TreeNode` methods at the
top are the *only* stateful, untested entry points — everything below them
is pure and covered).

```
flowchart TD
    subgraph stateful["Tree / TreeNode methods (stateful, NOT covered)"]
        getNewPairs["Tree.getNewPairs<br/>Tree.mergeChildrenUB"]
        mergeNodes["Tree.mergeNodes"]
        sprMoves["TreeNode.do_spr_search<br/>detach_subtree / add_subtree"]
        optTimesWhole["Tree.optTimes"]
    end

    getNewPairs --> calcSingleDLogL
    mergeNodes --> calcSingleDLogL

    calcSingleDLogL --> calcTInit
    calcSingleDLogL -->|sequential=True| optimiseT3LeafStarSequential
    calcSingleDLogL -->|sequential=False, or as fallback| optimiseT3LeafStar
    calcSingleDLogL --> getLoglikAndGradStarTree

    optimiseT3LeafStarSequential --> getOptTime2LeafTree
    getOptTime2LeafTree --> der2LeafTree
    optimiseT3LeafStarSequential -->|scipy.minimize| getLoglikAndGradStarTree

    optimiseT3LeafStar -->|scipy.minimize| getLoglikAndGradStarTree

    getLoglikAndGradStarTree --> findNodeLtqsGivenLeafs

    optTimesWhole --> optimiseTStar
    optimiseTStar --> findNodeLtqsGivenLeafs
    optimiseTStar --> findOptimalTSingleCell
    findOptimalTSingleCell --> derLogLikStar
    optimiseTStar -->|EM fallback: scipy.minimize| logLGradStarTreeLogT
    optimiseTStar --> getLoglikAndGradStarTree

    sprMoves --> getLtqsAsIfRoot
    sprMoves --> subtract_contrib_ltqs
    sprMoves --> add_contrib_ltqs
    getNewPairs --> getLtqsAsIfRoot_vectorized
    mergeNodes --> getLtqsAfterChildUpdate

    getLtqsAfterChildUpdate -.->|algebraically equivalent to, see test_incremental_updates.py| subtract_contrib_ltqs
    subtract_contrib_ltqs -.-> add_contrib_ltqs
```

Note: `logLGradStarTreeLogT` is a **separate, self-contained
reimplementation** of the same log-likelihood/gradient as
`getLoglikAndGradStarTree` (it doesn't call it, or `findNodeLtqsGivenLeafs`
— it recomputes the weighted average inline). It exists because
`optimiseTStar`'s gradient-based fallback path needs one function that does
both the E-step (recompute `xr_g`/`W_g`) and the log-likelihood/gradient in a
single call for `scipy.optimize.minimize`.

Separate from the tree-merge path, and independent of it:

```
flowchart LR
    read_and_filter --> neg_loglik_grad_using_measurements_and_errors
    reorderChildrenRoot --> get_pdist_inds
    getNewPairs --> getApproxNNs
    getApproxNNs --> getNNssklearn
    benchmark["_core/approx_nn.py `__main__` benchmark"] --> getFracCorrectNNs
```
