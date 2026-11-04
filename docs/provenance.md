# Provenance

- Upstream: https://github.com/dhdegroot/Bonsai-data-representation (via fork mlbelab/bonsai),
  commit b16ef6f, licence CC BY-NC 4.0. `src/bonsai/_core/` is upstream code moved with packaging
  edits only (see git history and docs/deviations.md).
- Template: scverse/cookiecutter-scverse (commit in `.cruft.json`).
- Golden reference `tests/data/golden_b16ef6f/`: unmodified upstream b16ef6f, Python 3.14,
  numpy 2.4.6, scipy 1.17.1 (requirements.txt pins), command
  `python bonsai/bonsai_main.py --config_filepath config.yaml --step all` on
  `tests/data/simulated_binary_6_gens_samplingNoise` (examples/1_simple_example config:
  zscore_cutoff 1.0, use_knn 10, UB_ellipsoid_size 1.0, nnn_n_randommoves 10, nnn_n_randomtrees 2),
  2026-10-13. Two independent runs bitwise identical. finalLoglik -65745.68334074633.
- Phase 2 (2026-10): `tests/conftest.py` precision guard, `bench/metadata.py` and
  `bench.run.count_compilations` reproduce the behaviour of helpers in the internal mlbelab project rose
  (`tests/conftest.py`, `bench/metadata.py`, `rose/core/compile_count.py`); no licence constraint (internal).
- `bonsai._ref.preprocess`: a vectorised float64 NumPy port of upstream's `read_and_filter`
  (`_core/dataprocessing.py` DP:2792-3135) and the `√geneVariances` rescaling in `SCData.__init__`
  (DP:204-230), reproducing both bitwise on the kept fixtures.
- `bonsai._ref.{star,pair,tree}` and `bonsai._engine.*`: re-implementations, in NumPy (fp64
  reference) and JAX (device kernels), of upstream b16ef6f's node-message, star-EM, 3-leaf pair and
  whole-tree log-likelihood formulas, licence CC BY-NC 4.0. Upstream's own code is not copied; each
  formula is cited inline to its upstream source lines, e.g. `_core/tree_helpers.py` (`TH:`) for
  node messages (`findNodeLtqsGivenLeafs`, TH:5257-5294), the star log-likelihood and gradient
  (`getLoglikAndGradStarTree`, TH:5063-5115), and the pair solvers and box
  (`calcSingleDLogL` and related, TH:4822-4923, TH:5166-5226, TH:5362-5408); `_core/helpers.py`
  (`HE:`) for `calcTInit` (HE:788-794); and `_core/dataprocessing.py` (`DP:`) for preprocessing
  (above).
- `MATMUL_PRECISION` (`src/bonsai/_engine/numerics.py`): the explicit-precision matmul pattern
  (passing `precision=jax.lax.Precision.HIGHEST` so float32 matmuls do not lower to TF32 on GPU) is
  the one used by the internal mlbelab project rose (`rose.core.numerics`); no licence constraint
  (internal).
- `bonsai._ref.greedy`, `bonsai._engine.greedy` and `bonsai._engine.build`: re-implementations, in NumPy (fp64
  reference) and JAX with host NumPy (engine), of upstream b16ef6f's greedy merge loop and redo starry, licence
  CC BY-NC 4.0: `mergeChildrenRecursive` (TH:968-1012), `mergeChildrenUB` (TH:1039-1620),
  `getOptimalDLogLPairInfoNew` and `mergeNodes` (TH:2381-2686) and `mergeZeroTimeChilds` (TH:789-807). Upstream's
  code is not copied; the differences are listed in docs/deviations.md ("Phase 2 M2 greedy engine").
