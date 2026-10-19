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
