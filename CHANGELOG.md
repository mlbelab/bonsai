# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `bonsai.pp.read_sanity` and `bonsai.pp.from_arrays` to load Bonsai inputs into an AnnData object.
- `bonsai.tl.reconstruct` to run the (unchanged) upstream Bonsai tree-reconstruction algorithm on
  an AnnData object, and `bonsai.tl.node_posteriors` for per-node posterior means/variances.
- `bonsai.io.read_results` to load an upstream Bonsai result folder into AnnData.
- `bonsai run` and `bonsai config` command-line entry points (same options as upstream's
  `bonsai_main.py`/`create_config_file.py`).
- A golden regression test against a frozen upstream reference run (`tests/data/golden_b16ef6f`).
