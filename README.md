# bonsai

[![PyPI][badge-pypi]][link-pypi]

[badge-pypi]: https://img.shields.io/pypi/v/bonsai.svg
[link-pypi]: https://pypi.org/project/bonsai

An [scverse](https://scverse.org)/[AnnData](https://anndata.readthedocs.io)-compatible package for
*Bonsai*, the Bayesian maximum-likelihood cell-tree reconstruction method of de Groot et al. This
package wraps the original algorithm (unchanged, see [`docs/deviations.md`][deviations] and
[`docs/provenance.md`][provenance]) behind a Python/AnnData API and a CLI, derived from upstream
[`dhdegroot/Bonsai-data-representation`][upstream] (commit `b16ef6f`), licence
[CC BY-NC 4.0](https://github.com/mlbelab/bonsai/blob/main/LICENSE).

If you use this package, please cite the *Bonsai* paper and the software, see
[`CITATION.cff`](https://github.com/mlbelab/bonsai/blob/main/CITATION.cff):

> de Groot, D.H., Morillo Leonardo, S.X., Pachkov, M., van Nimwegen, E. (2025). *Bonsai: Tree
> representations for distortion-free visualization and exploratory analysis of single-cell omics
> data.* bioRxiv. https://doi.org/10.1101/2025.05.08.652944

*Bonsai-scout*, upstream's interactive visualization app, is **not** included in this package; use
[upstream's `bonsai_scout`][upstream-scout] on a result folder produced here (same `edgeInfo.txt`/
`vertInfo.txt`/`tree.nwk`/`metadata.json` layout).

## Getting started

Please refer to the [documentation][link-docs], in particular the [API documentation][link-api].
For a worked tour of the package on example datasets, see
[`docs/notebooks/bonsai_tour.ipynb`][link-tour].

## Installation

You need Python >=3.12.

Install the latest release from PyPI (once published):

```bash
pip install bonsai
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/mlbelab/bonsai.git
```

For datasets of a few thousand cells and larger, *Bonsai* benefits from parallel computation via
`mpi4py`/OpenMPI. Install the `mpi` extra once OpenMPI (and `mpicc`) are available:

```bash
pip install "bonsai[mpi] @ git+https://github.com/mlbelab/bonsai.git"
```

Even without a working `mpi4py` installation, *Bonsai* runs fine on a single core.

## Input: preprocessing with *Sanity*

*Bonsai* needs, for each cell, a vector of most-likely feature values with error bars. For
scRNA-seq data we recommend preprocessing raw counts with
[*Sanity*](https://github.com/jmbreda/Sanity) first.

**Important: run *Sanity* with the flags `-e 1 -v_m MAP`.** `-e 1` produces *Sanity*'s extended
output; `-v_m MAP` makes the inferred expression values correspond to a single guess of the
gene-variance rather than an average over it (see the SI of the *Bonsai* paper for why this is the
preferred mode). `-v_m` also accepts `MLE` and `EAP`: all three fix a single prior variance per
gene and return the matching posterior — *Bonsai* needs this because it divides out the prior
before inference (the tree replaces *Sanity*'s simple Gaussian prior). The fourth option, `MARG`,
marginalises over the gene-variance and **cannot** be used as *Bonsai* input; `bonsai.pp.read_sanity`
recognises such output and raises.

`bonsai.pp.read_sanity` also reads output from *Sanity* versions before 2.0 (run with
`-e 1 -max_v true`, which wrote files with a `_vmax` suffix) — it tells the two apart by whether a
`sanity_command.txt` file is present, so no argument needs to change when you upgrade *Sanity*.

## Quick start (Python API)

```python
import bonsai

adata = bonsai.pp.read_sanity("sanity_out/")
bonsai.tl.reconstruct(adata)
print(adata.uns["bonsai"]["newick"])
```

For data that is not *Sanity* output, provide your own per-cell mean/SD layers and mark them with
`bonsai.pp.from_arrays(adata, mean_layer, sd_layer=None)` instead of `read_sanity`.

See [`docs/api.md`][link-api] for the full reference (`bonsai.pp`, `bonsai.tl`, `bonsai.io`).

### AnnData slots

| Slot | Contents |
|---|---|
| `layers["bonsai_mean"]` | *Sanity* posterior means δ |
| `layers["bonsai_sd"]` | *Sanity* posterior SDs ε |
| `var["bonsai_prior_var"]` | *Sanity* per-gene prior variance |
| `var["bonsai_mu"]` | *Sanity* per-gene mean |
| `uns["bonsai_input"]` | input kind: `sanity_posterior` or `likelihood` |
| `uns["bonsai"]` | result: newick, edges, nodes, loglik, genes_used, config, workdir |
| `var["bonsai_used"]` | genes that passed the z-score filter |
| `obs["bonsai_node"]` | `vertInd` of each cell |

`bonsai.tl.node_posteriors(adata)` returns an AnnData of per-node posterior means/variances (nodes
as obs, used genes as var); it needs `reconstruct(..., keep_workdir=True)`.

## Command-line interface

Create a config file, then run the reconstruction from it:

```bash
bonsai config --new_yaml_path c.yaml
bonsai run --config_filepath c.yaml
```

`bonsai config` accepts the same options as upstream's `create_config_file.py` (dataset,
data_folder, zscore_cutoff, use_knn, UB_ellipsoid_size, nnn_n_randommoves, nnn_n_randomtrees, …);
run `bonsai config --help` for the full list. Paths given on the command line or in the YAML config
are relative to the **current directory**, not the repository.

`bonsai run` also accepts `--step {all,preprocess,core_calc,metadata}` to split the computation
(useful on an HPC cluster, since `core_calc` is the most time-consuming step but needs the least
memory) and `--pickup_intermediate` to resume a run that was interrupted, from the latest stored
intermediate state.

For larger datasets, run under MPI:

```bash
mpiexec -n <NUMBER_OF_CORES> bonsai run --config_filepath c.yaml
```

## Result folder

A completed run's `results_folder` contains a `final_bonsai_*` subdirectory with:

- `tree.nwk`: the tree in Newick format, full floating-point precision.
- `edgeInfo.txt`, `vertInfo.txt`, `metadata.json`: a more human-readable form of the same tree.
  `edgeInfo.txt` has one row per edge (`vertInd` of each endpoint, branch length — written with 9
  significant digits, less precise than `tree.nwk`); `vertInfo.txt` maps each `vertInd` to its
  `vertName` (used to join cells back to the original data); `metadata.json` records dataset
  metadata, including where the original expression data live.
- `posterior_ltqs_vertByGene.npy`, `posterior_ltqsVars_vertByGene.npy`: inferred gene-expression
  posterior means/variances for every node (including internal nodes), read by
  `bonsai.tl.node_posteriors`.

`intermediate_bonsai_*` subdirectories hold the tree state after each reconstruction step, useful
for resuming a failed run; they are safe to delete once the run has completed.
`bonsai.io.read_results(folder, adata=None)` loads a result folder (with or without an existing
`adata`) into `uns["bonsai"]`, `obs["bonsai_node"]`, `var["bonsai_used"]`.

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

See [`CITATION.cff`](https://github.com/mlbelab/bonsai/blob/main/CITATION.cff).

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/mlbelab/bonsai/issues
[changelog]: https://github.com/mlbelab/bonsai/blob/main/docs/changelog.md
[link-docs]: https://github.com/mlbelab/bonsai/tree/main/docs
[link-api]: https://github.com/mlbelab/bonsai/blob/main/docs/api.md
[link-tour]: https://github.com/mlbelab/bonsai/blob/main/docs/notebooks/bonsai_tour.ipynb
[upstream]: https://github.com/dhdegroot/Bonsai-data-representation
[upstream-scout]: https://github.com/dhdegroot/Bonsai-data-representation#bonsai-scout-visualizing-the-bonsai-results
[deviations]: https://github.com/mlbelab/bonsai/blob/main/docs/deviations.md
[provenance]: https://github.com/mlbelab/bonsai/blob/main/docs/provenance.md
