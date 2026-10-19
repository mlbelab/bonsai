"""Prepare a Sanity input (TSV counts) from the control cells of a large CRISPR-perturbation h5ad.

The source file (~13 GB, ~1M cells) is read entirely through ``h5py``: ``obs`` columns (small) are
read in full to pick cells, and only the picked rows of ``raw/X`` (a CSR group) are read, by
slicing ``indptr`` for each picked row and reading the corresponding ``data``/``indices`` range
directly from disk. The full expression matrix is never loaded or densified.

Subsamples control cells (``obs["genetic_perturbation_strategy"] == "control"``) stratified by
``obs["annotation"]``, keeps genes detected in at least ``--min-cells-per-gene`` of the sampled
cells, and writes a tab-separated raw-count matrix (genes x cells) that Sanity's ``-f`` flag
accepts, plus the sampled cells' metadata for notebook plotting.

``raw.var_names`` in this file are Ensembl gene ids (``ENSG...``), not gene symbols; gene symbols
live in ``raw/var/gene_name`` (no duplicates in this file), which is what this script writes as
``GeneID``.

Usage
-----
    python prepare_perturb_ctrl_data.py <perturb_atlas.h5ad> <output_dir> [--n-cells 600] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def _decode(arr: np.ndarray) -> np.ndarray:
    """Decode a bytes array of fixed/variable-length HDF5 strings to Python str."""
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def _read_categorical(group: h5py.Group, name: str) -> pd.Categorical:
    """Read an anndata-style {categories, codes} categorical column as a pandas Categorical."""
    cats = _decode(group[name]["categories"][:])
    codes = group[name]["codes"][:]
    return pd.Categorical.from_codes(codes, categories=cats)


def main() -> None:
    """Parse CLI args, subsample control cells, read their raw counts via h5py, and write TSV/CSV."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_h5ad", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--n-cells", type=int, default=600)
    p.add_argument("--min-cells-per-gene", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5ad, "r") as f:
        strategy = _read_categorical(f["obs"], "genetic_perturbation_strategy")
        is_control = np.asarray(strategy == "control")
        n_control = int(is_control.sum())
        print(f"{n_control} control cells of {len(is_control)} total")

        annotation = _read_categorical(f["obs"], "annotation")
        batch = f["obs/batch"][:]
        sample = _read_categorical(f["obs"], "sample")
        bc_wells = _decode(f["obs/bc_wells"][:])

        control_idx = np.flatnonzero(is_control)
        obs_ctrl = pd.DataFrame(
            {"annotation": annotation[control_idx], "batch": batch[control_idx], "sample": sample[control_idx]},
            index=bc_wells[control_idx],
        )

        # Stratify by annotation among control cells only, same under-fill-small-strata rule as
        # prepare_organoid_data.py: request ~n_cells/n_strata per stratum, take the whole stratum
        # if it is smaller than that.
        strata = obs_ctrl.groupby("annotation", observed=True)
        n_strata = strata.ngroups
        per_stratum = max(1, args.n_cells // n_strata)
        picked_local = []
        strata_report = []
        for key, idx in strata.indices.items():
            k = min(per_stratum, len(idx))
            chosen = rng.choice(idx, size=k, replace=False)
            picked_local.append(chosen)
            strata_report.append((key, len(idx), k))
        picked_local = np.unique(np.concatenate(picked_local))
        if len(picked_local) > args.n_cells:
            picked_local = rng.choice(picked_local, size=args.n_cells, replace=False)
        picked_local = np.sort(picked_local)  # sort for efficient, in-order CSR row reads below

        picked_global = control_idx[picked_local]
        obs_sub = obs_ctrl.iloc[picked_local].copy()
        assert obs_sub["annotation"].notna().all(), "picked cells must all have a non-NaN annotation"
        assert np.all(np.asarray(strategy)[picked_global] == "control"), "picked cells must all be controls"

        print(f"requested ~{per_stratum} cells per stratum across {n_strata} annotation strata")
        for key, available, k in sorted(strata_report, key=lambda row: str(row[0])):
            print(f"  annotation={key}: {k}/{per_stratum} requested ({available} available)")

        # The h5ad index (`raw/var/feature_id`) holds Ensembl gene ids (ENSG...), not gene symbols,
        # so we use `raw/var/gene_name` instead (checked unique below).
        raw_var_index = _decode(f["raw/var/feature_id"][:])
        gene_name = _decode(f["raw/var/gene_name"][:])
        assert raw_var_index[0].startswith("ENSG"), "raw.var_names are Ensembl ids, not gene symbols"
        assert pd.Series(gene_name).duplicated().sum() == 0, "raw var gene_name has duplicates"
        print("raw.var_names (the h5ad index) are Ensembl gene ids; using raw/var/gene_name (unique) as GeneID")

        indptr = f["raw/X/indptr"][:]
        data_ds, indices_ds = f["raw/X/data"], f["raw/X/indices"]
        n_genes = f["raw/X"].attrs["shape"][1]

        rows_data, rows_indices, rows_indptr = [], [], [0]
        for row in picked_global:
            start, end = int(indptr[row]), int(indptr[row + 1])
            rows_data.append(data_ds[start:end])
            rows_indices.append(indices_ds[start:end])
            rows_indptr.append(rows_indptr[-1] + (end - start))

        # All nonzero entries of every picked row (not a subsample: with ~600 picked cells this is
        # all of them) must be integer-valued, confirming raw.X holds raw counts.
        all_data = np.concatenate(rows_data)
        assert np.all(all_data == np.rint(all_data)), "raw.X is not integer-valued"

        mat = sparse.csr_matrix(
            (all_data, np.concatenate(rows_indices), np.array(rows_indptr)),
            shape=(len(picked_global), n_genes),
        )

    detected = np.asarray((mat > 0).sum(axis=0)).ravel()
    gene_mask = detected >= args.min_cells_per_gene
    counts = mat[:, gene_mask].toarray()
    genes = gene_name[gene_mask]

    df = pd.DataFrame(counts.T, index=genes, columns=obs_sub.index)
    df.index.name = "GeneID"
    df.to_csv(args.output_dir / "counts.tsv", sep="\t")
    obs_sub.to_csv(args.output_dir / "cell_meta.csv")

    print(f"wrote {df.shape[1]} cells x {df.shape[0]} genes to {args.output_dir}")


if __name__ == "__main__":
    main()
