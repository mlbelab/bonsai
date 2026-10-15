"""Prepare a Sanity input (TSV counts) from the brain-organoid h5ad for the tour notebook.

Subsamples cells stratified by (age, lineage), keeps genes detected in at least ~10 of the
sampled cells, and writes a tab-separated raw-count matrix that Sanity's ``-f`` flag accepts
(genes x cells, as required by Sanity's TSV format).

With the defaults (``--n-cells 500`` over the dataset's ~23 non-empty age x lineage strata), each
stratum requests about 22 cells; small strata are under-filled (fewer than 22 cells with that age
and lineage), so the total sampled is usually somewhat below ``--n-cells``.

Usage
-----
    python prepare_organoid_data.py <input.h5ad> <output_dir> [--n-cells 500] [--seed 0]

Writes ``<output_dir>/counts.tsv`` and ``<output_dir>/cell_meta.csv`` (the sampled cells' obs
columns used for notebook plotting).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    """Parse CLI args, subsample and filter the input h5ad, and write the Sanity-input TSV/CSV."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_h5ad", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--n-cells", type=int, default=500)
    p.add_argument("--min-cells-per-gene", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read in backed mode; obs is small, so subset first and only densify the sampled counts.
    a = ad.read_h5ad(args.input_h5ad, backed="r")

    obs = a.obs.dropna(subset=["age", "lineage"]).copy()
    strata = obs.groupby(["age", "lineage"], observed=True)
    n_strata = strata.ngroups
    per_stratum = max(1, args.n_cells // n_strata)

    # `strata.indices` gives positions within the filtered `obs`, not within the unfiltered `a` —
    # go through cell-name labels so the final selection indexes `a` correctly regardless of the
    # two frames' row order.
    picked_names = []
    strata_report = []
    for (age, lineage), idx in strata.indices.items():
        k = min(per_stratum, len(idx))
        chosen = rng.choice(idx, size=k, replace=False)
        names = obs.index[chosen]
        picked_names.append(names)
        strata_report.append((age, lineage, len(idx), k))
    picked_names = pd.Index(np.concatenate(picked_names)).unique()
    if len(picked_names) > args.n_cells:
        picked_names = pd.Index(rng.choice(picked_names, size=args.n_cells, replace=False))

    sub = a[picked_names].to_memory()
    assert sub.obs["lineage"].notna().all(), "picked cells must all have a non-NaN lineage"
    assert sub.obs["age"].notna().all(), "picked cells must all have a non-NaN age"
    picked_pairs = set(zip(sub.obs["age"], sub.obs["lineage"], strict=True))
    requested_pairs = {(age, lineage) for age, lineage, _, _ in strata_report}
    assert picked_pairs <= requested_pairs, "picked (age, lineage) pairs must come from the requested strata"

    print(f"requested ~{per_stratum} cells per stratum across {n_strata} (age, lineage) strata")
    for age, lineage, available, k in sorted(strata_report):
        print(f"  age={age}, lineage={lineage}: {k}/{per_stratum} requested ({available} available)")
    counts = sub.layers["counts"]
    counts = np.asarray(counts.todense() if hasattr(counts, "todense") else counts)

    detected = (counts > 0).sum(axis=0)
    gene_mask = detected >= args.min_cells_per_gene
    counts = counts[:, gene_mask]
    genes = sub.var_names[gene_mask]

    df = pd.DataFrame(counts.T, index=genes, columns=sub.obs_names)
    df.index.name = "GeneID"
    df.to_csv(args.output_dir / "counts.tsv", sep="\t")
    sub.obs.to_csv(args.output_dir / "cell_meta.csv")

    print(f"wrote {df.shape[1]} cells x {df.shape[0]} genes to {args.output_dir}")


if __name__ == "__main__":
    main()
