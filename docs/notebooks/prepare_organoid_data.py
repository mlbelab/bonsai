r"""Prepare a Sanity input (TSV counts) from an organoid h5ad for the tour notebook.

Subsamples cells stratified by one or more ``obs`` columns (default ``age,lineage``, the brain
dataset), keeps genes detected in at least ~10 of the sampled cells, and writes a tab-separated
raw-count matrix that Sanity's ``-f`` flag accepts (genes x cells, as required by Sanity's TSV
format).

With the defaults (``--n-cells 500`` over the brain dataset's ~23 non-empty age x lineage strata),
each stratum requests about 22 cells; small strata are under-filled (fewer than 22 cells with that
age and lineage), so the total sampled is usually somewhat below ``--n-cells``.

Usage
-----
    python prepare_organoid_data.py <input.h5ad> <output_dir> [--n-cells 500] [--seed 0]
    python prepare_organoid_data.py <eec.h5ad> <output_dir> --strata-cols cell_type_initial \\
        --counts-source X --round

Writes ``<output_dir>/counts.tsv`` and ``<output_dir>/cell_meta.csv`` (the sampled cells' obs
columns used for notebook plotting). ``--counts-source X`` reads raw counts from ``.X`` instead of
``layers["counts"]``; ``--round`` rounds (``np.rint``, clipped at 0) and reports the fraction of
entries changed, for datasets whose "counts" include fractional multi-mapping splits.
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
    p.add_argument("--strata-cols", default="age,lineage", help="comma-separated obs columns to stratify on")
    p.add_argument("--counts-source", choices=["layers:counts", "X"], default="layers:counts")
    p.add_argument("--round", action="store_true", help="round counts to integers (np.rint, clipped at 0)")
    args = p.parse_args()
    strata_cols = args.strata_cols.split(",")

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read in backed mode; obs is small, so subset first and only densify the sampled counts.
    a = ad.read_h5ad(args.input_h5ad, backed="r")

    obs = a.obs.dropna(subset=strata_cols).copy()
    strata = obs.groupby(strata_cols, observed=True)
    n_strata = strata.ngroups
    per_stratum = max(1, args.n_cells // n_strata)

    # `strata.indices` gives positions within the filtered `obs`, not within the unfiltered `a` —
    # go through cell-name labels so the final selection indexes `a` correctly regardless of the
    # two frames' row order.
    picked_names = []
    strata_report = []
    for key, idx in strata.indices.items():
        key = key if isinstance(key, tuple) else (key,)
        k = min(per_stratum, len(idx))
        chosen = rng.choice(idx, size=k, replace=False)
        names = obs.index[chosen]
        picked_names.append(names)
        strata_report.append((*key, len(idx), k))
    picked_names = pd.Index(np.concatenate(picked_names)).unique()
    if len(picked_names) > args.n_cells:
        picked_names = pd.Index(rng.choice(picked_names, size=args.n_cells, replace=False))

    sub = a[picked_names].to_memory()
    for col in strata_cols:
        assert sub.obs[col].notna().all(), f"picked cells must all have a non-NaN {col}"
    picked_pairs = set(zip(*(sub.obs[col] for col in strata_cols), strict=True))
    requested_pairs = {tuple(row[: len(strata_cols)]) for row in strata_report}
    assert picked_pairs <= requested_pairs, "picked strata must come from the requested strata"

    print(f"requested ~{per_stratum} cells per stratum across {n_strata} {tuple(strata_cols)} strata")
    for row in sorted(strata_report):
        *key, available, k = row
        print(f"  {dict(zip(strata_cols, key, strict=True))}: {k}/{per_stratum} requested ({available} available)")
    # Real ERCC spike-in probe names only (e.g. "ERCC-00002"); this must not match human genes
    # that merely start with "ERCC" (ERCC1-8, ERCC6L, ERCC6L2, ERCC8-AS1, BIVM-ERCC5, PERCC1).
    spike_in = sub.var_names.str.match(r"^ERCC-\d{5}$")
    if spike_in.any():
        print(f"dropping {spike_in.sum()} spike-in/ERCC var(s)")
        sub = sub[:, ~spike_in].copy()

    counts = sub.X if args.counts_source == "X" else sub.layers["counts"]
    counts = np.asarray(counts.todense() if hasattr(counts, "todense") else counts)

    if args.round:
        rounded = np.clip(np.rint(counts), 0, None)
        changed = rounded != counts
        changed_frac_all = np.mean(changed)
        nonzero = counts != 0
        changed_frac_nonzero = np.mean(changed[nonzero]) if nonzero.any() else float("nan")
        print(
            f"rounding changed {changed_frac_all:.4%} of all count entries "
            f"({changed_frac_nonzero:.4%} of nonzero entries)"
        )
        counts = rounded

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
