"""Preprocessing: get Bonsai inputs into an AnnData object."""

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core.helpers import parse_sanity_command_file, read_ids, resolve_sanity_filepaths

LEGACY_METHOD = "MLE"  # Sanity < 2.0 run with -max_v reports the maximum-likelihood gene variance


def read_sanity(path: str | Path) -> AnnData:
    """Read a Sanity output folder (>=2.0 or legacy `_vmax`) into a cells x genes AnnData.

    Sets ``layers["bonsai_mean"]`` (posterior means δ), ``layers["bonsai_sd"]`` (posterior SDs ε),
    ``var["bonsai_prior_var"]`` (Sanity gene variance), ``var["bonsai_mu"]`` (gene mean) and
    ``uns["bonsai_input"]``.
    """
    path = Path(path)
    files = resolve_sanity_filepaths(str(path))  # raises BonsaiError (ValueError) like upstream
    mean = np.loadtxt(files["means"], delimiter="\t", ndmin=2).T
    sd = np.loadtxt(files["stds"], delimiter="\t", ndmin=2).T
    cmd = path / "sanity_command.txt"
    version, method = parse_sanity_command_file(str(cmd)) if cmd.exists() else (None, LEGACY_METHOD)
    # Generic names match upstream's fallback (dataprocessing.py: 'Cell_'+str(ind), 'Gene_'+str(ind)).
    cell_ids = read_ids(str(path / "cellID.txt")) or [f"Cell_{i}" for i in range(mean.shape[0])]
    gene_ids = read_ids(str(path / "geneID.txt")) or [f"Gene_{i}" for i in range(mean.shape[1])]
    a = AnnData(obs=pd.DataFrame(index=cell_ids), var=pd.DataFrame(index=gene_ids))
    a.layers["bonsai_mean"], a.layers["bonsai_sd"] = mean, sd
    a.var["bonsai_prior_var"] = np.loadtxt(files["gene_variances"], ndmin=1)
    a.var["bonsai_mu"] = np.loadtxt(files["gene_means"], ndmin=1)
    a.uns["bonsai_input"] = {
        "kind": "sanity_posterior",
        "sanity_version": ".".join(map(str, version)) if version else "<2.0",
        "sanity_method": method,
    }
    return a


def from_arrays(adata: AnnData, mean_layer: str, sd_layer: str | None = None) -> None:
    """Mark general (non-Sanity) input: per-cell feature estimates with optional SDs (likelihood)."""
    for layer in (mean_layer, sd_layer):
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"layer {layer!r} not in adata.layers")
    adata.uns["bonsai_input"] = {"kind": "likelihood", "mean_layer": mean_layer, "sd_layer": sd_layer}
