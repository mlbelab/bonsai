"""Preprocessing: get Bonsai inputs into an AnnData object."""

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core.helpers import parse_sanity_command_file, read_ids, resolve_sanity_filepaths

LEGACY_METHOD = "MLE"  # Sanity < 2.0 run with -max_v reports the maximum-likelihood gene variance


def read_sanity(path: str | Path) -> AnnData:
    """Read a Sanity output folder (>=2.0 or legacy `_vmax`) into a cells x genes AnnData.

    Sanity must have been run with ``-e 1`` and ``-v_m MAP`` (or ``MLE``/``EAP``); ``-v_m MARG``
    output is rejected (see the package README for why).

    Parameters
    ----------
    path
        Path to a Sanity output folder. Whether it is >=2.0 or legacy ``_vmax`` output is detected
        by the presence of ``sanity_command.txt``.

    Returns
    -------
    AnnData (cells x genes) with:

    - ``layers["bonsai_mean"]``: Sanity posterior means (delta).
    - ``layers["bonsai_sd"]``: Sanity posterior SDs (epsilon).
    - ``var["bonsai_prior_var"]``: Sanity per-gene prior variance.
    - ``var["bonsai_mu"]``: Sanity per-gene mean.
    - ``uns["bonsai_input"]``: ``{"kind": "sanity_posterior", "sanity_version": ..., "sanity_method": ...}``.

    ``obs_names``/``var_names`` come from ``cellID.txt``/``geneID.txt`` if present, else upstream's
    generic fallback names (``Cell_<i>``, ``Gene_<i>``).

    Raises
    ------
    ValueError
        ``sanity_command.txt`` records ``-v_m MARG`` output, or an expected Sanity output file is
        missing (via ``bonsai._core.helpers.resolve_sanity_filepaths``).
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
    """Mark general (non-Sanity) input: per-cell feature estimates with optional SDs (likelihood).

    Use this instead of :func:`read_sanity` when the mean/SD layers were computed some other way.
    Sets ``adata.uns["bonsai_input"]`` in place; does not otherwise modify ``adata``.

    Parameters
    ----------
    adata
        Cells x genes; must already have ``mean_layer`` (and ``sd_layer``, if given) in
        ``adata.layers``.
    mean_layer
        Name of the layer in ``adata.layers`` holding per-cell, per-gene mean estimates.
    sd_layer
        Name of the layer holding the matching standard deviations, or ``None`` (default) for a
        pure-likelihood model with no per-estimate uncertainty.

    Returns
    -------
    None. Sets ``adata.uns["bonsai_input"] = {"kind": "likelihood", "mean_layer": mean_layer,
    "sd_layer": sd_layer}``.

    Raises
    ------
    ValueError
        ``mean_layer`` or ``sd_layer`` is not a key of ``adata.layers``.
    """
    for layer in (mean_layer, sd_layer):
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"layer {layer!r} not in adata.layers")
    adata.uns["bonsai_input"] = {"kind": "likelihood", "mean_layer": mean_layer, "sd_layer": sd_layer}
