"""Write AnnData inputs as an upstream Bonsai data folder (genes x cells TSV)."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ._core.helpers import write_ids

FMT = "%.17g"  # round-trips float64 exactly


def _dense(x) -> np.ndarray:
    return np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float64)


def _layers(adata: AnnData) -> tuple[str, str | None]:
    info = adata.uns.get("bonsai_input")
    if info is None:
        raise ValueError("adata.uns['bonsai_input'] missing: use bonsai.pp.read_sanity or bonsai.pp.from_arrays")
    if info["kind"] == "sanity_posterior":
        return "bonsai_mean", "bonsai_sd"
    return info["mean_layer"], info["sd_layer"]


def validate_inputs(adata: AnnData) -> tuple[np.ndarray, np.ndarray | None]:
    """Dense float64 (cells x genes) mean and SD; raise ValueError on non-finite or SD <= 0."""
    mean_layer, sd_layer = _layers(adata)  # raises if adata.uns["bonsai_input"] is missing
    info = adata.uns["bonsai_input"]
    mean = _dense(adata.layers[mean_layer])
    if not np.isfinite(mean).all():
        raise ValueError(f"layer {mean_layer!r} has non-finite values")
    sd = None
    if sd_layer is not None:
        sd = _dense(adata.layers[sd_layer])
        if not (np.isfinite(sd).all() and (sd > 0).all()):
            raise ValueError(f"layer {sd_layer!r} must be finite and > 0")
    if info["kind"] == "sanity_posterior":
        prior_var = adata.var["bonsai_prior_var"].to_numpy()
        if not (np.isfinite(prior_var).all() and (prior_var > 0).all()):
            raise ValueError("var['bonsai_prior_var'] must be finite and > 0")
        mu = adata.var["bonsai_mu"].to_numpy()
        if not np.isfinite(mu).all():
            raise ValueError("var['bonsai_mu'] has non-finite values")
    return mean, sd


def write_data_folder(adata: AnnData, folder: Path) -> dict:
    """Write the inputs in the format upstream reads; return the matching config keys."""
    mean, sd = validate_inputs(adata)  # validate before creating the folder, so a failure leaves nothing behind
    folder = Path(folder)
    folder.mkdir(parents=True)
    write_ids(str(folder / "cellID.txt"), list(adata.obs_names))
    write_ids(str(folder / "geneID.txt"), list(adata.var_names))
    info = adata.uns["bonsai_input"]
    if info["kind"] == "sanity_posterior":
        assert sd is not None, "validate_inputs always returns an SD array for sanity_posterior input"
        np.savetxt(folder / "delta.txt", mean.T, fmt=FMT, delimiter="\t")
        np.savetxt(folder / "d_delta.txt", sd.T, fmt=FMT, delimiter="\t")
        np.savetxt(folder / "variance.txt", adata.var["bonsai_prior_var"].to_numpy(), fmt=FMT)
        np.savetxt(folder / "mu.txt", adata.var["bonsai_mu"].to_numpy(), fmt=FMT)
        (folder / "sanity_command.txt").write_text(
            f"# Sanity version: 2.0.0\n# Method: {info['sanity_method']}\n"
            "# written by bonsai._io_folder.write_data_folder\n"
        )
        return {"data_folder": str(folder), "filenames_data": "delta.txt,d_delta.txt", "input_is_sanity_output": True}
    np.savetxt(folder / "features.txt", mean.T, fmt=FMT, delimiter="\t")
    names = "features.txt"
    if sd is not None:
        np.savetxt(folder / "standard_deviations.txt", sd.T, fmt=FMT, delimiter="\t")
        names += ",standard_deviations.txt"
    return {"data_folder": str(folder), "filenames_data": names, "input_is_sanity_output": False}
