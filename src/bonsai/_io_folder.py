"""Write AnnData inputs as an upstream Bonsai data folder (genes x cells TSV)."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ._core.helpers import write_ids

FMT = "%.17g"  # round-trips float64 exactly; used only for the small per-gene vectors below
_KINDS = ("sanity_posterior", "likelihood")


def _dense(x) -> np.ndarray:
    """Dense array in its own dtype (sparse densified to its own dtype); no float64 copy."""
    return np.asarray(x.toarray() if sp.issparse(x) else x)


def _write_matrix(path: Path, x: np.ndarray) -> None:
    """Write x (cells x genes) as genes x cells TSV, one gene (column) at a time.

    Each row is cast to float64 individually and formatted with Python's shortest round-trip
    ``repr`` (equivalent to ``%.17g`` for float64 but roughly half the text size). This avoids
    densifying/copying the whole matrix to float64 at once; a float32 value's float64 cast is
    always exact, so float32 input still round-trips exactly.
    """
    with path.open("w") as f:
        for col in range(x.shape[1]):
            row = np.asarray(x[:, col], dtype=np.float64)
            f.write("\t".join(map(repr, row.tolist())) + "\n")


def _layers(adata: AnnData) -> tuple[str, str | None]:
    info = adata.uns.get("bonsai_input")
    if info is None:
        raise ValueError("adata.uns['bonsai_input'] missing: use bonsai.pp.read_sanity or bonsai.pp.from_arrays")
    kind = info.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"adata.uns['bonsai_input']['kind'] must be one of {_KINDS}, got {kind!r}")
    if kind == "sanity_posterior":
        if "sanity_method" not in info:
            raise ValueError("adata.uns['bonsai_input'] is missing 'sanity_method'")
        return "bonsai_mean", "bonsai_sd"
    for key in ("mean_layer", "sd_layer"):
        if key not in info:
            raise ValueError(f"adata.uns['bonsai_input'] is missing {key!r}")
    return info["mean_layer"], info["sd_layer"]


def validate_inputs(adata: AnnData) -> tuple[np.ndarray, np.ndarray | None]:
    """Dense (cells x genes) mean and SD in their original dtype.

    Raises ``ValueError`` for a missing/malformed ``uns["bonsai_input"]`` (missing keys, an
    unknown ``kind``, a layer that no longer exists), and for non-finite mean/SD/mu or a negative
    SD. ``sd == 0`` and a non-positive ``bonsai_prior_var`` are accepted (upstream clamps them);
    see ``docs/deviations.md``.
    """
    mean_layer, sd_layer = _layers(adata)  # raises ValueError on a missing/malformed bonsai_input
    info = adata.uns["bonsai_input"]
    if mean_layer not in adata.layers:
        raise ValueError(f"layer {mean_layer!r} not in adata.layers")
    mean = _dense(adata.layers[mean_layer])
    if not np.isfinite(mean).all():
        raise ValueError(f"layer {mean_layer!r} has non-finite values")
    sd = None
    if sd_layer is not None:
        if sd_layer not in adata.layers:
            raise ValueError(f"layer {sd_layer!r} not in adata.layers")
        sd = _dense(adata.layers[sd_layer])
        if not (np.isfinite(sd).all() and (sd >= 0).all()):
            raise ValueError(f"layer {sd_layer!r} must be finite and >= 0")
    if info["kind"] == "sanity_posterior":
        if "bonsai_prior_var" not in adata.var:
            raise ValueError("var['bonsai_prior_var'] missing")
        prior_var = adata.var["bonsai_prior_var"].to_numpy()
        if not np.isfinite(prior_var).all():
            raise ValueError("var['bonsai_prior_var'] has non-finite values")
        if "bonsai_mu" not in adata.var:
            raise ValueError("var['bonsai_mu'] missing")
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
        _write_matrix(folder / "delta.txt", mean)
        _write_matrix(folder / "d_delta.txt", sd)
        np.savetxt(folder / "variance.txt", adata.var["bonsai_prior_var"].to_numpy(), fmt=FMT)
        np.savetxt(folder / "mu.txt", adata.var["bonsai_mu"].to_numpy(), fmt=FMT)
        (folder / "sanity_command.txt").write_text(
            f"# Sanity version: 2.0.0\n# Method: {info['sanity_method']}\n"
            "# written by bonsai._io_folder.write_data_folder\n"
        )
        return {"data_folder": str(folder), "filenames_data": "delta.txt,d_delta.txt", "input_is_sanity_output": True}
    _write_matrix(folder / "features.txt", mean)
    names = "features.txt"
    if sd is not None:
        _write_matrix(folder / "standard_deviations.txt", sd)
        names += ",standard_deviations.txt"
    return {"data_folder": str(folder), "filenames_data": names, "input_is_sanity_output": False}
