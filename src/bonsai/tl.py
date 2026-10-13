"""Tools: reconstruct the Bonsai tree for an AnnData object."""

import re
import shutil
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from anndata import AnnData
from ruamel.yaml import YAML

from . import io
from ._core.main import run_from_config
from ._io_folder import write_data_folder

_BAD_NAME = re.compile(r"[\s(),:;]")
_RESERVED_CELL_NAME = re.compile(r"^(root|internal_\d+)$")


def _check_names(names, what):
    if not names.is_unique:
        raise ValueError(f"{what} names must be unique")
    bad = [n for n in names if _BAD_NAME.search(str(n))]
    if bad:
        raise ValueError(f"{what} name {bad[0]!r} contains whitespace or one of '(),:;'")
    if what == "cell":
        reserved = [n for n in names if _RESERVED_CELL_NAME.match(str(n))]
        if reserved:
            raise ValueError(
                f"cell name {reserved[0]!r} collides with upstream's internal vertex names ('root', 'internal_<n>')"
            )


def reconstruct(
    adata: AnnData,
    *,
    zscore_cutoff: float = 1.0,
    rescale_by_var: bool = True,
    use_knn: int = 10,
    UB_ellipsoid_size: float = 1.0,
    nnn_n_randommoves: int = 1000,
    nnn_n_randomtrees: int = 10,
    spr_strategy: str = "large_tree",
    verbose: bool = False,
    workdir: str | Path | None = None,
    keep_workdir: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Reconstruct the maximum-likelihood Bonsai tree (upstream algorithm, unchanged).

    Parameters mirror the upstream YAML config (defaults = upstream template defaults).
    Writes ``uns["bonsai"]``, ``obs["bonsai_node"]`` and ``var["bonsai_used"]``.

    Validation order (before anything is written): cell count, then cell/gene names, then
    workdir emptiness.
    """
    if adata.n_obs < 3:
        raise ValueError("need at least 3 cells")
    _check_names(adata.obs_names, "cell")
    _check_names(adata.var_names, "gene")
    tmp = workdir is None
    if tmp:
        wd = Path(tempfile.mkdtemp(prefix="bonsai_"))
        existed = False  # ours: mkdtemp already created it, but it is fully disposable
    else:
        assert workdir is not None, "tmp is False only when workdir was given"
        wd = Path(workdir)
        existed = wd.exists()
        if existed and any(wd.iterdir()):
            raise ValueError(f"workdir {wd} exists and is not empty")
    adata = adata.copy() if copy else adata
    try:
        config = {
            "dataset": "bonsai",
            "results_folder": str(wd / "results"),
            "verbose": verbose,
            "zscore_cutoff": zscore_cutoff,
            "rescale_by_var": rescale_by_var,
            "use_knn": use_knn,
            "UB_ellipsoid_size": UB_ellipsoid_size,
            "nnn_n_randommoves": nnn_n_randommoves,
            "nnn_n_randomtrees": nnn_n_randomtrees,
            "spr_strategy": spr_strategy,
            "pickup_intermediate": False,
            "tmp_folder": "",
            **write_data_folder(adata, wd / "data"),
        }
        cfg = wd / "config.yaml"
        with cfg.open("w") as f:
            YAML().dump(config, f)
        run_from_config(cfg)
        io.read_results(wd / "results", adata)
        adata.uns["bonsai"]["config"] = config
        adata.uns["bonsai"]["workdir"] = str(wd) if keep_workdir else None
    finally:
        if not keep_workdir:
            if existed:
                # wd was the user's own (empty) directory: clear what we put in it, keep wd itself.
                for child in wd.iterdir():
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
            else:
                shutil.rmtree(wd, ignore_errors=True)
    return adata if copy else None


def node_posteriors(adata: AnnData) -> AnnData:
    """Posterior means/variances for every tree node (obs = nodes, var = genes used).

    Reads the files upstream wrote; requires ``reconstruct(..., keep_workdir=True)``.
    """
    wd = adata.uns.get("bonsai", {}).get("workdir")
    if not wd:
        raise ValueError("node posteriors need the work directory: run reconstruct(..., keep_workdir=True)")
    final = io._final_folder(Path(wd) / "results")
    nodes = adata.uns["bonsai"]["nodes"]
    var = cast(pd.DataFrame, adata.var)  # in-memory AnnData: var is always a DataFrame, never Dataset2D
    out = AnnData(
        X=np.load(final / "posterior_ltqs_vertByGene.npy"),
        obs=nodes.set_index(nodes["name"].astype(str)),
        var=var.loc[var["bonsai_used"].to_numpy(), []],
    )
    out.layers["var"] = np.load(final / "posterior_ltqsVars_vertByGene.npy")
    return out
