"""Tools: reconstruct the Bonsai tree for an AnnData object."""

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from ruamel.yaml import YAML

from . import io
from ._core.main import run_from_config
from ._io_folder import write_data_folder

_BAD_CELL_NAME = re.compile(r"[\s(),:;]")
_BAD_GENE_NAME = re.compile(r"[\t\n]")
_RESERVED_CELL_NAME = re.compile(r"^(root|internal_\d+)$")


def _check_names(names, what):
    if not names.is_unique:
        raise ValueError(f"{what} names must be unique")
    if what == "gene":
        # Genes are never newick labels (they only index per-gene TSV columns), so only a
        # tab/newline (which would corrupt that TSV) is refused; see docs/deviations.md.
        bad = [n for n in names if _BAD_GENE_NAME.search(str(n))]
        if bad:
            raise ValueError(f"gene name {bad[0]!r} contains a tab or newline")
        return
    bad = [n for n in names if _BAD_CELL_NAME.search(str(n))]
    if bad:
        raise ValueError(f"{what} name {bad[0]!r} contains whitespace or one of '(),:;'")
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

    Parameters mirror the upstream YAML config (defaults = upstream template defaults). Needs
    ``adata.uns["bonsai_input"]`` set by :func:`bonsai.pp.read_sanity` or
    :func:`bonsai.pp.from_arrays`.

    Validation order (before anything is written): cell count, then cell/gene names, then
    workdir emptiness, then input layers/columns (``bonsai._io_folder.validate_inputs``).

    The work directory holds Bonsai's inputs and outputs as plain text: roughly
    ``2 * n_cells * n_genes`` numeric values for the mean/SD matrices alone, plus upstream's own
    intermediate and result files. At 100k cells x 5k genes this is tens of GB. Pass ``workdir=``
    on a filesystem with enough free space, or point ``TMPDIR`` there before calling with
    ``workdir=None``.

    Parameters
    ----------
    adata
        Cells x genes; ``adata.uns["bonsai_input"]`` must already be set.
    zscore_cutoff
        Signal-to-noise cutoff on gene variance. A gene is kept only if the mean over cells of
        ``mean**2 / var`` exceeds ``zscore_cutoff**2`` (``_core/dataprocessing.py:read_and_filter``,
        ``zscoreSq = np.sum(means**2 / vars) / nCells``, ``var = max(sd**2, 1e-12)``). A negative
        value keeps (almost) all genes, since the cutoff is then treated as 0.
    rescale_by_var
        Whether each gene's coordinate is rescaled by its inferred variance before the knn/SPR
        search, i.e. whether a prior is placed that expression changes are more likely in genes
        that already vary a lot across the dataset (``_core/config.py``'s ``--rescale_by_var``
        help text).
    use_knn
        Number of nearest neighbours used to find candidate pairs to merge (candidate generation,
        not tree initialisation). ``-1`` considers all pairs of leaves instead.
    UB_ellipsoid_size
        Size of the root-position/precision ellipsoid used for a computational upper bound (UB) on
        how much a candidate merge could increase the log-likelihood, letting Bonsai skip
        recomputing merges whose UB cannot beat the current best. A value ``< 0`` disables this
        shortcut entirely. It is a computational optimisation only: it changes neither the
        candidate pairs considered nor the final result, only how much work is spent choosing
        between them.
    nnn_n_randommoves
        Number of random nearest-neighbour-interchange (NNI) moves performed (per random starting
        tree) before continuing with greedy NNI.
    nnn_n_randomtrees
        Number of random starting trees (each built via ``nnn_n_randommoves`` random NNI moves);
        the one with the highest log-likelihood is kept and refined greedily.
    spr_strategy
        Upstream SPR candidate/target selection strategy (``"large_tree"``, ``"long_branch_sorted"``
        or ``"super_sure"``; see ``_core/dataprocessing.py:do_spr_moveset``).
    verbose
        Forwarded to upstream as ``args.verbose``; only gates extra ``mp_print``/logging calls, not
        any decision (see ``docs/deviations.md``). Default differs from the upstream config
        template's default of ``True``.
    workdir
        Directory for inputs/intermediate/output files. If ``None`` (default), a temporary
        directory is created and removed again unless ``keep_workdir=True``. If given and it
        already exists, it must be empty.
    keep_workdir
        If ``True``, keep ``workdir`` (resolved to an absolute path in
        ``adata.uns["bonsai"]["workdir"]``) after the run; needed by
        :func:`node_posteriors`. If ``False`` (default), the directory (or just the files written
        into a pre-existing one) is removed, even if the run raised.
    copy
        If ``True``, operate on and return a copy of ``adata``, leaving the original untouched. If
        ``False`` (default), ``adata`` is modified in place and ``None`` is returned.

    Returns
    -------
    The modified (or copied) AnnData if ``copy=True``, else ``None``. On success, writes
    ``uns["bonsai"]`` (newick, edges, nodes, loglik, genes_used, config, workdir),
    ``obs["bonsai_node"]`` (vertInd of each cell) and ``var["bonsai_used"]`` (genes kept by
    upstream's filters: the ``zscore_cutoff`` above and, for ``sanity_posterior`` input, a second
    per-cell check that ``prior_var - var >= 1e-4`` everywhere, else the gene is dropped;
    ``_core/dataprocessing.py:read_and_filter``, ``divide_out_prior_cutoff``).

    Raises
    ------
    ValueError
        Fewer than 3 cells; a duplicate, reserved or otherwise invalid cell/gene name (see
        ``_check_names``); a non-empty pre-existing ``workdir``; or an invalid/missing input layer
        or column (``bonsai._io_folder.validate_inputs``).
    """
    if adata.n_obs < 3:
        raise ValueError("need at least 3 cells")
    _check_names(adata.obs_names, "cell")
    _check_names(adata.var_names, "gene")
    adata = adata.copy() if copy else adata  # before any directory is created, so a failing copy leaves nothing behind
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
        adata.uns["bonsai"]["workdir"] = str(wd.resolve()) if keep_workdir else None
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

    Reads ``posterior_ltqs_vertByGene.npy``/``posterior_ltqsVars_vertByGene.npy`` that upstream
    already wrote into the kept work directory; it does not recompute them.

    Parameters
    ----------
    adata
        An AnnData produced by :func:`reconstruct(..., keep_workdir=True) <reconstruct>` (or by
        :func:`bonsai.io.read_results` on such a kept results folder, with
        ``uns["bonsai"]["workdir"]`` still pointing at it).

    Returns
    -------
    AnnData with one row per tree node (``obs`` indexed by node name, ordered by ``vertInd``) and
    one column per gene used by the reconstruction (``var_names`` = ``uns["bonsai"]["genes_used"]``,
    in the ``.npy`` column order). ``X`` holds posterior means; ``layers["var"]`` holds posterior
    variances.

    Raises
    ------
    ValueError
        ``adata.uns["bonsai"]["workdir"]`` is missing/falsy, or ``nodes["vertInd"]`` is not a
        ``0..n-1`` permutation (so it cannot be used to align rows to the ``.npy`` files).
    """
    wd = adata.uns.get("bonsai", {}).get("workdir")
    if not wd:
        raise ValueError("node posteriors need the work directory: run reconstruct(..., keep_workdir=True)")
    final = io._final_folder(Path(wd) / "results")
    nodes = adata.uns["bonsai"]["nodes"].sort_values("vertInd")
    if not (nodes["vertInd"].to_numpy() == np.arange(len(nodes))).all():
        raise ValueError("nodes['vertInd'] is not a 0..n-1 permutation; cannot align to the .npy row order")
    genes_used = adata.uns["bonsai"]["genes_used"]
    out = AnnData(
        X=np.load(final / "posterior_ltqs_vertByGene.npy"),
        obs=nodes.set_index(nodes["name"].astype(str)),
        var=pd.DataFrame(index=pd.Index(genes_used, dtype=object)),
    )
    out.layers["var"] = np.load(final / "posterior_ltqsVars_vertByGene.npy")
    return out
