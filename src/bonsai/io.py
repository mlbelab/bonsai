"""Read upstream Bonsai result folders into AnnData."""

import json
from pathlib import Path
from typing import SupportsFloat, cast

import pandas as pd
from anndata import AnnData


def _final_folder(folder: Path) -> Path:
    if (folder / "vertInfo.txt").exists():
        return folder
    found = sorted(folder.glob("final_bonsai*"))
    if len(found) != 1:
        raise ValueError(f"expected one final_bonsai* folder in {folder}, found {len(found)}")
    return found[0]


def read_results(folder: str | Path, adata: AnnData | None = None) -> AnnData:
    """Load a Bonsai result folder into ``uns["bonsai"]``, ``obs["bonsai_node"]``, ``var["bonsai_used"]``.

    Note: ``edgeInfo.txt`` branch lengths are written upstream with 9 significant digits
    (``%.8e``); ``tree.nwk`` carries full precision. Consumers wanting exact lengths should
    parse the newick rather than ``uns["bonsai"]["edges"]["t"]``.

    Parameters
    ----------
    folder
        A results folder, either a ``final_bonsai_*`` folder directly (detected by the presence of
        ``vertInfo.txt``) or its parent (containing exactly one ``final_bonsai_*`` subfolder).
    adata
        If given, the result is attached to this AnnData in place (and it is also returned); its
        ``obs_names`` must all be present among the result's cell names. If ``None`` (default), a
        new AnnData is created from the result's own cell/gene lists
        (``metadata.json["cellIds"]``/``["geneIds"]``).

    Returns
    -------
    The (possibly newly created) AnnData, with:

    - ``uns["bonsai"]``: ``newick``, ``edges`` (DataFrame: parent, child, t), ``nodes`` (DataFrame:
      vertInd, name), ``loglik``, ``genes_used``, ``upstream_commit``.
    - ``obs["bonsai_node"]``: each cell's ``vertInd``.
    - ``var["bonsai_used"]``: whether each gene is in ``genes_used``.

    Raises
    ------
    ValueError
        ``folder`` is not a result folder and does not contain exactly one ``final_bonsai_*``
        subfolder, or ``adata.obs_names`` has cells missing from the result's ``vertInfo.txt``
        (in which case ``adata`` is left unmodified: no ``uns``/``obs``/``var`` keys are set).
    """
    final = _final_folder(Path(folder))
    meta = json.loads((final / "metadata.json").read_text())
    # keep_default_na=False: cell/gene names like "NA", "NaN", "null" must stay strings, not become NaN.
    nodes = pd.read_csv(
        final / "vertInfo.txt",
        sep="\t",
        keep_default_na=False,
        na_values=[],
        dtype={"vertName": str},
    )[["vertInd", "vertName"]].rename(columns={"vertName": "name"})
    edges = pd.read_csv(final / "edgeInfo.txt", sep="\t", header=None, names=["parent", "child", "t"])
    raw_loglik = pd.read_csv(final / "metadata_bonsai.txt", index_col=0).loc["metadata", "finalLoglik"]
    loglik = float(cast(SupportsFloat, raw_loglik))
    if adata is None:
        adata = AnnData(obs=pd.DataFrame(index=meta["cellIds"]), var=pd.DataFrame(index=meta["geneIds"]))
    vert = nodes.set_index("name")["vertInd"]
    missing = adata.obs_names.difference(vert.index)
    if len(missing):
        raise ValueError(f"cells missing from {final / 'vertInfo.txt'}: {list(missing)[:5]}")
    adata.uns["bonsai"] = {
        "newick": (final / "tree.nwk").read_text().strip(),
        "edges": edges,
        "nodes": nodes,
        "loglik": loglik,
        "genes_used": list(meta["geneIds"]),
        "upstream_commit": "b16ef6f",
    }
    adata.obs["bonsai_node"] = vert.reindex(adata.obs_names).to_numpy()
    adata.var["bonsai_used"] = adata.var_names.isin(meta["geneIds"])
    return adata
