"""Read upstream Bonsai result folders into AnnData."""

import json
from pathlib import Path

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

    Without ``adata``, returns a new AnnData with the result's cells (obs) and used genes (var).

    Note: ``edgeInfo.txt`` branch lengths are written upstream with 9 significant digits
    (``%.8e``); ``tree.nwk`` carries full precision. Consumers wanting exact lengths should
    parse the newick rather than ``uns["bonsai"]["edges"]["t"]``.
    """
    final = _final_folder(Path(folder))
    meta = json.loads((final / "metadata.json").read_text())
    nodes = pd.read_csv(final / "vertInfo.txt", sep="\t")[["vertInd", "vertName"]].rename(columns={"vertName": "name"})
    edges = pd.read_csv(final / "edgeInfo.txt", sep="\t", header=None, names=["parent", "child", "t"])
    loglik = float(pd.read_csv(final / "metadata_bonsai.txt", index_col=0).loc["metadata", "finalLoglik"])
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
