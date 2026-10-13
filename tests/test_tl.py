import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

import bonsai

from .golden import FIXTURE, GOLDEN, assert_trees_match, load_tree

GOLDEN_KW = {
    "zscore_cutoff": 1.0,
    "use_knn": 10,
    "UB_ellipsoid_size": 1.0,
    "nnn_n_randommoves": 10,
    "nnn_n_randomtrees": 2,
}


@pytest.fixture(scope="session")
def reconstructed(tmp_path_factory):
    a = bonsai.pp.read_sanity(FIXTURE)
    wd = tmp_path_factory.mktemp("tl") / "wd"
    bonsai.tl.reconstruct(a, workdir=wd, keep_workdir=True, **GOLDEN_KW)
    return a, wd


@pytest.mark.slow
def test_reconstruct_matches_golden(reconstructed):
    a, wd = reconstructed
    (final,) = (wd / "results").glob("final_bonsai*")
    assert_trees_match(final)  # topology exact; loglik/lengths bitwise or derived rtol
    assert a.uns["bonsai"]["loglik"] == load_tree(final).loglik  # uns carries exactly what upstream wrote
    assert a.uns["bonsai"]["newick"] == (final / "tree.nwk").read_text().strip()


@pytest.mark.slow
def test_uns_obs_var_slots(reconstructed):
    a, _ = reconstructed
    gold = bonsai.io.read_results(GOLDEN)
    assert a.uns["bonsai"]["newick"].count("Cell") == 64
    assert set(a.obs["bonsai_node"]) <= set(a.uns["bonsai"]["nodes"]["vertInd"])
    names = a.uns["bonsai"]["nodes"].set_index("vertInd")["name"]
    assert (names[a.obs["bonsai_node"]].to_numpy() == a.obs_names.to_numpy()).all()
    assert a.var["bonsai_used"].sum() == 1131
    assert list(a.var_names[a.var["bonsai_used"]]) == gold.uns["bonsai"]["genes_used"]


@pytest.mark.slow
def test_node_posteriors(reconstructed):
    a, wd = reconstructed
    n = bonsai.tl.node_posteriors(a)
    (final,) = (wd / "results").glob("final_bonsai*")
    assert n.shape == (len(a.uns["bonsai"]["nodes"]), 1131)
    assert list(n.obs_names) == list(a.uns["bonsai"]["nodes"]["name"].astype(str))
    assert list(n.var_names) == a.uns["bonsai"]["genes_used"]
    np.testing.assert_array_equal(n.X, np.load(final / "posterior_ltqs_vertByGene.npy"))
    np.testing.assert_array_equal(n.layers["var"], np.load(final / "posterior_ltqsVars_vertByGene.npy"))


def test_read_results_missing_cell_leaves_adata_untouched():
    a = bonsai.pp.read_sanity(FIXTURE)[:5].copy()
    a.obs_names = ["not_a_tree_cell"] + list(a.obs_names[1:])
    with pytest.raises(ValueError, match="missing"):
        bonsai.io.read_results(GOLDEN, a)
    assert "bonsai" not in a.uns
    assert "bonsai_node" not in a.obs


def test_read_results_cell_named_na(tmp_path):
    """A cell literally named 'NA' must stay a string through vertInfo.txt/metadata.json/tree.nwk,
    not get turned into NaN by pandas' default NA sniffing (mutant: drop keep_default_na=False)."""
    import shutil

    dest = tmp_path / "golden"
    shutil.copytree(GOLDEN, dest)
    old_name = "Cell011001"
    (dest / "vertInfo.txt").write_text((dest / "vertInfo.txt").read_text().replace(old_name, "NA"))
    (dest / "tree.nwk").write_text((dest / "tree.nwk").read_text().replace(old_name, "NA"))
    meta_path = dest / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["cellIds"] = [("NA" if c == old_name else c) for c in meta["cellIds"]]
    meta_path.write_text(json.dumps(meta))

    a = bonsai.io.read_results(dest)
    assert "NA" in set(a.obs_names)
    assert a.uns["bonsai"]["nodes"]["name"].isna().sum() == 0
    assert "NA" in set(a.uns["bonsai"]["nodes"]["name"])


def test_node_posteriors_needs_workdir():
    a = bonsai.io.read_results(GOLDEN)
    a.uns["bonsai"]["workdir"] = None
    with pytest.raises(ValueError, match="keep_workdir"):
        bonsai.tl.node_posteriors(a)


def _fake_run(cfg):
    """Stand-in for the pipeline: place the golden final folder where upstream would write it."""
    import shutil

    from ruamel.yaml import YAML

    results = Path(YAML().load(Path(cfg).read_text())["results_folder"])
    shutil.copytree(GOLDEN, results / "final_bonsai_zscore1.0")


@pytest.mark.parametrize("copy", [True, False])
def test_copy_semantics(monkeypatch, tmp_path, copy):
    a = bonsai.pp.read_sanity(FIXTURE)
    monkeypatch.setattr(bonsai.tl, "run_from_config", _fake_run)
    out = bonsai.tl.reconstruct(a, workdir=tmp_path / "wd", copy=copy)
    if copy:
        assert out is not a and "bonsai" in out.uns and "bonsai" not in a.uns
        assert "bonsai_node" not in a.obs
        assert "bonsai_used" not in a.var
    else:
        assert out is None and "bonsai" in a.uns and "bonsai_node" in a.obs
    assert not (tmp_path / "wd").exists()  # keep_workdir=False removes it


@pytest.mark.parametrize("name", ["a b", "a(b", "a,b", "a:b", "a;b", "a)b", "a\tb"])
def test_bad_cell_names_rejected(name, tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)[:5].copy()
    a.obs_names = [name] + list(a.obs_names[1:])
    with pytest.raises(ValueError, match="name"):
        bonsai.tl.reconstruct(a, workdir=tmp_path / "wd")


def test_duplicate_cell_names_rejected(tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)[:5].copy()
    a.obs_names = ["x"] * 5
    with pytest.raises(ValueError, match="unique"):
        bonsai.tl.reconstruct(a, workdir=tmp_path / "wd")


def test_nonempty_workdir_rejected(tmp_path):
    (tmp_path / "wd").mkdir()
    (tmp_path / "wd" / "old").write_text("x")
    a = bonsai.pp.read_sanity(FIXTURE)[:5].copy()
    with pytest.raises(ValueError, match="workdir"):
        bonsai.tl.reconstruct(a, workdir=tmp_path / "wd")


@pytest.mark.parametrize("name", ["root", "internal_0", "internal_12"])
def test_reserved_cell_names_rejected(name, tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)[:5].copy()
    a.obs_names = [name] + list(a.obs_names[1:])
    with pytest.raises(ValueError, match="internal vertex"):
        bonsai.tl.reconstruct(a, workdir=tmp_path / "wd")


def test_gene_name_with_colon_accepted(tmp_path, monkeypatch):
    """Gene names are not newick labels: 'chr1:100-200' (unlike a cell name) is fine."""
    import shutil

    from ruamel.yaml import YAML

    def fake_run(cfg):
        results = Path(YAML().load(Path(cfg).read_text())["results_folder"])
        shutil.copytree(GOLDEN, results / "final_bonsai_zscore1.0")

    a = bonsai.pp.read_sanity(FIXTURE)
    a.var_names = ["chr1:100-200"] + list(a.var_names[1:])
    monkeypatch.setattr(bonsai.tl, "run_from_config", fake_run)
    bonsai.tl.reconstruct(a, workdir=tmp_path / "wd")  # must not raise


def test_bad_gene_name_rejected_through_reconstruct(tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)[:5].copy()
    a.var_names = ["a\tb"] + list(a.var_names[1:])
    with pytest.raises(ValueError, match="tab or newline"):
        bonsai.tl.reconstruct(a, workdir=tmp_path / "wd")


def test_preexisting_empty_workdir_survives(monkeypatch, tmp_path):
    """A user-created empty workdir is not deleted by keep_workdir=False; only its contents are."""
    wd = tmp_path / "wd"
    wd.mkdir()
    a = bonsai.pp.read_sanity(FIXTURE)
    monkeypatch.setattr(bonsai.tl, "run_from_config", _fake_run)
    bonsai.tl.reconstruct(a, workdir=wd, copy=True)
    assert wd.exists() and wd.is_dir()
    assert list(wd.iterdir()) == []


def _fake_run_then_fail(cfg):
    """Stand-in for a pipeline that writes something, then blows up before completing."""
    from ruamel.yaml import YAML

    results = Path(YAML().load(Path(cfg).read_text())["results_folder"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "partial.txt").write_text("x")
    raise RuntimeError("boom")


def test_reconstruct_cleans_preexisting_workdir_on_failure(monkeypatch, tmp_path):
    """A pre-existing empty workdir survives a failed run: contents are cleared, dir stays, and
    the half-finished `uns["bonsai"]` is never attached."""
    wd = tmp_path / "wd"
    wd.mkdir()
    a = bonsai.pp.read_sanity(FIXTURE)
    monkeypatch.setattr(bonsai.tl, "run_from_config", _fake_run_then_fail)
    with pytest.raises(RuntimeError, match="boom"):
        bonsai.tl.reconstruct(a, workdir=wd)
    assert wd.exists() and wd.is_dir()
    assert list(wd.iterdir()) == []
    assert "bonsai" not in a.uns


def test_reconstruct_removes_temp_workdir_on_failure(monkeypatch):
    """With workdir=None, the temp dir reconstruct created itself is fully removed on failure."""
    captured = {}

    def fake_run(cfg):
        captured["wd"] = Path(cfg).parent
        _fake_run_then_fail(cfg)

    a = bonsai.pp.read_sanity(FIXTURE)
    monkeypatch.setattr(bonsai.tl, "run_from_config", fake_run)
    with pytest.raises(RuntimeError, match="boom"):
        bonsai.tl.reconstruct(a)
    assert "wd" in captured
    assert not captured["wd"].exists()
    assert "bonsai" not in a.uns


def test_too_few_cells_rejected(tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)[:2].copy()
    with pytest.raises(ValueError, match="at least 3 cells"):
        bonsai.tl.reconstruct(a, workdir=tmp_path / "wd")


def test_validation_before_any_write(tmp_path):
    """Cell-count, name and workdir checks must all happen before any file is created."""
    wd = tmp_path / "wd"
    a = bonsai.pp.read_sanity(FIXTURE)[:5].copy()
    a.obs_names = ["x"] * 5
    with pytest.raises(ValueError, match="unique"):
        bonsai.tl.reconstruct(a, workdir=wd)
    assert not wd.exists()  # duplicate-name rejection happened before workdir/data were touched


def test_run_configs_roundtrip_matches_golden(tmp_path, monkeypatch):
    """ruamel's dump of tl.reconstruct's config dict reads back identical to the golden config
    for every key emitted by tl.reconstruct (Run_Configs.__init__ coerces unknown/missing keys to
    its own defaults, so comparing the raw YAML would be misleading)."""
    from bonsai._core.helpers import Run_Configs

    a = bonsai.pp.read_sanity(FIXTURE)
    monkeypatch.setattr(bonsai.tl, "run_from_config", _fake_run)
    bonsai.tl.reconstruct(a, workdir=tmp_path / "wd", keep_workdir=True, **GOLDEN_KW)
    mine = Run_Configs(str(tmp_path / "wd" / "config.yaml"))
    gold = Run_Configs(str(GOLDEN / "config.yaml"))
    # "verbose" is excluded on purpose: default False here, True in the golden template.
    shared = [
        "zscore_cutoff",
        "rescale_by_var",
        "use_knn",
        "UB_ellipsoid_size",
        "nnn_n_randommoves",
        "nnn_n_randomtrees",
        "spr_strategy",
        "pickup_intermediate",
        "tmp_folder",
        "input_is_sanity_output",
    ]
    for key in shared:
        assert getattr(mine, key) == getattr(gold, key), key


def test_reconstruct_h5ad_roundtrip(tmp_path, monkeypatch):
    """A reconstructed AnnData's bonsai slots survive write_h5ad/read_h5ad (uns["bonsai"] holds
    DataFrames, scalars and a dict config; workdir is None here since keep_workdir=False)."""
    a = bonsai.pp.read_sanity(FIXTURE)
    monkeypatch.setattr(bonsai.tl, "run_from_config", _fake_run)
    bonsai.tl.reconstruct(a, workdir=tmp_path / "wd", **GOLDEN_KW)

    path = tmp_path / "a.h5ad"
    a.write_h5ad(path)
    b = anndata.read_h5ad(path)

    assert b.uns["bonsai"]["workdir"] is None
    for key in ("newick", "loglik", "upstream_commit"):
        assert b.uns["bonsai"][key] == a.uns["bonsai"][key]
    assert list(b.uns["bonsai"]["genes_used"]) == list(a.uns["bonsai"]["genes_used"])
    pd.testing.assert_frame_equal(b.uns["bonsai"]["nodes"], a.uns["bonsai"]["nodes"])
    pd.testing.assert_frame_equal(b.uns["bonsai"]["edges"], a.uns["bonsai"]["edges"])
    assert b.uns["bonsai"]["config"] == a.uns["bonsai"]["config"]
    np.testing.assert_array_equal(b.obs["bonsai_node"].to_numpy(), a.obs["bonsai_node"].to_numpy())
    np.testing.assert_array_equal(b.var["bonsai_used"].to_numpy(), a.var["bonsai_used"].to_numpy())
