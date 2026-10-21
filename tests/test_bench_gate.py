import json

import numpy as np
import pytest
from bench import gate as gate_mod
from bench.gate import gate_verdict, load_arm, main
from bench.retime import RetimeResult
from bench.run import leaf_sha256
from bench.simulators.brownian import simulate

from bonsai._ref.preprocess import preprocess


def test_gate_verdict():
    v = gate_verdict(L_jax=-10.0, L_up=-9.0, n_cells=1000, n_genes=2000, q_jax=0.745, q_up=0.75)
    assert v["a1_per_entry"] == pytest.approx(-5e-7, rel=1e-15, abs=0) and v["a1_pass"] and v["a7_pass"]
    assert not gate_verdict(-12.1, -10.0, 1000, 2000)["a1_pass"]  # -1.05e-6 per entry
    assert not gate_verdict(0, 0, 10, 10, q_jax=0.735, q_up=0.75)["a7_pass"]
    assert gate_verdict(0, 0, 10, 10)["a7_pass"] is None


def _run_dir(path, cfg, newick, sha, genes):
    path.mkdir()
    run = {"stage_newick": {"greedy": newick, "redo_starry": newick}, "leaf_sha256": sha, "genes_used": genes}
    (path / "results.json").write_text(json.dumps({"config": cfg, "runs": [run]}))
    return path


@pytest.mark.parametrize("field", ["leaf_sha256", "genes_used"])
def test_gate_refuses_mismatched_arms(tmp_path, field, monkeypatch):
    monkeypatch.setattr(gate_mod, "retime_subprocess", lambda *a, **k: pytest.fail("re-timer reached"))
    nwk, cfg = "((a:1,b:1):1,c:1);", {"run": {"seed": 0}}
    a = _run_dir(tmp_path / "jax", cfg, nwk, "h0", ["g0", "g1"])
    other = {"leaf_sha256": "h1", "genes_used": ["g0", "g2"]}
    up = {"leaf_sha256": "h0", "genes_used": ["g0", "g1"]} | {field: other[field]}
    b = _run_dir(tmp_path / "up", cfg, nwk, up["leaf_sha256"], up["genes_used"])
    with pytest.raises(ValueError, match=field):
        main(["--jax", str(a), "--upstream", str(b), "--out", str(tmp_path / "gate.json")])
    assert not (tmp_path / "gate.json").exists()


def test_load_arm_from_upstream_results_folder(tmp_path):
    """An upstream results folder: redo-starry and greedy trees, hash of delta*.npy (genes × cells) by cellIds."""
    rng = np.random.default_rng(1)
    x, var = rng.normal(size=(4, 3)), rng.uniform(0.1, 1, size=(4, 3))  # cells × genes
    cells, genes = ["d", "a", "c", "b"], ["g0", "g1", "g2"]
    proc = tmp_path / "zscorefiltered_1.000_and_processed"
    proc.mkdir()
    np.save(proc / "delta.npy", x.T)
    np.save(proc / "delta_vars.npy", var.T)
    (proc / "metadata.json").write_text(json.dumps({"cellIds": cells, "geneIds": genes}))
    for d, nwk in [
        ("intermediate_bonsai_zscore1.0", "(a:1,b:1,c:1,d:1);"),
        ("intermediate_bonsai_zscore1.0_redoStarry", "((a:1,b:1):1,c:1,d:1);"),
        ("intermediate_bonsai_zscore1.0_redoStarry_optTimes", "((a:2,b:1):1,c:1,d:1);"),
    ]:
        (tmp_path / d).mkdir()
        (tmp_path / d / "tree.nwk").write_text(nwk + "\n")
    arm = load_arm(tmp_path)
    assert arm["stage_newick"] == {"greedy": "(a:1,b:1,c:1,d:1);", "redo_starry": "((a:1,b:1):1,c:1,d:1);"}
    assert arm["leaf_sha256"] == leaf_sha256(x, var, cells) and arm["genes_used"] == genes


def _sim_arms(tmp_path):
    """Both arms = the simulator's true tree (24 cells × 60 genes, seed 0) as bench run dirs."""
    cfg = {
        "run": {"name": "t", "seed": 0, "backend": "jax"},
        "data": {"kind": "simulate", "n_cells": 24, "n_genes": 60},
    }
    sim = simulate(seed=0, n_cells=24, n_genes=60)
    lv = preprocess(sim.adata)
    sha, genes = leaf_sha256(lv.x, lv.var, list(sim.adata.obs_names)), list(sim.adata.var_names[lv.genes])
    nwk = sim.adata.uns["bonsai_truth"]["newick"]
    return _run_dir(tmp_path / "jax", cfg, nwk, sha, genes), _run_dir(tmp_path / "up", cfg, nwk, sha, genes), genes


@pytest.mark.slow
def test_gate_cli_identical_trees(tmp_path):
    """Both arms = the simulator's true tree: same re-timer input, so A1 per entry is exactly 0, A7 diff 0,
    and the two re-time results are identical (one deterministic CPU computation, twice). The re-timed
    tree has edges below the 2e-6 snap, so A15 contraction makes the contracted quartet score lower than the
    raw one (1.0: the true tree itself) and the contracted RF positive."""
    a, b, genes = _sim_arms(tmp_path)
    out = main(["--jax", str(a), "--upstream", str(b), "--out", str(tmp_path / "gate.json")])
    g = json.loads(out.read_text())
    assert g["a1_per_entry"] == 0.0 and g["a7_diff"] == 0.0
    assert g["arms"]["jax"]["retime"] == g["arms"]["upstream"]["retime"]
    assert g["arms"]["jax"]["L_raw_stage"]["redo_starry"] == g["arms"]["jax"]["L_raw_stage"]["greedy"]
    assert g["verdict"] == "PASS" and g["invalid_arms"] == [] and g["n_cells"] == 24 and g["n_genes"] == len(genes)
    arm = g["arms"]["jax"]
    assert arm["quartet_accuracy"] < arm["quartet_accuracy_raw"] == 1.0 and arm["rf_normalised"] > 0
    assert arm["rf_normalised_raw"] == 0.0


def _fake_retime(monkeypatch, L, status):
    """retime_subprocess stub: call i returns L[i] and status[i] (arms in order jax, upstream) and the input
    times with one internal edge set to 1e-7 (below the 2e-6 snap; its parent is not the root, so the
    contraction removes an unrooted split)."""
    calls = []

    def fake(parent, t, x, var, device=None):
        i = len(calls)
        calls.append(1)
        tt = np.array(t, dtype=float)
        grand = np.where(parent >= 0, parent[np.maximum(parent, 0)], -1) >= 0  # parent is not the root
        tt[np.flatnonzero(grand & np.isin(np.arange(len(parent)), parent))[0]] = 1e-7
        return RetimeResult(tt, L[i], L[i] - 1.0, 0.0, 0.0, 0.0, 1, True, False, 1, 0, status[i], "")

    monkeypatch.setattr(gate_mod, "retime_subprocess", fake)


def test_gate_invalid_arm_makes_verdict_invalid(tmp_path, monkeypatch):
    a, b, _ = _sim_arms(tmp_path)
    _fake_retime(monkeypatch, [-100.0, -100.0], ["converged", "invalid"])
    g = gate_mod.gate(a, b)
    assert g["verdict"] == "INVALID" and g["invalid_arms"] == ["upstream"]
    for arm in g["arms"].values():  # contraction uses the re-timed times with the snap (A15)
        assert arm["quartet_accuracy"] < arm["quartet_accuracy_raw"] == 1.0


def test_gate_fail_verdict(tmp_path, monkeypatch):
    """Both arms valid, jax L lower than upstream by 2e-6 per entry (A1 slack 1e-6) -> FAIL."""
    a, b, genes = _sim_arms(tmp_path)
    _fake_retime(monkeypatch, [-100.0 - 2e-6 * 24 * len(genes), -100.0], ["converged", "converged"])
    g = gate_mod.gate(a, b)
    assert g["verdict"] == "FAIL" and not g["a1_pass"] and g["a7_pass"] and g["invalid_arms"] == []


def test_gate_rejects_upstream_folder_as_jax_arm(tmp_path, monkeypatch):
    """An upstream results folder has no config (load_arm gives config None): refused as the jax arm."""
    a, b, _ = _sim_arms(tmp_path)
    real = gate_mod.load_arm
    monkeypatch.setattr(gate_mod, "load_arm", lambda path: real(path) | ({"config": None} if path == a else {}))
    with pytest.raises(ValueError, match="jax arm must be a bench run dir"):
        gate_mod.gate(a, b)
