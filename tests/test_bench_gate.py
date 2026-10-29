import json
from pathlib import Path

import numpy as np
import pytest
from bench import gate as gate_mod
from bench.gate import gate_verdict, load_arm, main
from bench.retime import Endpoint, RetimeResult, times_of
from bench.run import leaf_sha256
from bench.simulators.brownian import simulate

from bonsai._ref.preprocess import preprocess


def _rr(t, L=-1.0, status="converged", gap=0.0):
    """A RetimeResult with times ``t`` (other diagnostics neutral)."""
    return RetimeResult(
        t=np.array(t, float), L=L, L_start=L - 1.0, gap=gap, gap_newton=gap, gap_flat=0.0, gap_flat_sum=0.0,
        gap_flat_joint=0.0, gap_floor=0.0, n_floor=0, lambda_min=1.0, n_flat=0,
        n_free=1, asym=0.0, sum_abs_pg=0.0, extra_dL=0.0, nit=1, restarts=0, hess_s=0.0, eigh_s=0.0, status=status,
        reason="" if status == "converged" else "x",
    )  # fmt: skip


def _ep(parent, t, nit=7):
    """An Endpoint at times ``t`` (scipy status 0, two runs recorded)."""
    s = np.log(np.clip(np.asarray(t, float)[np.asarray(parent) >= 0], np.exp(-15.0), np.exp(15.0)))
    return Endpoint(s, -2.0, nit, 1, 0.0, True, "scipy status 0 (ok)", np.array([[5.0, -2.5, 0.0], [2.0, -2.0, 0.0]]))


def test_gate_verdict():
    v = gate_verdict(L_jax=-10.0, L_up=-9.0, n_cells=1000, n_genes=2000, q_jax=0.745, q_up=0.75)
    assert v["a1_per_entry"] == pytest.approx(-5e-7, rel=1e-15, abs=0) and v["a1_pass"] and v["a7_pass"]
    assert not gate_verdict(-12.1, -10.0, 1000, 2000)["a1_pass"]  # -1.05e-6 per entry
    assert not gate_verdict(0, 0, 10, 10, q_jax=0.735, q_up=0.75)["a7_pass"]
    assert gate_verdict(0, 0, 10, 10)["a7_pass"] is None


def _run_dir(path, cfg, newick, sha, genes, x=None, var=None, cells=None):
    """A bench run dir (A23: always carries its ``leaf_arrays.npz``, as a real one would)."""
    path.mkdir()
    if x is None:
        x, var, cells = np.zeros((1, max(len(genes), 1))), np.ones((1, max(len(genes), 1))), ["c0"]
    np.savez(path / "leaf_arrays.npz", x=x, var=var, cells=np.array(cells), genes=np.array(genes))
    run = {
        "stage_newick": {"greedy": newick, "redo_starry": newick},
        "leaf_sha256": sha,
        "genes_used": genes,
        "leaf_arrays": "leaf_arrays.npz",
    }
    (path / "results.json").write_text(json.dumps({"config": cfg, "runs": [run]}))
    return path


def test_load_leaf_arrays_rejects_a_baseline_predating_a23(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"config": {}, "runs": [{"leaf_sha256": "h"}]}))
    with pytest.raises(RuntimeError, match="predates A23"):
        gate_mod.load_leaf_arrays(tmp_path)


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
    """Both arms = the simulator's true tree (24 cells × 60 genes, seed 0) as bench run dirs, each
    carrying the same stored leaf arrays (A23: the gate reads these, not a fresh preprocess)."""
    cfg = {
        "run": {"name": "t", "seed": 0, "backend": "jax"},
        "data": {"kind": "simulate", "n_cells": 24, "n_genes": 60},
    }
    sim = simulate(seed=0, n_cells=24, n_genes=60)
    lv = preprocess(sim.adata)
    cells = list(sim.adata.obs_names)
    sha, genes = leaf_sha256(lv.x, lv.var, cells), list(sim.adata.var_names[lv.genes])
    nwk = sim.adata.uns["bonsai_truth"]["newick"]
    a = _run_dir(tmp_path / "jax", cfg, nwk, sha, genes, lv.x, lv.var, cells)
    b = _run_dir(tmp_path / "up", cfg, nwk, sha, genes, lv.x, lv.var, cells)
    return a, b, genes


def test_gate_refuses_arms_whose_leaf_arrays_differ_by_one_ulp(tmp_path, monkeypatch):
    """A23: a 1-ulp difference between the arms' stored arrays flips leaf_sha256, so the mismatched-arms
    check (not a fresh, SIMD-sensitive preprocess) refuses them."""
    monkeypatch.setattr(gate_mod, "retime_subprocess", lambda *a, **k: pytest.fail("re-timer reached"))
    nwk, cfg = "((a:1,b:1):1,c:1);", {"run": {"seed": 0}}
    rng = np.random.default_rng(0)
    x, var = rng.normal(size=(3, 2)), rng.uniform(0.1, 1, size=(3, 2))
    cells, genes = ["a", "b", "c"], ["g0", "g1"]
    x_ulp = x.copy()
    x_ulp[0, 0] = np.nextafter(x_ulp[0, 0], np.inf)
    sha, sha_ulp = leaf_sha256(x, var, cells), leaf_sha256(x_ulp, var, cells)
    assert sha != sha_ulp
    a = _run_dir(tmp_path / "jax", cfg, nwk, sha, genes, x, var, cells)
    b = _run_dir(tmp_path / "up", cfg, nwk, sha_ulp, genes, x_ulp, var, cells)
    with pytest.raises(ValueError, match="leaf_sha256"):
        gate_mod.gate(a, b)


def test_gate_refuses_upstream_arm_whose_npz_does_not_match_its_recorded_hash(tmp_path, monkeypatch):
    """A23 integrity check: both arms record the *same* leaf_sha256 (so the cross-arm check passes), but
    upstream's own stored npz does not hash to it (a corrupted or stale npz) -> refused."""
    monkeypatch.setattr(gate_mod, "retime_subprocess", lambda *a, **k: pytest.fail("re-timer reached"))
    monkeypatch.setattr(gate_mod, "load_data", lambda c: (None, None))  # cfg has no [data]; truth not needed here
    nwk, cfg = "((a:1,b:1):1,c:1);", {"run": {"seed": 0}}
    rng = np.random.default_rng(0)
    x, var = rng.normal(size=(3, 2)), rng.uniform(0.1, 1, size=(3, 2))
    cells, genes = ["a", "b", "c"], ["g0", "g1"]
    sha = leaf_sha256(x, var, cells)  # both arms claim this hash
    x_bad = x.copy()
    x_bad[0, 0] = np.nextafter(x_bad[0, 0], np.inf)  # upstream's actual npz no longer matches it
    a = _run_dir(tmp_path / "jax", cfg, nwk, sha, genes, x, var, cells)
    b = _run_dir(tmp_path / "up", cfg, nwk, sha, genes, x_bad, var, cells)
    with pytest.raises(ValueError, match="stored leaf arrays"):
        gate_mod.gate(a, b)


def test_gate_jax_arm_without_leaf_arrays_is_not_rejected_as_predating_a23(tmp_path, monkeypatch):
    """A future jax bench run dir never carries leaf_arrays.npz (its arrays are never used for scoring);
    load_arm(..., load_arrays=False) must not try to load it, so a missing npz there is not an error."""
    nwk, cfg = "((a:1,b:1):1,c:1);", {"run": {"seed": 0}}
    rng = np.random.default_rng(0)
    x, var = rng.normal(size=(3, 2)), rng.uniform(0.1, 1, size=(3, 2))
    cells, genes = ["a", "b", "c"], ["g0", "g1"]
    sha = leaf_sha256(x, var, cells)
    jax_dir = tmp_path / "jax"
    jax_dir.mkdir()
    run = {"stage_newick": {"greedy": nwk, "redo_starry": nwk}, "leaf_sha256": sha, "genes_used": genes}
    (jax_dir / "results.json").write_text(json.dumps({"config": cfg, "runs": [run]}))  # no "leaf_arrays"
    up = _run_dir(tmp_path / "up", cfg, nwk, sha, genes, x, var, cells)
    monkeypatch.setattr(gate_mod, "load_data", lambda c: (None, None))  # cfg has no [data]; truth not needed here
    monkeypatch.setattr(
        gate_mod,
        "retime_subprocess",
        lambda parent, t, x, var, device=None, endpoint=None: (_rr(t, 0.0), _ep(parent, t)),
    )
    gate_mod.gate(jax_dir, up)  # must not raise "baseline predates A23" for the jax arm


def test_load_arm_npz_genes_match_recorded_genes_used(tmp_path):
    a = _run_dir(tmp_path / "up", {}, "(a:1,b:1,c:1);", "sha", ["g0", "g1"], np.zeros((1, 2)), np.ones((1, 2)), ["c0"])
    arm = load_arm(a)
    assert arm["genes"] == arm["genes_used"] == ["g0", "g1"]


@pytest.mark.slow
def test_gate_cli_identical_trees(tmp_path):
    """Both arms = the simulator's true tree: same re-timer input, so A1 per entry is exactly 0, A7 diff 0,
    and the two re-time results are identical apart from wall times (one deterministic CPU computation, twice). The re-timed
    tree has edges below the 2e-6 snap, so A15 contraction makes the contracted quartet score lower than the
    raw one (1.0: the true tree itself) and the contracted RF positive."""
    a, b, genes = _sim_arms(tmp_path)
    out = main(["--jax", str(a), "--upstream", str(b), "--out", str(tmp_path / "gate.json")])
    g = json.loads(out.read_text())
    assert g["a1_per_entry"] == 0.0 and g["a7_diff"] == 0.0
    timing = {"hess_s", "eigh_s"}  # wall times
    jr, ur = ({k: v for k, v in g["arms"][a]["retime"].items() if k not in timing} for a in ("jax", "upstream"))
    assert jr == ur
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

    def fake(parent, t, x, var, device=None, endpoint=None):
        i = len(calls)
        calls.append(1)
        tt = np.array(t, dtype=float)
        grand = np.where(parent >= 0, parent[np.maximum(parent, 0)], -1) >= 0  # parent is not the root
        tt[np.flatnonzero(grand & np.isin(np.arange(len(parent)), parent))[0]] = 1e-7
        return _rr(tt, L[i], status[i]), _ep(parent, tt)

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
    monkeypatch.setattr(
        gate_mod,
        "load_arm",
        lambda path, load_arrays=True: real(path, load_arrays) | ({"config": None} if path == a else {}),
    )
    with pytest.raises(ValueError, match="jax arm must be a bench run dir"):
        gate_mod.gate(a, b)


def test_gate_truth_comes_from_the_upstream_arms_config(tmp_path, monkeypatch):
    """The jax arm's config may not pick the truth: a jax config with another seed/size is ignored."""
    a, b, _ = _sim_arms(tmp_path)
    res = json.loads((a / "results.json").read_text())
    res["config"] = {"run": {"seed": 9}, "data": {"kind": "simulate", "n_cells": 30, "n_genes": 60}}
    (a / "results.json").write_text(json.dumps(res))
    seen = []

    def spy(cfg):
        seen.append(cfg)
        raise RuntimeError("stop")

    monkeypatch.setattr(gate_mod, "load_data", spy)
    with pytest.raises(RuntimeError, match="stop"):
        gate_mod.gate(a, b)
    assert seen[0]["run"]["seed"] == 0 and seen[0]["data"]["n_cells"] == 24


def _stub(monkeypatch, calls, status="converged", gap=0.0):
    """retime_subprocess stub: records the endpoint it was given (None = L-BFGS would run) and returns ``status``."""

    def fake(parent, t, x, var, device=None, endpoint=None):
        calls.append(endpoint)
        ep = _ep(parent, t) if endpoint is None else endpoint
        return _rr(times_of(parent, ep.s), -1.0, status, gap), ep

    monkeypatch.setattr(gate_mod, "retime_subprocess", fake)


_TREE = (np.array([-1, 0, 0]), np.array([0.0, 0.1, 0.2]), ["r", "a", "b"], "h")
_XX = np.zeros((2, 1))


def test_endpoint_cache_and_acceptance_memo(tmp_path, monkeypatch):
    """A16f: the endpoint is cached with its restart history; the acceptance is memoised for the current acceptance
    code (a repeat spawns nothing) and reruns on the stored endpoint (no L-BFGS) when that code changes."""
    calls = []
    _stub(monkeypatch, calls)
    r1, hit1, key = gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    r2, hit2, key2 = gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    assert (hit1, hit2, key2) == (False, True, key) and calls == [None]
    for k, v in r1.__dict__.items():
        np.testing.assert_array_equal(getattr(r2, k), v)
    monkeypatch.setattr(gate_mod, "acceptance_hash", lambda: "new acceptance code")
    gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    (ep,) = calls[1:]
    np.testing.assert_array_equal(ep.s, _ep(*_TREE[:2]).s)
    np.testing.assert_array_equal(ep.history, _ep(*_TREE[:2]).history)  # the restart history is stored
    assert (ep.nit, ep.restarts, ep.status_ok, ep.message) == (7, 1, True, "scipy status 0 (ok)")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 3 and f"{key}.npz" in names and all(".tmp" not in n for n in names)  # endpoint + 2 memos


def test_acceptance_memo_is_keyed_on_the_acceptance_code(tmp_path, monkeypatch):
    """An INVALID result's endpoint is cached too (only endpoint fields in its entry); its memoised verdict is served
    while the acceptance code is unchanged and recomputed (here converged) once it changes. Mutant "acceptance
    cached with the endpoint" (memo not keyed on the acceptance code) serves the stale INVALID."""
    calls = []
    _stub(monkeypatch, calls, status="invalid", gap=1.0)
    rt, hit, key = gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    assert (rt.status, hit) == ("invalid", False) and (tmp_path / f"{key}.npz").exists()
    with np.load(tmp_path / f"{key}.npz") as z:
        assert not {"status", "gap", "reason", "L"} & set(z.files)  # only the endpoint
    _stub(monkeypatch, calls, status="converged")
    assert gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)[0].status == "invalid" and len(calls) == 1
    monkeypatch.setattr(gate_mod, "acceptance_hash", lambda: "new acceptance code")
    rt, hit, _ = gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    assert (rt.status, rt.gap, hit) == ("converged", 0.0, True) and calls[-1] is not None


def test_acceptance_hash_tracks_acceptance_code_and_constants(monkeypatch):
    h = gate_mod.acceptance_hash()
    r = gate_mod.retime
    monkeypatch.setattr(r, "HESS_MEM_FRAC", 0.8)  # batch sizing only: same result
    assert gate_mod.acceptance_hash() == h
    for name, v in [("CURV_REL", 1e-9), ("FLOOR_S", -8.0), ("LG_GRID", 7), ("L_MATCH_REL", 1e-8), ("GAP_FRAC", 2e-2)]:
        monkeypatch.setattr(r, name, v)
        assert gate_mod.acceptance_hash() != h, name
        monkeypatch.undo()

    def gap_terms(w, U, pg, tau, s, fi, L, Lg):  # a different acceptance part
        return None

    monkeypatch.setattr(r, "gap_terms", gap_terms)
    assert gate_mod.acceptance_hash() != h


def test_endpoint_is_stored_when_the_acceptance_fails(tmp_path, monkeypatch):
    """A16f detail: the subprocess saved its fresh endpoint before the acceptance crashed; retime_cached stores it and
    re-raises, and the next call re-accepts from it without L-BFGS."""
    import subprocess

    calls = []
    ep0 = _ep(*_TREE[:2], nit=99)

    def crash(parent, t, x, var, device=None, endpoint=None):
        calls.append(endpoint)
        err = subprocess.CalledProcessError(1, ["bench.retime"])
        err.endpoint = ep0
        raise err

    monkeypatch.setattr(gate_mod, "retime_subprocess", crash)
    with pytest.raises(subprocess.CalledProcessError):
        gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    _stub(monkeypatch, calls)
    rt, hit, _ = gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    assert hit is True and calls[0] is None and calls[1].nit == 99 and rt.status == "converged"


def test_retime_subprocess_attaches_the_saved_endpoint_on_failure(monkeypatch):
    import subprocess

    ep0 = _ep(*_TREE[:2], nit=42)

    def run(args, **k):
        np.savez(gate_mod.retime.endpoint_path(args[-1]), **gate_mod.asdict(ep0))
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(gate_mod.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as ei:
        gate_mod.retime_subprocess(*_TREE[:2], _XX, _XX)
    assert ei.value.endpoint.nit == 42
    np.testing.assert_array_equal(ei.value.endpoint.s, ep0.s)
    monkeypatch.setattr(
        gate_mod.subprocess, "run", lambda args, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, args))
    )
    with pytest.raises(subprocess.CalledProcessError) as ei:
        gate_mod.retime_subprocess(*_TREE[:2], _XX, _XX)
    assert ei.value.endpoint is None  # failed before L-BFGS finished


def test_retime_cache_off_by_default(monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    out = gate_mod.retime_cached(np.array([-1, 0]), np.array([0.0, 0.1]), ["r", "a"], "h", _XX, _XX, None, None)
    assert out[1:] == (None, None) and calls == [None]


def test_retime_cache_key_misses(monkeypatch):
    base = _TREE
    k = gate_mod.retime_key(*base)
    assert gate_mod.retime_key(base[0], np.nextafter(base[1], 1.0), base[2], "h") != k  # t at full precision
    assert gate_mod.retime_key(*base[:3], "h2") != k  # leaf hash
    assert gate_mod.retime_key(base[0], base[1], ["r", "b", "a"], "h") != k  # leaf names
    assert gate_mod.retime_key(np.array([-1, 0, 1]), *base[1:]) != k  # topology
    for name, v in [("GAP_FRAC", 2e-2), ("MAX_RESTARTS", 3), ("OPTIONS", gate_mod.retime.OPTIONS | {"gtol": 1e-9})]:
        monkeypatch.setattr(gate_mod.retime, name, v)  # an L-BFGS constant
        assert gate_mod.retime_key(*base) != k
        monkeypatch.undo()
    assert gate_mod.retime_key(*base) == k


def test_acceptance_change_keeps_the_endpoint_key_lbfgs_change_misses(monkeypatch):
    """A16f cache split: acceptance constants and code are not in the key; the L-BFGS part's code is."""
    k = gate_mod.retime_key(*_TREE)
    r = gate_mod.retime
    for name, v in [("CURV_REL", 1e-9), ("L_MATCH_REL", 1e-8), ("LG_GRID", 7), ("LG_ALPHA_MIN", 1e-2),
                    ("HESS_MEM_FRAC", 0.5), ("accept", lambda *a: None), ("gap_terms", lambda *a: None),
                    ("line_gain", lambda *a: None), ("joint_gain", lambda *a: None), ("FLOOR_S", -8.0), ("free_hessian", lambda *a: None), ("eigh", lambda *a: None)]:  # fmt: skip
        monkeypatch.setattr(r, name, v)
        assert gate_mod.retime_key(*_TREE) == k, name
    monkeypatch.undo()

    def lbfgs(parent, t, x, var):  # a different L-BFGS part
        return None

    def _objective(parent, t, x, var):
        return None

    for name, f in [("lbfgs", lbfgs), ("_objective", _objective)]:
        monkeypatch.setattr(r, name, f)
        assert gate_mod.retime_key(*_TREE) != k, name
        monkeypatch.undo()


def test_gate_reports_cache_hits_and_reuses_identical_arms(tmp_path, monkeypatch):
    """Both arms are the same tree: the second arm reuses the first's endpoint and memoised acceptance, and a rerun
    hits for both without spawning anything."""
    a, b, _ = _sim_arms(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    c = tmp_path / "cache"
    g1 = gate_mod.gate(a, b, cache_dir=c)
    assert [g1["arms"][k]["retime_cache_hit"] for k in ("jax", "upstream")] == [False, True]
    assert g1["acceptance"] == "A16f" and calls == [None]
    g2 = gate_mod.gate(a, b, cache_dir=c)
    assert [g2["arms"][k]["retime_cache_hit"] for k in ("jax", "upstream")] == [True, True] and calls == [None]
    assert gate_mod.gate(a, b)["arms"]["jax"]["retime_cache_hit"] is None


def test_upstream_only_fills_the_cache_for_the_later_gate(tmp_path, monkeypatch):
    a, b, _ = _sim_arms(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    c, out = tmp_path / "cache", tmp_path / "u.json"
    main(["--upstream", str(b), "--upstream-only", "--retime-cache", str(c), "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rep["status"] == "converged" and rep["arm"]["retime_cache_hit"] is False and calls == [None]
    assert rep["acceptance"] == "A16f"
    assert gate_mod.gate(a, b, cache_dir=c)["arms"]["upstream"]["retime_cache_hit"] is True
    with pytest.raises(SystemExit):
        main(["--upstream", str(b), "--upstream-only", "--out", str(out)])  # needs a cache


def test_code_hash_covers_the_lbfgs_part_and_both_trees(tmp_path, monkeypatch):
    from types import SimpleNamespace

    names = ("engine_tree", "numerics", "array_tree", "ref_tree", "ref_newick")
    real = [Path(getattr(gate_mod, n).__file__).as_posix().split("/")[-3:] for n in names]
    assert real == [
        ["bonsai", "_engine", "tree.py"],
        ["bonsai", "_engine", "numerics.py"],
        ["bonsai", "_engine", "array_tree.py"],
        ["bonsai", "_ref", "tree.py"],
        ["bonsai", "_ref", "newick.py"],
    ]
    assert set(gate_mod._ENDPOINT_CODE) == {"Endpoint", "times_of", "_objective", "_bound", "lbfgs"}
    files = [tmp_path / f"{n}.py" for n in names]
    for f in files:
        f.write_text("x = 1\n")
    for name, f in zip(names, files, strict=True):
        monkeypatch.setattr(gate_mod, name, SimpleNamespace(__file__=str(f)))
    h0 = gate_mod.code_hash()
    for f in files:
        f.write_text(f.read_text() + "# edit\n")
        h1 = gate_mod.code_hash()
        assert h1 != h0
        h0 = h1


def test_retime_cache_key_misses_on_code_hash_and_time_box(monkeypatch):
    k = gate_mod.retime_key(*_TREE)
    monkeypatch.setattr(gate_mod, "code_hash", lambda: "other")
    assert gate_mod.retime_key(*_TREE) != k
    monkeypatch.undo()
    monkeypatch.setattr(gate_mod, "LOG_T_MIN", gate_mod.LOG_T_MIN - 1.0)
    assert gate_mod.retime_key(*_TREE) != k
    monkeypatch.undo()
    monkeypatch.setattr(gate_mod, "LOG_T_MAX", gate_mod.LOG_T_MAX + 1.0)
    assert gate_mod.retime_key(*_TREE) != k


def test_retime_cache_entry_with_another_code_hash_is_a_miss(tmp_path, monkeypatch):
    """Belt and braces: an entry under the right key whose stored code hash differs runs L-BFGS again."""
    calls = []
    _stub(monkeypatch, calls)
    gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    f = tmp_path / f"{gate_mod.retime_key(*_TREE)}.npz"
    with np.load(f) as z:
        d = {k: z[k] for k in z.files} | {"code_hash": np.array("stale")}
    np.savez(f, **d)
    assert gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)[1] is False and calls == [None, None]


def _a16e_entry(cache, parent, t, names, leaf, code="a16e-code", status="converged"):
    """A pre-A16f cache entry: a converged RetimeResult in the A16e layout plus its code hash, under the A16e key."""
    key = gate_mod._key(parent, t, names, leaf, gate_mod._A16E_CONSTANTS, code)
    tt = np.array([0.0, np.exp(-15.0), 0.3])  # edge 1 at the box floor (exp of the bound)
    old = {"t": tt, "L": -1.5, "L_start": -3.0, "gap": 1e-6, "sum_abs_pg": 1e-3, "extra_dL": 2e-9, "cg_iters": 40,
           "cg_converged": True, "curvature_stop": False, "nit": 1234, "restarts": 2, "status": status, "reason": "",
           "code_hash": code}  # fmt: skip
    cache.mkdir(exist_ok=True)
    np.savez(cache / f"{key}.npz", **old)
    return key, tt


def test_a16e_cache_entry_is_migrated_to_an_endpoint(tmp_path, monkeypatch):
    """An A16e entry (converged result, A16e key) is read as the endpoint: s = log t, which recovers the box edge
    exactly (free_set compares s to the bound with <=), its counters kept, no L-BFGS run; it is then stored under
    the new key and served from there."""
    calls = []
    _stub(monkeypatch, calls)
    c = tmp_path / "cache"
    old_key, tt = _a16e_entry(c, *_TREE)
    _a16e_entry(c, np.array([-1, 0, 1]), *_TREE[1:], code="other-code")  # another tree, another code hash
    rt, hit, key = gate_mod.retime_cached(*_TREE, _XX, _XX, None, c)
    (ep,) = calls
    assert hit is True and ep is not None and key != old_key
    assert ep.s[0] == -15.0 and ep.s[1] == pytest.approx(np.log(0.3), rel=1e-15, abs=0)
    assert (ep.L_start, ep.nit, ep.restarts, ep.extra_dL, ep.status_ok) == (-3.0, 1234, 2, 2e-9, True)
    assert ep.history.shape == (0, 3) and old_key in ep.message
    assert (c / f"{key}.npz").exists() and (c / f"{old_key}.npz").exists()
    assert gate_mod.retime_cached(*_TREE, _XX, _XX, None, c)[1] is True and len(calls) == 1  # memo hit
    new = gate_mod.retime.load_endpoint(c / f"{key}.npz")  # the migrated new-format entry
    np.testing.assert_array_equal(new.s, ep.s)
    assert old_key in new.message


def test_a16e_entry_of_another_tree_or_invalid_is_not_used(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    c = tmp_path / "cache"
    _a16e_entry(c, *_TREE[:3], "other-leaves")
    assert gate_mod.retime_cached(*_TREE, _XX, _XX, None, c)[1] is False and calls == [None]
    d = tmp_path / "cache2"
    _a16e_entry(d, *_TREE, status="invalid")
    assert gate_mod.retime_cached(*_TREE, _XX, _XX, None, d)[1] is False and calls == [None, None]


def test_recheck_rewrites_reports_from_cached_endpoints(tmp_path, monkeypatch):
    """--recheck: every existing report of the manifest is re-evaluated from the cached endpoints (no L-BFGS),
    tagged A16f, the old report kept under ``superseded`` (the original one on a repeat); lines without a
    report are skipped."""
    from bench.sweep import write_manifest

    a, b, _ = _sim_arms(tmp_path)
    calls = []
    _stub(monkeypatch, calls, status="invalid", gap=1.0)
    c = tmp_path / "cache"
    up_out, g_out = tmp_path / "up.json", tmp_path / "g" / "gate.json"
    g_out.parent.mkdir()
    main(["--upstream", str(b), "--upstream-only", "--retime-cache", str(c), "--out", str(up_out)])
    main(["--jax", str(a), "--upstream", str(b), "--retime-cache", str(c), "--out", str(g_out)])
    old_up, old_g = json.loads(up_out.read_text()), json.loads(g_out.read_text())
    assert old_g["verdict"] == "INVALID"
    lines = [
        {"id": 0, "kind": "upstream", "case": "x", "gate_out": str(up_out), "cache": str(c)},
        {"id": 1, "kind": "jax", "case": "x", "gate_out": str(g_out), "cache": str(c)},
        {"id": 2, "kind": "jax", "case": "y", "gate_out": str(tmp_path / "missing.json"), "cache": str(c)},
    ]
    write_manifest(lines, tmp_path / "w.tsv")
    _stub(monkeypatch, calls, status="converged")
    monkeypatch.setattr(gate_mod, "acceptance_hash", lambda: "A16f fixed")  # the acceptance code changed
    n = len(calls)
    assert main(["--recheck", "--manifest", str(tmp_path / "w.tsv"), "--lines", "1-2"]) == [g_out]
    assert json.loads(up_out.read_text()) == old_up  # line 0 not selected
    # one acceptance from the cached endpoint; the gate's second arm (same tree) reuses its memo
    assert len(calls) == n + 1 and calls[-1] is not None
    assert main(["--recheck", "--manifest", str(tmp_path / "w.tsv")]) == [up_out]  # g_out's hash is current now
    assert len(calls) == n + 1  # everything memoised now
    up, g = json.loads(up_out.read_text()), json.loads(g_out.read_text())
    assert (up["acceptance"], up["status"], up["superseded"]) == ("A16f", "converged", old_up)
    assert (g["acceptance"], g["verdict"], g["superseded"]) == ("A16f", "PASS", old_g)
    assert up["acceptance_hash"] == g["acceptance_hash"] == "A16f fixed"
    assert main(["--recheck", "--manifest", str(tmp_path / "w.tsv")]) == []  # stored hashes current: skipped
    assert json.loads(g_out.read_text()) == g
    monkeypatch.setattr(gate_mod, "acceptance_hash", lambda: "A16f fixed again")  # acceptance changed again
    assert main(["--recheck", "--manifest", str(tmp_path / "w.tsv")]) == [up_out, g_out]
    g2 = json.loads(g_out.read_text())
    assert g2["superseded"] == old_g and g2["acceptance_hash"] == "A16f fixed again"
    with pytest.raises(SystemExit):
        main(["--recheck"])  # needs --manifest


def test_real_subprocess_accepts_the_cached_endpoint(tmp_path, monkeypatch):
    """End to end through ``python -m bench.retime``: the second call sends the cached endpoint (``ep_`` fields),
    the subprocess skips L-BFGS and returns the same endpoint and, on CPU, the same acceptance."""
    from ._trees import bm_tree

    parent, t, x, var = bm_tree(6, 8, np.random.default_rng(0))
    names = [f"n{i}" for i in range(len(parent))]
    r1, hit1, key = gate_mod.retime_cached(parent, t, names, "h", x, var, None, tmp_path)
    monkeypatch.setattr(gate_mod, "acceptance_hash", lambda: "force a fresh acceptance")  # else the memo answers
    r2, hit2, _ = gate_mod.retime_cached(parent, t, names, "h", x, var, None, tmp_path)
    assert (hit1, hit2) == (False, True) and r1.status == "converged", r1.reason
    np.testing.assert_array_equal(r2.t, r1.t)
    assert (r2.L, r2.gap, r2.nit, r2.restarts, r2.status) == (r1.L, r1.gap, r1.nit, r1.restarts, r1.status)
    ep = gate_mod.retime.load_endpoint(tmp_path / f"{key}.npz")
    assert ep.history.shape == (r1.restarts + 1, 3) and ep.nit == r1.nit == int(ep.history[:, 0].sum())


def test_parse_lines():
    assert gate_mod.parse_lines("0,3-5, 9") == {0, 3, 4, 5, 9}


def test_acceptance_memo_is_keyed_on_every_endpoint_field(tmp_path, monkeypatch):
    """Another endpoint under the same endpoint key (a rerun with another s, or another extra_dL, which accept reads)
    misses the memo. Reviewer mutant "memo key without endpoint s" fails here."""
    from dataclasses import replace

    calls = []
    _stub(monkeypatch, calls)
    _, _, key = gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
    ep = gate_mod.retime.load_endpoint(tmp_path / f"{key}.npz")
    code = gate_mod.code_hash()
    for other in (replace(ep, s=ep.s + 1e-3), replace(ep, extra_dL=ep.extra_dL + 1.0)):
        assert gate_mod._acceptance_file(tmp_path, key, other) != gate_mod._acceptance_file(tmp_path, key, ep)
        np.savez(tmp_path / f"{key}.npz", **gate_mod.asdict(other), code_hash=code)  # the cached endpoint changes
        n = len(calls)
        gate_mod.retime_cached(*_TREE, _XX, _XX, None, tmp_path)
        assert len(calls) == n + 1 and np.array_equal(calls[-1].s, other.s)  # re-accepted, not memo-served
        assert calls[-1].extra_dL == other.extra_dL
