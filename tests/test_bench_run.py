import json
import logging
import subprocess

import numpy as np
import pytest
from bench import run
from bench.simulators.brownian import simulate

CONFIGS = run.metadata.REPO / "bench" / "configs"


def _write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


@pytest.mark.parametrize("name", ["smoke_upstream", "upstream_sim_1k", "upstream_sim_3k", "upstream_sim_10k"])
def test_committed_configs_load(name):
    cfg = run.load_config(CONFIGS / f"{name}.toml")
    assert cfg["run"]["backend"] == "upstream" and cfg["data"]["kind"] == "simulate"


def test_real_config_needs_a_path(tmp_path):
    with pytest.raises(run.ConfigError, match="--data-path"):
        run.load_config(CONFIGS / "real_upstream.toml")
    cfg = run.load_config(CONFIGS / "real_upstream.toml", data_path=str(tmp_path / "x.h5ad"))
    with pytest.raises(run.ConfigError, match="does not exist"):
        run.load_data(cfg)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[run]\nname='a'\nseed=0\n", "backend"),
        ("[run]\nname='a'\nseed=0\nbackend='jax'\n[data]\nkind='simulate'\n", "backend must be"),
        ("[run]\nname='a'\nseed=0\nbackend='upstream'\n[data]\nkind='csv'\n", "kind must be"),
        ("[run]\nname='a'\nseed=0\nbackend='upstream'\n[data]\nkind='simulate'\nn_cells=5\n", "n_genes"),
        ("[run\n", "invalid TOML"),
    ],
)
def test_invalid_configs_raise(tmp_path, text, match):
    with pytest.raises(run.ConfigError, match=match):
        run.load_config(_write(tmp_path, text))


def test_real_h5ad_without_bonsai_slots_is_rejected(tmp_path):
    a = simulate(5, 4, seed=0).adata
    del a.uns["bonsai_input"]
    a.uns.pop("bonsai_truth")
    a.write_h5ad(tmp_path / "x.h5ad")
    cfg = run.load_config(CONFIGS / "real_upstream.toml", data_path=str(tmp_path / "x.h5ad"))
    with pytest.raises(run.ConfigError, match="read_sanity"):
        run.load_data(cfg)


def test_stage_log_keeps_stages_in_order_and_writes_partial_results(tmp_path):
    h = run._StageLog(tmp_path / "partial.json")
    logger = logging.getLogger("bench_test_stage")
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    for msg in (
        "SPR moves took 2.5 seconds.",
        "Optimization of diffusion times took 1e-3 seconds.",
        "Optimization of diffusion times took 1 seconds.",
        "Loglikelihood of tree: -5",
    ):
        logger.info(msg)
    logger.removeHandler(h)
    want = [["SPR moves", 2.5], ["Optimization of diffusion times", 0.001], ["Optimization of diffusion times", 1.0]]
    assert h.stages == want  # the two time optimisations stay separate
    assert json.loads((tmp_path / "partial.json").read_text()) == {"stages_s": want}


def test_data_path_is_rejected_for_simulated_data(tmp_path):
    with pytest.raises(run.ConfigError, match="only for"):
        run.load_config(CONFIGS / "upstream_sim_1k.toml", data_path=str(tmp_path / "x.h5ad"))


def test_git_dirty_ignores_untracked_files(tmp_path, monkeypatch):
    """bench/results/ is committed only in Task 5, so an untracked file must not mark the tree dirty."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("x")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    (repo / "untracked.txt").write_text("y")  # e.g. a benchmark output file not yet committed
    monkeypatch.setattr(run.metadata, "REPO", repo)
    assert run.metadata.collect(CONFIGS / "smoke_upstream.toml", seed=0)["git_dirty"] is False
    (repo / "tracked.txt").write_text("changed")
    assert run.metadata.collect(CONFIGS / "smoke_upstream.toml", seed=0)["git_dirty"] is True


def test_timed_blocks_and_metadata_fields(tmp_path):
    import jax.numpy as jnp

    out, s = run.timed(lambda a: a * 2, jnp.ones(3))
    assert s >= 0 and np.asarray(out).tolist() == [2.0, 2.0, 2.0]
    meta = run.metadata.collect(CONFIGS / "smoke_upstream.toml", seed=0)
    for key in ("git_sha", "gpus", "versions", "seed", "config_sha256", "config_text", "timestamp_utc", "threads"):
        assert key in meta
    assert meta["versions"]["bonsai"] is not None


@pytest.mark.slow
def test_smoke_run_end_to_end(tmp_path):
    """Upstream backend on 24 simulated cells: results/metadata JSON written; the A1 log-likelihood
    (fp64 reference on full-precision newick times) equals upstream's finalLoglik to 1e-10 relative
    (the M1 gate's reference-vs-upstream bound; both are the same sum in fp64)."""
    out = run.main(["--config", str(CONFIGS / "smoke_upstream.toml"), "--out", str(tmp_path)])
    res = json.loads((out / "results.json").read_text())
    meta = json.loads((out / "metadata.json").read_text())
    (r,) = res["runs"]
    assert r["kind"] == "cold" and r["wall_s"] > 0 and "SPR moves" in [name for name, _ in r["stages_s"]]
    assert json.loads((out / "run0_partial.json").read_text())["stages_s"] == r["stages_s"]
    assert "took" in (out / "run0_upstream.log").read_text()
    ll = r["loglik"]
    assert ll["convention"].startswith("A1")
    assert abs(ll["value"] - ll["upstream_finalLoglik"]) <= 1e-10 * abs(ll["upstream_finalLoglik"])
    assert ll["per_entry"] == pytest.approx(ll["value"] / (24 * r["n_genes_used"]), rel=1e-15)
    assert 0.0 <= r["truth"]["quartet_accuracy"] <= 1.0 and 0.0 <= r["truth"]["rf_normalised"] <= 1.0
    assert meta["seed"] == 0
