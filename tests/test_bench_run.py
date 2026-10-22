import json
import logging
import subprocess

import numpy as np
import pytest
from bench import gate, run
from bench.metrics import quartet_accuracy, rf_distance
from bench.simulators.brownian import simulate

from bonsai._engine.build import greedy_tree
from bonsai._engine.greedy import GreedyConfig
from bonsai._ref import pair as ref_pair
from bonsai._ref.newick import children_lists, parse_newick

from ._trees import bm_tree

CONFIGS = run.metadata.REPO / "bench" / "configs"
SMOKE = CONFIGS / "smoke_upstream.toml"
SMOKE_JAX = CONFIGS / "smoke_jax.toml"
SMOKE_ADATA = simulate(n_cells=24, n_genes=60, seed=0).adata
SMOKE_CELLS = set(SMOKE_ADATA.obs_names)


def _write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def _record(msg):
    """A ``logging.LogRecord`` carrying ``msg`` as its already-formatted message."""
    return logging.LogRecord(name="myapp", level=logging.INFO, pathname="", lineno=0, msg=msg, args=(), exc_info=None)


def preprocessed_sorted(adata):
    """``(x, var, cell_names)`` from the independent fp64 preprocessing port (bitwise upstream, M1)."""
    from bonsai._ref.preprocess import preprocess

    leaves = preprocess(adata)
    return leaves.x, leaves.var, list(adata.obs_names)


@pytest.mark.parametrize(
    "name",
    [
        "smoke_upstream",
        "upstream_sim_1k",
        "upstream_sim_3k",
        "upstream_sim_10k",
        "upstream_redo_sim_1k",
        "upstream_redo_sim_10k",
    ],
)
def test_committed_configs_load(name):
    cfg = run.load_config(CONFIGS / f"{name}.toml")
    assert cfg["run"]["backend"] == "upstream" and cfg["data"]["kind"] == "simulate"


def test_golden_config_loads():
    cfg = run.load_config(CONFIGS / "upstream_redo_golden.toml")
    assert cfg["run"]["stop_after"] == "redo_starry" and cfg["data"]["kind"] == "real"


def test_real_config_needs_a_path(tmp_path):
    with pytest.raises(run.ConfigError, match="--data-path"):
        run.load_config(CONFIGS / "real_upstream.toml")
    # (item 8) the path is checked inside load_config, before any output directory exists.
    with pytest.raises(run.ConfigError, match="does not exist"):
        run.load_config(CONFIGS / "real_upstream.toml", data_path=str(tmp_path / "x.h5ad"))


def test_real_data_path_is_checked_before_any_output_is_written(tmp_path):
    """(item 8) a bad --data-path fails before main() creates its output directory."""
    out = tmp_path / "out"
    with pytest.raises(run.ConfigError, match="does not exist"):
        run.main(
            [
                "--config",
                str(CONFIGS / "real_upstream.toml"),
                "--data-path",
                str(tmp_path / "missing.h5ad"),
                "--out",
                str(out),
            ]
        )
    assert not out.exists()


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[run]\nname='a'\nseed=0\n", "backend"),
        ("[run]\nname='a'\nseed=0\nbackend='torch'\n[data]\nkind='simulate'\n", "backend must be"),
        ("[run]\nname='a'\nseed=0\nbackend='upstream'\n[data]\nkind='csv'\n", "kind must be"),
        ("[run]\nname='a'\nseed=0\nbackend='upstream'\n[data]\nkind='simulate'\nn_cells=5\n", "n_genes"),
        ("[run\n", "invalid TOML"),
    ],
)
def test_invalid_configs_raise(tmp_path, text, match):
    with pytest.raises(run.ConfigError, match=match):
        run.load_config(_write(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        "[run]\nname='a'\nseed=0\nbackend='upstream'\nn_warmm=1\n[data]\nkind='simulate'\nn_cells=5\nn_genes=4\n",
        "[run]\nname='a'\nseed=0\nbackend='upstream'\n[data]\nkind='simulate'\nn_cells=5\nn_genes=4\ntypo=1\n",
        "[run]\nname='a'\nseed=0\nbackend='upstream'\n[data]\nkind='simulate'\nn_cells=5\nn_genes=4\n"
        "[upstream]\nnnn_n_randommovs=1\n",
    ],
)
def test_unknown_config_keys_are_rejected(tmp_path, text):
    with pytest.raises(run.ConfigError, match="unknown key"):
        run.load_config(_write(tmp_path, text))


def test_real_h5ad_without_bonsai_slots_is_rejected(tmp_path):
    a = simulate(5, 4, seed=0).adata
    del a.uns["bonsai_input"]
    a.uns.pop("bonsai_truth")
    a.write_h5ad(tmp_path / "x.h5ad")
    cfg = run.load_config(CONFIGS / "real_upstream.toml", data_path=str(tmp_path / "x.h5ad"))
    with pytest.raises(run.ConfigError, match="read_sanity"):
        run.load_data(cfg)


def test_stage_log_writes_an_empty_partial_file_immediately(tmp_path):
    """(item 9) a run capped before its first stage completes still leaves a partial file."""
    p = tmp_path / "partial.json"
    run._StageLog(p)
    assert json.loads(p.read_text()) == {"stages_s": []}


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


def test_stage_log_raises_only_on_stop_marker():
    log = run._StageLog(stop_after="redo_starry")
    log.emit(_record("Redoing starry nodes took 1.5 seconds."))  # recorded, no raise
    assert log.stages == [["Redoing starry nodes", 1.5]]
    with pytest.raises(run._StopAfterStage):
        log.emit(_record(run.STOP_MARKER))
    run._StageLog(stop_after=None).emit(_record(run.STOP_MARKER))  # never raises


def test_set_overrides_parse_and_validate(tmp_path):
    cfg = run.load_config(SMOKE, sets=["run.seed=2", "run.stop_after=redo_starry", "upstream.use_knn=5"])
    assert (cfg["run"]["seed"], cfg["run"]["stop_after"], cfg["upstream"]["use_knn"]) == (2, "redo_starry", 5)
    for bad in ["run.stop_after=nni", "nosuch.key=1", "run.nosuch=1", "run.seed"]:
        with pytest.raises(run.ConfigError):
            run.load_config(SMOKE, sets=[bad])


def test_leaf_sha256_is_order_canonical():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))
    var = rng.uniform(0.5, 2.0, size=(5, 3))
    names = ["e", "d", "c", "b", "a"]
    perm = [4, 3, 2, 1, 0]  # same cells, rows permuted (with their names)
    h = run.leaf_sha256(x, var, names)
    assert run.leaf_sha256(x[perm], var[perm], [names[i] for i in perm]) == h
    x_changed = x.copy()
    x_changed[0, 0] = np.nextafter(x_changed[0, 0], np.inf)  # one changed float flips the hash
    assert run.leaf_sha256(x_changed, var, names) != h
    var_changed = var.copy()
    var_changed[0, 0] = np.nextafter(var_changed[0, 0], np.inf)  # ``var`` is hashed too, not just ``x``
    assert run.leaf_sha256(x, var_changed, names) != h


def test_run_upstream_raises_if_stop_marker_never_fires(monkeypatch):
    """A run that somehow completes without hitting STOP_MARKER must never be labelled stage-limited."""
    monkeypatch.setattr(run.tl, "reconstruct", lambda *a, **k: None)  # "succeeds" without raising _StopAfterStage
    with pytest.raises(RuntimeError, match="stop_after"):
        run.run_upstream({}, {}, out=None, tag="run0", stop_after="redo_starry")


@pytest.mark.slow
def test_upstream_stop_after_redo_starry_smoke(tmp_path):
    out = run.main(["--config", str(SMOKE), "--set", "run.stop_after=redo_starry", "--out", str(tmp_path)])
    r = json.loads((out / "results.json").read_text())["runs"][0]
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["sets"] == ["run.stop_after=redo_starry"]
    names = [s for s, _ in r["stages_s"]]
    # Upstream logs a "Storing tree in folder" housekeeping stage after both the greedy and the
    # redo-starry stage, so the trailing two are not the stage names themselves (deviation from the
    # plan's literal example); the last real stage is still "Redoing starry nodes", after the greedy one.
    assert names[-2:] == ["Redoing starry nodes", "Storing tree in folder"]
    assert names.index("First greedy maximisation of tree likelihood") < names.index("Redoing starry nodes")
    assert "Optimization of diffusion times" not in names
    assert r["stage_newick"]["greedy"] != r["stage_newick"]["redo_starry"]  # redo starry actually ran a merge
    p, t, nm = parse_newick(r["stage_newick"]["redo_starry"])
    assert r["newick"] == r["stage_newick"]["redo_starry"]
    assert {n for n, k in zip(nm, children_lists(p), strict=True) if not k} == set(SMOKE_CELLS)
    # redo starry only applies merges with dLogL > 1e-6 and collapses zero edges, so it strictly raises L on
    # this fixture (measured: greedy -274.4696 vs redo-starry -273.0296); catches "greedy" harvested from the
    # redo-starry folder (which would make this an equality, not a strict increase).
    assert r["loglik"]["value"] > r["loglik"]["greedy_stage"]
    assert not any(tmp_path.rglob("run0_workdir"))  # workdir removed after harvest
    assert "upstream_finalLoglik" not in r["loglik"] and len(r["genes_used"]) == r["n_genes_used"]
    assert r["leaf_sha256"] == run.leaf_sha256(*preprocessed_sorted(SMOKE_ADATA))  # same as the port's arrays
    # A23: the processed leaf arrays travel with the baseline, saved before the workdir is removed.
    assert r["leaf_arrays"] == "run0_leaf_arrays.npz"
    with np.load(out / r["leaf_arrays"]) as z:
        assert run.leaf_sha256(z["x"], z["var"], list(z["cells"])) == r["leaf_sha256"]
        assert list(z["genes"]) == r["genes_used"]


def test_leaf_arrays_maps_rows_by_cell_ids_not_obs_order():
    """(A-1) rows come from ``cell_ids`` order, which may differ from any other cell ordering."""
    parent, _, names = parse_newick("((a:1,b:1)x:1,c:1)root;")
    x = np.array([[1.0], [2.0], [3.0]])  # row i belongs to cell_ids[i]
    var = np.array([[10.0], [20.0], [30.0]])
    X, V = run._leaf_arrays(parent, names, ["c", "a", "b"], x, var)
    want_x = {"a": 2.0, "b": 3.0, "c": 1.0}
    want_v = {"a": 20.0, "b": 30.0, "c": 10.0}
    for i, n in enumerate(names):
        if n in want_x:
            assert X[i, 0] == want_x[n] and V[i, 0] == want_v[n]
        else:
            assert X[i, 0] == 0.0 and V[i, 0] == 1.0  # internal/root default


def test_leaf_arrays_rejects_cell_id_mismatch():
    """(A-1) a leaf name missing from (or extra in) ``cell_ids`` raises instead of defaulting silently."""
    parent, _, names = parse_newick("((a:1,b:1)x:1,c:1)root;")
    x = var = np.ones((3, 1))
    with pytest.raises(ValueError, match="do not match"):
        run._leaf_arrays(parent, names, ["a", "b", "d"], x, var)  # "d" instead of "c"
    with pytest.raises(ValueError, match="do not match"):
        run._leaf_arrays(parent, names, ["a", "b"], x[:2], var[:2])  # "c" missing


def test_leaf_arrays_rejects_a_cell_name_on_a_non_leaf_node():
    """(A-1) a cell name must label a leaf; if upstream ever names an internal node after a cell, raise."""
    parent, _, names = parse_newick("((a:1,b:1)x:1,c:1)root;")
    names = [n if n != "x" else "a" for n in names]  # "a" now also labels the internal node x
    x = var = np.ones((3, 1))
    with pytest.raises(ValueError, match="has children"):
        run._leaf_arrays(parent, names, ["a", "b", "c"], x, var)


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


def test_count_compilations_counts_a_jit_compile():
    import jax
    import jax.numpy as jnp

    with run.count_compilations() as c:
        jax.jit(lambda x: x + 1)(jnp.ones(3))
    assert c.count >= 1


def test_timed_blocks_and_metadata_fields(tmp_path):
    import jax.numpy as jnp

    out, s = run.timed(lambda a: a * 2, jnp.ones(3))
    assert s >= 0 and np.asarray(out).tolist() == [2.0, 2.0, 2.0]
    meta = run.metadata.collect(CONFIGS / "smoke_upstream.toml", seed=0)
    for key in (
        "git_sha",
        "gpus",
        "versions",
        "seed",
        "config_sha256",
        "config_text",
        "timestamp_utc",
        "threads",
        "cpu_model",
    ):
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
    # (item 1) the full-precision upstream tree is stored, re-scorable without rerunning the baseline.
    # Upstream's tree may have polytomies, so only the root count and leaf count are checked structurally.
    parent, _, names = parse_newick(r["newick"])
    kids = children_lists(parent)
    assert int((parent < 0).sum()) == 1
    leaves = {names[i] for i in range(len(parent)) if not kids[i]}
    assert len(leaves) == 24
    # The real point of storing the newick: re-scoring it from scratch against the true tree (the
    # same config/seed the smoke config uses) reproduces the quartet/RF numbers written by main(),
    # without rerunning upstream.
    sim = simulate(n_cells=24, n_genes=60, seed=0)
    assert leaves == set(sim.adata.obs_names)
    truth, scored = (sim.parent, sim.names), (parent, names)
    acc, n_q = quartet_accuracy(truth, scored, seed=0)
    assert acc == r["truth"]["quartet_accuracy"] and n_q == r["truth"]["n_quartets"]
    assert rf_distance(truth, scored) == r["truth"]["rf_normalised"]
    assert json.loads((out / "run0_partial.json").read_text())["stages_s"] == r["stages_s"]
    assert "took" in (out / "run0_upstream.log").read_text()
    ll = r["loglik"]
    assert ll["convention"].startswith("A1")
    assert abs(ll["value"] - ll["upstream_finalLoglik"]) <= 1e-10 * abs(ll["upstream_finalLoglik"])
    assert ll["per_entry"] == pytest.approx(ll["value"] / (24 * r["n_genes_used"]), rel=1e-15)
    assert 0.0 <= r["truth"]["quartet_accuracy"] <= 1.0 and 0.0 <= r["truth"]["rf_normalised"] <= 1.0
    assert meta["seed"] == 0
    # A23: a full (non-stage-limited) run also saves its leaf arrays before the workdir is removed.
    with np.load(out / r["leaf_arrays"]) as z:
        assert run.leaf_sha256(z["x"], z["var"], list(z["cells"])) == r["leaf_sha256"]


@pytest.mark.parametrize("name", ["smoke_jax", "jax_sim_1k", "jax_sim_10k", "jax_golden", "jax_real"])
def test_committed_jax_configs_load(name):
    cfg = run.load_config(CONFIGS / f"{name}.toml", data_path="tests/data" if name == "jax_real" else None)
    assert cfg["run"]["backend"] == "jax" and "upstream_dir" not in cfg["run"]
    assert {"mode", "rescore_top", "merges_per_round"} <= set(cfg["jax"])  # the A17 settings are explicit


def test_load_config_jax_section():
    for bad in ["jax.nosuch=1", "run.stop_after=redo_starry", "jax.mode"]:
        with pytest.raises(run.ConfigError):
            run.load_config(SMOKE_JAX, sets=[bad])
    cfg = run.load_config(SMOKE_JAX, sets=["jax.mode=joint", "jax.rescore_top=64", "run.upstream_dir=x"])
    assert (cfg["jax"]["mode"], cfg["jax"]["rescore_top"], cfg["run"]["upstream_dir"]) == ("joint", 64, "x")
    with pytest.raises(run.ConfigError, match="jax backend only"):
        run.load_config(SMOKE, sets=["run.upstream_dir=x"])


def _fake_baseline(path, backend="upstream", **cfg_sets):
    """A bench-dir-shaped baseline (results.json only) with the smoke config's data/seed (plus overrides)."""
    cfg = run.load_config(SMOKE, sets=[f"{k}={v}" for k, v in cfg_sets.items()])
    path.mkdir()
    (path / "results.json").write_text(json.dumps({"config": cfg, "backend": backend, "runs": [{"stage_newick": {}}]}))
    return path


def _jax_main(tmp_path, upstream_dir, *sets):
    return run.main(
        ["--config", str(SMOKE_JAX), "--out", str(tmp_path / "out"), "--set", f"run.upstream_dir={upstream_dir}"]
        + [x for s_ in sets for x in ("--set", s_)]
    )


def test_jax_upstream_dir_is_validated_before_any_output(tmp_path):
    (tmp_path / "empty").mkdir()
    bad = {
        "missing": (tmp_path / "nope", "existing directory"),
        "empty": (tmp_path / "empty", "neither a bench run dir"),
        "jax backend dir": (_fake_baseline(tmp_path / "j", backend="jax"), "not a stage-limited upstream baseline"),
        "seed": (_fake_baseline(tmp_path / "s"), "differ"),
        "size": (_fake_baseline(tmp_path / "n"), "differ"),
    }
    extra = {"seed": ["run.seed=1"], "size": ["data.n_cells=30"]}
    for key, (d, match) in bad.items():
        with pytest.raises(run.ConfigError, match=match):
            _jax_main(tmp_path, d, *extra.get(key, []))
    with pytest.raises(run.ConfigError, match="existing directory"):
        run.main(["--config", str(SMOKE_JAX), "--out", str(tmp_path / "out")])  # no upstream_dir at all
    assert not (tmp_path / "out").exists()  # nothing written by any of them


def test_raw_upstream_folder_refuses_simulated_data(tmp_path):
    raw = tmp_path / "raw" / "intermediate_bonsai_zscore1.0_redoStarry"
    raw.mkdir(parents=True)
    (raw / "tree.nwk").write_text("(a:1,b:1)root;")
    with pytest.raises(run.ConfigError, match="simulated"):
        _jax_main(tmp_path, tmp_path / "raw")


def _noisy_leaves(n=24):
    _, _, x, var = bm_tree(n, 60, np.random.default_rng(11))
    return x[:n].copy(), np.full_like(x[:n], 1e6), [f"cell{i}" for i in range(n)]


def test_false_stop_diagnostic_is_read_only_and_flags_false_stops(monkeypatch):
    """var = 1e6: no pair passes in fp64, the stop checks run and find no false stop. With every host re-score faked
    to 1.0 the stop is flagged; the trees are identical in all cases (the diagnostic never writes the store)."""
    x, var, names = _noisy_leaves()
    base = greedy_tree(x, var, names, GreedyConfig())
    assert base.stats["n_stop_events"] == base.stats["n_band_rows"] == base.stats["n_false_stop_events"] == 0
    on = greedy_tree(x, var, names, GreedyConfig(false_stop_check=True))
    assert on.stats["n_stop_events"] > 0 and on.stats["n_band_rows"] > 0 and on.stats["n_false_stop_events"] == 0
    assert on.stats["stop_check_s"] > 0 and on.stage_newick == base.stage_newick
    real = ref_pair.pair_score
    monkeypatch.setattr(ref_pair, "pair_score", lambda *a, **k: real(*a, **k)._replace(dlogl=1.0))
    fake = greedy_tree(x, var, names, GreedyConfig(false_stop_check=True))
    assert fake.stats["n_false_stop_events"] == fake.stats["n_stop_events"] > 0
    assert fake.stage_newick == base.stage_newick


def test_false_stop_band_rows_are_exactly_the_fresh_rows_within_the_resolution(monkeypatch):
    """Exact count: with the resolution pinned to R (the median of 1e-6 - d over the fresh stop-round rows), the
    diagnostic re-scores exactly the fresh rows with 1e-6 - d <= R, read off the engine's own stop-round trace; a
    band-less version would re-score all of them."""
    from bonsai._engine import greedy as eg
    from bonsai._engine.greedy import NodeTable, greedy_star
    from bonsai._ref.star import star_em

    x, var, _ = _noisy_leaves()
    n = len(x)

    def run_star(cfg, trace=None):
        table = NodeTable(x, var)
        table.t[:n] = star_em(x, var, tol=cfg.star_tol)[0]
        return greedy_star(table, n, None, True, cfg, trace=trace)

    tr: list = []
    run_star(GreedyConfig(), tr)
    d = np.array([v[0] for v in tr[-1][0].values()])  # fresh rows of the final stop round, none host-scored
    assert len(d) == n * (n - 1) // 2 and not any(v[2] for v in tr[-1][0].values())
    gap = 1e-6 - d
    R = float(np.median(gap))
    expected = int(np.sum((gap <= R) & (d <= 1e-6)))
    assert 0 < expected < len(d)
    monkeypatch.setattr(eg, "pair_round_resolution", lambda *a, **k: R)
    st = run_star(GreedyConfig(false_stop_check=True))
    assert st.n_stop_events == 1 and st.n_band_rows == expected


@pytest.mark.slow
def test_jax_backend_smoke(tmp_path):
    """24 cells x 60 genes: upstream baseline (stage-limited) -> jax arm on its saved leaf arrays (A23) -> gate CLI."""
    up = run.main(["--config", str(SMOKE), "--set", "run.stop_after=redo_starry", "--out", str(tmp_path / "up")])
    up_run = json.loads((up / "results.json").read_text())["runs"][0]
    args = ["--config", str(SMOKE_JAX), "--set", f"run.upstream_dir={up}", "--out", str(tmp_path / "jax")]
    out = run.main(args)
    runs = json.loads((out / "results.json").read_text())["runs"]
    assert [s for s, _ in runs[0]["stages_s"]] == ["star_em", "greedy", "redo_starry"]
    assert runs[0]["compile_count"] > 0 and runs[1]["compile_count"] == 0  # a warm run compiles nothing
    assert runs[0]["newick"] == runs[1]["newick"] and [r["kind"] for r in runs] == ["cold", "warm"]
    assert {"quartet_accuracy", "quartet_accuracy_raw", "rf_normalised"} <= set(runs[0]["truth"])
    assert runs[0]["leaf_sha256"] == up_run["leaf_sha256"] and runs[0]["genes_used"] == up_run["genes_used"]
    assert runs[0]["leaf_sha256"] == run.leaf_sha256(*preprocessed_sorted(SMOKE_ADATA))
    assert runs[0]["greedy_config"]["false_stop_check"] is True
    assert {
        "n_reopts",
        "n_rebuilds",
        "n_rescored",
        "n_host_rescored",
        "n_threshold_rescored",
        "n_band_rows",
        "n_stop_events",
    } <= set(runs[0]["stats"])
    assert runs[0]["loglik"]["value"] >= runs[0]["loglik"]["greedy_stage"] - 1e-9
    # A23: the jax arm is scored on the baseline's arrays, so tampering with them is caught, not silently used
    with np.load(up / up_run["leaf_arrays"]) as z:
        arrays = dict(z)
    arrays["x"] = arrays["x"] + 1e-9
    np.savez(up / up_run["leaf_arrays"], **arrays)
    with pytest.raises(ValueError, match="leaf_sha256"):
        run.main(args)
    np.savez(up / up_run["leaf_arrays"], **(arrays | {"x": arrays["x"] - 1e-9}))
    rep = json.loads(
        gate.main(["--jax", str(out), "--upstream", str(up), "--out", str(tmp_path / "gate.json")]).read_text()
    )
    assert rep["verdict"] in ("PASS", "FAIL") and rep["n_cells"] == 24
