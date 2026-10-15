"""Benchmark runner: ``python -m bench.run --config bench/configs/<name>.toml [--data-path P] [--out DIR]``.

Writes ``<out>/<run.name>/<UTC timestamp>/``: ``metadata.json`` (first), per run ``run<i>_upstream.log``
(upstream's own log) and ``run<i>_partial.json`` (stage times so far, rewritten after every stage), and
``results.json`` (rewritten after every completed run). Only the upstream
backend exists until M2; its per-stage wall times come from upstream's own "... took X seconds."
log lines. Log-likelihoods follow amendment A1 (see ``LOGLIK_CONVENTION``).
"""

import argparse
import json
import logging
import os
import re
import resource
import shutil
import tempfile
import time
import tomllib
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import jax
import numpy as np

from bonsai import pp, tl
from bonsai._ref.newick import children_lists, parse_newick
from bonsai._ref.tree import tree_loglik

from . import metadata
from .metrics import quartet_accuracy, rf_distance
from .simulators.brownian import simulate

LOGLIK_CONVENTION = (
    "A1: sum over internal nodes of upstream's per-node terms (finalLoglik convention: 2x the Gaussian "
    "log-likelihood, no 2*pi terms, no loglikVarCorr), fp64 reference, full-precision newick times, "
    "upstream's preprocessed leaf arrays"
)
BACKENDS = ("upstream",)
DATA_KINDS = ("simulate", "real")
_STAGE = re.compile(r"^(?P<stage>.+?) took (?P<s>[0-9.eE+-]+) seconds\.?$")


class ConfigError(ValueError):
    """Invalid or incomplete benchmark config."""


def _write_json(path: Path, data) -> None:
    """Write JSON atomically (tmp file + ``os.replace``): a SIGKILL mid-write never leaves truncated JSON."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=float) + "\n")
    os.replace(tmp, path)


def load_config(path: Path, data_path: str | None = None) -> dict:
    """Parse and validate a TOML config; ``data_path`` overrides ``[data].path``."""
    try:
        cfg = tomllib.loads(Path(path).read_text())
    except FileNotFoundError:
        raise ConfigError(f"config not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {path}: {e}") from None
    for key in ("name", "seed", "backend"):
        if key not in cfg.get("run", {}):
            raise ConfigError(f"missing [run].{key}")
    if cfg["run"]["backend"] not in BACKENDS:
        raise ConfigError(f"[run].backend must be one of {BACKENDS}, got {cfg['run']['backend']!r}")
    data = cfg.setdefault("data", {})
    if data.get("kind") not in DATA_KINDS:
        raise ConfigError(f"[data].kind must be one of {DATA_KINDS}, got {data.get('kind')!r}")
    if data_path is not None:
        if data["kind"] == "simulate":
            raise ConfigError("--data-path is only for [data].kind = 'real'; this config simulates its data")
        data["path"] = data_path
    if data["kind"] == "simulate":
        for key in ("n_cells", "n_genes"):
            if key not in data:
                raise ConfigError(f"missing [data].{key} for kind = 'simulate'")
    elif not data.get("path"):
        raise ConfigError("[data].kind = 'real' needs [data].path or --data-path (an .h5ad file or a Sanity folder)")
    cfg["run"].setdefault("n_warm", 0)
    cfg.setdefault("upstream", {})
    return cfg


def load_data(cfg: dict):
    """(adata, truth or None). truth = (parent, names) of the true tree (simulator only)."""
    data = dict(cfg["data"])
    kind = data.pop("kind")
    if kind == "simulate":
        sim = simulate(seed=cfg["run"]["seed"], **data)
        return sim.adata, (sim.parent, sim.names)
    path = Path(data["path"])
    if not path.exists():
        raise ConfigError(f"data path does not exist: {path}")
    adata = pp.read_sanity(path) if path.is_dir() else ad.read_h5ad(path)
    if "bonsai_input" not in adata.uns:
        raise ConfigError(
            f"{data['path']} has no uns['bonsai_input']: prepare it with bonsai.pp.read_sanity or bonsai.pp.from_arrays"
        )
    return adata, None


def timed(fn, *args, **kwargs):
    """(result, seconds) with ``jax.block_until_ready`` at both boundaries (async dispatch)."""
    jax.block_until_ready(args)
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args, **kwargs))
    return out, time.perf_counter() - t0


_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"  # private JAX event name (jax 0.11)


@contextmanager
def count_compilations():
    """Counts XLA backend compilations inside the block (``with count_compilations() as c: ...; c.count``).

    Behaviour reproduced from rose ``rose.core.compile_count`` (mlbelab internal). The listener
    is process-global: it sees compilations from every thread while active.
    """
    box = SimpleNamespace(count=0)

    def on_event(event, duration_secs, **_):
        if event == _COMPILE_EVENT:
            box.count += 1

    jax.monitoring.register_event_duration_secs_listener(on_event)
    try:
        yield box
    finally:
        jax.monitoring.unregister_event_duration_listener(on_event)


def device_peak_bytes() -> int | None:
    """Peak bytes in use on the default device so far (None on CPU, where JAX does not report it)."""
    stats = jax.devices()[0].memory_stats()
    return None if stats is None else stats.get("peak_bytes_in_use")


class _StageLog(logging.Handler):
    """Collects upstream's '<stage> took <s> seconds.' log lines (helpers.py logger 'myapp').

    Stages are kept in order of occurrence ("Optimization of diffusion times" happens twice).
    With ``partial`` set, every new stage rewrites that JSON file, so a run killed by its time cap
    still leaves the stage times it reached.
    """

    def __init__(self, partial: Path | None = None):
        super().__init__(logging.INFO)
        self.stages: list[list] = []  # [stage, seconds], in order
        self.partial = partial

    def emit(self, record):
        m = _STAGE.match(record.getMessage().strip())
        if m:
            self.stages.append([m["stage"], float(m["s"])])
            if self.partial is not None:
                _write_json(self.partial, {"stages_s": self.stages})


def _leaf_arrays(parent: np.ndarray, names: list[str], cell_ids: list[str], x: np.ndarray, var: np.ndarray):
    """Node-indexed ``(X, V)`` for ``tree_loglik``: newick leaves get ``x``/``var`` rows by ``cell_ids`` order.

    ``cell_ids`` must be exactly the newick's leaf-name set (upstream's own
    ``processed/metadata.json`` ``cellIds``, i.e. ``delta.npy``'s row order); a leaf name missing
    from ``cell_ids``, or vice versa, raises instead of silently defaulting to ``x=0, var=1``. A
    name present in ``cell_ids`` must also label a leaf (no children) in the newick.
    """
    kids = children_lists(parent)
    leaves = {names[i] for i in range(len(parent)) if not kids[i]}
    if leaves != set(cell_ids):
        raise ValueError(
            "newick leaf names do not match the processed cell order: "
            f"missing from newick {set(cell_ids) - leaves}, unexpected in newick {leaves - set(cell_ids)}"
        )
    row = {c: i for i, c in enumerate(cell_ids)}
    X, V = np.zeros((len(parent), x.shape[1])), np.ones((len(parent), x.shape[1]))
    for node, name in enumerate(names):
        if name in row:
            if kids[node]:
                raise ValueError(f"cell node {name!r} has children in the upstream tree")
            X[node], V[node] = x[row[name]], var[row[name]]
    return X, V


def run_upstream(adata, upstream_kw: dict, out: Path | None = None, tag: str = "run0") -> dict:
    """One ``tl.reconstruct`` run (backend="upstream"): wall time, stage times, A1 log-likelihood.

    With ``out``, upstream's own log goes to ``out/<tag>_upstream.log`` and the stage times so far
    to ``out/<tag>_partial.json`` as they happen.
    """
    log = _StageLog(None if out is None else out / f"{tag}_partial.json")
    logger = logging.getLogger("myapp")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    handlers: list[logging.Handler] = [log]
    if out is not None:
        fh = logging.FileHandler(out / f"{tag}_upstream.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        handlers.append(fh)
    for h in handlers:
        logger.addHandler(h)
    # Under `out` (not node-local /tmp): a SLURM kill skips this function's `finally` below, so the
    # workdir would otherwise leak with no trace; keeping it next to the run's other output files
    # means a killed baseline still leaves its intermediate state for inspection.
    wd = Path(tempfile.mkdtemp(prefix="bonsai_bench_")) if out is None else out / f"{tag}_workdir"
    if out is not None:
        wd.mkdir()
    try:
        a = adata.copy()
        t0 = time.perf_counter()
        tl.reconstruct(a, workdir=wd, keep_workdir=True, **upstream_kw)
        wall = time.perf_counter() - t0
        (processed,) = wd.rglob("zscorefiltered_*_and_processed")
        x, var = np.load(processed / "delta.npy").T, np.load(processed / "delta_vars.npy").T  # cells × genes
    finally:
        for h in handlers:
            logger.removeHandler(h)
            h.close()
        logger.setLevel(old_level)
    shutil.rmtree(wd, ignore_errors=True)  # only reached on success; a failure or kill leaves wd behind
    res = a.uns["bonsai"]
    parent, t, names = parse_newick(res["newick"])
    cell_ids = json.loads((processed / "metadata.json").read_text())["cellIds"]  # delta.npy's own row order
    X, V = _leaf_arrays(parent, names, cell_ids, x, var)
    L = tree_loglik(parent, t, X, V)
    return {
        "wall_s": wall,
        "stages_s": log.stages,
        # Linux: ru_maxrss is KiB and process-wide (not per-stage), so it includes the simulator's own
        # memory, and a warm run's value already carries the cold run's peak (RUSAGE_SELF never resets).
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "device_peak_bytes": device_peak_bytes(),
        "n_genes_used": int(x.shape[1]),
        "loglik": {
            "convention": LOGLIK_CONVENTION,
            "value": L,
            "per_entry": L / (a.n_obs * x.shape[1]),
            "upstream_finalLoglik": res["loglik"],
        },
        # Full-precision newick: kept re-scorable (item 1) because M2 may change how quartets treat
        # zero-length edges or polytomies, so the baselines must not fix the metric at this commit.
        "newick": res["newick"],
        "tree": (parent, names),
    }


def main(argv: list[str] | None = None) -> Path:
    """Run one config; return the output folder."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--out", type=Path, default=metadata.REPO / "bench" / "results")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.data_path)
    meta = metadata.collect(args.config, cfg["run"]["seed"])
    out = args.out / cfg["run"]["name"] / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")  # first, so a capped run keeps it
    adata, truth = load_data(cfg)
    runs = []
    for i in range(1 + cfg["run"]["n_warm"]):
        r = run_upstream(adata, cfg["upstream"], out, tag=f"run{i}")
        tree = r.pop("tree")
        r["kind"] = "cold" if i == 0 else "warm"  # upstream compiles nothing; "cold" = first run in the process
        if truth is not None:
            acc, n_q = quartet_accuracy(truth, tree, seed=cfg["run"]["seed"])
            r["truth"] = {"quartet_accuracy": acc, "n_quartets": n_q, "rf_normalised": rf_distance(truth, tree)}
        runs.append(r)
        results = {"config": cfg, "backend": cfg["run"]["backend"], "n_cells": adata.n_obs, "runs": runs}
        _write_json(out / "results.json", results)  # after every run
    return out


if __name__ == "__main__":
    print(main())
