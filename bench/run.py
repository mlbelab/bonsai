"""Benchmark runner: ``python -m bench.run --config bench/configs/<name>.toml [--data-path P] [--out DIR] [--set K=V]*``.

Writes ``<out>/<run.name>/<UTC timestamp>/``: ``metadata.json`` (first), per run ``run<i>_upstream.log``
(upstream's own log) and ``run<i>_partial.json`` (stage times so far, rewritten after every stage), and
``results.json`` (rewritten after every completed run). The upstream backend's per-stage wall
times come from upstream's own "... took X seconds." log lines. Log-likelihoods follow amendment A1
(see ``LOGLIK_CONVENTION``).

``backend = "jax"`` (M2) runs ``bonsai._engine.build.greedy_tree`` on the leaf arrays saved by its
upstream baseline (A23; ``[run] upstream_dir`` = a bench run dir or a raw upstream results folder,
read with ``bench.gate.load_arm``), never on regenerated data. ``[jax]`` keys are ``GreedyConfig``
fields; ``n_warm`` warm runs repeat in the same process (cold = first, with the compilations).

``[run] stop_after = "redo_starry"`` (A22) stops the upstream run right after the redo-starry tree is
stored, without touching ``_core``: a log handler raises once it sees ``STOP_MARKER``, and the
harvested greedy/redo-starry stage trees are returned instead of a finished run.

Before the workdir is removed, the run's processed leaf arrays (cells × genes float64 ``x``/``var``,
cell names, ``genes_used``: exactly what ``leaf_sha256`` hashes) are saved as ``<tag>_leaf_arrays.npz``
next to the other per-run output files and recorded as ``results.json``'s ``leaf_arrays`` (A23): the
gate reads them from there instead of regenerating leaf data, which is not bit-identical across CPU
SIMD targets.
"""

import argparse
import dataclasses
import hashlib
import inspect
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
from bonsai._engine.build import greedy_tree
from bonsai._engine.greedy import GreedyConfig, GreedyStats
from bonsai._engine.opt_times import T_SNAP
from bonsai._ref.newick import children_lists, parse_newick
from bonsai._ref.tree import tree_loglik

from . import metadata
from .metrics import contract_zero_edges, quartet_accuracy, rf_distance
from .simulators.brownian import simulate

LOGLIK_CONVENTION = (
    "A1: sum over internal nodes of upstream's per-node terms (finalLoglik convention: 2x the Gaussian "
    "log-likelihood, no 2*pi terms, no loglikVarCorr), fp64 reference, full-precision newick times, "
    "upstream's preprocessed leaf arrays"
)
BACKENDS = ("upstream", "jax")
DATA_KINDS = ("simulate", "real")
_STAGE = re.compile(r"^(?P<stage>.+?) took (?P<s>[0-9.eE+-]+) seconds\.?$")

_RUN_KEYS = {"name", "seed", "backend", "n_warm", "stop_after", "upstream_dir"}
_DATA_KEYS = {"kind", "path"} | (set(inspect.signature(simulate).parameters) - {"seed"})
_UPSTREAM_KEYS = set(inspect.signature(tl.reconstruct).parameters) - {"adata", "workdir", "keep_workdir", "copy"}
_STOP_AFTER_VALUES = ("redo_starry",)
_JAX_KEYS = {f.name for f in dataclasses.fields(GreedyConfig)}
_SET_SECTIONS = ("run", "data", "upstream", "jax")
STOP_MARKER = "Starting final optimization of all diffusion times."


class ConfigError(ValueError):
    """Invalid or incomplete benchmark config."""


class _StopAfterStage(Exception):
    """Raised by ``_StageLog`` to unwind out of ``tl.reconstruct`` once the requested stage is reached."""


def _parse_set_value(raw: str):
    """``raw`` as a TOML value (``"2"`` -> ``2``, ``"true"`` -> ``True``, ...), else the bare string."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_sets(cfg: dict, sets) -> None:
    """Apply ``SECTION.KEY=VALUE`` overrides to ``cfg`` in place, before validation."""
    for item in sets:
        keypath, sep, raw = item.partition("=")
        if not sep or "." not in keypath:
            raise ConfigError(f"--set {item!r} is not SECTION.KEY=VALUE")
        section, _, key = keypath.partition(".")
        if section not in _SET_SECTIONS:
            raise ConfigError(f"--set {item!r}: unknown section {section!r}")
        cfg.setdefault(section, {})[key] = _parse_set_value(raw)


def _write_json(path: Path, data) -> None:
    """Write JSON atomically (tmp file + ``os.replace``): a SIGKILL mid-write never leaves truncated JSON."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=float) + "\n")
    os.replace(tmp, path)


def _reject_unknown_keys(section: dict, known: set[str], name: str) -> None:
    extra = set(section) - known
    if extra:
        raise ConfigError(f"unknown key(s) in [{name}]: {sorted(extra)}")


def load_config(path: Path, data_path: str | None = None, sets=()) -> dict:
    """Parse and validate a TOML config; ``data_path`` overrides ``[data].path``.

    ``sets`` are ``SECTION.KEY=VALUE`` overrides (CLI ``--set``, repeatable), applied to the parsed
    dict before validation.
    """
    try:
        cfg = tomllib.loads(Path(path).read_text())
    except FileNotFoundError:
        raise ConfigError(f"config not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {path}: {e}") from None
    _apply_sets(cfg, sets)
    _reject_unknown_keys(cfg.get("run", {}), _RUN_KEYS, "run")
    for key in ("name", "seed", "backend"):
        if key not in cfg.get("run", {}):
            raise ConfigError(f"missing [run].{key}")
    if cfg["run"]["backend"] not in BACKENDS:
        raise ConfigError(f"[run].backend must be one of {BACKENDS}, got {cfg['run']['backend']!r}")
    stop_after = cfg["run"].get("stop_after")
    if stop_after is not None and stop_after not in _STOP_AFTER_VALUES:
        raise ConfigError(f"[run].stop_after must be one of {_STOP_AFTER_VALUES}, got {stop_after!r}")
    if cfg["run"]["backend"] == "jax" and stop_after is not None:
        raise ConfigError("[run].stop_after is for the upstream backend only")
    if cfg["run"]["backend"] != "jax" and "upstream_dir" in cfg["run"]:
        raise ConfigError("[run].upstream_dir is for the jax backend only")
    data = cfg.setdefault("data", {})
    _reject_unknown_keys(data, _DATA_KEYS, "data")
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
    else:
        if not data.get("path"):
            raise ConfigError(
                "[data].kind = 'real' needs [data].path or --data-path (an .h5ad file or a Sanity folder)"
            )
        if not Path(data["path"]).exists():  # (item 8) checked here, before any output dir/metadata.json exists
            raise ConfigError(f"data path does not exist: {data['path']}")
    cfg["run"].setdefault("n_warm", 0)
    cfg.setdefault("upstream", {})
    _reject_unknown_keys(cfg["upstream"], _UPSTREAM_KEYS, "upstream")
    _reject_unknown_keys(cfg.get("jax", {}), _JAX_KEYS, "jax")
    return cfg


def load_data(cfg: dict):
    """(adata, truth or None). truth = (parent, names) of the true tree (simulator only)."""
    data = dict(cfg["data"])
    kind = data.pop("kind")
    if kind == "simulate":
        sim = simulate(seed=cfg["run"]["seed"], **data)
        return sim.adata, (sim.parent, sim.names)
    path = Path(data["path"])  # existence already checked by load_config (item 8)
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
    With ``partial`` set, the file is written immediately (empty) and every new stage rewrites
    it, so a run killed before its first stage completes still leaves one (item 9).

    With ``stop_after`` set (A22), ``emit`` raises ``_StopAfterStage`` the moment it sees
    ``STOP_MARKER`` (the line upstream logs right after storing the requested stage's tree); the
    exception propagates out of ``logging.Handler.handle`` and up through ``tl.reconstruct``.
    """

    def __init__(self, partial: Path | None = None, stop_after: str | None = None):
        super().__init__(logging.INFO)
        self.stages: list[list] = []  # [stage, seconds], in order
        self.partial = partial
        self.stop_after = stop_after
        if self.partial is not None:
            _write_json(self.partial, {"stages_s": self.stages})

    def emit(self, record):
        msg = record.getMessage().strip()
        m = _STAGE.match(msg)
        if m:
            self.stages.append([m["stage"], float(m["s"])])
            if self.partial is not None:
                _write_json(self.partial, {"stages_s": self.stages})
        if self.stop_after is not None and msg == STOP_MARKER:
            raise _StopAfterStage(msg)


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


def leaf_sha256(x: np.ndarray, var: np.ndarray, cell_names: list[str]) -> str:
    """SHA-256 of the float64 bytes of ``x`` then ``var``, rows sorted by ``cell_names`` (order-canonical)."""
    order = np.argsort(cell_names, kind="stable")
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(x, dtype=np.float64)[order]).tobytes())
    h.update(np.ascontiguousarray(np.asarray(var, dtype=np.float64)[order]).tobytes())
    return h.hexdigest()


_GREEDY_FOLDER = re.compile(r"intermediate_bonsai(_zscore[0-9.]+)?")


def run_upstream(
    adata, upstream_kw: dict, out: Path | None = None, tag: str = "run0", stop_after: str | None = None
) -> dict:
    """One ``tl.reconstruct`` run (backend="upstream"): wall time, stage times, A1 log-likelihood.

    With ``out``, upstream's own log goes to ``out/<tag>_upstream.log`` and the stage times so far
    to ``out/<tag>_partial.json`` as they happen.

    With ``stop_after="redo_starry"`` (A22), the run is stopped right after the redo-starry tree is
    stored (no ``_core`` change: a log handler raises on ``STOP_MARKER``); the result then carries
    ``"stage_newick"`` (``"greedy"``, ``"redo_starry"``), ``loglik["greedy_stage"]``, and no
    ``upstream_finalLoglik`` (no final stage ran). Every result, stopped or not, carries
    ``"genes_used"`` and ``"leaf_sha256"``. Raises ``RuntimeError`` if ``stop_after`` is set but the
    run completed without ever hitting ``STOP_MARKER`` (never silently mislabel a full run).

    With ``out``, the processed leaf arrays are also saved as ``<tag>_leaf_arrays.npz`` and recorded
    as ``"leaf_arrays"`` in the returned dict (A23); without ``out`` there is no results folder to
    save them into.
    """
    if stop_after is not None and stop_after not in _STOP_AFTER_VALUES:
        raise ValueError(f"stop_after must be one of {_STOP_AFTER_VALUES}, got {stop_after!r}")
    log = _StageLog(None if out is None else out / f"{tag}_partial.json", stop_after=stop_after)
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
        stopped = False
        try:
            tl.reconstruct(a, workdir=wd, keep_workdir=True, **upstream_kw)
        except _StopAfterStage:
            stopped = True  # the workdir (kept: keep_workdir=True) is harvested below
        wall = time.perf_counter() - t0
        if stop_after is not None and not stopped:
            # A full run that somehow finished without ever hitting STOP_MARKER must never be
            # mislabeled as a stage-limited baseline (it would carry a full-run newick/loglik under
            # the stage-limited result shape, silently poisoning the M2 gate's inputs).
            raise RuntimeError(
                f"stop_after={stop_after!r} requested but the run completed without hitting {STOP_MARKER!r}"
            )
        (processed,) = wd.rglob("zscorefiltered_*_and_processed")
        x, var = np.load(processed / "delta.npy").T, np.load(processed / "delta_vars.npy").T  # cells × genes
        cell_ids = json.loads((processed / "metadata.json").read_text())["cellIds"]  # delta.npy's own row order
    finally:
        for h in handlers:
            logger.removeHandler(h)
            h.close()
        logger.setLevel(old_level)
    result = {
        "wall_s": wall,
        "stages_s": log.stages,
        # Linux: ru_maxrss is KiB and process-wide (not per-stage), so it includes the simulator's own
        # memory, and a warm run's value already carries the cold run's peak (RUSAGE_SELF never resets).
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "device_peak_bytes": device_peak_bytes(),
        "n_genes_used": int(x.shape[1]),
        "leaf_sha256": leaf_sha256(x, var, cell_ids),
    }
    if stop_after is None:
        res = a.uns["bonsai"]
        parent, t, names = parse_newick(res["newick"])
        X, V = _leaf_arrays(parent, names, cell_ids, x, var)
        L = tree_loglik(parent, t, X, V)
        result |= {
            "genes_used": res["genes_used"],
            # Full-precision newick: kept re-scorable (item 1) because M2 may change how quartets treat
            # zero-length edges or polytomies, so the baselines must not fix the metric at this commit.
            "newick": res["newick"],
            "loglik": {
                "convention": LOGLIK_CONVENTION,
                "value": L,
                "per_entry": L / (a.n_obs * x.shape[1]),
                "upstream_finalLoglik": res["loglik"],
            },
            "tree": (parent, names),
        }
    else:
        results_dir = wd / "results"
        redo_dirs = sorted(results_dir.glob("intermediate_bonsai*_redoStarry"))
        if len(redo_dirs) != 1:
            raise RuntimeError(
                f"expected exactly one redo-starry result folder in {results_dir}, found {len(redo_dirs)}"
            )
        greedy_dirs = [p for p in results_dir.iterdir() if _GREEDY_FOLDER.fullmatch(p.name)]
        if len(greedy_dirs) != 1:
            raise RuntimeError(f"expected exactly one greedy result folder in {results_dir}, found {len(greedy_dirs)}")
        redo_nwk = (redo_dirs[0] / "tree.nwk").read_text().strip()
        greedy_nwk = (greedy_dirs[0] / "tree.nwk").read_text().strip()
        genes_used = list(json.loads((redo_dirs[0] / "metadata.json").read_text())["geneIds"])
        parent, t, names = parse_newick(redo_nwk)
        X, V = _leaf_arrays(parent, names, cell_ids, x, var)
        L = tree_loglik(parent, t, X, V)
        pg, tg, ng = parse_newick(greedy_nwk)
        Xg, Vg = _leaf_arrays(pg, ng, cell_ids, x, var)
        L_greedy = tree_loglik(pg, tg, Xg, Vg)
        result |= {
            "genes_used": genes_used,
            "newick": redo_nwk,
            "stage_newick": {"greedy": greedy_nwk, "redo_starry": redo_nwk},
            "loglik": {
                "convention": LOGLIK_CONVENTION,
                "value": L,
                "per_entry": L / (a.n_obs * x.shape[1]),
                "greedy_stage": L_greedy,
            },
            "tree": (parent, names),
        }
    if out is not None:
        # A23: the simulator's leaf arrays are not bit-identical across CPU SIMD targets, so the gate
        # must never regenerate them; save exactly what leaf_sha256 hashed, next to the workdir it came
        # from, before that workdir (and the processed/ arrays inside it) is removed below.
        leaf_arrays_name = f"{tag}_leaf_arrays.npz"
        np.savez(out / leaf_arrays_name, x=x, var=var, cells=np.array(cell_ids), genes=np.array(result["genes_used"]))
        result["leaf_arrays"] = leaf_arrays_name
    shutil.rmtree(wd, ignore_errors=True)  # only reached on success; a failure or kill leaves wd behind
    return result


def _check_jax_baseline(cfg: dict) -> None:
    """Validate ``[run].upstream_dir`` for a jax run: an upstream bench run dir or a raw upstream results folder.

    A bench dir must come from the upstream backend with stage trees, and its ``[data]`` and seed must equal the
    jax config's (the truth tree and the leaf arrays both belong to the baseline). A raw folder has no config, so
    a simulated ``[data]`` is refused (no truth without it).
    """
    d = cfg["run"].get("upstream_dir")
    if not d or not Path(d).is_dir():
        raise ConfigError("backend jax needs [run].upstream_dir (--set run.upstream_dir=DIR): an existing directory")
    d = Path(d)
    if (d / "results.json").exists():
        res = json.loads((d / "results.json").read_text())
        if res.get("backend") != "upstream" or "stage_newick" not in res["runs"][0]:
            raise ConfigError(
                f"{d} is not a stage-limited upstream baseline (backend upstream, stop_after redo_starry)"
            )
        base = res["config"]
        if base["data"] != cfg["data"] or base["run"]["seed"] != cfg["run"]["seed"]:
            raise ConfigError(
                f"jax [data]/seed differ from the baseline's ({d}): {cfg['data']}, seed {cfg['run']['seed']} vs "
                f"{base['data']}, seed {base['run']['seed']}"
            )
    elif not list(d.glob("intermediate_bonsai*_redoStarry/tree.nwk")):
        raise ConfigError(f"{d} is neither a bench run dir (results.json) nor an upstream results folder")
    elif cfg["data"]["kind"] == "simulate":
        raise ConfigError("a raw upstream results folder has no simulator config: simulated [data] is refused")


def run_jax(arm: dict, jax_kw: dict) -> dict:
    """One ``greedy_tree`` run on the upstream baseline's leaf arrays (``arm`` from ``bench.gate.load_arm``, A23).

    Result: ``wall_s``, ``stages_s`` (star_em, greedy, redo_starry; ``block_until_ready`` inside
    ``greedy_tree``), ``compile_count`` (the whole call), ``device_peak_bytes`` (process-wide peak, None on
    CPU), ``n_genes_used``, ``genes_used`` and ``leaf_sha256`` (the baseline's; the arrays are hashed here
    and must match its recorded hash), ``loglik`` (A1 on the redo-starry tree, ``greedy_stage`` too),
    ``newick``, ``stage_newick``, ``greedy_config`` (the effective ``GreedyConfig``) and ``stats`` (the
    ``GreedyStats`` counters incl. the false-stop diagnostic, ``n_merges``, redo passes). ``tree`` =
    (parent, t, names) of the redo-starry tree, for the caller to pop.

    ``device_peak_bytes`` is the per-process peak so far (a running maximum over runs sharing the process). For a
    raw upstream results folder the recorded hash is computed from the same arrays, so the hash check is a
    self-comparison there; for a bench dir it checks the stored npz against the recorded value.
    """
    x, var, cells = arm["x"], arm["var"], list(arm["cells"])
    sha = leaf_sha256(x, var, cells)
    if sha != arm["leaf_sha256"]:
        raise ValueError("the baseline's leaf arrays do not match its recorded leaf_sha256")
    if list(arm["genes"]) != list(arm["genes_used"]):
        raise ValueError("the baseline's stored genes differ from its recorded genes_used")
    cfg = GreedyConfig(**jax_kw)
    with count_compilations() as c:
        t0 = time.perf_counter()
        built = greedy_tree(x, var, cells, cfg)
        wall = time.perf_counter() - t0
    L = {}
    for stage, nwk in built.stage_newick.items():
        p, t, nm = parse_newick(nwk)
        L[stage] = tree_loglik(p, t, *_leaf_arrays(p, nm, cells, x, var))
    s = built.stats
    return {
        "wall_s": wall,
        "stages_s": [[k, s["stage_s"][k]] for k in ("star_em", "greedy", "redo_starry")],
        "compile_count": c.count,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "device_peak_bytes": device_peak_bytes(),
        "n_genes_used": int(x.shape[1]),
        "genes_used": list(arm["genes_used"]),
        "leaf_sha256": sha,
        "loglik": {
            "convention": LOGLIK_CONVENTION,
            "value": L["redo_starry"],
            "per_entry": L["redo_starry"] / (len(cells) * x.shape[1]),
            "greedy_stage": L["greedy"],
        },
        "newick": built.stage_newick["redo_starry"],
        "stage_newick": built.stage_newick,
        "greedy_config": dataclasses.asdict(cfg),
        "stats": {k: s[k] for k in GreedyStats._fields if k != "events"}
        | {
            "n_merges": sum(e.kind == "merge" for e in s["events"]),
            "passes": s["passes"],
            "zero_edges_collapsed": s["zero_edges_collapsed"],
        },
        "tree": (built.parent, built.t, built.names),
    }


def _truth_metrics(truth, tree, seed: int) -> dict:
    """Quartet accuracy and RF against the truth (A15 on the tree's own times; ``_raw``: uncontracted)."""
    parent, t, names = tree
    pc, _, nc = contract_zero_edges(parent, t, names, snap=T_SNAP)
    acc, n_q = quartet_accuracy(truth, (pc, nc), seed=seed)
    return {
        "quartet_accuracy": acc,
        "n_quartets": n_q,
        "quartet_accuracy_raw": quartet_accuracy(truth, (parent, names), seed=seed)[0],
        "rf_normalised": rf_distance(truth, (pc, nc)),
        "rf_normalised_raw": rf_distance(truth, (parent, names)),
    }


def main(argv: list[str] | None = None) -> Path:
    """Run one config; return the output folder."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--out", type=Path, default=metadata.REPO / "bench" / "results")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.data_path, sets=args.sets)
    is_jax = cfg["run"]["backend"] == "jax"
    if is_jax:
        _check_jax_baseline(cfg)  # before any output exists
    meta = metadata.collect(args.config, cfg["run"]["seed"])
    meta["sets"] = list(args.sets)  # provenance: --set overrides are not reflected in config_text/config_sha256
    out = args.out / cfg["run"]["name"] / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    _write_json(out / "metadata.json", meta)  # first, so a capped run keeps it
    if is_jax:
        from .gate import load_arm  # lazy: bench.gate imports this module

        arm = load_arm(Path(cfg["run"]["upstream_dir"]))  # A23: the baseline's own leaf arrays, never regenerated
        n_cells = len(arm["cells"])
        # the truth belongs to the baseline (its config), whose [data]/seed equal this run's (checked above)
        truth = load_data(arm["config"])[1] if arm["config"] is not None and cfg["data"]["kind"] == "simulate" else None
        if truth is not None:
            leaves = {n for n, k in zip(truth[1], children_lists(truth[0]), strict=True) if not k}
            if leaves != set(arm["cells"]):
                raise ValueError("the truth tree's leaves differ from the baseline's cells")
    else:
        adata, truth = load_data(cfg)
        n_cells = adata.n_obs
    runs = []
    for i in range(1 + cfg["run"]["n_warm"]):
        if is_jax:
            r = run_jax(arm, cfg.get("jax", {}))
        else:
            r = run_upstream(adata, cfg["upstream"], out, tag=f"run{i}", stop_after=cfg["run"].get("stop_after"))
        tree = r.pop("tree")
        r["kind"] = "cold" if i == 0 else "warm"  # upstream compiles nothing; "cold" = first run in the process
        if truth is not None:
            if is_jax:
                r["truth"] = _truth_metrics(truth, tree, cfg["run"]["seed"])
            else:
                acc, n_q = quartet_accuracy(truth, tree, seed=cfg["run"]["seed"])
                r["truth"] = {"quartet_accuracy": acc, "n_quartets": n_q, "rf_normalised": rf_distance(truth, tree)}
        runs.append(r)
        results = {"config": cfg, "backend": cfg["run"]["backend"], "n_cells": n_cells, "runs": runs}
        _write_json(out / "results.json", results)  # after every run
    return out


if __name__ == "__main__":
    print(main())
