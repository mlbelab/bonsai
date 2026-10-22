"""M2 quality gate (A1, A7 after redo starry; A15, A16).

``python -m bench.gate --jax JAX_RUN_DIR --upstream (UPSTREAM_RUN_DIR | UPSTREAM_RESULTS_FOLDER) --out FILE
[--device IDX]``

Both arms' redo-starry trees are re-timed by the same float64 re-timer (``bench.retime``, in a
``JAX_ENABLE_X64=1`` subprocess) on the same leaf arrays, then compared by A1 (re-timed L per
entry) and, with a simulated truth, A7 (A15 quartets). Arms whose ``leaf_sha256`` or
``genes_used`` differ are refused. Any INVALID re-time makes the verdict ``"INVALID"``.

A23: the gate never regenerates leaf data (the simulator's leaf arrays are not bit-identical across
CPU SIMD targets). Both arms are scored on the upstream arm's own stored arrays (``load_leaf_arrays``,
``leaf_arrays.npz`` for a bench run dir, or upstream's still-on-disk processed folder for a raw
results folder); ``leaf_sha256`` stays a strict integrity check.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from bonsai._engine.opt_times import T_SNAP
from bonsai._ref.newick import parse_newick
from bonsai._ref.tree import tree_loglik

from . import metadata
from .metrics import contract_zero_edges, quartet_accuracy, rf_distance
from .retime import RetimeResult, load
from .run import _GREEDY_FOLDER, _leaf_arrays, _write_json, leaf_sha256, load_data

A1_MIN_PER_ENTRY = -1e-6  # A1: (L_jax − L_up)/(N·G) ≥ −1e-6
A7_MIN_DIFF = -0.01  # A7: quartet accuracy at most 1 percentage point below upstream


def gate_verdict(L_jax, L_up, n_cells, n_genes, q_jax=None, q_up=None) -> dict:
    """A1 on the re-timed L per entry; A7 on quartet accuracy (None without quartets)."""
    a1 = (L_jax - L_up) / (n_cells * n_genes)
    a7 = None if q_jax is None or q_up is None else q_jax - q_up
    return {
        "a1_per_entry": a1,
        "a1_pass": bool(a1 >= A1_MIN_PER_ENTRY),
        "a7_diff": a7,
        "a7_pass": None if a7 is None else bool(a7 >= A7_MIN_DIFF),
    }


def retime_subprocess(parent, t, x, var, device: str | None = None) -> RetimeResult:
    """``bench.retime`` in a ``JAX_ENABLE_X64=1`` subprocess (on GPU ``device`` if given)."""
    env = dict(os.environ, JAX_ENABLE_X64="1")
    if device is not None:
        env.update(CUDA_VISIBLE_DEVICES=str(device), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    with tempfile.TemporaryDirectory(prefix="bonsai_retime_") as d:
        inp, out = Path(d) / "in.npz", Path(d) / "out.npz"
        np.savez(inp, parent=parent, t=t, x=x, var=var)
        subprocess.run(
            [sys.executable, "-m", "bench.retime", str(inp), str(out)], cwd=metadata.REPO, env=env, check=True
        )
        return load(out)


def _unique(paths, what: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {what}, found {len(paths)}")
    return paths[0]


def load_leaf_arrays(results_dir: Path) -> tuple[np.ndarray, np.ndarray, list, list]:
    """(x, var, cells, genes) from a bench run dir's stored leaf arrays.

    A23's ``<tag>_leaf_arrays.npz``, recorded as ``results.json``'s ``runs[0]["leaf_arrays"]``. Raises ``RuntimeError`` if the run predates A23: the workdir its leaf arrays came from is already
    gone, so there is nothing left to score the gate on, and it must not be silently regenerated.
    """
    results_dir = Path(results_dir)
    res = json.loads((results_dir / "results.json").read_text())
    name = res["runs"][0].get("leaf_arrays")
    if name is None:
        raise RuntimeError(f"{results_dir}: baseline predates A23; rerun it")
    with np.load(results_dir / name) as z:
        return z["x"], z["var"], list(z["cells"]), list(z["genes"])


def load_arm(path: Path, load_arrays: bool = True) -> dict:
    """{"stage_newick", "leaf_sha256", "genes_used", "config"} of a bench run dir or upstream results folder, plus "x"/"var"/"cells"/"genes" when ``load_arrays``.

    A bench run dir's ``x``/``var``/``cells``/``genes`` come from ``load_leaf_arrays`` (A23) and are
    only read when ``load_arrays`` (default ``True``): the jax arm's own arrays are never used for
    scoring (``gate()`` scores both arms on the upstream arm's arrays, loaded with ``load_arrays=True``),
    so a jax bench run dir without a stored ``leaf_arrays.npz`` must not raise "baseline predates A23" —
    only its recorded ``leaf_sha256``/``genes_used`` are checked against upstream's. A raw upstream
    results folder (no ``results.json``) always has its processed arrays on disk and returns them
    regardless of ``load_arrays``.
    """
    path = Path(path)
    if (path / "results.json").exists():
        res = json.loads((path / "results.json").read_text())
        r = res["runs"][0]
        out = {k: r[k] for k in ("stage_newick", "leaf_sha256", "genes_used")} | {"config": res["config"]}
        if load_arrays:
            x, var, cells, genes = load_leaf_arrays(path)
            out |= {"x": x, "var": var, "cells": cells, "genes": genes}
        return out
    redo = _unique(path.glob("intermediate_bonsai*_redoStarry/tree.nwk"), "redo-starry tree")
    greedy = _unique(
        (p for p in path.iterdir() if p.is_dir() and _GREEDY_FOLDER.fullmatch(p.name)), "greedy-stage folder"
    )
    proc = _unique(path.glob("zscorefiltered_*_and_processed"), "processed-data folder")
    meta = json.loads((proc / "metadata.json").read_text())
    x, var = np.load(proc / "delta.npy").T, np.load(proc / "delta_vars.npy").T  # cells × genes
    return {
        "stage_newick": {"greedy": (greedy / "tree.nwk").read_text().strip(), "redo_starry": redo.read_text().strip()},
        "leaf_sha256": leaf_sha256(x, var, meta["cellIds"]),
        "genes_used": list(meta["geneIds"]),
        "config": None,
        "x": x,
        "var": var,
        "cells": meta["cellIds"],
        "genes": list(meta["geneIds"]),
    }


def _arm_report(arm: dict, cells, x, var, truth, seed: int, device) -> tuple[dict, RetimeResult]:
    stages = {k: parse_newick(v) for k, v in arm["stage_newick"].items()}
    L_raw = {k: tree_loglik(p, t, *_leaf_arrays(p, nm, cells, x, var)) for k, (p, t, nm) in stages.items()}
    p, t, nm = stages["redo_starry"]
    X, V = _leaf_arrays(p, nm, cells, x, var)
    rt = retime_subprocess(p, t, X, V, device)
    out = {"L_raw_stage": L_raw, "retime": {k: v for k, v in rt.__dict__.items() if k != "t"}}
    if truth is not None:
        pc, _, nc = contract_zero_edges(p, rt.t, nm, snap=T_SNAP)
        out["quartet_accuracy"], out["n_quartets"] = quartet_accuracy(truth, (pc, nc), seed=seed)
        out["quartet_accuracy_raw"] = quartet_accuracy(truth, (p, nm), seed=seed)[0]
        out["rf_normalised"] = rf_distance(truth, (pc, nc))
        out["rf_normalised_raw"] = rf_distance(truth, (p, nm))
    return out, rt


def gate(jax_dir: Path, upstream: Path, device: str | None = None) -> dict:
    """Gate report of one case; raises ``ValueError`` if the arms' leaf arrays differ.

    A23: both arms are scored on the upstream arm's own stored leaf arrays, never on arrays
    regenerated here (the simulator's leaf arrays are not bit-identical across CPU SIMD targets).
    """
    arms = {"jax": load_arm(jax_dir, load_arrays=False), "upstream": load_arm(upstream)}
    for key in ("leaf_sha256", "genes_used"):
        if arms["jax"][key] != arms["upstream"][key]:
            raise ValueError(f"arms differ in {key}: refusing to gate")
    cfg = arms["jax"]["config"]
    if cfg is None:
        raise ValueError(f"jax arm must be a bench run dir (with results.json), got {jax_dir}")
    _, truth = load_data(cfg)
    cells, genes = arms["upstream"]["cells"], arms["upstream"]["genes"]
    x, var = arms["upstream"]["x"], arms["upstream"]["var"]
    if leaf_sha256(x, var, cells) != arms["upstream"]["leaf_sha256"]:
        raise ValueError("upstream's stored leaf arrays do not match its recorded leaf_sha256")
    seed = cfg["run"]["seed"]
    both = {k: _arm_report(a, cells, x, var, truth, seed, device) for k, a in arms.items()}
    rep, rts = {k: r for k, (r, _) in both.items()}, {k: rt for k, (_, rt) in both.items()}
    q = {k: rep[k].get("quartet_accuracy") for k in rep}
    v = gate_verdict(rts["jax"].L, rts["upstream"].L, len(cells), len(genes), q["jax"], q["upstream"])
    invalid = [k for k, r in rts.items() if r.status != "converged"]
    v["verdict"] = "INVALID" if invalid else "PASS" if v["a1_pass"] and v["a7_pass"] is not False else "FAIL"
    return {
        "jax": str(jax_dir),
        "upstream": str(upstream),
        "n_cells": len(cells),
        "n_genes": len(genes),
        "invalid_arms": invalid,
        "arms": rep,
    } | v


def main(argv: list[str] | None = None) -> Path:
    """Gate one case; write the report to ``--out`` and return its path."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jax", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    _write_json(args.out, gate(args.jax, args.upstream, args.device))
    return args.out


if __name__ == "__main__":
    print(main())
