"""M2 gate sweep (A17): manifest of SLURM array lines and the aggregator with the R1 -> R2 -> R3 decisions.

``python -m bench.sweep manifest --wave 1 --out w1.tsv [--root DIR] [--golden-upstream DIR]`` and
``manifest --wave 2 --mode MODE --out w2.tsv`` (MODE = R1's winner; optional ``--r3-M``, default 256) write a header
plus one line per array task. ``bench/slurm/gate.sbatch`` runs line ``SLURM_ARRAY_TASK_ID`` (0-based over the data
lines of the file). Output paths depend only on (case, setting) under ``--root``, so the waves share the cache and
``aggregate`` reads both.

- Wave 1: ``upstream`` lines (9: re-time just the upstream arm into the shared cache) then the R1 ``jax`` lines (18:
  mode sequential/joint at M=256, m=1; ``bench.run`` then ``bench.gate``).
- Wave 2 (needs R1's winner): R3 ``jax`` lines (m in {4, 16} at ``--r3-M``, 18); R2 quality ``jax`` lines for M in
  {64, 1024} on the 6 non-10k cases (12); and one ``timing`` line per 10k seed (3): the jax arm for M in {64, 256, 1024}
  back to back on one GPU with ``false_stop_check=false`` (each in its own run dir), then the quality gates of M=64
  and 1024. M=256's quality gate is R1's. R2 timing comes only from these interleaved runs.

``python -m bench.sweep aggregate --manifest w1.tsv [w2.tsv] --out measurements.md`` reads every ``gate.json`` and
``results.json`` and writes the per-case tables and the A17 decisions (``decide``).
"""

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

from . import metadata

COLUMNS = (
    "id kind case setting mode rescore_top merges seed config data_path upstream run_name run_out gate_out gate_base"
    " cache sets"
).split()
SEEDS = (0, 1, 2)
CASES = ("golden", *(f"sim{k}_s{s}" for k in ("1k", "10k") for s in SEEDS), "eec_vasa_sub", "perturb_ctrl_sub")
REAL = ("eec_vasa_sub", "perturb_ctrl_sub")
BASELINES = metadata.REPO.parent / "bonsai-m2-baselines2" / "bench" / "results"
DATA = metadata.REPO.parent / "data"
DEFAULT_ROOT = metadata.REPO.parent / "m2-gate-sweep"
CONFIG = {"golden": "jax_golden", "1k": "jax_sim_1k", "10k": "jax_sim_10k", "real": "jax_real"}
MIN_MEAN_DIFF = 1e-7  # R1: means within this -> sequential
R2_TOL = 0.10  # R2: M=256 wins if within 10% of the best median
TIMING_M = (64, 256, 1024)


def setting_name(mode: str, M: int, m: int) -> str:
    """Setting label used in run names and output paths."""
    return f"{mode}_M{M}_m{m}"


def resolve_upstream(case: str, baselines: Path, golden: Path | None) -> Path:
    """The case's upstream arm: exactly one timestamp dir per simulated seed, else raise."""
    if case == "golden":
        g = None if golden is None else Path(golden)
        if g is None or not g.is_absolute() or not (g / "results.json").is_file():
            raise RuntimeError(
                f"--golden-upstream must be an existing absolute run dir containing results.json, got {golden!r}"
            )
        return g
    if case in REAL:
        return DATA / case / "bonsai_workdir" / "results"
    size, seed = case[3:].split("_s")
    hits = sorted(glob.glob(str(baselines / f"upstream_redo_sim_{size}_s{seed}" / "*")))
    hits = [h for h in hits if Path(h).is_dir()]
    if len(hits) != 1:
        raise RuntimeError(f"{case}: expected exactly one upstream run dir under {baselines}, found {hits}")
    return Path(hits[0])


def _config(case: str) -> str:
    return CONFIG["golden" if case == "golden" else "real" if case in REAL else case[3:].split("_")[0]]


def _jax_line(c: str, ups: dict, root: Path, mode: str, M: int, m: int, extra=()) -> dict:
    """One ``bench.run`` + ``bench.gate`` line."""
    name = f"{c}_{setting_name(mode, M, m)}"
    sets = [f"run.name={name}", f"run.upstream_dir={ups[c]}", f"jax.mode={mode}", f"jax.rescore_top={M}"]
    sets += [f"jax.merges_per_round={m}", *extra]
    if c.startswith("sim"):
        sets.insert(0, f"run.seed={c.rsplit('_s', 1)[1]}")
    return _common(c, ups, root, mode, M, m) | {
        "kind": "jax", "setting": setting_name(mode, M, m), "run_name": name, "sets": ";".join(sets),
        "gate_out": str(root / "gates" / c / setting_name(mode, M, m) / "gate.json"),
    }  # fmt: skip


def _common(c: str, ups: dict, root: Path, mode: str, M, m: int) -> dict:
    return {
        "case": c, "mode": mode, "rescore_top": M, "merges": m,
        "seed": c.rsplit("_s", 1)[1] if c.startswith("sim") else 0,
        "config": f"bench/configs/{_config(c)}.toml",
        "data_path": str(DATA / c) if c in REAL else "",
        "upstream": str(ups[c]), "run_out": str(root / "runs"), "gate_base": str(root / "gates" / c),
        "cache": str(root / "retime_cache"),
    }  # fmt: skip


def _timing_line(c: str, ups: dict, root: Path, mode: str) -> dict:
    """R2 timing line of one 10k seed: M in TIMING_M back to back (the shell adds run.name and jax.rescore_top)."""
    sets = [f"run.upstream_dir={ups[c]}", f"jax.mode={mode}", "jax.merges_per_round=1", "jax.false_stop_check=false"]
    sets.insert(0, f"run.seed={c.rsplit('_s', 1)[1]}")
    return _common(c, ups, root, mode, ",".join(map(str, TIMING_M)), 1) | {
        "kind": "timing", "setting": f"{mode}_timing", "run_name": f"{c}_{mode}", "sets": ";".join(sets),
        "gate_out": str(root / "timing" / f"{c}_{mode}.json"),
    }  # fmt: skip


def build_manifest(
    root: Path, baselines: Path, golden: Path | None, wave: int = 1, mode: str = "sequential", r3_M: int = 256
) -> list[dict]:
    """Manifest lines of ``wave`` (see the module docstring); ``id`` is the array index within the wave."""
    root = Path(root).resolve()
    ups = {c: resolve_upstream(c, baselines, golden) for c in CASES}
    if wave == 1:
        lines = [
            {"kind": "upstream", "case": c, "setting": "upstream", "upstream": str(ups[c]), "gate_out": str(root / "upstream" / f"{c}.json"), "cache": str(root / "retime_cache")}
            for c in CASES
        ]  # fmt: skip
        lines += [_jax_line(c, ups, root, md, 256, 1) for c in CASES for md in ("sequential", "joint")]
    else:
        big = [c for c in CASES if c.startswith("sim10k")]
        lines = [_jax_line(c, ups, root, mode, r3_M, m) for c in CASES for m in (4, 16)]
        lines += [_jax_line(c, ups, root, mode, M, 1) for c in CASES if c not in big for M in (64, 1024)]
        lines += [_timing_line(c, ups, root, mode) for c in big]
    for i, ln in enumerate(lines):
        ln["id"] = i
    return lines


def write_manifest(lines: list[dict], out: Path) -> None:
    """Write ``lines`` as a header plus tab-separated rows."""
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(lines)


def read_manifest(path: Path) -> list[dict]:
    """Rows of a manifest as dicts of strings."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------------------------- decisions


def _eligible(rows: dict, cases, key) -> tuple[bool, list[str]]:
    """A17 eligibility of one setting: A1 on all cases, A7 on the simulated ones, no jax arm INVALID."""
    why = []
    for c in cases:
        r = rows.get((c, *key))
        if r is None or r["status"] == "missing":
            why.append(f"{c}: MISSING")
        elif r["status"] != "ok":
            why.append(f"{c}: {r['status'].upper()}")
        elif not r["a1_pass"]:
            why.append(f"{c}: A1 fail")
        elif c.startswith("sim") and r["a7_pass"] is not True:
            why.append(f"{c}: A7 fail" if r["a7_pass"] is False else f"{c}: A7 missing")
    return not why, why


def _median_greedy(timings: dict, cases, mode: str, M: int) -> float | None:
    ts = [timings.get((c, mode, M)) for c in cases if c.startswith("sim10k")]
    return None if not ts or any(t is None for t in ts) else statistics.median(ts)


def decide(table: list[dict], timings: dict | None = None, cases=CASES) -> dict:
    """A17 R1 -> R2 -> R3 on ``table`` rows ``{case, mode, M, m, status, a1_pass, a7_pass, L_pe}``.

    ``timings[(case, mode, M)]`` is the warm greedy-stage time of the interleaved check-off R2 runs (10k seeds).

    ``status`` is ``ok`` / ``jax_invalid`` / ``upstream_invalid`` / ``missing``. An INVALID upstream arm
    blocks every decision (A17: resolved before any mode is judged). Exact ties on the R2 median take the
    smaller M. R3 only reports pass/fail per m (m = 1 stays the default).
    """
    rows = {(r["case"], r["mode"], r["M"], r["m"]): r for r in table}
    out = {
        "blocked": sorted({r["case"] for r in table if r["status"] == "upstream_invalid"}),
        "R1": None,
        "R2": None,
        "R3": {},
    }
    out["notes"] = notes = []
    if out["blocked"]:
        notes.append(f"upstream INVALID on {out['blocked']}: resolve before any decision")
        return out
    elig = {}
    for md in ("sequential", "joint"):
        ok, why = _eligible(rows, cases, (md, 256, 1))
        elig[md] = ok
        if not ok:
            notes.append(f"R1 {md} ineligible: {why}")
    if all(elig.values()):
        mean = {md: statistics.fmean(rows[(c, md, 256, 1)]["L_pe"] for c in cases) for md in elig}
        out["R1_means"] = mean
        mode = (
            "sequential"
            if abs(mean["joint"] - mean["sequential"]) <= MIN_MEAN_DIFF or mean["sequential"] > mean["joint"]
            else "joint"
        )
    elif any(elig.values()):
        mode = next(md for md, ok in elig.items() if ok)
    else:
        notes.append("R1: neither mode eligible; no default, the controller decides")
        return out
    out["R1"] = mode
    el2 = {M: _eligible(rows, cases, (mode, M, 1)) for M in (64, 256, 1024)}
    med = {M: _median_greedy(timings or {}, cases, mode, M) for M, (ok, _) in el2.items() if ok}
    for M, (ok, why) in el2.items():
        if not ok:
            notes.append(f"R2 M={M} ineligible: {why}")
        elif med[M] is None:
            notes.append(f"R2 M={M} eligible but a 10k warm greedy time is missing")
    med = {M: t for M, t in med.items() if t is not None}
    out["R2_medians"] = med
    if not med:
        notes.append("R2: no eligible M with timings; no default")
        return out
    best = min(sorted(med), key=lambda M: med[M])
    out["R2"] = 256 if 256 in med and med[256] <= (1 + R2_TOL) * med[best] else best
    for m in (4, 16):
        ok, why = _eligible(rows, cases, (mode, out["R2"], m))
        out["R3"][m] = {"pass": ok, "why": why}
    return out


# ---------------------------------------------------------------------------------------------- aggregation


def _warm_greedy(res: dict | None) -> float | None:
    """Median over the warm runs of the greedy-stage wall time (gross)."""
    ts = [] if res is None else [dict(r["stages_s"])["greedy"] for r in res["runs"] if r["kind"] == "warm"]
    return statistics.median(ts) if ts else None


def _results(run_dir) -> dict | None:
    p = Path(run_dir) / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_row(line: dict) -> dict:
    """One jax manifest line -> a flat row (``status`` missing if there is no gate.json)."""
    row = {
        "case": line["case"],
        "setting": line["setting"],
        "mode": line["mode"],
        "M": int(line["rescore_top"]),
        "m": int(line["merges"]),
    }
    gp = Path(line["gate_out"])
    if not gp.exists():
        return row | {"status": "missing"}
    g = json.loads(gp.read_text())
    inv = g["invalid_arms"]
    res = _results(g["jax"])
    stats = res["runs"][0]["stats"] if res else {}
    arms = {k: g["arms"][k] for k in ("jax", "upstream")}
    return row | {
        "status": "upstream_invalid" if "upstream" in inv else "jax_invalid" if inv else "ok",
        "verdict": g["verdict"],
        "a1": g["a1_per_entry"], "a1_pass": g["a1_pass"], "a7": g["a7_diff"], "a7_pass": g["a7_pass"],
        "L_pe": arms["jax"]["retime"]["L"] / (g["n_cells"] * g["n_genes"]),
        "arms": arms,
        "stats": {k: stats.get(k) for k in ("n_stop_events", "n_false_stop_events", "n_band_rows", "stop_check_s")},
        "greedy_s": _warm_greedy(res),
    }  # fmt: skip


def _f(v, fmt="{:.6g}") -> str:
    return "-" if v is None else fmt.format(v) if isinstance(v, float) else str(v)


def _table(rows: list[list], head: list[str]) -> str:
    return "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(r) + " |" for r in rows)]
    )


def render(rows: list[dict], dec: dict, upstream_rows: list[dict], timings: dict) -> str:
    """The measurements markdown."""
    done = [r for r in rows if r["status"] != "missing"]
    gate = [[r["case"], r["setting"], r["verdict"], _f(r["a1"], "{:.3e}"), _f(r["a7"], "{:.4f}")]
            + [_f(r["arms"][a].get(k), "{:.4f}") for k in ("quartet_accuracy", "quartet_accuracy_raw", "rf_normalised", "rf_normalised_raw") for a in ("jax", "upstream")]
            for r in done]  # fmt: skip
    arms = [[r["case"], r["setting"], a]
            + [_f(r["arms"][a]["L_raw_stage"]["greedy"], "{:.8g}"), _f(r["arms"][a]["L_raw_stage"]["redo_starry"], "{:.8g}"),
               r["arms"][a]["retime"]["status"], _f(r["arms"][a]["retime"]["gap"], "{:.3e}"), _f(r["arms"][a]["retime"]["sum_abs_pg"], "{:.3e}"),
               str(r["arms"][a]["retime"]["cg_iters"]), str(r["arms"][a]["retime"]["restarts"]), _f(r["arms"][a].get("retime_cache_hit"))]
            for r in done for a in ("jax", "upstream")]  # fmt: skip
    stats = [[r["case"], r["setting"], *(_f(r["stats"][k]) for k in ("n_stop_events", "n_false_stop_events", "n_band_rows", "stop_check_s")),
              _f(r["greedy_s"], "{:.1f}")] for r in done]  # fmt: skip
    bad = [
        f"- {r['case']} {r['setting']}: {r['status'].upper()}" + (f" ({r['verdict']})" if r.get("verdict") else "")
        for r in rows
        if r["status"] not in ("ok",)
    ]
    bad += [f"- upstream-only {r['case']}: {r['status'].upper()}" for r in upstream_rows if r["status"] != "converged"]
    parts = [
        "# M2 gate sweep measurements (A17)\n",
        "## Gate (A1 per entry on re-timed L, A7 diff; quartets/RF contracted then raw; jax/upstream pairs)\n",
        _table(
            gate,
            [
                "case",
                "setting",
                "verdict",
                "A1/entry",
                "A7 diff",
                "q jax",
                "q up",
                "q raw jax",
                "q raw up",
                "RF jax",
                "RF up",
                "RF raw jax",
                "RF raw up",
            ],
        ),
        "\n## Arms: raw-stage L, re-timer status\n",
        _table(
            arms,
            [
                "case",
                "setting",
                "arm",
                "L greedy",
                "L redo",
                "retime",
                "gap",
                "sum|pg|",
                "cg_iters",
                "restarts",
                "cache hit",
            ],
        ),
        "\n## False-stop diagnostic and warm greedy-stage time (includes the diagnostic when it ran)\n",
        _table(stats, ["case", "setting", "stops", "false stops", "band rows", "check s", "warm greedy s"]),
        "\n## R2 timing (interleaved check-off runs): warm greedy-stage s\n",
        _table(
            [[c, md, str(M), _f(t, "{:.1f}")] for (c, md, M), t in sorted(timings.items())], ["case", "mode", "M", "s"]
        ),
        "\n## INVALID / missing\n",
        "\n".join(bad) or "none",
        "\n## A17 decisions\n",
        "```json\n" + json.dumps(dec, indent=2, default=str) + "\n```",
    ]
    return "\n".join(parts) + "\n"


def _timing_expansion(ln: dict) -> tuple[list[dict], dict]:
    """A ``timing`` line -> (pseudo jax lines of its M=64/1024 quality gates, {(case, mode, M): warm greedy s})."""
    c, md = ln["case"], ln["mode"]
    pseudo = [
        ln | {"kind": "jax", "setting": setting_name(md, M, 1), "rescore_top": M, "merges": 1,
              "gate_out": str(Path(ln["gate_base"]) / setting_name(md, M, 1) / "gate.json")}
        for M in TIMING_M if M != 256
    ]  # fmt: skip
    p, times = Path(ln["gate_out"]), {}
    if p.exists():
        for M, run_dir in json.loads(p.read_text()).items():
            t = _warm_greedy(_results(run_dir))
            if t is not None:
                times[(c, md, int(M))] = t
    return pseudo, times


def aggregate(manifests: list[Path], out: Path) -> dict:
    """Write the measurements markdown for the wave ``manifests`` to ``out``; return the A17 decisions."""
    lines = [ln for m in manifests for ln in read_manifest(m)]
    jax, timings = [ln for ln in lines if ln["kind"] == "jax"], {}
    for ln in lines:
        if ln["kind"] == "timing":
            pseudo, t = _timing_expansion(ln)
            jax += pseudo
            timings |= t
    rows = [load_row(ln) for ln in jax]
    ups = []
    for ln in lines:
        if ln["kind"] == "upstream":
            p = Path(ln["gate_out"])
            ups.append({"case": ln["case"], "status": json.loads(p.read_text())["status"] if p.exists() else "missing"})
    dec = decide(rows, timings)
    out.write_text(render(rows, dec, ups, timings))
    return dec


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--wave", type=int, choices=(1, 2), required=True)
    m.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="output root (runs, gates, cache, logs); outside the repo"
    )
    m.add_argument("--baselines", type=Path, default=BASELINES)
    m.add_argument("--golden-upstream", type=Path, default=None)
    m.add_argument("--mode", choices=("sequential", "joint"), default=None, help="R1's winner (wave 2 only)")
    m.add_argument("--r3-M", type=int, default=256)
    a = sub.add_parser("aggregate")
    a.add_argument("--manifest", required=True, type=Path, nargs="+")
    a.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "manifest":
        if (args.wave == 2) != (args.mode is not None):
            ap.error("--mode is required for --wave 2 and not allowed for --wave 1")
        lines = build_manifest(args.root, args.baselines, args.golden_upstream, args.wave, args.mode, args.r3_M)
        write_manifest(lines, args.out)
        kinds = {k: [ln["id"] for ln in lines if ln["kind"] == k] for k in ("upstream", "jax", "timing")}
        print(f"{args.out}: {len(lines)} lines; " + ", ".join(f"{k} {v[0]}-{v[-1]}" for k, v in kinds.items() if v))
    else:
        print(json.dumps(aggregate(args.manifest, args.out), default=str))


if __name__ == "__main__":
    main()
