import json
from pathlib import Path

import pytest
from bench import sweep
from bench.sweep import CASES

SETTINGS = [("sequential", 256, 1), ("joint", 256, 1), ("sequential", 64, 1), ("sequential", 1024, 1)] + [
    ("sequential", 256, m) for m in (4, 16)
]


class _Rows(list):
    timings: dict


def decide(rows):
    return sweep.decide(rows, rows.timings)


def _table(L=None, greedy=None, bad=None, tbad=None):
    """Synthetic A17 table for the 9 cases and the six default settings. ``L[(mode, M, m)]`` is L/(N·G)
    (default -1), ``greedy[M]`` the warm greedy time of the interleaved 10k timing runs (default 10 s, both modes);
    ``bad[(case, mode, M, m)]`` overrides a row, ``tbad[(case, mode, M)]`` a timing (None removes it)."""
    L, greedy, rows = L or {}, greedy or {}, _Rows()
    for c in CASES:
        for md, M, m in SETTINGS:
            r = {"case": c, "mode": md, "M": M, "m": m, "status": "ok", "a1_pass": True, "a7_pass": True if c.startswith("sim") else None,
                 "L_pe": L.get((md, M, m), -1.0)}  # fmt: skip
            rows.append(r | (bad or {}).get((c, md, M, m), {}))
    rows.timings = {
        (c, md, M): greedy.get(M, 10.0) for c in CASES if c.startswith("sim10k") for md in ("sequential", "joint") for M in (64, 256, 1024)
    }  # fmt: skip
    for k, v in (tbad or {}).items():
        rows.timings.pop(k) if v is None else rows.timings.__setitem__(k, v)
    return rows


def _drop(rows, **kw):
    out = _Rows(r for r in rows if not all(r[k] == v for k, v in kw.items()))
    out.timings = rows.timings
    return out


def test_r1_higher_mean_wins_but_ties_go_to_sequential():
    assert decide(_table(L={("joint", 256, 1): -1.0 + 1e-3}))["R1"] == "joint"
    assert decide(_table(L={("joint", 256, 1): -1.0 + 1e-6}))["R1"] == "joint"
    assert decide(_table(L={("joint", 256, 1): -1.0 + 5e-8}))["R1"] == "sequential"  # within 1e-7
    assert decide(_table(L={("sequential", 256, 1): -1.0 + 1e-3}))["R1"] == "sequential"


@pytest.mark.parametrize(
    "bad",
    [
        {"a1_pass": False},  # A1 on any of the 9 cases, golden and real included
        {"status": "jax_invalid"},
        {"status": "missing"},
    ],
)
@pytest.mark.parametrize("case", ["golden", "eec_vasa_sub", "sim10k_s2"])
def test_r1_ineligible_mode_loses_to_the_other(bad, case):
    rows = _table(
        L={("sequential", 256, 1): 5.0}, bad={(case, "sequential", 256, 1): bad}
    )  # sequential better, but ineligible
    d = decide(rows)
    assert d["R1"] == "joint" and any("sequential" in n for n in d["notes"])


def test_a7_is_required_only_on_simulated_cases():
    d = decide(_table(bad={("sim1k_s0", "joint", 256, 1): {"a7_pass": False}}))
    assert d["R1"] == "sequential" and any("sim1k_s0: A7 fail" in n for n in d["notes"])
    assert (
        decide(_table(bad={("sim1k_s0", "joint", 256, 1): {"a7_pass": None}}))["R1"] == "sequential"
    )  # absent A7 is not a pass
    assert (
        decide(_table(L={("joint", 256, 1): 5.0}, bad={("golden", "joint", 256, 1): {"a7_pass": None}}))["R1"]
        == "joint"
    )


def test_r1_no_eligible_mode_and_upstream_invalid_block_everything():
    rows = _table(bad={("golden", md, 256, 1): {"a1_pass": False} for md in ("sequential", "joint")})
    d = decide(rows)
    assert d["R1"] is None and d["R2"] is None and "controller decides" in " ".join(d["notes"])
    d = decide(
        _table(bad={("sim1k_s1", "sequential", 64, 1): {"status": "upstream_invalid"}})
    )  # even in an unrelated setting
    assert d["blocked"] == ["sim1k_s1"] and d["R1"] is None


def test_missing_row_is_ineligible():
    rows = _drop(_table(), case="sim10k_s1", mode="joint", M=256, m=1)
    assert decide(rows)["R1"] == "sequential"


def test_r2_prefers_256_within_ten_percent_else_the_fastest():
    assert decide(_table(greedy={64: 9.1, 256: 10.0, 1024: 12.0}))["R2"] == 256  # 10.0 <= 1.1 * 9.1 = 10.01
    assert decide(_table(greedy={64: 9.0, 256: 10.0, 1024: 12.0}))["R2"] == 64  # 10.0 > 9.9
    assert decide(_table(greedy={64: 9.0, 256: 12.0, 1024: 20.0}))["R2"] == 64
    assert decide(_table(greedy={64: 20.0, 256: 12.0, 1024: 9.0}))["R2"] == 1024
    assert decide(_table(greedy={64: 10.0, 256: 12.0, 1024: 10.0}))["R2"] == 64  # exact tie -> smaller M


def test_r2_uses_the_median_over_the_three_10k_seeds_and_only_eligible_M():
    rows = _table(greedy={64: 5.0, 256: 10.0, 1024: 10.0}, tbad={("sim10k_s0", "sequential", 64): 50.0})
    assert decide(rows)["R2"] == 64  # median(50, 5, 5) = 5
    rows = _table(greedy={64: 5.0, 256: 10.0, 1024: 10.0}, bad={("sim1k_s2", "sequential", 64, 1): {"a1_pass": False}})
    d = decide(rows)
    assert d["R2"] == 256 and 64 not in d["R2_medians"]  # ineligible fast M is excluded
    rows = _table(greedy={64: 5.0}, tbad={("sim10k_s2", "sequential", 64): None})
    d = decide(rows)
    assert d["R2"] == 256 and 64 not in d["R2_medians"] and any("time is missing" in n for n in d["notes"])


def test_r2_and_r3_follow_the_r1_mode():
    rows = _table(
        L={("joint", 256, 1): 1.0}, greedy={64: 1.0}
    )  # joint wins R1; its R2 timings are used, not sequential's
    rows += [r | {"mode": "joint"} for r in rows if r["M"] != 256 or r["m"] != 1]
    rows = _drop(rows, mode="sequential", M=64)
    for c in ("sim10k_s0", "sim10k_s1", "sim10k_s2"):
        rows.timings[(c, "sequential", 64)] = 100.0
        rows.timings[(c, "joint", 64)] = 1.0
    d = decide(rows)
    assert d["R1"] == "joint" and d["R2"] == 64
    assert d["R3"][4]["why"] == [f"{c}: MISSING" for c in CASES]  # joint M=64 m=4 is not in this table
    rows = _table(L={("joint", 256, 1): 1.0})
    rows += [r | {"mode": "joint"} for r in rows if r["M"] != 256 or r["m"] != 1]
    assert decide(rows)["R3"] == {4: {"pass": True, "why": []}, 16: {"pass": True, "why": []}}


def test_r3_reports_each_m_separately_at_r2s_M():
    rows = _table(greedy={64: 5.0, 256: 10.0, 1024: 10.0})  # R2 = 64; R3 m values are judged at M = 64 (absent here)
    d = decide(rows)
    assert d["R2"] == 64 and not d["R3"][4]["pass"] and "MISSING" in d["R3"][4]["why"][0]
    d = decide(_table(bad={("sim1k_s0", "sequential", 256, 16): {"a1_pass": False}}))
    assert d["R3"][4]["pass"] and not d["R3"][16]["pass"] and d["R3"][16]["why"] == ["sim1k_s0: A1 fail"]


def _bases(tmp_path):
    base = tmp_path / "b"
    for size in ("1k", "10k"):
        for s in range(3):
            (base / f"upstream_redo_sim_{size}_s{s}" / "20261022T000000Z").mkdir(parents=True)
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "results.json").write_text("{}")
    return base, gold


def _sets(ln):
    return dict(s.split("=", 1) for s in ln["sets"].split(";"))


def test_manifest_wave_1(tmp_path):
    base, gold = _bases(tmp_path)
    lines = sweep.build_manifest(tmp_path / "root", base, gold, 1)
    assert [ln["kind"] for ln in lines] == ["upstream"] * 9 + ["jax"] * 18 and [ln["id"] for ln in lines] == list(
        range(27)
    )
    assert {(ln["mode"], ln["rescore_top"], ln["merges"]) for ln in lines[9:]} == {
        ("sequential", 256, 1),
        ("joint", 256, 1),
    }
    ln = next(ln for ln in lines if ln["case"] == "sim10k_s2" and ln.get("mode") == "joint")
    sets = _sets(ln)
    assert sets["run.seed"] == "2" and sets["jax.mode"] == "joint" and sets["jax.rescore_top"] == "256"
    assert "jax.false_stop_check" not in sets  # R1 arms keep the false-stop counters
    assert sets["run.upstream_dir"] == ln["upstream"] and ln["upstream"].endswith(
        "upstream_redo_sim_10k_s2/20261022T000000Z"
    )
    gold_ln = next(ln for ln in lines if ln["case"] == "golden" and ln["kind"] == "jax")
    assert "run.seed" not in gold_ln["sets"] and gold_ln["config"].endswith("jax_golden.toml")
    real = next(ln for ln in lines if ln["case"] == "eec_vasa_sub" and ln["kind"] == "jax")
    assert real["data_path"].endswith("data/eec_vasa_sub") and real["upstream"].endswith(
        "eec_vasa_sub/bonsai_workdir/results"
    )
    sweep.write_manifest(lines, tmp_path / "m.tsv")
    assert sweep.read_manifest(tmp_path / "m.tsv")[10]["sets"] == lines[10]["sets"]


def test_manifest_wave_2_and_stable_paths(tmp_path):
    base, gold = _bases(tmp_path)
    w1 = sweep.build_manifest(tmp_path / "root", base, gold, 1)
    w2 = sweep.build_manifest(tmp_path / "root", base, gold, 2, "joint")
    kinds = [ln["kind"] for ln in w2]
    assert kinds == ["jax"] * 30 + ["timing"] * 3 and [ln["id"] for ln in w2] == list(range(33))
    jax = [ln for ln in w2 if ln["kind"] == "jax"]
    assert {ln["mode"] for ln in w2} == {"joint"}
    assert sorted({(ln["rescore_top"], ln["merges"]) for ln in jax}) == [(64, 1), (256, 4), (256, 16), (1024, 1)]
    assert {ln["case"] for ln in jax if ln["rescore_top"] in (64, 1024)}.isdisjoint(
        {"sim10k_s0", "sim10k_s1", "sim10k_s2"}
    )
    paths = [ln["gate_out"] for ln in w1[9:] + w2]
    assert len(set(paths)) == len(paths)  # wave 2 never reuses a wave 1 path
    t = next(ln for ln in w2 if ln["kind"] == "timing" and ln["case"] == "sim10k_s1")
    sets = _sets(t)
    assert t["rescore_top"] == "64,256,1024" and sets["jax.false_stop_check"] == "false" and sets["run.seed"] == "1"
    assert "run.name" not in sets and "jax.rescore_top" not in sets and sets["jax.mode"] == "joint"
    assert all("false_stop_check" not in ln["sets"] for ln in jax)
    # the same case/setting gets the same path whichever wave builds it
    again = sweep.build_manifest(tmp_path / "root", base, gold, 2, "joint", r3_M=256)
    assert [ln["gate_out"] for ln in again] == [ln["gate_out"] for ln in w2]


def test_resolve_upstream_needs_exactly_one_dir(tmp_path, monkeypatch):
    (tmp_path / "upstream_redo_sim_1k_s0").mkdir()
    with pytest.raises(RuntimeError, match="exactly one"):
        sweep.resolve_upstream("sim1k_s0", tmp_path, None)
    (tmp_path / "upstream_redo_sim_1k_s0" / "a").mkdir()
    (tmp_path / "upstream_redo_sim_1k_s0" / "b").mkdir()
    with pytest.raises(RuntimeError, match="exactly one"):
        sweep.resolve_upstream("sim1k_s0", tmp_path, None)
    (tmp_path / "rel").mkdir()
    (tmp_path / "rel" / "results.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    for bad in (None, "", tmp_path, "rel"):  # unset, empty, no results.json, not absolute
        with pytest.raises(RuntimeError, match="golden-upstream"):
            sweep.resolve_upstream("golden", tmp_path, bad)


def _arm(hit):
    return {
        "L_raw_stage": {"greedy": -9.0, "redo_starry": -8.0},
        "retime": {"L": -100.0, "status": "converged", "gap": 1e-5, "sum_abs_pg": 1e-9, "restarts": 1},
        "retime_cache_hit": hit, "quartet_accuracy": 0.9, "quartet_accuracy_raw": 0.95, "rf_normalised": 0.1, "rf_normalised_raw": 0.05,
    }  # fmt: skip


def _gate(tmp_path, name, greedy_warm):
    run = tmp_path / f"run_{name}"
    run.mkdir()
    stats = {"n_stop_events": 3, "n_false_stop_events": 1, "n_band_rows": 5, "stop_check_s": 2.0}
    runs = [
        {"kind": k, "stages_s": [["greedy", g]], "stats": stats} for k, g in (("cold", 99.0), ("warm", greedy_warm))
    ]
    (run / "results.json").write_text(json.dumps({"runs": runs}))
    gate = {"jax": str(run), "n_cells": 10, "n_genes": 10, "invalid_arms": [], "verdict": "PASS", "a1_per_entry": 0.0, "a1_pass": True,
            "a7_diff": None, "a7_pass": None, "arms": {"jax": _arm(False), "upstream": _arm(True)}}  # fmt: skip
    gp = tmp_path / f"gate_{name}.json"
    gp.write_text(json.dumps(gate))
    return run, gp


def test_aggregate_reads_both_waves_and_the_timing_runs(tmp_path):
    w1, w2 = [], []
    _, gp = _gate(tmp_path, "r1", 30.0)
    w1.append(
        {
            "id": 0,
            "kind": "jax",
            "case": "golden",
            "setting": "sequential_M256_m1",
            "mode": "sequential",
            "rescore_top": 256,
            "merges": 1,
            "gate_out": str(gp),
        }
    )
    w1.append(
        {
            "id": 1,
            "kind": "jax",
            "case": "golden",
            "setting": "joint_M256_m1",
            "mode": "joint",
            "rescore_top": 256,
            "merges": 1,
            "gate_out": str(tmp_path / "none.json"),
        }
    )
    w1.append({"id": 2, "kind": "upstream", "case": "golden", "gate_out": str(tmp_path / "u.json")})
    base = tmp_path / "gates" / "sim10k_s0"
    runs = {}
    for M, g in ((64, 7.0), (256, 8.0), (1024, 9.0)):
        runs[str(M)], gate_p = _gate(tmp_path, f"t{M}", g)
        if M != 256:
            (base / f"sequential_M{M}_m1").mkdir(parents=True)
            (base / f"sequential_M{M}_m1" / "gate.json").write_text(gate_p.read_text())
    (tmp_path / "timing.json").write_text(json.dumps({k: str(v) for k, v in runs.items()}))
    w2.append(
        {
            "id": 0,
            "kind": "timing",
            "case": "sim10k_s0",
            "mode": "sequential",
            "gate_base": str(base),
            "gate_out": str(tmp_path / "timing.json"),
        }
    )
    sweep.write_manifest(w1, tmp_path / "w1.tsv")
    sweep.write_manifest(w2, tmp_path / "w2.tsv")
    dec = sweep.aggregate([tmp_path / "w1.tsv", tmp_path / "w2.tsv"], tmp_path / "out.md")
    md = (tmp_path / "out.md").read_text()
    assert "- golden joint_M256_m1: MISSING" in md and "- upstream-only golden: MISSING" in md
    assert (
        "| sim10k_s0 | sequential_M64_m1 |" in md and "| sim10k_s0 | sequential_M1024_m1 |" in md
    )  # quality gates of the timing line
    assert "| sim10k_s0 | sequential | 64 | 7.0 |" in md and "| 30.0 |" in md  # gross times, no net correction
    assert (
        "sequential_M256_m1 |" not in md.split("## INVALID")[0].split("sim10k_s0")[-1]
    )  # no gate for the timing run of M=256
    assert "- sim10k_s0 sequential_M256_m1" not in md  # the M=256 timing run has no gate of its own
    assert dec["R1"] is None and dec["blocked"] == []  # a partial table decides nothing


def test_manifest_cli_requires_golden_and_uses_an_absolute_root(tmp_path, monkeypatch):
    base, gold = _bases(tmp_path)
    with pytest.raises(RuntimeError, match="golden-upstream"):
        sweep.main(["manifest", "--wave", "1", "--out", str(tmp_path / "w.tsv"), "--baselines", str(base)])
    assert not (tmp_path / "w.tsv").exists()
    monkeypatch.chdir(tmp_path)
    args = ["manifest", "--wave", "1", "--out", "w.tsv", "--baselines", str(base), "--golden-upstream", str(gold)]
    sweep.main([*args, "--root", "rel_root"])
    rows = sweep.read_manifest(tmp_path / "w.tsv")
    assert all(Path(r["gate_out"]).is_absolute() and r["gate_out"].startswith(str(tmp_path)) for r in rows)
    assert all(r["cache"].startswith(str(tmp_path)) for r in rows)


@pytest.mark.parametrize("invalid", [["upstream"], ["jax", "upstream"]])
def test_upstream_invalid_arm_blocks_decisions_through_aggregate(tmp_path, invalid):
    """load_row maps invalid_arms containing 'upstream' (alone or with 'jax') to upstream_invalid, which blocks."""
    _, gp = _gate(tmp_path, "x", 1.0)
    g = json.loads(gp.read_text()) | {"invalid_arms": invalid, "verdict": "INVALID"}
    gp.write_text(json.dumps(g))
    line = {"id": 0, "kind": "jax", "case": "golden", "setting": "sequential_M256_m1", "mode": "sequential", "rescore_top": 256, "merges": 1, "gate_out": str(gp)}  # fmt: skip
    assert sweep.load_row(line)["status"] == "upstream_invalid"
    sweep.write_manifest([line], tmp_path / "m.tsv")
    assert sweep.aggregate([tmp_path / "m.tsv"], tmp_path / "o.md")["blocked"] == ["golden"]
    g["invalid_arms"] = ["jax"]
    gp.write_text(json.dumps(g))
    assert sweep.load_row(line)["status"] == "jax_invalid"
