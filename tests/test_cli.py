import subprocess
import sys

import pytest

from .golden import FIXTURE, assert_trees_match
from .test_golden_core import CONFIG


@pytest.mark.slow
def test_cli_run_matches_golden(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG.format(data=FIXTURE, results=tmp_path / "results"))
    subprocess.run([sys.executable, "-m", "bonsai.cli", "run", "--config_filepath", str(cfg)], check=True)
    (final,) = (tmp_path / "results").glob("final_bonsai*")
    assert_trees_match(final)


def test_cli_config_writes_yaml(tmp_path):
    out = tmp_path / "c.yaml"
    subprocess.run([sys.executable, "-m", "bonsai.cli", "config", "--new_yaml_path", str(out)], check=True)
    assert "zscore_cutoff" in out.read_text()


def test_cli_run_rejects_unknown_option(tmp_path):
    """`parse_known_args` would otherwise silently drop a typo'd option instead of erroring."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("dataset: x\n")
    result = subprocess.run(
        [sys.executable, "-m", "bonsai.cli", "run", "--config_filepath", str(cfg), "--typo_option", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
