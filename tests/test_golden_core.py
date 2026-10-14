import pytest

from bonsai._core.main import run_from_config

from .golden import FIXTURE, assert_trees_match

# Mirrors tests/data/golden_b16ef6f/config.yaml (dataset and verbose included, so this *is* the
# golden run config, not just an equivalent one). The deprecated skip_* keys and start_time are
# left out: skip_* default to False in Run_Configs, matching the golden config's explicit False
# values, and start_time is a timestamp that Run_Configs stamps itself and plays no role in the
# computation.
CONFIG = """\
dataset: simulated_binary_6_gens_samplingNoise
data_folder: {data}
filenames_data: delta_vmax.txt,d_delta_vmax.txt
results_folder: {results}
verbose: true
input_is_sanity_output: true
zscore_cutoff: 1.0
rescale_by_var: true
nnn_n_randommoves: 10
nnn_n_randomtrees: 2
use_knn: 10
UB_ellipsoid_size: 1.0
pickup_intermediate: false
tmp_folder: ''
"""


@pytest.fixture(scope="session")
def core_result(tmp_path_factory):
    d = tmp_path_factory.mktemp("core")
    cfg = d / "config.yaml"
    cfg.write_text(CONFIG.format(data=FIXTURE, results=d / "results"))
    run_from_config(cfg)
    (final,) = (d / "results").glob("final_bonsai*")
    return final


@pytest.mark.slow
def test_core_matches_golden(core_result):
    assert_trees_match(core_result)


@pytest.mark.slow
def test_second_in_process_run_is_identical(core_result, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG.format(data=FIXTURE, results=tmp_path / "results"))
    run_from_config(cfg)
    (final,) = (tmp_path / "results").glob("final_bonsai*")
    assert (final / "tree.nwk").read_bytes() == (core_result / "tree.nwk").read_bytes()
