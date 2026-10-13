from bonsai._core import globals as bs_glob


def test_reset_restores_defaults_and_clears_monkey_patched_attrs():
    bs_glob.nCells = 42
    bs_glob.nGenes = 7
    bs_glob.nNodes = 13
    bs_glob.max_node_ind = 99
    bs_glob.mem_friendly = True
    bs_glob.nwk_counter = 1
    bs_glob.nwk_folder = "/tmp/whatever"

    bs_glob.reset()

    assert bs_glob.nCells is None
    assert bs_glob.nGenes is None
    assert bs_glob.nNodes is None
    assert bs_glob.max_node_ind == 0
    assert bs_glob.mem_friendly is False
    assert not hasattr(bs_glob, "nwk_counter")
    assert not hasattr(bs_glob, "nwk_folder")
