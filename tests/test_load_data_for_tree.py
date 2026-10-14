"""`load_data_for_tree` wraps its data-loading logic in a `try`/`except Exception`, which used to
let upstream's `exit()` (`SystemExit`, a `BaseException`) propagate unimpeded. Now that the
"no data was found" exit is a `raise BonsaiError(...)` (a `ValueError`/`Exception`), the broad
`except Exception` would silently swallow it unless `except BonsaiError: raise` precedes it.
"""

import json
import types

import pytest

from bonsai._core.dataprocessing import load_data_for_tree
from bonsai._core.helpers import BonsaiError


def test_load_data_for_tree_raises_when_no_data_found(tmp_path):
    orig_folder = tmp_path / "orig"
    orig_folder.mkdir()
    tree_folder = tmp_path / "tree"
    tree_folder.mkdir()
    metadata = {
        "pathToOrigData": None,
        "dataset": "test",
        "nCells": 1,
        "nGenes": 0,
        "cellIds": ["Cell_0"],
        "geneIds": [],
        "loglik": None,
        "loglikVarCorr": None,
        "geneVariances": None,
        "geneDiffusionScaling": None,
        "processedDatafolder": str(orig_folder),  # empty: no delta.npy, no delta_vmax.txt
        "results_folder": str(tmp_path),
    }
    (tree_folder / "metadata.json").write_text(json.dumps(metadata))
    scData = types.SimpleNamespace(metadata=None, tree=types.SimpleNamespace(nNodes=0), originalData=None)

    with pytest.raises(BonsaiError, match="No data was found"):
        load_data_for_tree(scData, str(tree_folder), {}, get_all_data=True, load_data=True)
