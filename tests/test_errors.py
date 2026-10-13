import pytest

from bonsai._core.helpers import BonsaiError, resolve_sanity_filepaths


def test_marg_sanity_output_raises(tmp_path):
    (tmp_path / "sanity_command.txt").write_text(
        "# Sanity version: 2.0.0\n# Method: MARG\nSanity -f count_table.tsv -e 1 -d output_folder -v_m MARG\n"
    )
    for name in ("delta.txt", "d_delta.txt", "variance.txt", "mu.txt"):
        (tmp_path / name).write_text("")

    with pytest.raises(BonsaiError, match="MARG"):
        resolve_sanity_filepaths(str(tmp_path))
