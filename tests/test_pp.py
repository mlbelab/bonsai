import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

import bonsai
from bonsai._io_folder import validate_inputs, write_data_folder

from .golden import FIXTURE


def _txt(name):
    return np.loadtxt(FIXTURE / name, delimiter="\t")


def test_read_sanity_legacy_layout():
    a = bonsai.pp.read_sanity(FIXTURE)
    assert a.shape == (64, 12304)
    assert list(a.obs_names) == (FIXTURE / "cellID.txt").read_text().split()
    assert list(a.var_names) == (FIXTURE / "geneID.txt").read_text().split()
    np.testing.assert_array_equal(a.layers["bonsai_mean"], _txt("delta_vmax.txt").T)
    np.testing.assert_array_equal(a.layers["bonsai_sd"], _txt("d_delta_vmax.txt").T)
    np.testing.assert_array_equal(a.var["bonsai_prior_var"], _txt("variance_vmax.txt"))
    np.testing.assert_array_equal(a.var["bonsai_mu"], _txt("mu_vmax.txt"))
    assert a.uns["bonsai_input"]["kind"] == "sanity_posterior"


def _sanity2_folder(tmp_path, method="MAP"):
    rng = np.random.default_rng(0)
    g, c = 5, 4
    (tmp_path / "sanity_command.txt").write_text(f"# Sanity version: 2.0.0\n# Method: {method}\nSanity -f x\n")
    np.savetxt(tmp_path / "delta.txt", rng.normal(size=(g, c)), delimiter="\t")
    np.savetxt(tmp_path / "d_delta.txt", rng.uniform(0.1, 1, size=(g, c)), delimiter="\t")
    np.savetxt(tmp_path / "variance.txt", rng.uniform(1, 2, size=g))
    np.savetxt(tmp_path / "mu.txt", rng.normal(size=g))
    (tmp_path / "cellID.txt").write_text("\n".join(f"c{i}" for i in range(c)) + "\n")
    (tmp_path / "geneID.txt").write_text("\n".join(f"g{i}" for i in range(g)) + "\n")
    return tmp_path


def test_read_sanity_v2_layout(tmp_path):
    a = bonsai.pp.read_sanity(_sanity2_folder(tmp_path))
    assert a.shape == (4, 5)
    assert a.uns["bonsai_input"]["sanity_method"] == "MAP"


def test_read_sanity_marg_refused(tmp_path):
    with pytest.raises(ValueError, match="MARG"):
        bonsai.pp.read_sanity(_sanity2_folder(tmp_path, "MARG"))


def test_write_then_read_is_bitwise(tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)
    write_data_folder(a, tmp_path / "data")
    b = bonsai.pp.read_sanity(tmp_path / "data")
    for k in ("bonsai_mean", "bonsai_sd"):
        np.testing.assert_array_equal(a.layers[k], b.layers[k])
    for k in ("bonsai_prior_var", "bonsai_mu"):
        np.testing.assert_array_equal(a.var[k].to_numpy(), b.var[k].to_numpy())
    assert list(a.obs_names) == list(b.obs_names) and list(a.var_names) == list(b.var_names)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_bad_sd_rejected(bad):
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    a.layers["bonsai_sd"][0, 0] = bad
    with pytest.raises(ValueError, match="bonsai_sd"):
        validate_inputs(a)


def test_zero_sd_accepted():
    """Upstream clamps var = max(sd**2, 1e-12), so sd == 0 passes through (docs/deviations.md)."""
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    a.layers["bonsai_sd"][0, 0] = 0.0
    validate_inputs(a)  # must not raise


def test_nonfinite_mean_rejected():
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    a.layers["bonsai_mean"][1, 2] = np.nan
    with pytest.raises(ValueError, match="bonsai_mean"):
        validate_inputs(a)


def test_sparse_layers_densified():
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    dense = validate_inputs(a)
    a.layers["bonsai_mean"] = sp.csr_matrix(a.layers["bonsai_mean"])
    got = validate_inputs(a)
    np.testing.assert_array_equal(got[0], dense[0])


def test_from_arrays_marks_likelihood_input():
    a = AnnData(np.ones((3, 2)))
    a.layers["m"] = np.zeros((3, 2))
    bonsai.pp.from_arrays(a, "m")
    assert a.uns["bonsai_input"] == {"kind": "likelihood", "mean_layer": "m", "sd_layer": None}


def test_write_data_folder_sanity_config_and_files(tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    out = write_data_folder(a, tmp_path / "data")
    assert out == {
        "data_folder": str(tmp_path / "data"),
        "filenames_data": "delta.txt,d_delta.txt",
        "input_is_sanity_output": True,
    }
    for name in ("delta.txt", "d_delta.txt", "variance.txt", "mu.txt", "sanity_command.txt"):
        assert (tmp_path / "data" / name).exists()


def test_write_data_folder_likelihood_with_sd_config_and_files(tmp_path):
    a = AnnData(np.ones((3, 2)))
    a.layers["m"] = np.full((3, 2), 2.0)
    a.layers["s"] = np.ones((3, 2))
    bonsai.pp.from_arrays(a, "m", "s")
    out = write_data_folder(a, tmp_path / "data")
    assert out == {
        "data_folder": str(tmp_path / "data"),
        "filenames_data": "features.txt,standard_deviations.txt",
        "input_is_sanity_output": False,
    }
    for name in ("features.txt", "standard_deviations.txt"):
        assert (tmp_path / "data" / name).exists()


def test_write_data_folder_likelihood_without_sd_config_and_files(tmp_path):
    a = AnnData(np.ones((3, 2)))
    a.layers["m"] = np.full((3, 2), 2.0)
    bonsai.pp.from_arrays(a, "m")
    out = write_data_folder(a, tmp_path / "data")
    assert out == {
        "data_folder": str(tmp_path / "data"),
        "filenames_data": "features.txt",
        "input_is_sanity_output": False,
    }
    assert (tmp_path / "data" / "features.txt").exists()
    assert not (tmp_path / "data" / "standard_deviations.txt").exists()


def test_read_sanity_missing_ids_get_upstream_generic_names(tmp_path):
    folder = _sanity2_folder(tmp_path)
    (folder / "cellID.txt").unlink()
    (folder / "geneID.txt").unlink()
    a = bonsai.pp.read_sanity(folder)
    assert list(a.obs_names) == [f"Cell_{i}" for i in range(a.n_obs)]
    assert list(a.var_names) == [f"Gene_{i}" for i in range(a.n_vars)]


def test_nonfinite_prior_var_rejected():
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    a.var.loc[a.var_names[0], "bonsai_prior_var"] = np.nan
    with pytest.raises(ValueError, match="bonsai_prior_var"):
        validate_inputs(a)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_prior_var_accepted(bad):
    """Upstream drops genes where prior_var - var < 1e-4 rather than rejecting them up front
    (docs/deviations.md); only non-finite prior_var is refused here."""
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    a.var.loc[a.var_names[0], "bonsai_prior_var"] = bad
    validate_inputs(a)  # must not raise


def test_bad_mu_rejected():
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    a.var.loc[a.var_names[0], "bonsai_mu"] = np.nan
    with pytest.raises(ValueError, match="bonsai_mu"):
        validate_inputs(a)


def test_unknown_kind_raises_valueerror():
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    a.uns["bonsai_input"]["kind"] = "bogus"
    with pytest.raises(ValueError, match="kind"):
        validate_inputs(a)


def test_sanity_input_missing_sanity_method_raises_valueerror():
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    del a.uns["bonsai_input"]["sanity_method"]
    with pytest.raises(ValueError, match="sanity_method"):
        validate_inputs(a)


def test_from_arrays_missing_layer_raises_valueerror():
    """A layer deleted after from_arrays() must raise ValueError, not KeyError."""
    a = AnnData(np.ones((3, 2)))
    a.layers["m"] = np.zeros((3, 2))
    bonsai.pp.from_arrays(a, "m")
    del a.layers["m"]
    with pytest.raises(ValueError, match="not in adata.layers"):
        validate_inputs(a)


def test_sanity_posterior_missing_prior_var_raises_valueerror():
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    del a.var["bonsai_prior_var"]
    with pytest.raises(ValueError, match="bonsai_prior_var"):
        validate_inputs(a)


def test_sanity_posterior_missing_mu_raises_valueerror():
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    del a.var["bonsai_mu"]
    with pytest.raises(ValueError, match="bonsai_mu"):
        validate_inputs(a)


def test_write_data_folder_float32_layer_roundtrips_exactly(tmp_path):
    """A float32 layer's values, each cast up to float64, must parse back exactly (item 10:
    row-by-row shortest-repr writer instead of a full float64 densify)."""
    a = AnnData(np.ones((3, 2), dtype=np.float32))
    a.layers["m"] = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float32)
    bonsai.pp.from_arrays(a, "m")
    write_data_folder(a, tmp_path / "data")
    written = np.loadtxt(tmp_path / "data" / "features.txt", delimiter="\t")
    np.testing.assert_array_equal(written, a.layers["m"].T.astype(np.float64))


def test_write_data_folder_rewritten_delta_not_larger_than_original(tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)
    write_data_folder(a, tmp_path / "data")
    rewritten = (tmp_path / "data" / "delta.txt").stat().st_size
    original = (FIXTURE / "delta_vmax.txt").stat().st_size
    assert rewritten <= original


def test_write_data_folder_failed_validation_leaves_no_folder(tmp_path):
    a = bonsai.pp.read_sanity(FIXTURE)[:5, :20].copy()
    a.layers["bonsai_sd"][0, 0] = -1.0
    dest = tmp_path / "data"
    with pytest.raises(ValueError, match="bonsai_sd"):
        write_data_folder(a, dest)
    assert not dest.exists()
    # A retry with valid data must not hit FileExistsError from a leftover folder.
    a.layers["bonsai_sd"][0, 0] = 1.0
    write_data_folder(a, dest)
    assert dest.exists()
