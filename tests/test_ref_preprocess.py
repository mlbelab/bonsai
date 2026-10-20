"""The vectorised preprocessing port reproduces upstream read_and_filter + rescaling bitwise (A5)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from bench.simulators.brownian import simulate

from bonsai import pp
from bonsai._core.helpers import BonsaiError
from bonsai._io_folder import write_data_folder
from bonsai._ref import preprocess as preprocess_mod
from bonsai._ref.preprocess import preprocess

from ._upstream import upstream_leaves
from .golden import FIXTURE


def _assert_bitwise(leaves, up):
    x, var, gv, keep = up
    np.testing.assert_array_equal(np.flatnonzero(leaves.genes), keep)
    for got, want in ((leaves.x, x), (leaves.var, var), (leaves.gene_var, gv)):
        assert got.dtype == want.dtype == np.float64
        assert np.array_equal(got, want)  # bitwise, not allclose


@pytest.mark.parametrize("kw", [{}, {"rescale_by_var": False}, {"zscore_cutoff": 0.5}, {"all_genes": True}])
def test_fixture_bitwise(kw):
    leaves = preprocess(pp.read_sanity(FIXTURE), **kw)
    up = upstream_leaves(
        FIXTURE,
        "delta_vmax.txt",
        "d_delta_vmax.txt",
        True,
        cutoff=kw.get("zscore_cutoff", 1.0),
        rescale_by_var=kw.get("rescale_by_var", True),
        all_genes=kw.get("all_genes", False),
    )
    _assert_bitwise(leaves, up)


def test_loglik_var_corr_matches_upstream_definition():
    leaves = preprocess(pp.read_sanity(FIXTURE))
    n = leaves.x.shape[0]
    assert leaves.loglik_var_corr == -(n - 1) * np.sum(np.log(leaves.gene_var))  # DP:208-210


def _likelihood_adata(with_sd: bool):
    sim = simulate(30, 40, seed=1)
    a = sim.adata.copy()
    a.layers["m"], a.layers["s"] = sim.lik_mean, np.sqrt(sim.lik_var)
    pp.from_arrays(a, "m", "s" if with_sd else None)
    return a


@pytest.mark.parametrize("with_sd", [True, False])
def test_likelihood_inputs_bitwise(tmp_path, with_sd):
    """Non-Sanity input through upstream's text folder (written by tl.reconstruct's own writer)."""
    a = _likelihood_adata(with_sd)
    files = write_data_folder(a, tmp_path / "data")["filenames_data"].split(",")
    up = upstream_leaves(tmp_path / "data", files[0], files[1] if with_sd else None, False)
    _assert_bitwise(preprocess(a), up)


def test_float32_layers_match_upstream_float64_text_path(tmp_path):
    """Review focus 5: h5ad layers are often float32; upstream reads their exact float64 text."""
    a = simulate(30, 40, seed=2).adata
    for k in ("bonsai_mean", "bonsai_sd"):
        a.layers[k] = a.layers[k].astype(np.float32)
    write_data_folder(a, tmp_path / "data")
    _assert_bitwise(preprocess(a), upstream_leaves(tmp_path / "data", "delta.txt", "d_delta.txt", True))


def test_closed_form_round_trip_to_simulated_likelihood():
    """preprocess(Sanity posterior) = (m, s2)/v_g: precision addition undone exactly (maths ref §1).

    Tolerance: ≤ 8 rounded operations per entry; v − eps² has condition κ = (s2 + v)/v, so
    |rel. error| ≤ 8·eps64·κ (s2) and 8·eps64·(κ·|m − mu| + |mu|)/|m| (m); /v and /√v add 2 ulps.
    """
    sim = simulate(200, 50, seed=3)
    leaves = preprocess(sim.adata, zscore_cutoff=-1)
    g = leaves.genes
    v, mu = sim.adata.var["bonsai_prior_var"].to_numpy()[g], sim.adata.var["bonsai_mu"].to_numpy()[g]
    m, s2 = sim.lik_mean[:, g], sim.lik_var[:, g]
    kappa = (s2 + v) / v
    e = 10 * np.finfo(float).eps
    assert np.all(np.abs(leaves.var * v - s2) <= e * kappa * s2)
    assert np.all(np.abs(leaves.x * np.sqrt(v) - m) <= e * (kappa * np.abs(m - mu) + np.abs(mu)))


def test_fewer_than_two_genes_raises():
    with pytest.raises(BonsaiError, match="at least 2"):
        preprocess(pp.read_sanity(FIXTURE), zscore_cutoff=1e6)


_N_TRICKY = 40  # cells; small enough to keep text I/O fast, large enough for a real BFGS fit


def _tricky_adata(kind: str, seed: int = 3) -> AnnData:
    """A handful of ordinary genes plus genes built to sit exactly on each filter's edge.

    - ``allzero``: delta/mean 0 everywhere. z == 0 exactly for every kind, so it is the probe for
      the strict ``>`` in the z-score filters (sanity DP:2869, likelihood-with-SD DP:2968) and for
      the no-SD ``var > 0`` filter (DP:3016): it must be dropped even at ``zscore_cutoff <= 0``
      (``zsq_cut == 0``), where a ``>=`` mutant would wrongly keep it.
    - ``tau_gene`` (sanity only): v_g = 1.0, one cell has sd**2 = 0.9995 so
      D = v_g - sigma2_post = 5e-4 for that cell, strictly between tau = 1e-4 and a mutated 1e-3;
      the rest of the cells are far from the tau boundary. Kept at tau = 1e-4, dropped at 1e-3.
    - ``clamp_gene`` (sanity and likelihood-with-SD): sd == 0 for every cell, so the returned
      ``var`` is 1e-12 (clamped) vs 0 (unclamped) -- a guaranteed, large bitwise difference that
      does not depend on where the gene lands relative to any cutoff.
    - ``zero_z`` (likelihood-with-SD only): a constant mean with ordinary per-cell variances. BFGS
      converges (v_hat -> 0 is the true MLE for a noiseless constant signal), and the z-score's
      v_hat/(v_hat + sigma^2) factor is then exactly 0, so z == 0 exactly: the same strict-``>``
      probe as ``allzero``, through the likelihood-with-SD branch's own formula.
    """
    sim = simulate(_N_TRICKY, 8, seed=seed)
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(_N_TRICKY)]
    if kind == "sanity":
        a = sim.adata.copy()
        tau_sd = np.full(_N_TRICKY, 0.1)
        tau_sd[0] = np.sqrt(0.9995)
        extra_mean = np.stack(
            [np.zeros(_N_TRICKY), rng.normal(0, 3.0, _N_TRICKY), rng.normal(0, 1.0, _N_TRICKY)], axis=1
        )
        extra_sd = np.stack([np.zeros(_N_TRICKY), tau_sd, np.zeros(_N_TRICKY)], axis=1)
        names = list(a.var_names) + ["allzero", "tau_gene", "clamp_gene"]
        out = AnnData(obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=names))
        out.layers["bonsai_mean"] = np.concatenate([a.layers["bonsai_mean"], extra_mean], axis=1)
        out.layers["bonsai_sd"] = np.concatenate([a.layers["bonsai_sd"], extra_sd], axis=1)
        out.var["bonsai_prior_var"] = np.concatenate([a.var["bonsai_prior_var"].to_numpy(), [1.0, 1.0, 1.0]])
        out.var["bonsai_mu"] = np.concatenate([a.var["bonsai_mu"].to_numpy(), [0.0, 0.0, 0.5]])
        out.uns["bonsai_input"] = dict(a.uns["bonsai_input"])
        return out
    if kind in ("lik_sd", "lik_nosd"):
        with_sd = kind == "lik_sd"
        m_ord = sim.lik_mean
        extras_m = [np.zeros(_N_TRICKY)]
        extras_s2 = [np.full(_N_TRICKY, 0.1)]
        names = [f"gene{i}" for i in range(m_ord.shape[1])] + ["allzero"]
        if with_sd:
            extras_m += [np.full(_N_TRICKY, 3.0), np.full(_N_TRICKY, 1e300), rng.normal(0, 1.0, _N_TRICKY)]
            extras_s2 += [np.linspace(0.1, 2.0, _N_TRICKY), np.full(_N_TRICKY, 1e-10), np.zeros(_N_TRICKY)]
            names += ["zero_z", "bfgs_fail", "clamp_gene"]
        m = np.concatenate([m_ord, np.stack(extras_m, axis=1)], axis=1)
        a = AnnData(obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=names))
        a.layers["m"] = m
        if with_sd:
            a.layers["s"] = np.sqrt(np.concatenate([sim.lik_var, np.stack(extras_s2, axis=1)], axis=1))
        pp.from_arrays(a, "m", "s" if with_sd else None)
        return a
    raise ValueError(kind)


_TRICKY_FILES = {
    "sanity": ("delta.txt", "d_delta.txt", True),
    "lik_sd": ("features.txt", "standard_deviations.txt", False),
    "lik_nosd": ("features.txt", None, False),
}


@pytest.mark.parametrize("cutoff", [1.0, -1.0])
@pytest.mark.parametrize("kind", ["sanity", "lik_sd", "lik_nosd"])
def test_tricky_edges_bitwise(tmp_path, kind, cutoff):
    """Edge-case genes (clamp, tau, exact-zero z) that the smooth fixture/simulator never hit.

    Confirmed against upstream by direct probe: all 6 (kind, cutoff) combinations below are
    bitwise-equal to ``upstream_leaves``, with the all-zero gene dropped at every cutoff
    (including ``cutoff <= 0``, where ``zsq_cut == 0`` and the filter is a strict ``>``).
    """
    a = _tricky_adata(kind)
    means_file, sd_file, sanity = _TRICKY_FILES[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # upstream warns when no stds file is given (lik_nosd)
        leaves = preprocess(a, zscore_cutoff=cutoff)
        write_data_folder(a, tmp_path / "data")
        up = upstream_leaves(tmp_path / "data", means_file, sd_file, sanity, cutoff=cutoff)
    _assert_bitwise(leaves, up)
    assert "allzero" not in a.var_names[leaves.genes]


def test_bfgs_failure_drops_the_gene_like_upstreams_continue(monkeypatch):
    """DP:2955-2959: upstream discards a gene whose BFGS fit does not converge (``continue``).

    A real non-convergent-but-finite BFGS fit could not be constructed (3000 random trials
    always converged or degenerated to nan, which already drops the gene through the z filter
    regardless of the ``continue``); the failure path is pinned directly by forcing every fit to
    report ``success=False`` with an otherwise unremarkable ``x``. Removing ``if not res.success:
    continue`` (preprocess.py) would let these genes through the z filter (real data clears it
    easily) and preprocess would return 5 genes instead of raising.
    """
    sim = simulate(_N_TRICKY, 5, seed=13)
    a = sim.adata.copy()
    a.layers["m"], a.layers["s"] = sim.lik_mean, np.sqrt(sim.lik_var)
    pp.from_arrays(a, "m", "s")

    class _Failed:
        success = False
        x = np.array([0.0])
        message = "mocked failure"

    monkeypatch.setattr(preprocess_mod, "minimize", lambda *a, **k: _Failed())
    with pytest.raises(BonsaiError, match="at least 2"):
        preprocess(a)
