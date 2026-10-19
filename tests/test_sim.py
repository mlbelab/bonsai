"""The simulator reproduces its declared statistics (spec M0 gate).

4σ bounds: each check compares a sample mean/variance with its exact expectation; the bound is 4
standard errors of that estimator (two-sided false-alarm rate 6e-5 per check, fixed seeds).
"""

import numpy as np
import pytest
from bench.simulators.brownian import S2_MAX, S2_MIN, V_MIN, random_binary_tree, simulate

from bonsai._io_folder import validate_inputs
from bonsai._ref.newick import parse_newick
from bonsai._ref.tree import postorder

N, G, MEAN_T = 2000, 200, 0.1


@pytest.fixture(scope="module")
def sim():
    return simulate(N, G, seed=0, mean_t=MEAN_T)


def test_tree_is_a_rooted_binary_tree(sim):
    parent = sim.parent
    _, kids = postorder(parent)  # raises on forests/cycles
    assert len(parent) == 2 * N - 1 and parent[-1] == -1
    assert all(len(k) == 0 for k in kids[:N]) and all(len(k) == 2 for k in kids[N:])
    assert np.all(parent[:-1] > np.arange(2 * N - 2))  # parents after children


def test_cherry_count_matches_random_joining(sim):
    """Cherry count (internal nodes with two leaf children) for uniform random joining.

    McKenzie & Steel (2000): for a tree grown by repeatedly joining two uniformly random active
    nodes, E[#cherries] = N/3 and Var[#cherries] = 2N/45 for N >= 5. A caterpillar tree (join a
    fresh leaf to the current spine each step) has O(1) cherries, far outside this band, so this
    also catches a wrong tree-growth process (e.g. the reviewer's caterpillar swap).
    """
    parent = sim.parent
    _, kids = postorder(parent)
    c = sum(1 for n in range(N, len(parent)) if len(kids[n]) == 2 and all(k < N for k in kids[n]))
    assert abs(c - N / 3) <= 4 * np.sqrt(2 * N / 45)


def test_branch_lengths_are_exponential(sim):
    t = sim.t[:-1]
    n = len(t)
    assert sim.t[-1] == 0.0
    assert abs(t.mean() - MEAN_T) <= 4 * MEAN_T / np.sqrt(n)  # Exp: sd = mean
    assert abs(t.var() - MEAN_T**2) <= 4 * MEAN_T**2 * np.sqrt(8 / n)  # Var(s²) ≈ (μ4 − σ⁴)/n, μ4 = 9σ⁴


def test_brownian_increments_are_standard_normal(sim):
    """(z_c − z_parent)/sqrt(t_c·v_g) ~ iid N(0, 1) over edges and genes."""
    v = sim.adata.var["bonsai_prior_var"].to_numpy()
    u = (sim.z[:-1] - sim.z[sim.parent[:-1]]) / np.sqrt(sim.t[:-1, None] * v)
    assert abs(u.mean()) <= 4 / np.sqrt(u.size)
    assert abs(u.var() - 1) <= 4 * np.sqrt(2 / u.size)
    assert np.array_equal(sim.z[-1], sim.adata.var["bonsai_mu"].to_numpy())  # the root sits at mu_g


def test_measurement_noise_and_heteroscedastic_variances(sim):
    r = (sim.lik_mean - sim.z[:N]) / np.sqrt(sim.lik_var)
    assert abs(r.mean()) <= 4 / np.sqrt(r.size)
    assert abs(r.var() - 1) <= 4 * np.sqrt(2 / r.size)
    want = np.clip(1.0 / (sim.lib_size[:, None] * np.exp(sim.z[:N])), S2_MIN, S2_MAX)
    np.testing.assert_array_equal(sim.lik_var, want)  # the declared variance law, exactly


def test_sanity_posterior_inverts_upstream_prior_removal(sim):
    """Upstream's prior removal (DP:2916-2927) maps (delta, eps, v, mu) back to (m, s2).

    Per entry ≤ 8 rounded operations (square, sqrt, two subtractions, division, multiplication,
    addition); the subtraction v − eps² has condition number κ = v/(v − eps²) = (s2 + v)/v, so
    |rel. error| ≤ 8·eps64·κ for s2 and for the delta part of m; mu adds 8·eps64·|mu|.
    """
    a = sim.adata
    v, mu = a.var["bonsai_prior_var"].to_numpy(), a.var["bonsai_mu"].to_numpy()
    eps2 = np.maximum(a.layers["bonsai_sd"] ** 2, 1e-12)
    D = v - eps2
    assert D.min() >= 1e-4  # every gene survives upstream's divide_out_prior_cutoff (DP:2916)
    f = v / np.maximum(D, 1e-6)
    m, s2 = a.layers["bonsai_mean"] * f + mu, eps2 * f
    kappa = (sim.lik_var + v) / v
    e = 8 * np.finfo(float).eps
    assert np.all(np.abs(s2 - sim.lik_var) <= e * kappa * sim.lik_var)
    assert np.all(np.abs(m - sim.lik_mean) <= e * (kappa * np.abs(sim.lik_mean - mu) + np.abs(mu)))


def test_adata_slots_truth_and_determinism(sim):
    a = sim.adata
    validate_inputs(a)  # what tl.reconstruct checks first
    assert a.uns["bonsai_input"]["kind"] == "sanity_posterior"
    assert a.var["bonsai_prior_var"].min() >= V_MIN
    parent, t, names = parse_newick(a.uns["bonsai_truth"]["newick"])
    idx = np.array([int(n[4:]) for n in names if n.startswith("cell")])
    assert sorted(idx.tolist()) == list(range(N)) and len(parent) == 2 * N - 1
    again = simulate(N, G, seed=0, mean_t=MEAN_T)
    np.testing.assert_array_equal(again.adata.layers["bonsai_mean"], a.layers["bonsai_mean"])
    assert not np.array_equal(simulate(N, G, seed=1).parent, sim.parent)


def test_random_binary_tree_small_and_invalid():
    parent, t = random_binary_tree(2, np.random.default_rng(0), 1.0)
    assert parent.tolist() == [2, 2, -1] and t[-1] == 0.0
    with pytest.raises(ValueError):
        random_binary_tree(1, np.random.default_rng(0), 1.0)
