"""Brownian motion on a random binary tree with Sanity-like posterior output (phase 2 benchmarks).

Model, in Bonsai's own terms (maths ref §1): per gene g a prior variance v_g and mean mu_g; true
log expression z = mu_g + sqrt(v_g)·B, with B a standard Brownian motion along the tree started at
0 at the root (so z/sqrt(v_g) diffuses with N(parent, t·I), Bonsai's model after its √v_g
rescaling). Each cell has a library size L_c; the likelihood of a cell's log expression has mean
m = z + noise and variance s2 = clip(1/(L_c·exp(z)), S2_MIN, S2_MAX) (a Poisson delta-method
variance: low expression, small libraries ⇒ large s2). Sanity reports the posterior under the
prior N(mu_g, v_g): eps² = s2·v/(s2 + v), delta = (m − mu)·v/(s2 + v); exactly the inverse of
upstream's prior removal (DP:2916-2927).

ponytail: all node data (z) and the N×G intermediates (v, mu, lib, s2, m, eps2, delta) are held
in memory at once as float64; at 100k cells x 2k genes that is already ~12 GB or more, so 1M
cells is not feasible as written. Upgrade path (M5): chunk genes and stream the per-chunk arrays
instead of materialising all of them.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from anndata import AnnData

from bonsai._ref.newick import to_newick

S2_MIN, S2_MAX = 1e-4, 20.0  # S2_MAX keeps v − eps² = v²/(s2 + v) ≥ 1e-4 for v ≥ V_MIN (DP:2916)
V_MIN, V_MAX = 0.05, 10.0


@dataclass(frozen=True)
class Simulation:
    """One simulated dataset and its ground truth."""

    adata: AnnData  # cells × genes with the read_sanity slots and uns["bonsai_truth"]
    parent: np.ndarray  # (2N−1,) parent index, −1 at the root; leaves 0..N−1, root 2N−2
    t: np.ndarray  # (2N−1,) branch length above each node, 0 at the root
    names: list[str]  # (2N−1,) node names, in parent-array order (cells, then internal_i, then root)
    z: np.ndarray  # (2N−1, G) true log expression at every node
    lik_mean: np.ndarray  # (N, G) m
    lik_var: np.ndarray  # (N, G) s2
    lib_size: np.ndarray  # (N,) L_c


def random_binary_tree(n: int, rng: np.random.Generator, mean_t: float) -> tuple[np.ndarray, np.ndarray]:
    """Join two uniformly random active nodes until one is left; Exp(mean_t) branch lengths."""
    if n < 2:
        raise ValueError("need at least 2 leaves")
    parent = np.full(2 * n - 1, -1, dtype=np.int64)
    active = np.arange(n)
    m = n
    for new in range(n, 2 * n - 1):
        i = int(rng.integers(m))
        j = int(rng.integers(m - 1))
        j += j >= i
        parent[active[i]] = parent[active[j]] = new
        active[i] = new
        active[j] = active[m - 1]
        m -= 1
    t = rng.exponential(mean_t, size=2 * n - 1)
    t[-1] = 0.0
    return parent, t


def simulate(
    n_cells: int,
    n_genes: int,
    seed: int,
    mean_t: float = 0.1,
    v_median: float = 0.5,
    v_log_sd: float = 0.75,
    mu_mean: float = -9.0,
    mu_sd: float = 1.5,
    lib_median: float = 1e4,
    lib_log_sd: float = 0.5,
) -> Simulation:
    """Draw a tree, Brownian log expression and Sanity-like posteriors (see module docstring)."""
    rng = np.random.default_rng(seed)
    parent, t = random_binary_tree(n_cells, rng, mean_t)
    v = np.clip(v_median * np.exp(v_log_sd * rng.standard_normal(n_genes)), V_MIN, V_MAX)
    mu = mu_mean + mu_sd * rng.standard_normal(n_genes)
    z = np.empty((2 * n_cells - 1, n_genes))
    z[-1] = mu
    for c in range(2 * n_cells - 3, -1, -1):  # parents have larger indices than their children
        z[c] = z[parent[c]] + np.sqrt(t[c] * v) * rng.standard_normal(n_genes)
    lib = lib_median * np.exp(lib_log_sd * rng.standard_normal(n_cells))
    zl = z[:n_cells]
    s2 = np.clip(1.0 / (lib[:, None] * np.exp(zl)), S2_MIN, S2_MAX)
    m = zl + np.sqrt(s2) * rng.standard_normal(zl.shape)
    eps2 = s2 * v / (s2 + v)
    delta = (m - mu) * v / (s2 + v)
    cells = [f"cell{i}" for i in range(n_cells)]
    names = cells + [f"internal_{i}" for i in range(n_cells - 2)] + ["root"]
    a = AnnData(obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=[f"gene{g}" for g in range(n_genes)]))
    a.layers["bonsai_mean"], a.layers["bonsai_sd"] = delta, np.sqrt(eps2)
    a.var["bonsai_prior_var"], a.var["bonsai_mu"] = v, mu
    a.uns["bonsai_input"] = {"kind": "sanity_posterior", "sanity_version": "simulated", "sanity_method": "MAP"}
    a.uns["bonsai_truth"] = {"newick": to_newick(parent, t, names), "parent": parent, "t": t, "seed": seed}
    return Simulation(a, parent, t, names, z, m, s2, lib)
