"""Exact kNN in upstream's metric: float32 device search, float64 host coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bonsai._engine.knn import MIN_TIME, knn, knn_coords
from bonsai._ref.star import as_if_root, node_message

from ._bounds import EPS32, EPS64


def _unit(rng, C, G):
    p = rng.normal(size=(C, G))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _check_knn(P, k, block):
    """Returned neighbours are among the true k nearest up to the float32 distance error:
    d = |q|² + |p|² − 2q·p with an explicit-precision float32 matmul has |error| ≤ δ = 2(G+2)·eps32
    for unit vectors (G-term dot product, γ_G bound), so every returned j has d_ref(j) ≤ d_k + 2δ."""
    C, G = P.shape
    idx = np.asarray(knn(jnp.asarray(P, jnp.float32), k, block=block))
    d = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    dk = np.sort(d, axis=1)[:, k - 1]
    delta = 2 * (G + 2) * EPS32
    assert idx.shape == (C, k)
    assert np.all(idx != np.arange(C)[:, None]) and all(len(set(r)) == k for r in idx)
    assert np.all(np.take_along_axis(d, idx, axis=1) <= dk[:, None] + 2 * delta)
    return idx


@pytest.mark.parametrize(("C", "block"), [(300, 64), (1000, 1024), (1001, 256)])
def test_knn_matches_brute_force(C, block):
    _check_knn(_unit(np.random.default_rng(C), C, 50), 10, block)


@pytest.mark.parametrize("k", [0, 5])
def test_knn_invalid_k_raises(k):
    """k must satisfy 0 < k < C; k == C would silently return self as the last neighbour."""
    P = jnp.asarray(_unit(np.random.default_rng(0), 5, 3), jnp.float32)
    with pytest.raises(ValueError, match="0 < k < C"):
        knn(P, k, block=4)


def test_knn_nonunit_rows_needs_column_norm_term():
    """Rows scaled by factors in [0, 1] (including exact zeros), so every |q|², |p|², |q·p| ≤ 1
    still holds and the unit-vector δ bound from `_check_knn` is still valid. Unit-vector rows
    alone leave `+ sq[None, :]` untested (constant across columns there), so a mutant dropping it
    would still pass; non-unit rows catch it because the column norm then varies across points."""
    rng = np.random.default_rng(42)
    C, G = 200, 50
    P = _unit(rng, C, G)
    scale = rng.uniform(0, 1, size=C)
    scale[:5] = 0.0  # exact zero rows
    P *= scale[:, None]
    _check_knn(P, 10, 64)


def test_knn_smallest_valid_case_c2_k1():
    """C=2, k=1: the only valid neighbour of each point is the other point."""
    P = jnp.asarray(_unit(np.random.default_rng(7), 2, 5), jnp.float32)
    idx = np.asarray(knn(P, 1, block=4))
    assert idx.tolist() == [[1], [0]]


def test_duplicate_points_tie_break_by_lower_index():
    """Identical cells: equal distances; lax.top_k keeps the lower index first (deterministic)."""
    Q = np.zeros((6, 4))
    Q[:, 0] = 1.0
    Q[3:, 1] = 1.0
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    idx = np.asarray(knn(jnp.asarray(Q, jnp.float32), 2, block=4))
    assert idx.tolist() == [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]


def test_coords_match_upstream_definition():
    """AIR of each child (t ← max(t, 1e-4)) minus the parent's AIR mean, unit norm per child
    (TH:2115-2131), vectorised in float64 vs the reference AIR loop. Two cancellations: W_p − w̄_c
    (factor κ = W_p/W_{p\\c}) and x_AIR − x_p, whose relative error eps·(|x_AIR| + |x_p|)/‖x_AIR − x_p‖
    the normalisation spreads over the vector: ≤ 8·eps64·(1 + κ)·cond per entry."""
    rng = np.random.default_rng(1)
    C, G = 7, 30
    x, W, t = rng.normal(size=(C, G)), rng.uniform(1, 3, (C, G)), rng.uniform(0, 1, C)
    t[0] = 0.0  # exercises the minimum time
    xp, Wp = node_message(x, 1 / W, t)
    air = np.stack([as_if_root(x[i], W[i], max(t[i], MIN_TIME), xp, Wp)[0] for i in range(C)])
    want = (air - xp) / np.linalg.norm(air - xp, axis=1, keepdims=True)
    kappa = np.max(Wp / (Wp - 1 / (np.maximum(t, MIN_TIME)[:, None] + 1 / W)))
    cond = (np.abs(air) + np.abs(xp)).max(axis=1) / np.linalg.norm(air - xp, axis=1)
    got = knn_coords(x, W, t, xp, Wp)
    assert got.dtype == np.float64
    assert np.all(np.abs(got - want) <= 8 * EPS64 * (1 + kappa) * cond[:, None])


def test_zero_vector_child_stays_zero():
    """A child exactly at the parent's AIR mean has no direction: zeros, not NaN."""
    x = np.zeros((2, 3))
    W = np.ones((2, 3))
    xp, Wp = np.zeros(3), np.full(3, 4.0)
    assert np.array_equal(knn_coords(x, W, np.zeros(2), xp, Wp), np.zeros((2, 3)))


@pytest.mark.gpu
def test_knn_on_gpu_uses_true_float32(gpu_device):
    """On GPUs JAX's default float32 matmul is TF32 (10-bit mantissa, error ~1e-3 ≫ δ): passing the
    δ bound shows MATMUL_PRECISION is in effect."""
    with jax.default_device(gpu_device):
        _check_knn(_unit(np.random.default_rng(2), 4096, 256), 10, 1024)
