"""Exact kNN in upstream's metric: float32 device search, float64 host coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bench.run import count_compilations

from bonsai._engine.knn import MIN_TIME, knn, knn_coords, knn_query
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
    """k must satisfy 0 < k < count(valid); k == count(valid) would silently return an invalid
    row as the last neighbour."""
    P = jnp.asarray(_unit(np.random.default_rng(0), 5, 3), jnp.float32)
    with pytest.raises(ValueError, match="0 < k <"):
        knn(P, k, block=4)


def test_knn_rejects_k_not_below_valid_count():
    """5 valid rows out of 20: k=5 must raise, even though C=20 would allow it."""
    rng = np.random.default_rng(19)
    P = jnp.asarray(_unit(rng, 20, 5), jnp.float32)
    valid = np.arange(20) < 5
    with pytest.raises(ValueError, match="0 < k <"):
        knn(P, 5, valid)


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


def test_knn_masked_padding_matches_unpadded():
    """Padding to a bucket (C=37 into a 64-row buffer, invalid rows NaN) gives the same neighbour
    sets as the unpadded call, once a margin rules out fp32 rounding swapping the k/(k+1)
    boundary. A sorted index comparison (as the plan sketch does) only works when C is identical
    on both sides; here the two calls compile different shapes, so the sets are compared instead.
    Margin: the same expanded-form absolute bound as `_check_knn` (delta = 2(G+2)eps32, returned
    neighbours within dk + 2*delta), not a relative eps*max(d) bound."""
    rng = np.random.default_rng(5)
    C, G, k = 37, 16, 10
    pts = _unit(rng, C, G).astype(np.float32)
    pad = np.full((64, G), np.nan, np.float32)
    pad[:C] = pts
    valid = np.arange(64) < C
    d = ((pts[:, None, :].astype(np.float64) - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    dsorted = np.sort(d, axis=1)
    delta = 2 * (G + 2) * EPS32
    margin = 2 * delta
    assert np.all(dsorted[:, k] - dsorted[:, k - 1] > margin)
    unpadded = np.asarray(knn(jnp.asarray(pts), k))
    padded_full = np.asarray(knn(jnp.asarray(pad), k, valid))
    padded = padded_full[:C]
    assert all(set(a.tolist()) == set(b.tolist()) for a, b in zip(unpadded, padded, strict=True))
    # Invalid (padded NaN) rows' own query still returns well-formed, valid, non-self indices:
    # pins the row-zeroing step (without it, an invalid row's NaN query vector would poison its
    # own result with NaN before the column mask is applied).
    assert np.isin(padded_full[C:], np.flatnonzero(valid)).all()


def test_knn_excludes_invalid_middle_rows():
    """Invalid rows in the middle of the buffer (3 and 10) never come back as a neighbour of a
    valid row."""
    rng = np.random.default_rng(11)
    C, G, k = 40, 12, 5
    pts = _unit(rng, C, G)
    valid = np.ones(C, dtype=bool)
    valid[[3, 10]] = False
    idx = np.asarray(knn(jnp.asarray(pts, jnp.float32), k, valid))
    assert not np.isin(idx[valid], [3, 10]).any()


def test_knn_query_matches_fp64_bruteforce():
    """`knn_query` against 50 valid rows matches argsort of the fp64 squared distance, once a
    margin rules out fp32 rounding swapping the k/(k+1) boundary."""
    rng = np.random.default_rng(13)
    C, G, k = 50, 20, 10
    pts = _unit(rng, C, G)
    q = _unit(rng, 1, G)[0]
    d = ((pts - q[None, :]) ** 2).sum(-1)
    order = np.argsort(d, kind="stable")
    # Relative-form bound here (no cross term to cancel, unlike the blocked `knn` kernel):
    # (G+1)*eps32*d < 64*eps32*d for this test's G=20.
    margin = 64 * EPS32 * d.max()
    assert d[order[k]] - d[order[k - 1]] > margin
    idx = np.asarray(knn_query(jnp.asarray(pts, jnp.float32), jnp.ones(C, bool), jnp.asarray(q, jnp.float32), k))
    assert set(idx.tolist()) == set(order[:k].tolist())


def test_knn_query_k_equals_valid_count():
    """k == count(valid) is allowed (unlike `knn`): querying for every valid row is well-formed."""
    rng = np.random.default_rng(29)
    C, G = 6, 4
    pts = _unit(rng, C, G)
    q = _unit(rng, 1, G)[0]
    idx = np.asarray(knn_query(jnp.asarray(pts, jnp.float32), jnp.ones(C, bool), jnp.asarray(q, jnp.float32), C))
    assert sorted(idx.tolist()) == list(range(C))


def test_knn_query_self_match_if_valid():
    """Documents the A19 contract: `knn_query` does not exclude q's own row. If the caller queries
    a buffer whose freed row was already marked valid (wrong order -- A19 requires it stay invalid
    until after the query), that row comes back as its own zero-distance neighbour. Task 6b must
    query before validating, not after."""
    rng = np.random.default_rng(31)
    C, G = 8, 5
    pts = _unit(rng, C, G)
    valid = np.ones(C, dtype=bool)
    q = pts[3]  # q equals an already-valid row's own coordinate
    idx = np.asarray(knn_query(jnp.asarray(pts, jnp.float32), jnp.asarray(valid), jnp.asarray(q, jnp.float32), 1))
    assert idx.tolist() == [3]


def test_knn_query_excludes_invalid_rows():
    rng = np.random.default_rng(23)
    C, G, k = 20, 6, 3
    pts = _unit(rng, C, G)
    valid = np.ones(C, dtype=bool)
    valid[[1, 7]] = False
    q = _unit(rng, 1, G)[0]
    idx = np.asarray(knn_query(jnp.asarray(pts, jnp.float32), jnp.asarray(valid), jnp.asarray(q, jnp.float32), k))
    assert not np.isin(idx, [1, 7]).any()


def test_knn_compiles_once_per_bucket():
    """C=33 then C=60, both in a 64-row buffer: compiling for the bucket shape at C=33 leaves
    nothing new to compile at C=60, even though the valid count (and so which rows are masked)
    differs."""
    rng = np.random.default_rng(17)
    jax.clear_caches()
    buf1_np = np.zeros((64, 10), np.float32)
    buf1_np[:33] = _unit(rng, 33, 10)
    buf2_np = np.zeros((64, 10), np.float32)
    buf2_np[:60] = _unit(rng, 60, 10)
    buf1, buf2 = jnp.asarray(buf1_np), jnp.asarray(buf2_np)
    valid1, valid2 = np.arange(64) < 33, np.arange(64) < 60  # host arrays: no comparison to compile
    with count_compilations() as c:
        knn(buf1, 5, valid1).block_until_ready()
    assert c.count > 0
    with count_compilations() as c:
        knn(buf2, 5, valid2).block_until_ready()
    assert c.count == 0


@pytest.mark.gpu
def test_knn_on_gpu_uses_true_float32(gpu_device):
    """On GPUs JAX's default float32 matmul is TF32 (10-bit mantissa, error ~1e-3 ≫ δ): passing the
    δ bound shows MATMUL_PRECISION is in effect."""
    with jax.default_device(gpu_device):
        _check_knn(_unit(np.random.default_rng(2), 4096, 256), 10, 1024)
