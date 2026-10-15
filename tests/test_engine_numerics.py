import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bonsai._engine.numerics import MATMUL_PRECISION, dsum


def test_matmul_precision_is_highest():
    assert MATMUL_PRECISION == jax.lax.Precision.HIGHEST


def test_dsum_is_compensated_in_float32():
    """The default (x64-off) process: TwoSum keeps the bits that float32 sums lose in some orders
    (a plain sum's result depends on the backend's reduction order; dsum's does not)."""
    for a in ([1e8, 1.0, -1e8, 1.0], [1.0, 1e8, 1.0, -1e8], [1e8, 1.0, 1.0, -1e8, 1.0, 1.0]):
        a = jnp.asarray(a, jnp.float32)
        assert dsum(a).dtype == jnp.float32 and float(dsum(a)) == float(np.sum(np.asarray(a, np.float64)))


def test_dsum_inf_and_nan_match_jnp_sum():
    """inf − inf inside TwoSum's error term must not turn a legitimate ±inf result into NaN."""
    cases = {
        (float("inf"), 1.0): float("inf"),
        (float("-inf"), 1.0, 2.0): float("-inf"),
        (3e38, 3e38): float("inf"),
        (float("nan"), 1.0): float("nan"),
        (float("inf"), float("-inf")): float("nan"),
    }
    for a, expected in cases.items():
        got = float(dsum(jnp.asarray(a, jnp.float32)))
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == expected


def test_dsum_gradient_is_all_ones():
    a = jnp.asarray([1e8, 1.0, -1e8, 1.0, 2.0, 3.0], jnp.float32)
    grad = jax.grad(dsum)(a)
    np.testing.assert_array_equal(np.asarray(grad), np.ones(a.shape, np.float32))


def test_dsum_error_bound_on_many_terms():
    """|dsum − exact| ≤ eps32·|exact| + 4·log2(n)·eps32²·Σ|a| (one final rounding plus the
    second-order terms of the compensated pairwise tree), for each of 8 rows of 3000 terms. A plain
    float32 ``jnp.sum`` fails this bound, on CPU and on the L40S (checked while planning)."""
    rng = np.random.default_rng(0)
    a = (rng.normal(size=(8, 3000)) * 10.0 ** rng.integers(-3, 4, size=(8, 3000))).astype(np.float32)
    exact = a.astype(np.float64).sum(axis=1)
    got = np.asarray(dsum(jnp.asarray(a)), dtype=np.float64)
    eps = float(np.finfo(np.float32).eps)
    bound = eps * np.abs(exact) + 4 * np.log2(3000) * eps**2 * np.abs(a).astype(np.float64).sum(axis=1)
    assert np.all(np.abs(got - exact) <= bound)
    assert np.asarray(dsum(jnp.asarray(a), axis=0)).shape == (3000,)


@pytest.mark.x64
def test_float32_stays_float32_with_x64_enabled():
    """float32 input stays float32 when another library set JAX_ENABLE_X64=1 (see also Task 11)."""
    a = jnp.asarray([1e8, 1.0, -1e8, 1.0], jnp.float32)
    assert dsum(a).dtype == jnp.float32 and float(dsum(a)) == 2.0
