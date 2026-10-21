"""Vectorised star EM (``_engine.star.star_em``) vs the fp64 reference and the star KKT conditions (maths ref §9).

float32 cases run in the default lane; the float64 case (``x64`` marker) runs the same dtype-generic kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bench.run import count_compilations

from bonsai._engine.numerics import dsum
from bonsai._engine.star import STAR_T_MAX, _child_time, star_em
from bonsai._ref.star import _child_time as ref_child_time
from bonsai._ref.star import _star_dlogl, star_loglik_grad
from bonsai._ref.star import star_em as ref_star_em

from ._bounds import EPS32, ell_scale
from .test_ref_star import _kkt_violations, star_fixture


def _centred(rng=3, C=12, G=40):
    """Fixture centred by its column mean in float64 (centring contract of the engine kernels)."""
    x, var = star_fixture(np.random.default_rng(rng), C, G)
    return x - x.mean(axis=0), var


def _run(x, var, dtype=jnp.float32, **kw):
    return star_em(jnp.asarray(x, dtype), jnp.asarray(var, dtype), jnp.ones(len(x), bool), **kw)


@pytest.mark.x64
def test_engine_star_em_kkt_x64():
    """Same KKT-by-FD assertions as the reference test, judged on the engine's own float64 times."""
    xc, var = _centred()
    r = _run(xc, var, jnp.float64, tol=1e-12)
    t = np.asarray(r.t)
    assert bool(r.converged) and np.any(t == 0) and np.any(t > 0)
    assert _kkt_violations(xc, var, t) == []


@pytest.mark.x64
def test_child_time_clamps_only_on_strictly_negative_derivative():
    """D(0) == 0 exactly with D increasing at 0: the strict clamp (TH:4796) does not fire and both the
    engine and the reference step move to the positive root (a ``<=`` clamp would return 0)."""
    v, W = np.array([1.0, 100.0]), np.array([1e6, 1e6])
    d2 = np.array([0.0, 10099.989999])  # Σ_g w(w d² − 1 + w/W) rounds to exactly 0.0 in fp64 (both formulas)
    w = 1.0 / v
    assert _star_dlogl(0.0, v, d2, W) == 0.0 and _star_dlogl(1e-3, v, d2, W) > 0
    assert float(dsum(jnp.asarray(w * (w * d2 - 1.0 + w / W)))) == 0.0  # the engine's D(0)
    assert ref_child_time(1.0, v, d2, W) > 0
    assert float(_child_time(jnp.asarray(1.0), jnp.asarray(v), jnp.asarray(d2), jnp.asarray(W))) > 0


def test_engine_star_em_fp32_value_matches_reference():
    xc, var = _centred()
    r = _run(xc, var, tol=1e-4)
    t_ref, ell_ref, ok = ref_star_em(xc, var, tol=1e-12)
    assert bool(r.converged) and ok
    ell_eng = star_loglik_grad(xc, var, np.asarray(r.t, np.float64))[0]  # fp64 value at the fp32 times
    # float32 resolution of ell: 16·eps32·Σ|terms| (8 ops per term + input rounding, 2x margin). The stop
    # rule adds only second order: ell is stationary at the optimum, so |Δt| < 1e-4 costs O(H·1e-8).
    assert ell_eng >= ell_ref - 16 * EPS32 * ell_scale(xc, var, t_ref)


def test_engine_star_em_stops_on_absolute_change():
    """Stop rule max_c |Δt_c| < tol, absolute (TH:5000-5004): the sweep that stopped changed t by < tol,
    the one before by ≥ tol. Data ×10 puts the times near 50-100, where a relative rule would stop early."""
    xc, var = _centred()
    xc = 10 * xc
    tol = 1e-4
    r = _run(xc, var, tol=tol)
    k = int(r.sweeps)
    assert bool(r.converged) and k >= 3 and float(r.t.max()) > 10
    prev, prev2 = _run(xc, var, tol=tol, max_sweeps=k - 1), _run(xc, var, tol=tol, max_sweeps=k - 2)
    assert float(jnp.max(jnp.abs(r.t - prev.t))) < tol
    assert float(jnp.max(jnp.abs(prev.t - prev2.t))) >= tol


def test_engine_star_em_no_sign_change():
    """A child with d² ~ 1e14 has its root near t ~ d², beyond every bracket (≤ 1e12 from t = 1): it takes
    the last bracket and the sweeps end on max t ≥ 1e8, not converged (upstream: its own 1e8 exit)."""
    xc, var = _centred()
    xc[0], var[0] = 1e7, 1e-4
    r = _run(xc, var)
    assert not bool(r.converged) and float(r.t[0]) >= STAR_T_MAX and int(r.sweeps) == 1


def test_engine_star_em_padding_and_compiles():
    """C = 12 in a 16-row buffer with NaN padding: t[:12] bitwise equal to the unpadded call (``dsum``
    pads to 16 with zeros either way), padded t = 0; C = 13 in the same bucket adds no compile."""
    xc, var = _centred()
    unpadded = _run(xc, var)

    def padded(x, v):
        n = len(x)
        xp, vp = np.full((16, x.shape[1]), np.nan), np.full((16, x.shape[1]), np.nan)
        xp[:n], vp[:n] = x, v
        f = jnp.float32
        return star_em(jnp.asarray(xp, f), jnp.asarray(vp, f), jnp.asarray(np.arange(16) < n))

    jax.clear_caches()
    # bitwise holds on CPU (same pairwise dsum tree); if it flakes on a jax upgrade, compare to 1 ulp, CPU only
    with count_compilations() as c:
        r = padded(xc, var)
        r.t.block_until_ready()
    assert c.count >= 1
    assert np.array_equal(np.asarray(r.t[:12]), np.asarray(unpadded.t))
    assert np.all(np.asarray(r.t[12:]) == 0) and int(r.sweeps) == int(unpadded.sweeps)
    x13, v13 = _centred(rng=4, C=13)
    with count_compilations() as c:
        r13 = padded(x13, v13)
        r13.t.block_until_ready()
    assert c.count == 0 and bool(r13.converged) and np.all(np.asarray(r13.t[13:]) == 0)


def test_engine_star_em_flags_non_convergence():
    xc, var = _centred()
    r = _run(xc, var, max_sweeps=2)
    assert not bool(r.converged) and int(r.sweeps) == 2
