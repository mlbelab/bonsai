"""Whole-tree branch-length optimisation on the device: L-BFGS on log t (maths ref §10)."""

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, Int

from .array_tree import ArrayTree
from .numerics import dsum
from .tree import tree_loglik_grad

LOG_T_MIN, LOG_T_MAX = -15.0, 15.0  # upstream box (TH:3462), t in [3.1e-7, 3.3e6]
T_SNAP = 2e-6  # upstream sets t < 1e-6 + 1e-6 to exactly 0 after optimising (TH:3467)

_ell = eqx.filter_jit(lambda tree: tree_loglik_grad(tree).ell)  # ell only: XLA drops the n×G messages


@eqx.filter_custom_vjp
def _neg_loglik(s, args):
    """−(L_tree(s) − L_tree(s0)) at t = exp(clip(s)) for the non-root edges (one s per node row)."""
    return _eval(s, args)[0]


def _eval(s, args):
    """Objective and gradient of the minimisation.

    The objective is the change from the start, formed per node in the compute dtype and summed
    with ``dsum``: it resolves differences far below eps·|L| without float64 on the device (A8).
    The gradient is the projected gradient at the clipped point (scipy's pgtol quantity): zero
    only where it points out of the box. A plain zero slope outside the box froze edges that a
    line-search step had carried past a bound even when L pulled them back inside, and the gtol
    rule then reported such a point as converged.
    """
    tree, ell0 = args
    edge = tree.parent >= 0
    t = jnp.where(edge, jnp.exp(jnp.clip(s, LOG_T_MIN, LOG_T_MAX)), 0.0)
    e = tree_loglik_grad(eqx.tree_at(lambda tr: tr.t, tree, t))
    g = -e.grad * t  # d(−L)/ds = −t·dL/dt
    out = ((s <= LOG_T_MIN) & (g > 0)) | ((s >= LOG_T_MAX) & (g < 0))
    return -dsum(e.ell - ell0), jnp.where(edge & ~out, g, 0.0)


@_neg_loglik.def_fwd
def _neg_loglik_fwd(perturbed, s, args):
    return _eval(s, args)


@_neg_loglik.def_bwd
def _neg_loglik_bwd(g, ct, perturbed, s, args):
    return ct * g


class _GtolLBFGS(optx.AbstractMinimiser):
    """optx.LBFGS whose termination also fires once max|d(−L)/ds| ≤ gtol (scipy's pgtol rule).

    optimistix's own rule (Cauchy: small Δf and small Δy together) stays in place. Reads
    ``state.f_info.grad``, the gradient at the last accepted iterate (optimistix quasi-Newton
    internals, optimistix 0.1.x; A9 pins ``optimistix<0.2``).
    """

    inner: optx.LBFGS
    gtol: float
    rtol: float
    atol: float
    norm: Callable[[Any], Array]

    def __init__(self, rtol: float, atol: float, gtol: float):
        self.inner = optx.LBFGS(rtol=rtol, atol=atol)
        self.gtol, self.rtol, self.atol, self.norm = gtol, rtol, atol, self.inner.norm

    def init(self, *args):
        return self.inner.init(*args)

    def step(self, *args):
        return self.inner.step(*args)

    def terminate(self, fn, y, args, options, state, tags):
        done, result = self.inner.terminate(fn, y, args, options, state, tags)
        small = jnp.max(jnp.abs(state.f_info.grad)) <= self.gtol  # f_info is zeros before the first step
        return done | (small & ~state.first_step), result

    def postprocess(self, *args):
        return self.inner.postprocess(*args)


class TimesResult(NamedTuple):
    tree: ArrayTree  # with optimised t (snapped to 0 below T_SNAP if requested)
    L: float  # L_tree at the returned t (A1 convention, float64 host sum)
    steps: Int[Array, ""]
    converged: bool  # True: the gtol rule or optimistix's own Cauchy rule (small Δf and Δs) fired
    # (in fp32 the Cauchy rule fires at the objective's ΔL resolution floor, not at gtol); False:
    # max_steps was reached or the result is non-finite.


def optimise_times(
    tree: ArrayTree,
    rtol: float = 1e-6,
    atol: float = 0.0,
    gtol: float = 1e-6,
    max_steps: int = 100,
    snap_zero: bool = True,
) -> TimesResult:
    """Maximise L_tree over all branch lengths, L-BFGS on s = log t within upstream's box.

    Defaults mirror upstream's final ``optTimes`` (tol=1e-6 ⇒ ftol=gtol=1e-6, maxiter=100;
    main.py:274, 430): ``gtol`` is scipy's pgtol on d(−L)/ds; optimistix's Cauchy rule (small Δf
    *and* small Δs, relative to the objective measured from the start) never stops earlier than
    upstream's ftol. Starts from log clip(t, 1e-4, 1e4) as upstream (TH:3461).
    ``max_steps`` counts optimistix steps, one objective evaluation each (line-search trials
    included), so 100 steps buy fewer L-BFGS iterations than scipy's maxiter = 100.
    """
    s0 = jnp.log(jnp.clip(tree.t, 1e-4, 1e4))
    start = eqx.tree_at(lambda tr: tr.t, tree, jnp.where(tree.parent >= 0, jnp.exp(s0), 0.0))
    args = (tree, _ell(start))
    sol: optx.Solution = optx.minimise(
        _neg_loglik, _GtolLBFGS(rtol, atol, gtol), s0, args=args, max_steps=max_steps, throw=False
    )
    t = jnp.where(tree.parent >= 0, jnp.exp(jnp.clip(sol.value, LOG_T_MIN, LOG_T_MAX)), 0.0)
    if snap_zero:
        t = jnp.where(t < T_SNAP, 0.0, t)
    out = eqx.tree_at(lambda tr: tr.t, tree, t)
    L = float(np.sum(np.asarray(_ell(out), dtype=np.float64)))  # as TreeEval.L
    return TimesResult(out, L, sol.stats["num_steps"], bool(sol.result == optx.RESULTS.successful))
