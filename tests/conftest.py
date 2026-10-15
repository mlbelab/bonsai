import jax
import pytest


@pytest.fixture(autouse=True)
def _guard_precision(request):
    """`x64` tests fail unless JAX_ENABLE_X64=1; all other tests fail if it is set. `gpu` tests need a GPU.

    fp32 and fp64 lanes are separate processes; x64 is never toggled in-process (rose convention).
    """
    x64 = bool(jax.config.jax_enable_x64)
    if request.node.get_closest_marker("x64") is not None:
        if not x64:
            pytest.fail("x64 test run without JAX_ENABLE_X64=1; run the x64 lane (-m x64)")
    elif x64:
        pytest.fail("non-x64 test run with JAX_ENABLE_X64=1; select x64 tests with -m x64")
    if request.node.get_closest_marker("gpu") is not None:
        request.getfixturevalue("gpu_device")


@pytest.fixture
def gpu_device():
    """The first GPU; fails (not skips) when a `gpu` test runs without one."""
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        pytest.fail("gpu test run without a GPU device")
    return gpus[0]
