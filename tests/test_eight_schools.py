import pathlib, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mojo_src"))

import eight_schools as es  # noqa: E402  (Mojo extension)
from pytensor_mojo.eight_schools_ref import (  # noqa: E402
    J, SIGMA, Y, build_model, compile_flat, point_to_dict,
)

RNG = np.random.default_rng(20261126)
POINTS = [RNG.normal(scale=s, size=J + 2) for s in [0.1, 0.5, 1.0, 2.0, 3.0] for _ in range(5)]


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture(scope="module")
def pymc_fns(model):
    return model.compile_logp(), model.compile_dlogp()


def mojo_eval(q):
    q = np.ascontiguousarray(q, dtype=np.float64)
    g = np.empty_like(q)
    lp = es.logp_dlogp(q.ctypes.data, g.ctypes.data, Y.ctypes.data, SIGMA.ctypes.data, J)
    return lp, g


@pytest.mark.parametrize("i", range(len(POINTS)))
def test_matches_pymc_compile_logp_dlogp(pymc_fns, i):
    logp_fn, dlogp_fn = pymc_fns
    q = POINTS[i]
    lp_ref = logp_fn(point_to_dict(q))
    g_ref = dlogp_fn(point_to_dict(q))
    lp, g = mojo_eval(q)
    np.testing.assert_allclose(lp, lp_ref, rtol=1e-10, atol=0)
    np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", [None, "NUMBA"])
def test_matches_flat_pytensor_fn(model, mode):
    f = compile_flat(model, mode=mode)
    for q in POINTS:
        lp_ref, g_ref = f(q)
        lp, g = mojo_eval(q)
        np.testing.assert_allclose(lp, lp_ref, rtol=1e-10)
        np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-12)


def test_batch_matches_single():
    Q = np.ascontiguousarray(np.stack(POINTS))
    G = np.empty_like(Q)
    L = np.empty(len(Q))
    es.logp_dlogp_batch(Q.ctypes.data, G.ctypes.data, L.ctypes.data, Y.ctypes.data, SIGMA.ctypes.data, J, len(Q))
    for k, q in enumerate(POINTS):
        lp, g = mojo_eval(q)
        assert L[k] == lp
        np.testing.assert_array_equal(G[k], g)


def scipy_logp(q):
    from scipy import stats
    tau = np.exp(q[1])
    return (stats.norm(0, 5).logpdf(q[0]) + stats.halfcauchy(scale=5).logpdf(tau) + q[1]
            + stats.norm(0, 1).logpdf(q[2:]).sum()
            + stats.norm(q[0] + tau * q[2:], SIGMA).logpdf(Y).sum())


def test_logp_matches_scipy_tightly():
    for q in POINTS:
        lp, _ = mojo_eval(q)
        np.testing.assert_allclose(lp, scipy_logp(q), rtol=1e-12)


def test_grad_matches_finite_differences():
    h = 1e-6
    for q in POINTS[:5]:
        _, g = mojo_eval(q)
        fd = np.array([(mojo_eval(q + h * e)[0] - mojo_eval(q - h * e)[0]) / (2 * h) for e in np.eye(J + 2)])
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)


def test_pymc_literal_params_float32_offset_documented():
    """With Python-literal scale params PyMC logp has a constant float32 rounding offset."""
    m = build_model(float64_params=False)
    f = m.compile_logp()
    offs = np.array([f(point_to_dict(q)) - mojo_eval(q)[0] for q in POINTS])
    f32 = lambda v: float(np.log(np.float32(v))) - float(np.log(v))  # noqa: E731
    # mu: -log(5); tau: +log(2) - log(5); everything else is float64.
    expected = -f32(5.0) + f32(2.0) - f32(5.0)
    np.testing.assert_allclose(offs, expected, rtol=0, atol=1e-11)
    assert abs(expected) > 1e-8
