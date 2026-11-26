"""C-ABI Mojo models (mojo_capi/*.dylib) vs PyMC logp/dlogp at rtol 1e-10."""
import ctypes, pathlib, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pytensor_mojo import eight_schools_ref as es8, hlr_ref  # noqa: E402

P = ctypes.c_void_p


def test_es8_capi_matches_pymc():
    lib = ctypes.CDLL(str(ROOT / "mojo_capi" / "libes8.dylib"))
    lib.es8_logp_grad.restype = ctypes.c_int32
    assert lib.es8_dim(None) == 10
    m = es8.build_model(); lpf, dlf = m.compile_logp(), m.compile_dlogp()
    rng = np.random.default_rng(11)
    for s in np.repeat([0.1, 0.5, 1.0, 2.0, 3.0], 5):
        x = rng.normal(scale=s, size=10); g = np.empty(10); lp = np.empty(1)
        assert lib.es8_logp_grad(None, P(x.ctypes.data), P(g.ctypes.data), P(lp.ctypes.data)) == 0
        p = es8.point_to_dict(x)
        np.testing.assert_allclose(lp[0], lpf(p), rtol=1e-10)
        np.testing.assert_allclose(g, dlf(p), rtol=1e-10, atol=1e-12)


@pytest.fixture(scope="module")
def hlr():
    d = hlr_ref.make_data()
    m = hlr_ref.build_model(d)
    return d, m.compile_logp(), m.compile_dlogp(), hlr_ref.MojoHLR(hlr_ref.pack(d))


def test_hlr_capi_matches_pymc(hlr):
    d, lpf, dlf, mj = hlr
    assert mj.dim == d["G"] + d["K"] + 3 == 28
    rng = np.random.default_rng(12)
    for s in np.repeat([0.1, 0.5, 1.0, 2.0], 5):
        x = rng.normal(scale=s, size=mj.dim)
        rc, lp, g = mj(x)
        p = hlr_ref.point_to_dict(x, d["G"], d["K"])
        assert rc == 0
        np.testing.assert_allclose(lp, lpf(p), rtol=1e-10)
        np.testing.assert_allclose(g, dlf(p), rtol=1e-10, atol=1e-9)


def test_hlr_handles_group_sizes_not_multiple_of_simd_width():
    d = hlr_ref.make_data(N=1003, G=7, K=3, seed=5)  # ragged groups exercise the scalar tail
    m = hlr_ref.build_model(d); lpf, dlf = m.compile_logp(), m.compile_dlogp()
    mj = hlr_ref.MojoHLR(hlr_ref.pack(d))
    x = np.random.default_rng(0).normal(size=mj.dim)
    rc, lp, g = mj(x)
    p = hlr_ref.point_to_dict(x, d["G"], d["K"])
    np.testing.assert_allclose(lp, lpf(p), rtol=1e-10)
    np.testing.assert_allclose(g, dlf(p), rtol=1e-10, atol=1e-9)
