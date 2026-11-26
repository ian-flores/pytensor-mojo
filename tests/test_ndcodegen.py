"""Phase 3: generated N-d / fused / SIMD Mojo code (pytensor_mojo.ndcodegen) vs C backend and PyMC."""
import pathlib, sys
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pytensor_mojo import eight_schools_ref as es8, hlr_ref  # noqa: E402
from pytensor_mojo.ndlinker import MOJO_ND, CapiModel, compile_model_capi  # noqa: E402

rng = np.random.default_rng(123)


def same_as_c(ins, outs, vals, rtol=1e-12):
    f = pytensor.function(ins, outs, mode=MOJO_ND)
    g = pytensor.function(ins, outs)
    for a, b in zip(f(*vals), g(*vals)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-13)
    return f


def test_nd_broadcasting_runtime_shapes():
    M, v, s, T = pt.dmatrix("M"), pt.dvector("v"), pt.dscalar("s"), pt.dtensor3("T")
    outs = [pt.exp(M) * v[None, :] + s, (M.T - v[:, None]) ** 2, pt.log1p(T ** 2) + v,
            (T * 2).sum(axis=1), T.sum(axis=(0, 2)), T.sum()]
    f = pytensor.function([M, v, s, T], outs, mode=MOJO_ND)
    g = pytensor.function([M, v, s, T], outs)
    for n in (13, 8, 1, 3):  # same compiled code, different runtime shapes (incl. < SIMD width)
        vals = [rng.normal(size=(3, n)), rng.normal(size=n), 0.7, rng.normal(size=(2, 5, n))]
        for a, b in zip(f(*vals), g(*vals)):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("sorted_idx", [False, True])
def test_gather_scatter_vs_c_backend(sorted_idx):
    a7 = pt.dvector("a7", shape=(7,))
    a, idx, w = pt.dvector("a"), pt.lvector("idx"), pt.dvector("w")
    # 61 elements: 7 full SIMD chunks + tail; sorted runs cross chunk boundaries
    iv = rng.integers(0, 7, size=61)
    iv = np.sort(iv) if sorted_idx else iv
    vals = [rng.normal(size=7), iv, rng.normal(size=61)]
    same_as_c([a, idx], [a[idx]], vals[:2])                                     # unfused gather
    same_as_c([a, idx, w], [pt.inc_subtensor(a[idx], w)], vals)                 # unfused scatter
    same_as_c([a, idx, w], [pt.sum((a[idx] - w) ** 2)], vals)                   # fused gather+sum
    same_as_c([a7, idx, w], [pytensor.grad(pt.sum((a7[idx] - w) ** 2), a7)], vals)  # fused scatter


def test_fused_rowdot_coldot_vs_c_backend():
    X = pt.constant(rng.normal(size=(41, 3)))
    b, y = pt.dvector("b", shape=(3,)), pt.dvector("y", shape=(41,))
    r = y - X @ b
    f = same_as_c([b, y], [pt.sum(r ** 2), pytensor.grad(pt.sum(r ** 2), b)],
                  [rng.normal(size=3), rng.normal(size=41)])
    src = f.vm.jit_fn.mojo_source
    assert "unsafe_load[width=W]" in src  # SIMD main loop emitted


def test_unsupported_op_raises_nd():
    x = pt.dvector("x")
    with pytest.raises(NotImplementedError, match="unsupported op"):
        pytensor.function([x], pt.sort(x), mode=MOJO_ND)


def _check_model(model, p2d, name, n=24):
    so, data, dim, info = compile_model_capi(model, name)
    cm = CapiModel(so, name, data)
    lpf, dlf = model.compile_logp(), model.compile_dlogp()
    r = np.random.default_rng(5)
    for s in np.repeat([0.1, 0.5, 1.0, 2.0], n // 4):
        x = r.normal(scale=s, size=dim)
        rc, lp, g = cm(x)
        assert rc == 0
        np.testing.assert_allclose(lp, lpf(p2d(x)), rtol=1e-10)
        np.testing.assert_allclose(g, dlf(p2d(x)), rtol=1e-10, atol=1e-9)
    return info


def test_generated_es8_matches_pymc():
    _check_model(es8.build_model(), es8.point_to_dict, "es8gen")


def test_generated_hlr_matches_pymc_and_is_fused():
    d = hlr_ref.make_data()
    info = _check_model(hlr_ref.build_model(d), lambda x: hlr_ref.point_to_dict(x, d["G"], d["K"]), "hlrgen")
    src = info["source"]
    # Fusion evidence: no N=10000-length workspace is materialized (small static stack arena).
    assert "unsafe_alloc[Float64](" not in src.split("def hlrgen_run")[0].split("def _run_hlrgen")[1]
    assert "Array[Float64, " in src
    assert "unsafe_load[width=W]" in src


def test_generated_hlr_ragged_groups():
    d = hlr_ref.make_data(N=1003, G=7, K=3, seed=5)
    _check_model(hlr_ref.build_model(d), lambda x: hlr_ref.point_to_dict(x, d["G"], d["K"]), "hlrgen_ragged", n=20)


def test_check_failure_returns_status():
    """CheckParameterValue failures (e.g. sigma <= 0 in a non-transformed param) give status 1."""
    s = pt.dscalar("s")
    import pymc as pm
    lp = pm.logp(pm.HalfNormal.dist(sigma=s), pt.constant(1.0))
    f = pytensor.function([s], lp, mode=MOJO_ND)
    assert np.isfinite(f(1.0))
    with pytest.raises(ValueError, match="status 1"):
        f(-1.0)
