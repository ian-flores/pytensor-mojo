"""Phase 7: vectorized exp, log, log1p, expm1, lgamma (pytensor_mojo.mojo_math).

Faster subset of bench/phase7_accuracy.py (which runs 10M points per function): 20k points per
function from the same random and edge-case sets, against the 40-digit Decimal reference.
Limits: <= 1 ulp for exp, log, log1p, expm1 (fdlibm algorithms, documented < 1 ulp; measured
max 0.87 ulp over 10M points); <= 4 ulp for lgamma at x > 0 (measured max about 3 ulp).
"""
import os, pathlib, subprocess, sys
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
import phase7_accuracy as p7  # noqa: E402
from pytensor_mojo.ndlinker import MOJO_ND  # noqa: E402

LIMIT = {"exp": 1.0, "log": 1.0, "log1p": 1.0, "expm1": 1.0, "gammaln": 4.0}


@pytest.mark.parametrize("name", list(LIMIT))
def test_accuracy_subset(name):
    rows = p7.measure(name, 20_000 if name != "gammaln" else 8_000, workers=4)
    for r in rows:
        assert r["max_ulp"] <= LIMIT[name], r


def test_special_values():
    x = pt.dvector("x")
    f = pytensor.function([x], [pt.exp(x), pt.log(x), pt.log1p(x), pt.expm1(x), pt.gammaln(x)], mode=MOJO_ND)
    inf, nan = np.inf, np.nan
    xs = np.array([-inf, -800.0, -1.0, -0.0, 0.0, 5e-324, 1.0, 2.0, 709.78, 710.0, inf, nan])
    e, l, l1, em, lg = f(xs)
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(e[[0, 1, 9, 10]], [0.0, 0.0, inf, inf])
        assert np.isnan(e[11]) and np.isfinite(e[8])
        assert np.isnan(l[2]) and l[3] == -inf and l[4] == -inf and l[10] == inf and np.isnan(l[11])
        np.testing.assert_allclose(l[5], np.log(5e-324), rtol=1e-15)          # subnormal input
        assert l1[2] == -inf and np.isnan(l1[11]) and l1[10] == inf and l1[4] == 0.0
        assert em[0] == -1.0 and em[1] == -1.0 and em[9] == inf and em[10] == inf and np.isnan(em[11])
        assert lg[6] == 0.0 and lg[7] == 0.0 and lg[10] == inf and np.isnan(lg[11])
        assert lg[4] == inf                                                     # pole at 0 (libm path)


def test_libm_fallback_flag_switches_generated_source():
    code = ("import sys; sys.path.insert(0, '.');"
            "import pytensor, pytensor.tensor as pt;"
            "from pytensor_mojo import mojo_math as mm;"
            "print(mm.LIBM, mm.call('exp', 'W', 'x'), mm.call('gammaln', 'W', 'x'))")
    env = dict(os.environ, PYTENSOR_MOJO_LIBM="1")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True).stdout
    assert out.split()[0] == "True" and "vexp" in out and "vlgamma" in out
    env["PYTENSOR_MOJO_LIBM"] = "0"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True).stdout
    assert out.split()[0] == "False" and "sexp" in out and "slgamma" in out


def test_isnan_op_is_bitwise():
    """Mojo's SIMD .ne() is ordered, so x != x was False for NaN; IsNan now tests bits."""
    x = pt.dvector("x")
    one, zero = pt.constant(np.float64(1)), pt.constant(np.float64(0))
    f = pytensor.function([x], pt.switch(pt.isnan(x), one, zero), mode=MOJO_ND)
    np.testing.assert_array_equal(f(np.array([np.nan, 1.0, np.inf, -np.nan, 0.0] * 3)), [1, 0, 0, 1, 0] * 3)
