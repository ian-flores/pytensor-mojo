import pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mojo_src"))
import q1_mod  # noqa: E402


def test_noops():
    assert q1_mod.noop0() == 0.0
    assert q1_mod.noop1(3.5) == 3.5


def test_zero_copy_reads_and_writes():
    x = np.arange(37, dtype=np.float64)
    assert q1_mod.first_span(x) == 0.0
    x[0] = 7.0
    assert q1_mod.first_ptr(x.ctypes.data) == 7.0
    assert q1_mod.sum_span(x) == x.sum()
    assert np.isclose(q1_mod.sum_ptr(x.ctypes.data, len(x)), x.sum())
    y = np.ones(37); out = np.zeros(37)
    q1_mod.axpy_ptr(2.0, x.ctypes.data, y.ctypes.data, out.ctypes.data, 37)
    np.testing.assert_array_equal(out, 2.0 * x + y)
    out[:] = 0
    q1_mod.axpy_span(2.0, x, y, out)
    np.testing.assert_array_equal(out, 2.0 * x + y)


def test_exp_accuracy_std_vs_libm():
    """Documents a Mojo 1.2 finding: std.math.exp(Float64) is not correctly rounded."""
    xs = np.linspace(-20, 20, 401)
    ref = np.exp(xs)
    std_err = np.array([abs(q1_mod.exp_std(x) - r) / r for x, r in zip(xs, ref)])
    libm_err = np.array([abs(q1_mod.exp_libm(x) - r) / r for x, r in zip(xs, ref)])
    assert libm_err.max() <= 2.3e-16
    print(f"std.math.exp max rel err {std_err.max():.2e}; libm {libm_err.max():.2e}")
    assert std_err.max() > 1e-13  # if this fails, Mojo fixed exp: drop the libm workaround
