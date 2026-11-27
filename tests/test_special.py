"""Phase 4: accuracy and gradients of generated special functions.

Accuracy thresholds (max over each grid; measured values are in docs/FINDINGS.md):
  * libm-backed ops (gammaln x>0, erf, expm1, pow, log1mexp pieces): <= 4 ulp (erfc: <= 6,
    macOS libm measured 5 ulp at x in (5, 26.5); scipy is ~90x worse there) vs a
    high-precision or correctly rounded reference. These are libm's documented sub-ulp to
    few-ulp accuracy; the grids include the zeros of lgamma (x = 1, 2) and erfc's tail.
  * psi x > 0: <= 20 ulp. In the recurrence region just outside the root window
    (x ~ 1.75, psi ~ 0.25) the result is log(y) - sum(1/k) with |result| ~ 0.1 |log y|,
    so the 0.5-ulp rounding of log(y) alone is ~8 ulp of the result.
  * negative x (psi by reflection, gammaln by libm): <= 2e-13 relative, at least 1e-3 from
    the poles; the problem's condition number there is ~ 1 / distance-to-pole.
"""
import pathlib, sys
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
import phase4_accuracy  # noqa: E402
from pytensor_mojo.ndlinker import MOJO_ND  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return {(r[0], r[1]): r for r in phase4_accuracy.run()}


def test_accuracy_table(rows):
    for (name, label), (_, _, n, max_ulp, _, max_rel, _) in rows.items():
        neg = label.startswith("negative")
        if neg:
            assert max_rel <= 2e-13, (name, label, max_rel)
        elif name == "psi":
            assert max_ulp <= 20, (name, label, max_ulp)
        else:
            limit = 6 if name == "erfc" else 4   # macOS libm erfc measured <= 5 ulp in the tail
            assert max_ulp <= limit, (name, label, max_ulp)


def test_special_values():
    x = pt.dvector("x")
    f = pytensor.function([x], [pt.psi(x), pt.gammaln(x)], mode=MOJO_ND)
    psi, lg = f(np.array([0.0, -0.0, -1.0, -2.0, np.inf, 1.0, 2.0]))
    assert psi[0] == -np.inf and psi[1] == np.inf and np.isnan(psi[2]) and np.isnan(psi[3])
    assert psi[4] == np.inf
    np.testing.assert_allclose(psi[5:], [-0.5772156649015329, 0.42278433509846713], rtol=1e-15)
    assert lg[0] == np.inf and lg[2] == np.inf and lg[5] == 0.0 and lg[6] == 0.0


GRAD_CASES = {
    "gammaln": (pt.gammaln, lambda r: np.exp(r.uniform(-5, 4, 61))),
    "erf": (pt.erf, lambda r: r.uniform(-4, 4, 61)),
    "erfc": (pt.erfc, lambda r: r.uniform(-4, 8, 61)),
    "expm1": (pt.expm1, lambda r: r.uniform(-20, 20, 61)),
    "log1mexp": (pt.log1mexp, lambda r: -np.exp(r.uniform(-8, 3, 61))),
    "sign*x^2": (lambda v: pt.sign(v) * v ** 2, lambda r: r.normal(size=61)),
}


@pytest.mark.parametrize("name", list(GRAD_CASES))
def test_gradients_vs_c_backend(name):
    op, gen = GRAD_CASES[name]
    x = pt.dvector("x")
    cost = pt.sum(op(x) * pt.arange(1, 62))
    outs = [cost, pytensor.grad(cost, x)]
    f = pytensor.function([x], outs, mode=MOJO_ND)
    g = pytensor.function([x], outs)
    xv = gen(np.random.default_rng(3))
    for a, b in zip(f(xv), g(xv)):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_pow_gradients_both_args():
    x, y = pt.dvector("x"), pt.dvector("y")
    cost = pt.sum(x ** y)
    outs = [cost, *pytensor.grad(cost, [x, y])]
    r = np.random.default_rng(4)
    xv, yv = np.exp(r.uniform(-3, 3, 37)), r.uniform(-4, 4, 37)
    for a, b in zip(pytensor.function([x, y], outs, mode=MOJO_ND)(xv, yv), pytensor.function([x, y], outs)(xv, yv)):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_psi_gradient_needs_trigamma_and_raises_clearly():
    """grad(psi) = tri_gamma; NUTS needs only first order, so TriGamma is not implemented."""
    x = pt.dvector("x")
    with pytest.raises(NotImplementedError, match="TriGamma"):
        pytensor.function([x], pytensor.grad(pt.sum(pt.psi(x)), x), mode=MOJO_ND)
