import pathlib, sys
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pytensor_mojo.eight_schools_ref import J, build_model  # noqa: E402
from pytensor_mojo.linker import MOJO  # noqa: E402


@pytest.fixture(scope="module")
def fns():
    m = build_model()
    lp = m.logp(jacobian=True)
    outs = [lp, *pt.grad(lp, m.value_vars)]
    return (pytensor.function(m.value_vars, outs, mode=MOJO),
            pytensor.function(m.value_vars, outs))


def test_eight_schools_generated_mojo_matches_c_backend(fns):
    f_mojo, f_c = fns
    rng = np.random.default_rng(7)
    for _ in range(25):
        scale = rng.choice([0.1, 1.0, 3.0])
        args = [np.array(rng.normal(scale=scale)), np.array(rng.normal(scale=scale)), rng.normal(scale=scale, size=J)]
        for a, b in zip(f_mojo(*args), f_c(*args)):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_generated_source_is_ascii_and_one_function(fns):
    src = fns[0].vm.jit_fn.mojo_source
    assert src.isascii()
    assert src.count("\ndef run(") == 1


def test_small_elementwise_graph():
    x = pt.dvector("x", shape=(5,)); s = pt.dscalar("s")
    out = [pt.sum(pt.exp(x) * s - pt.log1p(x * x)), pt.switch(x > 0, x, -x) / s]
    f = pytensor.function([x, s], out, mode=MOJO)
    g = pytensor.function([x, s], out)
    xv = np.array([-1.0, 0.5, 2.0, -0.25, 3.0])
    for a, b in zip(f(xv, 2.0), g(xv, 2.0)):
        np.testing.assert_allclose(a, b, rtol=1e-13)


def test_unsupported_op_raises():
    x = pt.dmatrix("x", shape=(3, 3))
    with pytest.raises(NotImplementedError):
        pytensor.function([x], pt.sum(x), mode=MOJO)
