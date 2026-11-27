"""Extension ops (pytensor_mojo/ndcodegen_ext.py) vs the C backend, plus their gradients.

These ops were added for the posteriordb sweep: IncSubtensor / SetSubtensor with constant
indices, stepped Subtensor, CumOp, AllocEmpty, SpecifyShape, max / min reductions, N-d All
and the scalar op Second. Model-level checks against PyMC are in tests/test_posteriordb.py.
"""
import pathlib, sys
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pytensor_mojo import ndcodegen_ext  # noqa: E402
from pytensor_mojo.ndcodegen import missing_ops  # noqa: E402
from pytensor_mojo.ndlinker import MOJO_ND  # noqa: E402

rng = np.random.default_rng(7)


def same_as_c(ins, outs, vals, rtol=1e-12):
    f = pytensor.function(ins, outs, mode=MOJO_ND)
    g = pytensor.function(ins, outs)
    for a, b in zip(f(*vals), g(*vals)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-13)
    return f


def _ops_in(f):
    return {type(n.op).__name__ for n in f.maker.fgraph.apply_nodes}


def test_inc_and_set_subtensor_constant_indices():
    x, y = pt.dvector("x", shape=(9,)), pt.dvector("y", shape=(3,))
    M = pt.dmatrix("M", shape=(4, 5))
    outs = [pt.inc_subtensor(x[2], y[0]), pt.set_subtensor(x[-1], y[1]), pt.inc_subtensor(x[1:4], y),
            pt.set_subtensor(x[6:], y * 2), pt.inc_subtensor(M[1], y[0]), pt.set_subtensor(M[:, 2], x[:4]),
            pt.inc_subtensor(M[1:3, 1:4], y)]
    f = same_as_c([x, y, M], outs, [rng.normal(size=9), rng.normal(size=3), rng.normal(size=(4, 5))])
    assert "IncSubtensor" in _ops_in(f)


def test_gradient_of_indexed_parameters():
    """The graph PyMC builds for beta[0] + beta[1] * x: grad is IncSubtensor into zeros."""
    b, xs = pt.dvector("b", shape=(3,)), pt.constant(rng.normal(size=50))
    lp = -pt.sum((b[0] + b[1] * xs + b[2] * xs ** 2 - 1.0) ** 2)
    same_as_c([b], [lp, pytensor.grad(lp, b)], [rng.normal(size=3)])


def test_stepped_subtensor_and_cumsum_gradient():
    x = pt.dvector("x", shape=(11,))
    M = pt.dmatrix("M", shape=(3, 6))
    # views of inputs are scaled so that the outputs are not bare views (DeepCopyOp)
    outs = [x[::-1] * 2, x[1::3] * 2, x[8:1:-2] * 2, M[:, ::-2] * 2, pt.cumsum(x), pt.cumprod(x[:5]), pt.cumsum(M, axis=0),
            pt.cumsum(M, axis=1)]   # flattened cumsum of a matrix needs Reshape (unsupported)
    same_as_c([x, M], outs, [rng.normal(size=11), rng.normal(size=(3, 6))])
    lp = pt.sum(pt.cumsum(pt.exp(x)) ** 2)
    same_as_c([x], [pytensor.grad(lp, x)], [rng.normal(size=11) * 0.3])


def test_ordered_transform_graph():
    """PyMC's ordered transform (backward + log-Jacobian) and its gradient."""
    import pymc as pm
    y = pt.dvector("y", shape=(4,))
    tr = pm.distributions.transforms.ordered
    x = tr.backward(y)
    lp = -pt.sum((x - pt.arange(4.0)) ** 2) + tr.log_jac_det(y).sum()
    f = same_as_c([y], [x, lp, pytensor.grad(lp, y)], [rng.normal(size=4)])
    assert not any(missing_ops(f.maker.fgraph))


def test_max_min_reductions_and_logsumexp():
    x = pt.dvector("x", shape=(13,))
    M = pt.dmatrix("M", shape=(4, 7))
    outs = [pt.max(x), pt.min(x), pt.max(M, axis=0), pt.min(M, axis=1), pt.logsumexp(M, axis=0),
            pt.logsumexp(x)]
    same_as_c([x, M], outs, [rng.normal(size=13), rng.normal(size=(4, 7))])
    lp = pt.logsumexp(x * 3.0)
    same_as_c([x], [pytensor.grad(lp, x)], [rng.normal(size=13)])


def test_all_over_matrix_and_check():
    M = pt.dmatrix("M", shape=(5, 6))
    s = M.sum()
    outs = [pt.all(pt.ge(M, -10.0)) * s, pt.all(pt.ge(M, 0.0)) * s]
    same_as_c([M], outs, [rng.normal(size=(5, 6))])


def test_alloc_empty_and_specify_shape():
    x = pt.dvector("x", shape=(6,))
    e = pt.empty((6,), dtype="float64")
    out = pt.set_subtensor(e[:], x * 2) + pt.specify_shape(x, (6,))
    same_as_c([x], [out], [rng.normal(size=6)])


def test_materialized_length_one_dim_broadcasts_in_nd_elemwise():
    """Regression: a computed (1, n) buffer broadcast against (m, n) (logsumexp along axis 0)."""
    M = pt.dmatrix("M", shape=(4, 7))
    same_as_c([M], [pt.logsumexp(M, axis=0), pt.exp(M - pt.switch(pt.isinf(M.max(axis=0, keepdims=True)), 0.0,
                                                                   M.max(axis=0, keepdims=True)))],
              [rng.normal(size=(4, 7))])


def test_length_one_view_broadcasts_in_fused_loop():
    """Regression: a length-1 view q[1:] feeding a fused loop must broadcast, not stride."""
    q = pt.dvector("q", shape=(2,))
    X = pt.constant(rng.normal(size=(37, 1)))
    lp = pt.sum((q[0] + pt.dot(X, q[1:]) - 0.5) ** 2)
    same_as_c([q], [lp, pytensor.grad(lp, q)], [rng.normal(size=2)])


def test_disabled_flag_restores_baseline_census():
    b = pt.dvector("b", shape=(3,))
    g = pytensor.grad(pt.sum(b[0] * b[1]), b)
    fg = pytensor.function([b], g, mode=MOJO_ND).maker.fgraph
    assert missing_ops(fg) == ([], [])
    was = ndcodegen_ext.DISABLED
    ndcodegen_ext.DISABLED = True
    try:
        assert "IncSubtensor" in missing_ops(fg)[0]
    finally:
        ndcodegen_ext.DISABLED = was


@pytest.mark.parametrize("w", ["1", "8"])
def test_second_scalar_op(w):
    from pytensor.scalar import basic as ps
    assert ndcodegen_ext.scalar_table(w)[ps.Second]("a", "b") == "b"
