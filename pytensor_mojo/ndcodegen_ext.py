"""Extension ops for ``ndcodegen`` (added by the posteriordb sweep).

Kept in a separate module so that adding ops does not touch the existing dispatch in
``ndcodegen.Codegen.node``. ``ndcodegen`` consults this module in three places:

- ``Codegen.node`` calls :func:`claims` first and, when it returns True, :func:`emit`;
- ``Codegen.scalar_expr`` merges :func:`scalar_table` into its scalar-op table;
- ``missing_ops`` treats claimed nodes and :func:`scalar_table` ops as supported.

Ops handled here:

- ``IncSubtensor`` / ``SetSubtensor`` with constant integer indices and constant slices
  (the gradient of ``beta[i]``, ``x[a:b]`` and PyMC's ordered transform);
- ``Subtensor`` with a non-unit constant step (``x[::-1]``, the gradient of ``cumsum``) on
  static dims; unit-step Subtensor stays in ``ndcodegen``;
- ``CumOp`` (cumsum / cumprod) along one axis or flattened;
- ``AllocEmpty`` (zero-filled workspace);
- ``SpecifyShape`` (a no-op view);
- ``CAReduce`` with ``Maximum`` / ``Minimum`` (``max`` / ``min``, used by ``logsumexp``);
- ``All`` over a materialized N-d (ndim >= 2) bool array, e.g. the parameter check of a
  Bernoulli likelihood with matrix-shaped data (1-D ``All`` is fused by ``ndcodegen``);
- the scalar op ``Second`` (``second(a, b) = b`` broadcast to ``a``).
"""
from __future__ import annotations

import os

from pytensor.graph.basic import Constant
from pytensor.scalar import basic as ps
from pytensor.tensor.basic import AllocEmpty
from pytensor.tensor.elemwise import CAReduce
from pytensor.tensor.extra_ops import CumOp
from pytensor.tensor.shape import SpecifyShape
from pytensor.tensor.subtensor import IncSubtensor, Subtensor, indices_from_subtensor

# Set PYTENSOR_MOJO_NO_EXT=1 to disable every op in this module (baseline census).
DISABLED = os.environ.get("PYTENSOR_MOJO_NO_EXT", "0") == "1"

# Tensor ops handled here, with a note for the generated docs table.
TENSOR_NOTES = {
    "IncSubtensor": "set/inc into a copy of the base, constant int indices and constant slices (`ndcodegen_ext`)",
    "Subtensor (stepped)": "non-unit constant step on static dims, as a strided view (`ndcodegen_ext`)",
    "CumOp": "cumsum / cumprod along one axis or flattened (`ndcodegen_ext`)",
    "AllocEmpty": "zero-filled workspace (`ndcodegen_ext`)",
    "SpecifyShape": "no-op view (`ndcodegen_ext`)",
    "CAReduce[Maximum/Minimum]": "max / min over all or some axes (`ndcodegen_ext`)",
    "All (N-d)": "AND over a materialized bool array of ndim >= 2 (`ndcodegen_ext`)",
}

# ------------------------------------------------------------------ scalar ops
_SCALAR = {
    ps.Second: lambda w: (lambda a, b: b),
}


def scalar_table(w):
    """Extra scalar-op templates, keyed by PyTensor scalar op class (same shape as ``_sop_table``)."""
    if DISABLED:
        return {}
    return {cls: f(w) for cls, f in _SCALAR.items()}


# ------------------------------------------------------------------ claiming
def _const_int(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, Constant):
        return int(v.data)
    try:
        return int(v)
    except TypeError:
        return None


def _indices_static(x, idx):
    """True if every index is a constant int or a slice of constants (None allowed)."""
    for it in idx:
        if isinstance(it, slice):
            for b in (it.start, it.stop, it.step):
                if b is not None and _const_int(b) is None:
                    return False
        elif _const_int(it) is None:
            return False
    return True


def _has_step(idx):
    return any(isinstance(it, slice) and _const_int(it.step) not in (None, 1) for it in idx)


def _stepped_dims_static(x, idx):
    return all(x.type.shape[d] is not None for d, it in enumerate(idx)
               if isinstance(it, slice) and _const_int(it.step) not in (None, 1))


def claims(node) -> bool:
    """Whether this module emits ``node`` (checked before ``ndcodegen``'s own dispatch)."""
    if DISABLED:
        return False
    op = node.op
    if isinstance(op, IncSubtensor):
        idx = indices_from_subtensor(node.inputs[2:], op.idx_list)
        x = node.inputs[0]
        return x.type.dtype == "float64" and _indices_static(x, idx) and _stepped_dims_static(x, idx)
    if isinstance(op, Subtensor):
        idx = indices_from_subtensor(node.inputs[1:], op.idx_list)
        x = node.inputs[0]
        return _has_step(idx) and _indices_static(x, idx) and _stepped_dims_static(x, idx)
    if isinstance(op, CumOp):
        return op.mode in ("add", "mul") and node.inputs[0].type.dtype == "float64"
    if isinstance(op, AllocEmpty):
        return op.dtype == "float64"
    if isinstance(op, SpecifyShape):
        return True
    if isinstance(op, CAReduce) and isinstance(op.scalar_op, (ps.Maximum, ps.Minimum)):
        return node.inputs[0].type.dtype == "float64"
    if isinstance(op, CAReduce) and isinstance(op.scalar_op, ps.AND):
        x = node.inputs[0]
        return x.type.ndim >= 2 and (op.axis is None or len(op.axis) == x.type.ndim)
    return False


def emit(cg, node):
    op = node.op
    if isinstance(op, IncSubtensor):
        return _inc_subtensor(cg, node)
    if isinstance(op, Subtensor):
        return _stepped_subtensor(cg, node)
    if isinstance(op, CumOp):
        return _cumop(cg, node)
    if isinstance(op, AllocEmpty):
        return _alloc_empty(cg, node)
    if isinstance(op, SpecifyShape):
        cg.vals[node.outputs[0]] = cg.vals[node.inputs[0]]
        return None
    if isinstance(op, CAReduce) and isinstance(op.scalar_op, ps.AND):
        return _reduce_all_nd(cg, node)
    if isinstance(op, CAReduce):
        return _reduce_maxmin(cg, node)
    raise NotImplementedError(f"ndcodegen_ext: {op}")


# ------------------------------------------------------------------ helpers
def _nd():
    from . import ndcodegen as nd
    return nd


def _materialize(cg, var):
    """Return a Buf for ``var`` (Scal / Bcast values are written into a fresh buffer)."""
    nd = _nd()
    val = cg.vals[var]
    if isinstance(val, nd.Buf):
        return val
    shape = ["1"] * var.type.ndim
    ptr = cg.alloc("float64", shape)
    b = nd.Buf(ptr, "float64", shape, cg.c_strides(shape))
    cg.copy_into(b, val, shape)
    return b


def _view(cg, val, x, idx):
    """(shape, strides, offset) of ``val[idx]`` for constant indices; ``val`` is a Buf of ``x``."""
    nd = _nd()
    shape, st, off = [], [], val.offset
    d = 0
    for it in idx:
        n = val.shape[d]
        stride = val.strides[d]
        if isinstance(it, slice):
            start, stop, step = (_const_int(b) for b in (it.start, it.stop, it.step))
            if step in (None, 1):
                s0 = "0" if start is None else (str(start) if start >= 0 else f"({n} - {-start})")
                s1 = n if stop is None else (str(stop) if stop >= 0 else f"({n} - {-stop})")
                if x.type.shape[d] is not None:
                    a, b, _ = slice(start, stop, 1).indices(int(x.type.shape[d]))
                    s0, s1 = str(a), str(max(b, a))
                off = nd._add(off, nd._mul(s0, stride))
                shape.append(f"({s1} - {s0})" if s0 != "0" else s1)
                st.append(stride)
            else:
                ns = int(x.type.shape[d])
                a, b, s = slice(start, stop, step).indices(ns)
                length = len(range(a, b, s))
                off = nd._add(off, nd._mul(str(a), stride))
                shape.append(str(length))
                st.append(nd._mul(f"({s})", stride))
        else:
            i = _const_int(it)
            ie = str(i) if i >= 0 else f"({n} - {-i})"
            off = nd._add(off, nd._mul(ie, stride))
        d += 1
    shape += list(val.shape[d:])
    st += list(val.strides[d:])
    return shape, st, off


def _loop(cg, shape):
    idx = [cg.fresh("j") for _ in shape]
    for ix, n in zip(idx, shape):
        cg.emit(f"for {ix} in range({n}):")
        cg.ind += 1
    return idx


def _src_expr(cg, yvar, yval, out_ndim, idx):
    """Read ``y`` broadcast to the view's shape at loop indices ``idx``."""
    nd = _nd()
    if not isinstance(yval, nd.Buf):
        return cg.scalar_of(yval)
    pad = out_ndim - yval.ndim
    own = idx[pad:]
    use = ["0" if yvar.type.shape[d] == 1 else i for d, i in enumerate(own)]
    return cg.scalar_of(yval, use)


# ------------------------------------------------------------------ ops
def _inc_subtensor(cg, node):
    nd = _nd()
    x, y = node.inputs[:2]
    idx = indices_from_subtensor(node.inputs[2:], node.op.idx_list)
    xb = _materialize(cg, x)
    shape = [str(s) if s is not None else e for s, e in zip(x.type.shape, xb.shape)]
    ptr = cg.alloc("float64", shape)
    out = nd.Buf(ptr, "float64", shape, cg.c_strides(shape))
    cg.copy_into(out, xb, shape)
    vshape, vst, voff = _view(cg, out, x, idx)
    yval = cg.vals[y]
    opr = "=" if node.op.set_instead_of_inc else "+="
    view = nd.Buf(ptr, "float64", vshape, vst, voff)
    if not vshape:
        cg.emit(f"{ptr}[unsafe_offset={voff}] {opr} {cg.scalar_of(yval) if not isinstance(yval, nd.Buf) else cg.scalar_of(yval, ['0'] * yval.ndim)}")
    else:
        li = _loop(cg, vshape)
        cg.emit(f"{ptr}[unsafe_offset={view.index(li)}] {opr} {_src_expr(cg, y, yval, len(vshape), li)}")
        cg.ind -= len(vshape)
    cg.vals[node.outputs[0]] = out


def _stepped_subtensor(cg, node):
    nd = _nd()
    x = node.inputs[0]
    val = cg.vals[x]
    if not isinstance(val, nd.Buf):
        raise NotImplementedError("stepped Subtensor of non-buffer")
    idx = indices_from_subtensor(node.inputs[1:], node.op.idx_list)
    shape, st, off = _view(cg, val, x, idx)
    o = node.outputs[0]
    if not shape:
        s = cg.fresh("s")
        cg.emit(f"var {s} = {val.ptr}[unsafe_offset={off}]")
        cg.vals[o] = nd.Scal(s, val.dtype)
    else:
        cg.vals[o] = nd.Buf(val.ptr, val.dtype, shape, st, off, mask=val.mask)


def _cumop(cg, node):
    nd = _nd()
    (x,), (o,) = node.inputs, node.outputs
    xb = _materialize(cg, x)
    axis = node.op.axis
    if axis is None:
        n = nd._prod(xb.shape) if xb.shape else "1"
        shape = [str(o.type.shape[0]) if o.type.shape[0] is not None else n]
        ptr = cg.alloc("float64", shape)
        # C-order flatten of x into the output, then a running sum / product
        tmp = nd.Buf(ptr, "float64", list(xb.shape), cg.c_strides(list(xb.shape)))
        cg.copy_into(tmp, xb, list(xb.shape))
        out = nd.Buf(ptr, "float64", shape, ["1"])
        ax, oshape = 0, shape
    else:
        oshape = [str(s) if s is not None else e for s, e in zip(o.type.shape, xb.shape)]
        ptr = cg.alloc("float64", oshape)
        out = nd.Buf(ptr, "float64", oshape, cg.c_strides(oshape))
        cg.copy_into(out, xb, oshape)
        ax = axis % len(oshape)
    opr = "+=" if node.op.mode == "add" else "*="
    li = _loop(cg, oshape)
    prev = list(li)
    prev[ax] = f"({li[ax]} - 1)"
    cg.emit(f"if {li[ax]} > 0:")
    cg.emit(f"    {ptr}[unsafe_offset={out.index(li)}] {opr} {ptr}[unsafe_offset={out.index(prev)}]")
    cg.ind -= len(oshape)
    cg.vals[o] = out


def _alloc_empty(cg, node):
    nd = _nd()
    o = node.outputs[0]
    shape = []
    for d, s in zip(node.inputs, o.type.shape):
        if s is not None:
            shape.append(str(s))
        else:
            c = _const_int(d)
            shape.append(str(c) if c is not None else f"Int({cg.scalar_of(cg.vals[d])})")
    ptr = cg.alloc("float64", shape)
    b = nd.Buf(ptr, "float64", shape, cg.c_strides(shape))
    if shape:
        li = _loop(cg, shape)
        cg.emit(f"{ptr}[unsafe_offset={b.index(li)}] = 0.0")
        cg.ind -= len(shape)
        cg.vals[o] = b
    else:
        s = cg.fresh("s")
        cg.emit(f"var {s} = Float64(0)")
        cg.vals[o] = nd.Scal(s, "float64")


def _reduce_maxmin(cg, node):
    nd = _nd()
    (x,), (o,) = node.inputs, node.outputs
    val = cg.vals[x]
    is_max = isinstance(node.op.scalar_op, ps.Maximum)
    fn, init = ("max", "NEG_INF") if is_max else ("min", "POS_INF")
    if not isinstance(val, nd.Buf):
        cg.vals[o] = val if isinstance(val, nd.Scal) else val.scal
        return
    ndim = val.ndim
    axis = node.op.axis
    axes = tuple(range(ndim)) if axis is None else tuple(a % ndim for a in axis)
    keep = [d for d in range(ndim) if d not in axes]
    if not keep:
        s = cg.fresh("s")
        cg.emit(f"var {s} = Float64({init})")
        li = _loop(cg, val.shape)
        cg.emit(f"{s} = {fn}({s}, {cg.scalar_of(val, li)})")
        cg.ind -= ndim
        cg.vals[o] = nd.Scal(s, "float64")
        return
    oshape = [val.shape[d] for d in keep]
    ptr = cg.alloc("float64", oshape)
    ob = nd.Buf(ptr, "float64", oshape, cg.c_strides(oshape))
    li = _loop(cg, oshape)
    cg.emit(f"{ptr}[unsafe_offset={ob.index(li)}] = Float64({init})")
    cg.ind -= len(oshape)
    li = _loop(cg, val.shape)
    oi = ob.index([li[d] for d in keep])
    cg.emit(f"{ptr}[unsafe_offset={oi}] = {fn}({ptr}[unsafe_offset={oi}], {cg.scalar_of(val, li)})")
    cg.ind -= ndim
    cg.vals[o] = ob


def _reduce_all_nd(cg, node):
    nd = _nd()
    (x,), (o,) = node.inputs, node.outputs
    val = cg.vals[x]
    if not isinstance(val, nd.Buf):
        cg.vals[o] = val if isinstance(val, nd.Scal) else val.scal
        return
    s = cg.fresh("s")
    cg.emit(f"var {s} = True")
    li = _loop(cg, val.shape)
    cg.emit(f"if not Bool({cg.scalar_of(val, li)}):")
    cg.emit(f"    {s} = False")
    cg.ind -= val.ndim
    r = cg.fresh("s")
    cg.emit(f"var {r} = SIMD[DType.bool, 1]({s})")
    cg.vals[o] = nd.Scal(r, "bool")
