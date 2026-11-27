"""PyTensor FunctionGraph -> Mojo C-ABI shared library (Phase 3 codegen).

Buffer model
------------
Every array is a typed pointer plus a shape, strides and an element offset; each of
these is a Mojo expression string. Static dims (known from the PyTensor type) are
emitted as literals and runtime dims are read from a shape table. Views
(DimShuffle, Squeeze, Subtensor with constant indices, ScalarFromTensor,
CheckParameterValue) never copy. 0-d values are Mojo locals; arrays whose dims
are all broadcastable and whose value is a scalar are "Bcast" values.

Fusion
------
For each Elemwise node E whose output is 1-D of length n, a loop over n is emitted
that also absorbs, when they have no other clients:
  * producers: AdvancedSubtensor gather src[idx[i]], and the row mat-vec
    Squeeze(Dot(A(n,K), ExpandDims(b))), evaluated per element;
  * consumers: Sum over all axes, Squeeze(Dot(AT(K,n), ExpandDims(out))) (K
    accumulators) and AdvancedIncSubtensor scatter-add (accumulated into a zeroed
    temp; base + acc is added when the node is reached), seen through
    CheckParameterValue.
The loop body is emitted twice: at SIMD width W and at width 1 for the tail.

ABI of the generated library <m>
--------------------------------
  int32 <m>_run(int64* tab, int64* shp)          generic, used by MojoNDLinker
      tab: buffer addresses [inputs..., array constants..., outputs...]
      shp: dims of each input, then of each output, concatenated
  int32 <m>_dim(const double* data)              sampler ABI (flat 1-D input x)
  int32 <m>_logp_grad(const double* data, const double* x, double* grad, double* logp)
      data: [n_const, (offset, ndim, dims...) per const, payload...] as float64
            slots (int64 consts are stored bitwise)
Return status: 0 ok, 1 a CheckParameterValue / non-finite condition failed.
"""
from __future__ import annotations

import hashlib
import importlib  # noqa: F401
import math
import os
import pathlib
import subprocess
from dataclasses import dataclass, field

import numpy as np
from pytensor.graph.basic import Constant
from pytensor.raise_op import CheckAndRaise
from pytensor.scalar import basic as ps
from pytensor.scalar import math as psm
from pytensor.tensor.basic import Alloc, Join, MakeVector, ScalarFromTensor, TensorFromScalar
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Dot, Sum
from pytensor.tensor import subtensor as _st
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedSubtensor, Subtensor

# PyTensor 3.x folded the *1 variants into the general ops; keep both names when present.
AdvancedSubtensor1 = getattr(_st, "AdvancedSubtensor1", AdvancedSubtensor)
AdvancedIncSubtensor1 = getattr(_st, "AdvancedIncSubtensor1", AdvancedIncSubtensor)

CACHE = pathlib.Path(os.environ.get("PYTENSOR_MOJO_CACHE", pathlib.Path(__file__).parent / "__mojocache__"))
W = int(os.environ.get("PYTENSOR_MOJO_SIMD", "8"))
# Opt-in: x / splat(s) -> x * splat(1 / s) inside SIMD loops (changes rounding by <= 1-2 ulp).
RECIP = os.environ.get("PYTENSOR_MOJO_RECIP", "0") == "1"
# Opt-in: vectorized std.math.exp instead of per-lane libm exp (~1e-12 relative error in Mojo 1.2).
STDEXP = os.environ.get("PYTENSOR_MOJO_STDEXP", "0") == "1"

MTYPE = {"float64": "Float64", "int64": "Int64", "bool": "Bool"}
DT = {"float64": "DType.float64", "int64": "DType.int64", "bool": "DType.bool"}


def _flit(v) -> str:
    v = float(v)
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "POS_INF" if v > 0 else "NEG_INF"
    return repr(v)


def _prod(xs):
    xs = [x for x in xs if x != "1"]
    return " * ".join(f"({x})" for x in xs) if xs else "1"


def _add(a, b):
    if a == "0":
        return b
    if b == "0":
        return a
    return f"{a} + {b}"


def _mul(a, b):
    if a == "0" or b == "0":
        return "0"
    if a == "1":
        return b
    if b == "1":
        return a
    return f"({a}) * ({b})"


# ---------------------------------------------------------------- values
@dataclass
class Buf:
    ptr: str
    dtype: str
    shape: list
    strides: list
    offset: str = "0"
    mask: bool = False   # float64 storage holding 0/1, read as bool (packed bool constants)

    @property
    def ndim(self):
        return len(self.shape)

    def index(self, idx):
        e = self.offset
        for i, s in zip(idx, self.strides):
            e = _add(e, _mul(i, s))
        return e


@dataclass
class Scal:
    expr: str      # Mojo expression of width-1 SIMD (Float64/SIMD[bool,1])
    dtype: str


@dataclass
class Bcast:       # array of given ndim whose every element is a scalar value
    scal: Scal
    ndim: int


@dataclass
class ScalList:    # MakeVector of bool scalars (only consumed by All)
    items: list
    dtype: str


@dataclass
class Virtual:     # value consumed entirely inside a fused loop; never materialized
    pass


# ---------------------------------------------------------------- scalar ops
def _truediv(w):
    pre = f"SIMD[DType.float64, {w}]("

    def f(a, b):
        if RECIP and w != "1" and b.startswith(pre) and b.endswith(")") and b[len(pre):-1].isidentifier():
            return f"({a} * {pre}Float64(1.0) / {b[len(pre):-1]}))"
        return f"({a} / {b})"
    return f


def _sop_table(w):
    F = f"SIMD[DType.float64, {w}]"
    return {
        ps.Add: lambda *a: "(" + " + ".join(a) + ")",
        ps.Mul: lambda *a: "(" + " * ".join(a) + ")",
        ps.Sub: lambda a, b: f"({a} - {b})",
        ps.TrueDiv: _truediv(w),
        ps.Neg: lambda a: f"(-{a})",
        ps.Sqr: lambda a: f"({a} * {a})",
        ps.Sqrt: lambda a: f"sqrt({a})",
        ps.Exp: lambda a: f"vexp[{w}]({a})",
        ps.Log: lambda a: f"log({a})",
        ps.Log1p: lambda a: f"log1p({a})",
        ps.Reciprocal: lambda a: f"({F}(1.0) / {a})",
        ps.Identity: lambda a: a,
        ps.Abs: lambda a: f"abs({a})",
        ps.Maximum: lambda a, b: f"max({a}, {b})",
        ps.Minimum: lambda a, b: f"min({a}, {b})",
        psm.Sigmoid: lambda a: f"({F}(1.0) / ({F}(1.0) + vexp[{w}](-{a})))",
        psm.Softplus: lambda a: f"(max({a}, {F}(0.0)) + log1p(vexp[{w}](-abs({a}))))",
        ps.GE: lambda a, b: f"{a}.ge({b})",
        ps.GT: lambda a, b: f"{a}.gt({b})",
        ps.LE: lambda a, b: f"{a}.le({b})",
        ps.LT: lambda a, b: f"{a}.lt({b})",
        ps.EQ: lambda a, b: f"{a}.eq({b})",
        ps.NEQ: lambda a, b: f"{a}.ne({b})",
        ps.AND: lambda a, b: f"({a} & {b})",
        ps.OR: lambda a, b: f"({a} | {b})",
        ps.Invert: lambda a: f"(~{a})",
        ps.Switch: lambda c, a, b: f"{c}.select({a}, {b})",
        # Phase 4 special functions
        psm.GammaLn: lambda a: f"vlgamma[{w}]({a})",
        psm.Psi: lambda a: f"vpsi[{w}]({a})",
        psm.Erf: lambda a: f'vlibm1["erf", {w}]({a})',
        psm.Erfc: lambda a: f'vlibm1["erfc", {w}]({a})',
        ps.Expm1: lambda a: f'vlibm1["expm1", {w}]({a})',
        psm.Log1mexp: lambda a: f"vlog1mexp[{w}]({a})",
        ps.Pow: lambda a, b: f"vpow[{w}]({a}, {b})",
        ps.Sign: lambda a: f"vsign[{w}]({a})",
        ps.IsInf: lambda a: f"({a}.eq(SIMD[DType.float64, {w}](POS_INF)) | {a}.eq(SIMD[DType.float64, {w}](NEG_INF)))",
        ps.IsNan: lambda a: f"{a}.ne({a})",
        ps.Floor: lambda a: f'vlibm1["floor", {w}]({a})',
        ps.Ceil: lambda a: f'vlibm1["ceil", {w}]({a})',
        ps.Log2: lambda a: f'vlibm1["log2", {w}]({a})',
        ps.Log10: lambda a: f'vlibm1["log10", {w}]({a})',
        ps.Tanh: lambda a: f'vlibm1["tanh", {w}]({a})',
        ps.Cos: lambda a: f'vlibm1["cos", {w}]({a})',
        ps.Sin: lambda a: f'vlibm1["sin", {w}]({a})',
    }


HEADER = """# Generated by pytensor_mojo.ndcodegen -- do not edit.
from std.math import log, log1p, sqrt
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.collections import Array

comptime W = {W}
comptime POS_INF = Float64(1.0) / Float64(0.0)
comptime NEG_INF = Float64(-1.0) / Float64(0.0)
comptime NAN = Float64(0.0) / Float64(0.0)


@always_inline
def vexp[w: Int](x: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:
    # libm exp per lane: std.math.exp(Float64) is ~1e-12 off in Mojo 1.2.
    var r = SIMD[DType.float64, w]()
    comptime for l in range(w):
        r[l] = external_call["exp", Float64](x[l])
    return r


# ---- special functions (Phase 4). libm where it is accurate; psi implemented here.
comptime PI = 3.14159265358979323846264338327950288
comptime PSI_ROOT = 1.4616321449683622  # positive root of digamma, nearest double
comptime PSI_ROOT_C0 = -9.241265521729427e-17  # psi(PSI_ROOT) (root is not exactly representable)
# Taylor coefficients of psi about PSI_ROOT, highest order first: polygamma(n, root) / n!, n = 25..1
comptime PSI_C = [5.178575795222089e-05, -7.569179582195077e-05, 0.0001106337276874743, -0.00016170622091974825, 0.00023635601564027086, -0.0003454680251063082, 0.0005049532658346026, -0.000738070938996006, 0.0010788252019162975, -0.0015769367714301987, 0.00230512632673493, -0.0033698016554393312, 0.004926781395729858, -0.007204534386356875, 0.010538791616612182, -0.015424765904948968, 0.022597648232218125, -0.033161126474847376, 0.048804288164143145, -0.07219956125645476, 0.10782405069126239, -0.16394270544240658, 0.2584997609556512, -0.44276316898359214, 0.9676722454476214]


@always_inline
def _lgamma(x: Float64) -> Float64:
    # reentrant lgamma_r: plain lgamma writes the global signgam (data race across chains)
    var sg = Int32(0)
    return external_call["lgamma_r", Float64](x, Pointer(to=sg))


# Digamma. Reflection for x <= 0; Taylor series about the positive root for
# |x - root| < 0.25 (keeps relative accuracy where psi crosses zero); otherwise the
# recurrence psi(x) = psi(x + 1) - 1/x up to y >= 10 and the asymptotic series
# log(y) - 1/(2y) - sum B_2k / (2k y^2k) through B_14 (same as Cephes / PyTensor).
def _psi(x0: Float64) -> Float64:
    var x = x0
    if x <= 0.0:
        if x == external_call["floor", Float64](x):
            if x == 0.0:
                return NEG_INF if 1.0 / x > 0.0 else POS_INF  # scipy: psi(+0) = -inf, psi(-0) = +inf
            return NAN
        # psi(x) = psi(1 - x) - pi / tan(pi x), using exact sinpi/cospi argument reduction
        var c = external_call["__cospi", Float64](x) / external_call["__sinpi", Float64](x)
        return _psi(1.0 - x) - PI * c
    var d = x - PSI_ROOT
    if abs(d) < 0.25:
        var r = Float64(0)
        comptime for k in range(25):
            comptime ck = PSI_C[k]
            r = r * d + ck
        return r * d + PSI_ROOT_C0
    # Compensated recurrence: sum 1/y in double-double (TwoSum, plus the exact FMA residual
    # of each reciprocal), so the final cancellation log(y) - sum stays accurate.
    var s_hi = Float64(0)
    var s_lo = Float64(0)
    var y = x
    while y < 10.0:
        var r = 1.0 / y
        var r_lo = (-r).fma(y, 1.0) / y      # (1 - r*y) / y, exact numerator via fma
        var t = s_hi + r
        var bb = t - s_hi
        s_lo += ((s_hi - (t - bb)) + (r - bb)) + r_lo
        s_hi = t
        y += 1.0
    var R = 1.0 / y
    var R2 = R * R
    var ser = R2 * (1.0 / 12.0 - R2 * (1.0 / 120.0 - R2 * (1.0 / 252.0 - R2 * (1.0 / 240.0
        - R2 * (1.0 / 132.0 - R2 * (691.0 / 32760.0 - R2 / 12.0))))))
    return (log(y) - s_hi) + ((-0.5 * R - ser) - s_lo)


@always_inline
def vlgamma[w: Int](x: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:
    var r = SIMD[DType.float64, w]()
    comptime for l in range(w):
        r[l] = _lgamma(x[l])
    return r


@always_inline
def vpsi[w: Int](x: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:
    var r = SIMD[DType.float64, w]()
    comptime for l in range(w):
        r[l] = _psi(x[l])
    return r


@always_inline
def vlibm1[name: StaticString, w: Int](x: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:
    var r = SIMD[DType.float64, w]()
    comptime for l in range(w):
        r[l] = external_call[name, Float64](x[l])
    return r


@always_inline
def vpow[w: Int](x: SIMD[DType.float64, w], y: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:
    var r = SIMD[DType.float64, w]()
    comptime for l in range(w):
        r[l] = external_call["pow", Float64](x[l], y[l])
    return r


@always_inline
def vlog1mexp[w: Int](x: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:
    # log(1 - exp(x)) for x <= 0 (Maechler 2012): log(-expm1(x)) near 0, log1p(-exp(x)) otherwise
    var r = SIMD[DType.float64, w]()
    comptime for l in range(w):
        var v = x[l]
        if v > -0.6931471805599453:
            r[l] = log(-external_call["expm1", Float64](v))
        else:
            r[l] = log1p(-external_call["exp", Float64](v))
    return r


@always_inline
def vsign[w: Int](x: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:
    comptime F = SIMD[DType.float64, w]
    return x.gt(F(0)).select(F(1), x.lt(F(0)).select(F(-1), x))


@always_inline
def _p64(tab: Pointer[Int64, MutAnyOrigin], k: Int) -> Pointer[Float64, MutAnyOrigin]:
    return Pointer[Float64, MutAnyOrigin](unsafe_from_address=Int(tab[unsafe_offset=k]))


@always_inline
def _pi64(tab: Pointer[Int64, MutAnyOrigin], k: Int) -> Pointer[Int64, MutAnyOrigin]:
    return Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(tab[unsafe_offset=k]))

"""


@dataclass
class Plan:
    """Fusion plan for one 1-D Elemwise node."""
    node: object
    n: str = ""
    inlined: dict = field(default_factory=dict)     # input var -> ("gather"|"rowdot", producer nodes, info)
    sums: dict = field(default_factory=dict)        # out idx -> Sum node(s)
    ands: dict = field(default_factory=dict)        # out idx -> All (CAReduce AND) node(s)
    coldots: dict = field(default_factory=dict)     # out idx -> (Squeeze node, Dot node, AT var)
    scatters: dict = field(default_factory=dict)    # out idx -> AdvancedIncSubtensor node
    materialize: set = field(default_factory=set)   # out idx that must be stored


def _is_all(op):
    return isinstance(op, CAReduce) and isinstance(op.scalar_op, ps.AND)


def _clients(fg, v):
    return [c for c, _ in fg.clients[v]]


def _is_output(fg, v):
    return any(isinstance(c.op, str) or c == "output" for c, _ in fg.clients[v])


def _ds_is(node, op_pred):
    return node is not None and isinstance(node.op, DimShuffle) and op_pred(node.op)


def _expand1(op):   # (n,) -> (n, 1)
    return tuple(op.new_order) == (0, "x")


def _squeeze1(op):  # (m, 1) -> (m,)
    return tuple(op.new_order) == (0,) and op.input_ndim == 2


class Codegen:
    def __init__(self, fgraph, name, simd=W):
        self.fg = fgraph
        self.name = name
        self.W = simd
        self.lines = []
        self.vals = {}
        self.nbuf = 0
        self.consts = []          # (var, order) array constants in tab order
        self.const_order = {}     # var -> "C" | "F"
        self.frees = []
        self.n = 0
        self.done = set()         # nodes already produced by a fused loop
        self.skip = set()         # producer nodes inlined into a loop (never emitted)
        self.plans = {}
        self.shape_vars = []      # (input/output index, dim) -> name
        self.arena_sizes = []
        self.input_shape_names = []
        self.output_shape_exprs = []
        self.ind = 1

    # ------------------------------------------------------------ helpers
    def fresh(self, p="t"):
        self.n += 1
        return f"{p}{self.n}"

    def emit(self, s):
        self.lines.append("    " * self.ind + s)

    def alloc(self, dtype, shape):
        """Carve a float64 workspace buffer out of one per-call arena (allocated at the top).

        All shape expressions only reference input shape variables, which are defined
        before the arena, so every size can be hoisted. Sizes are rounded up to 8 elements.
        """
        if dtype != "float64":
            raise NotImplementedError("non-float64 workspace")
        p = self.fresh("w")
        off = " + ".join(self.arena_sizes) if self.arena_sizes else "0"
        self.emit(f"var {p} = arena.unsafe_offset({off})")
        self.arena_sizes.append(f"((max({_prod(shape)}, 1) + 7) // 8 * 8)")
        return p

    @staticmethod
    def c_strides(shape):
        st, acc = [], "1"
        for d in reversed(shape):
            st.append(acc)
            acc = _mul(acc, d)
        return list(reversed(st))

    def bshape(self, v, given):
        """Shape exprs for var v: static dims as literals, else the given runtime exprs."""
        return [str(s) if s is not None else g for s, g in zip(v.type.shape, given)]

    def scalar_of(self, val, idx_exprs=None):
        if isinstance(val, Scal):
            return val.expr
        if isinstance(val, Bcast):
            return val.scal.expr
        if isinstance(val, Buf):
            e = f"{val.ptr}[unsafe_offset={val.index(idx_exprs or ['0'] * val.ndim)}]"
            return f"{e}.ne(Float64(0))" if val.mask else e
        raise RuntimeError(f"cannot read {val}")

    # ------------------------------------------------------------ planning
    def plan(self):
        fg = self.fg
        for node in fg.toposort():
            if not isinstance(node.op, Elemwise) or node.outputs[0].type.ndim != 1:
                continue
            if node.outputs[0].type.shape[0] == 1:
                continue
            p = Plan(node)
            for x in node.inputs:
                if x.owner is None or len(_clients(fg, x)) != 1 or _is_output(fg, x):
                    continue
                o = x.owner
                if o in self.done or o in self.skip:
                    continue  # already claimed by an earlier fused loop
                if isinstance(o.op, (AdvancedSubtensor, AdvancedSubtensor1)) and len(o.inputs) == 2 \
                        and o.inputs[0].type.ndim == 1 and o.inputs[1].type.ndim == 1 \
                        and o.inputs[1].type.dtype.startswith("int"):
                    p.inlined[x] = ("gather", [o], (o.inputs[0], o.inputs[1]))
                elif _ds_is(o, _squeeze1) and o.inputs[0].owner is not None \
                        and isinstance(o.inputs[0].owner.op, Dot) and len(_clients(fg, o.inputs[0])) == 1:
                    dot = o.inputs[0].owner
                    A, b2 = dot.inputs
                    if A.type.ndim == 2 and _ds_is(b2.owner, _expand1) and len(_clients(fg, b2)) == 1:
                        p.inlined[x] = ("rowdot", [o, dot, b2.owner], (A, b2.owner.inputs[0]))
                        if isinstance(A, Constant):
                            self.const_order[A] = "F"
            for k, o in enumerate(node.outputs):
                if _is_output(fg, o):
                    p.materialize.add(k)
                    continue
                for c in _clients(fg, o):
                    tgt = o
                    if isinstance(c.op, CheckAndRaise) and c.inputs[0] is o:
                        # look through Check: its clients must all be Sums
                        cc = _clients(fg, c.outputs[0])
                        if cc and all(isinstance(z.op, Sum) and z.op.axis is None for z in cc) \
                                and not _is_output(fg, c.outputs[0]):
                            p.sums.setdefault(k, []).extend(cc)
                            continue
                        p.materialize.add(k)
                        continue
                    if isinstance(c.op, Sum) and (c.op.axis is None or tuple(c.op.axis) == (0,)):
                        p.sums.setdefault(k, []).append(c)
                        continue
                    if _is_all(c.op) and (c.op.axis is None or tuple(c.op.axis) == (0,)):
                        p.ands.setdefault(k, []).append(c)
                        continue
                    if isinstance(c.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1)) and len(c.inputs) == 3 \
                            and c.inputs[1] is o and c.inputs[0].type.ndim == 1 \
                            and c.inputs[0].type.shape[0] is not None and not c.op.set_instead_of_inc \
                            and c.inputs[2].type.ndim == 1:
                        p.scatters.setdefault(k, []).append(c)
                        continue
                    if _ds_is(c, _expand1):
                        dcl = _clients(fg, c.outputs[0])
                        if len(dcl) == 1 and isinstance(dcl[0].op, Dot) and dcl[0].inputs[1] is c.outputs[0]:
                            dot = dcl[0]
                            sq = _clients(fg, dot.outputs[0])
                            if len(sq) == 1 and _ds_is(sq[0], _squeeze1) and dot.inputs[0].type.ndim == 2:
                                p.coldots.setdefault(k, []).append((c, dot, sq[0]))
                                continue
                    p.materialize.add(k)
                    _ = tgt
            self.plans[node] = p
            for _, (kind, prods, _info) in p.inlined.items():
                self.skip.update(prods)
            for lst in p.sums.values():
                self.done.update(lst)
            for lst in p.ands.values():
                self.done.update(lst)
            for lst in p.coldots.values():
                for c, dot, sq in lst:
                    self.done.update([c, dot, sq])

    # ------------------------------------------------------------ binding
    def bind_tab(self, v, k, shape_exprs, dt=None):
        dt = dt or v.type.dtype
        if dt not in MTYPE:
            raise NotImplementedError(f"dtype {dt}")
        ptr = f"b{k}"
        fn = "_pi64" if dt == "int64" else "_p64"
        if dt == "bool":
            raise NotImplementedError("bool array inputs")
        self.emit(f"var {ptr} = {fn}(tab, {k})")
        order = self.const_order.get(v, "C")
        if order == "F":
            st = list(reversed(self.c_strides(list(reversed(shape_exprs)))))
        else:
            st = self.c_strides(shape_exprs)
        st = ["0" if v.type.shape[d] == 1 else s for d, s in enumerate(st)]
        b = Buf(ptr, dt, shape_exprs, st)
        if v.type.ndim == 0:
            s = self.fresh("s")
            self.emit(f"var {s} = {ptr}[unsafe_offset=0]")
            return Scal(s if dt != "bool" else s, dt)
        return b

    # ------------------------------------------------------------ main
    def generate(self):
        fg = self.fg
        self.plan()
        # tab layout: inputs, array constants, outputs
        consts = []
        seen = set()
        for node in fg.toposort():
            for x in node.inputs:
                if isinstance(x, Constant) and x.type.ndim > 0 and x not in seen \
                        and not (x.owner is None and node in self.skip and False):
                    seen.add(x)
                    consts.append(x)
        self.consts = consts
        nin, nc = len(fg.inputs), len(consts)
        self.emit("var status = Int32(0)")
        sidx = 0
        for i, v in enumerate(fg.inputs):
            names = []
            for d in range(v.type.ndim):
                nm = f"sh{i}_{d}"
                self.emit(f"var {nm} = Int(shp[unsafe_offset={sidx}])")
                names.append(nm)
                sidx += 1
            self.input_shape_names.append(names)
            self.vals[v] = self.bind_tab(v, i, self.bshape(v, names))
        # Array constants: integer arrays used only as gather/scatter indices stay int64;
        # everything else (float32 data, integer observations used in arithmetic) is packed
        # as float64 (exact for |v| < 2**53), so the kernel computes in float64 throughout.
        self.const_dtype = {}
        for c in consts:
            kind = np.asarray(c.data).dtype.kind
            index_only = all(
                (isinstance(cl.op, (AdvancedSubtensor, AdvancedSubtensor1)) and i >= 1) or
                (isinstance(cl.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1)) and i >= 2)
                for cl, i in fg.clients[c] if not isinstance(cl, str))
            self.const_dtype[c] = "int64" if kind in "iu" and index_only else ("bool" if kind == "b" else "float64")
        for j, c in enumerate(consts):
            dt = self.const_dtype[c]
            v = self.bind_tab(c, nin + j, [str(s) for s in c.data.shape], "float64" if dt == "bool" else dt)
            if dt == "bool" and isinstance(v, Buf):
                v.mask = True       # bool constants are packed as float64 0/1
            self.vals[c] = v
        self.out_base = nin + nc
        arena_at = len(self.lines)
        for node in fg.toposort():
            if node in self.skip or node in self.done:
                continue
            self.node(node)
        # outputs
        for j, o in enumerate(fg.outputs):
            self.ensure(o)
            val = self.vals[o]
            k = self.out_base + j
            fn = "_p64"
            if o.type.dtype != "float64":
                raise NotImplementedError("non-float64 outputs")
            ob = f"o{j}"
            self.emit(f"var {ob} = {fn}(tab, {k})")
            shape = self.shape_of(o)
            self.output_shape_exprs.append(shape)
            self.copy_into(Buf(ob, "float64", shape, self.c_strides(shape)), val, shape)
        total = " + ".join(self.arena_sizes) if self.arena_sizes else "8"
        try:
            static_total = int(eval(total, {"max": max}))  # noqa: S307  (only literals)
        except NameError:
            static_total = None
        if static_total is not None and static_total <= 8192:
            # small static workspace: stack array, no malloc per call
            self.lines[arena_at:arena_at] = [
                "    " + f"var arena_s = Array[Float64, {static_total}](uninitialized=True)",
                "    " + "var arena = arena_s.unsafe_ptr()",
            ]
        else:
            self.lines.insert(arena_at, "    " + f"var arena = unsafe_alloc[Float64]({total})")
            self.emit("arena.unsafe_free()")
        self.emit("return status")
        return "\n".join(self.lines) + "\n"

    def ensure(self, v):
        if v in self.vals:
            return
        if isinstance(v, Constant):
            if v.type.ndim == 0:
                dt = v.type.dtype
                if dt == "bool":
                    self.vals[v] = Scal("True" if v.data else "False", "bool")
                else:
                    # integer 0-d constants only ever feed float arithmetic here (index constants
                    # are read from v.data directly by Subtensor), so render them as Float64
                    self.vals[v] = Scal(f"Float64({_flit(v.data)})", "float64")
                return
        raise RuntimeError(f"value not computed: {v} (owner {v.owner})")

    def shape_of(self, v):
        val = self.vals[v]
        if isinstance(val, Buf):
            return list(val.shape)
        if isinstance(val, Scal):
            return []
        if isinstance(val, Bcast):
            return ["1"] * val.ndim
        raise RuntimeError(v)

    def copy_into(self, dst: Buf, val, shape):
        if not shape:
            self.emit(f"{dst.ptr}[unsafe_offset={dst.offset}] = {self.scalar_of(val)}")
            return
        idx = [self.fresh("j") for _ in shape]
        for d, (ix, n) in enumerate(zip(idx, shape)):
            self.emit(f"for {ix} in range({n}):")
            self.ind += 1
        self.emit(f"{dst.ptr}[unsafe_offset={dst.index(idx)}] = {self.scalar_of(val, idx)}")
        self.ind -= len(shape)

    # ------------------------------------------------------------ dispatch
    def node(self, node):
        op = node.op
        inl = self.plans[node].inlined if node in self.plans else {}
        for x in node.inputs:
            if x not in inl:
                self.ensure(x)
        if isinstance(op, Elemwise):
            return self.elemwise(node)
        if isinstance(op, DimShuffle):
            return self.dimshuffle(node)
        if isinstance(op, Subtensor):
            return self.subtensor(node)
        if isinstance(op, CheckAndRaise):
            return self.check(node)
        if isinstance(op, (ScalarFromTensor, TensorFromScalar)):
            self.vals[node.outputs[0]] = self.as_scal(self.vals[node.inputs[0]], node.inputs[0].type.dtype)
            return
        if isinstance(op, Sum) or (isinstance(op, CAReduce) and isinstance(op.scalar_op, ps.Add)):
            return self.reduce_sum(node)
        if isinstance(op, Dot):
            return self.dot(node)
        if isinstance(op, (AdvancedSubtensor, AdvancedSubtensor1)):
            return self.gather(node)
        if isinstance(op, (AdvancedIncSubtensor, AdvancedIncSubtensor1)):
            return self.scatter(node)
        if isinstance(op, Join):
            return self.join(node)
        if isinstance(op, Alloc):
            return self.alloc_op(node)
        if _is_all(op):
            return self.reduce_all(node)
        if isinstance(op, MakeVector):
            return self.make_vector(node)
        raise NotImplementedError(f"MojoND: unsupported op {op} ({type(op).__name__})")

    def as_scal(self, val, dtype):
        if isinstance(val, Scal):
            return val
        if isinstance(val, Bcast):
            return val.scal
        if isinstance(val, Buf):
            s = self.fresh("s")
            self.emit(f"var {s} = {self.scalar_of(val)}")
            return Scal(s, dtype)
        raise RuntimeError(val)

    # ------------------------------------------------------------ views
    def dimshuffle(self, node):
        (x,), (o,) = node.inputs, node.outputs
        val = self.vals[x]
        order = node.op.new_order
        if isinstance(val, (Scal, Bcast)):
            if o.type.ndim == 0:
                self.vals[o] = val if isinstance(val, Scal) else val.scal
            else:
                self.vals[o] = Bcast(val if isinstance(val, Scal) else val.scal, o.type.ndim)
            return
        if isinstance(val, Virtual):
            self.vals[o] = Virtual()
            return
        shape, st = [], []
        for d in order:
            if d == "x":
                shape.append("1"); st.append("0")
            else:
                shape.append(val.shape[d]); st.append(val.strides[d])
        if not shape:
            self.vals[o] = Scal(self.scalar_of(val), val.dtype)
            s = self.fresh("s")
            self.emit(f"var {s} = {self.vals[o].expr}")
            self.vals[o] = Scal(s, val.dtype)
            return
        self.vals[o] = Buf(val.ptr, val.dtype, shape, st, val.offset, mask=val.mask)

    def subtensor(self, node):
        x = node.inputs[0]
        val = self.vals[x]
        if not isinstance(val, Buf):
            raise NotImplementedError("Subtensor of non-buffer")
        from pytensor.tensor.subtensor import indices_from_subtensor
        idx = indices_from_subtensor(node.inputs[1:], node.op.idx_list)
        shape, st, off = [], [], val.offset
        d = 0
        for it in idx:
            if isinstance(it, slice):
                start, stop, step = (self.const_int(v) for v in (it.start, it.stop, it.step))
                n = val.shape[d]
                if step not in (None, 1):
                    raise NotImplementedError("Subtensor step")
                s0 = "0" if start is None else (str(start) if start >= 0 else f"({n} - {-start})")
                s1 = n if stop is None else (str(stop) if stop >= 0 else f"({n} - {-stop})")
                off = _add(off, _mul(s0, val.strides[d]))
                shape.append(f"({s1} - {s0})" if s0 != "0" else s1)
                st.append(val.strides[d])
            else:
                i = self.const_int(it)
                if i is None:
                    raise NotImplementedError("Subtensor with symbolic index")
                ie = str(i) if i >= 0 else f"({val.shape[d]} - {-i})"
                off = _add(off, _mul(ie, val.strides[d]))
            d += 1
        shape += val.shape[d:]
        st += val.strides[d:]
        o = node.outputs[0]
        shape = [str(s) if s is not None else e for s, e in zip(o.type.shape, shape)]
        if not shape:
            s = self.fresh("s")
            self.emit(f"var {s} = {val.ptr}[unsafe_offset={off}]")
            self.vals[o] = Scal(s, val.dtype)
        else:
            self.vals[o] = Buf(val.ptr, val.dtype, shape, st, off, mask=val.mask)

    @staticmethod
    def const_int(v):
        if v is None:
            return None
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, Constant):
            return int(v.data)
        return None

    def check(self, node):
        x, *conds = node.inputs
        for c in conds:
            cv = self.vals[c]
            e = self.scalar_of(cv) if not isinstance(cv, Buf) or cv.ndim == 0 else None
            if e is None:
                raise NotImplementedError("array-valued Check condition")
            self.emit(f"if not Bool({e}):")
            self.emit("    status = Int32(1)")
        self.vals[node.outputs[0]] = self.vals.get(x, Virtual())

    # ------------------------------------------------------------ elemwise
    def elemwise(self, node):
        out = node.outputs
        if node in self.plans:
            return self.fused_loop(self.plans[node])
        if all(o.type.ndim == 0 or all(s == 1 for s in o.type.shape) for o in out):
            args = [self.scalar_of(self.vals[x]) if not isinstance(self.vals[x], Buf)
                    else self.scalar_of(self.vals[x], ["0"] * self.vals[x].ndim) for x in node.inputs]
            res = self.scalar_expr(node.op.scalar_op, args, "1", [x.type.dtype for x in node.inputs], node.outputs[0].type.dtype)
            for o, e in zip(out, res):
                s = self.fresh("s")
                self.emit(f"var {s} = {e}")
                sc = Scal(s, o.type.dtype)
                self.vals[o] = sc if o.type.ndim == 0 else Bcast(sc, o.type.ndim)
            return
        return self.nd_elemwise(node)

    def scalar_expr(self, sop, args, w, types=None, otype=None):
        tbl = _sop_table(w)
        if isinstance(sop, ps.Composite):
            env = dict(zip(sop.fgraph.inputs, args))
            for nd in sop.fgraph.toposort():
                ins = []
                for x in nd.inputs:
                    if x in env:
                        ins.append(env[x])
                    elif isinstance(x, ps.ScalarConstant):
                        if x.type.dtype == "bool":
                            ins.append(f"SIMD[DType.bool, {w}](fill={'True' if x.data else 'False'})")
                        else:
                            ins.append(f"SIMD[DType.float64, {w}]({_flit(x.data)})")
                    else:
                        raise RuntimeError(x)
                outs = self.scalar_expr(nd.op, ins, w, [x.type.dtype for x in nd.inputs], nd.outputs[0].type.dtype)
                for o, e in zip(nd.outputs, outs):
                    t = self.fresh("t")
                    self.emit(f"var {t} = {e}")
                    env[o] = t
            return [env[o] for o in sop.fgraph.outputs]
        fn = tbl.get(type(sop))
        if fn is None:
            raise NotImplementedError(f"MojoND: unsupported scalar op {type(sop).__name__}")
        if otype == "bool" and isinstance(sop, (ps.Mul, ps.Add)):
            # PyTensor: Mul / Add of bools are logical and / or
            return ["(" + (" & " if isinstance(sop, ps.Mul) else " | ").join(args) + ")"]
        if isinstance(sop, ps.Switch) and types and types[0] != "bool":
            # PyTensor: a nonzero numeric condition is true (integer data is packed as float64)
            args = [f"({args[0]}).ne(SIMD[DType.float64, {w}](0))", *args[1:]]
        if types and otype and otype.startswith("float"):
            # PyTensor upcasts bool operands of arithmetic ops to float
            args = [f"{a}.cast[DType.float64]()" if t == "bool" and not (isinstance(sop, ps.Switch) and j == 0)
                    else a for j, (a, t) in enumerate(zip(args, types))]
        return [fn(*args)]

    def loader(self, val, i, w, dtype="float64"):
        """Return a Mojo expression for elements [i, i+w) of a 1-D value along its only dim."""
        if isinstance(val, Buf) and val.mask:
            raw = self.loader(Buf(val.ptr, "float64", val.shape, val.strides, val.offset), i, w, "float64")
            return f"{raw}.ne(SIMD[DType.float64, {w}](0))"
        F = f"SIMD[{DT[dtype]}, {w}]"
        if isinstance(val, Scal):
            return f"{F}({val.expr})"
        if isinstance(val, Bcast):
            return f"{F}({val.scal.expr})"
        if isinstance(val, Buf):
            st = val.strides[-1]
            base = val.index(["0"] * (val.ndim - 1) + [i])
            if st == "0":
                return f"{F}({val.ptr}[unsafe_offset={val.index(['0'] * val.ndim)}])"
            if st == "1":
                return f"{val.ptr}.unsafe_offset({base}).unsafe_load[width={w}]()"
            g = self.fresh("g")
            self.emit(f"var {g} = {F}()")
            self.emit(f"comptime for l in range({w}):")
            self.emit(f"    {g}[l] = {val.ptr}[unsafe_offset={val.index(['0'] * (val.ndim - 1) + [f'({i} + l)'])}]")
            return g
        raise RuntimeError(val)

    def fused_loop(self, p: Plan):
        node = p.node
        fg = self.fg
        o0 = node.outputs[0]
        # loop length: first non-broadcast 1-D input or inlined producer
        n = None
        for x in node.inputs:
            if x in p.inlined:
                kind, _, info = p.inlined[x]
                n = self.vals[info[1]].shape[0] if kind == "gather" else self.vals[info[0]].shape[0]
            else:
                v = self.vals[x]
                if isinstance(v, Buf) and v.shape[0] != "1":
                    n = v.shape[0]
            if n:
                break
        if o0.type.shape[0] is not None:
            n = str(o0.type.shape[0])
        if n is None:
            raise NotImplementedError("cannot infer loop length")
        # materialized outputs
        outbufs = {}
        for k in sorted(p.materialize):
            o = node.outputs[k]
            if o.type.dtype not in ("float64", "bool"):
                raise NotImplementedError(f"materialized {o.type.dtype} elemwise output")
            ptr = self.alloc("float64", [n])   # bool outputs are stored as float64 0/1 masks
            outbufs[k] = ptr
            self.vals[o] = Buf(ptr, "float64", [n], ["1"], mask=o.type.dtype == "bool")
        # accumulators
        accs = {}
        for k, lst in p.sums.items():
            a = self.fresh("acc")
            self.emit(f"var {a} = SIMD[DType.float64, W](0)")
            self.emit(f"var {a}_t = Float64(0)")
            accs[k] = a
        aaccs = {}
        for k, lst in p.ands.items():
            a = self.fresh("and")
            self.emit(f"var {a} = SIMD[DType.bool, W](fill=True)")
            self.emit(f"var {a}_t = True")
            aaccs[k] = a
        colacc = {}
        for k, lst in p.coldots.items():
            for (c, dot, sq) in lst:
                AT = self.vals[dot.inputs[0]]
                K = AT.shape[0]
                if not K.isdigit():
                    raise NotImplementedError("runtime K in fused coldot")
                names = [self.fresh("ca") for _ in range(int(K))]
                for nm in names:
                    self.emit(f"var {nm} = SIMD[DType.float64, W](0)")
                    self.emit(f"var {nm}_t = Float64(0)")
                colacc[(k, sq)] = (AT, names)
        scat = {}
        for k, lst in p.scatters.items():
            for c in lst:
                L = str(c.inputs[0].type.shape[0])
                ptr = self.alloc("float64", [L])
                self.emit(f"for j in range({L}):")
                self.emit(f"    {ptr}[unsafe_offset=j] = 0.0")
                scat[(k, c)] = (ptr, c.inputs[2])
        rowdot_k = {}
        for x, (kind, prods, info) in p.inlined.items():
            if kind == "rowdot":
                A, b = info
                K = self.vals[A].shape[1]
                if not K.isdigit():
                    raise NotImplementedError("runtime K in fused rowdot")
                bv = self.vals[b]
                names = []
                for kk in range(int(K)):
                    nm = self.fresh("bk")
                    self.emit(f"var {nm} = {self.scalar_of(bv, [str(kk)])}")
                    names.append(nm)
                rowdot_k[x] = names

        runs = {}
        for key in scat:
            rq, ra = self.fresh("rq"), self.fresh("ra")
            self.emit(f"var {rq} = -1")
            self.emit(f"var {ra} = SIMD[DType.float64, W](0)")
            runs[key] = (rq, ra)
        lane_cache = {}

        def idx_lanes(idxv, w, i):
            """SIMD-load W contiguous int64 indices once per chunk; None if not applicable."""
            if w == "1":
                return None
            idx = self.vals[idxv]
            if not (isinstance(idx, Buf) and idx.ndim == 1 and idx.strides[0] == "1"):
                return None
            if idxv not in lane_cache:
                ivec, same = self.fresh("iv"), self.fresh("same")
                self.emit(f"var {ivec} = {idx.ptr}.unsafe_offset({idx.index([i])}).unsafe_load[width={w}]()")
                self.emit(f"var {same} = {ivec}.eq(SIMD[DType.int64, {w}]({ivec}[0])).reduce_and()")
                lane_cache[idxv] = (ivec, same)
            return lane_cache[idxv]

        def body(w, i):
            lane_cache.clear()
            args = []
            for x in node.inputs:
                if x in p.inlined:
                    kind, _, info = p.inlined[x]
                    if kind == "gather":
                        src, idx = (self.vals[v] for v in info)
                        g = self.fresh("g")
                        F = f"SIMD[DType.float64, {w}]"
                        lanes = idx_lanes(info[1], w, i)
                        if lanes is not None:
                            ivec, same = lanes
                            se0 = src.index([f"Int({ivec}[0])"])
                            self.emit(f"var {g} = {F}({src.ptr}[unsafe_offset={se0}])")
                            self.emit(f"if not {same}:")
                            self.emit(f"    comptime for l in range({w}):")
                            sel = src.index([f"Int({ivec}[l])"])
                            self.emit(f"        {g}[l] = {src.ptr}[unsafe_offset={sel}]")
                        else:
                            self.emit(f"var {g} = {F}()")
                            self.emit(f"comptime for l in range({w}):")
                            ie = idx.index([f"({i} + l)"])
                            se = src.index([f"Int({idx.ptr}[unsafe_offset={ie}])"])
                            self.emit(f"    {g}[l] = {src.ptr}[unsafe_offset={se}]")
                        args.append(g)
                    else:
                        A = self.vals[info[0]]
                        e = self.fresh("rd")
                        F = f"SIMD[DType.float64, {w}]"
                        self.emit(f"var {e} = {F}(0)")
                        for kk, bn in enumerate(rowdot_k[x]):
                            col = Buf(A.ptr, A.dtype, [A.shape[0]], [A.strides[0]], _add(A.offset, _mul(str(kk), A.strides[1])))
                            ld = self.loader(col, i, w)
                            self.emit(f"{e} += {ld} * {F}({bn})")
                        args.append(e)
                else:
                    args.append(self.loader(self.vals[x], i, w, x.type.dtype if x.type.dtype in DT else "float64"))
            res = self.scalar_expr(node.op.scalar_op, args, w, [x.type.dtype for x in node.inputs], node.outputs[0].type.dtype)
            for k, ptr in outbufs.items():
                v = res[k]
                if node.outputs[k].type.dtype == "bool":
                    v = f"{v}.select(SIMD[DType.float64, {w}](1), SIMD[DType.float64, {w}](0))"
                self.emit(f"{ptr}.unsafe_offset({i}).unsafe_store({v})" if w != "1"
                          else f"{ptr}[unsafe_offset={i}] = {v}")
            for k, a in accs.items():
                self.emit(f"{a} += {res[k]}" if w != "1" else f"{a}_t += {res[k]}")
            for k, a in aaccs.items():
                self.emit(f"{a} = {a} & {res[k]}" if w != "1" else f"{a}_t = {a}_t and Bool({res[k]})")
            for (k, sq), (AT, names) in colacc.items():
                for kk, nm in enumerate(names):
                    row = Buf(AT.ptr, AT.dtype, [AT.shape[1]], [AT.strides[1]], _add(AT.offset, _mul(str(kk), AT.strides[0])))
                    ld = self.loader(row, i, w)
                    self.emit(f"{nm} += {ld} * {res[k]}" if w != "1" else f"{nm}_t += {ld} * {res[k]}")
            for (k, c), (ptr, idxv) in scat.items():
                idx = self.vals[idxv]
                tv = self.fresh("sv")
                self.emit(f"var {tv} = {res[k]}")
                lanes = idx_lanes(idxv, w, i)
                if lanes is not None:
                    # Run-length accumulate: when all lanes hit one index (sorted groups),
                    # keep the partial sum in a register and flush when the index changes.
                    ivec, same = lanes
                    rq, ra = runs[(k, c)]
                    self.emit(f"var {tv}_q = Int({ivec}[0])")
                    self.emit(f"if {same} and {tv}_q == {rq}:")
                    self.emit(f"    {ra} += {tv}")
                    self.emit("else:")
                    self.emit(f"    if {rq} >= 0:")
                    self.emit(f"        {ptr}[unsafe_offset={rq}] += {ra}.reduce_add()")
                    self.emit(f"    {rq} = -1")
                    self.emit(f"    if {same}:")
                    self.emit(f"        {rq} = {tv}_q")
                    self.emit(f"        {ra} = {tv}")
                    self.emit("    else:")
                    self.emit(f"        comptime for l in range({w}):")
                    self.emit(f"            {ptr}[unsafe_offset=Int({ivec}[l])] += {tv}[l]")
                else:
                    self.emit(f"comptime for l in range({w}):")
                    self.emit(f"    var q = Int({idx.ptr}[unsafe_offset={idx.index([f'({i} + l)'])}])")
                    self.emit(f"    {ptr}[unsafe_offset=q] += {tv}[l]")

        iv = self.fresh("i")
        self.emit(f"var {iv} = 0")
        self.emit(f"while {iv} + W <= {n}:")
        self.ind += 1
        body("W", iv)
        self.emit(f"{iv} += W")
        self.ind -= 1
        for key, (rq, ra) in runs.items():
            self.emit(f"if {rq} >= 0:")
            self.emit(f"    {scat[key][0]}[unsafe_offset={rq}] += {ra}.reduce_add()")
        self.emit(f"while {iv} < {n}:")
        self.ind += 1
        body("1", iv)
        self.emit(f"{iv} += 1")
        self.ind -= 1
        # finalize
        for k, lst in p.sums.items():
            s = self.fresh("s")
            self.emit(f"var {s} = {accs[k]}.reduce_add() + {accs[k]}_t")
            for snode in lst:
                self.vals[snode.outputs[0]] = Scal(s, "float64")
        for k, lst in p.ands.items():
            s = self.fresh("s")
            self.emit(f"var {s} = SIMD[DType.bool, 1]({aaccs[k]}.reduce_and() and {aaccs[k]}_t)")
            for snode in lst:
                self.vals[snode.outputs[0]] = Scal(s, "bool")
        for (k, sq), (AT, names) in colacc.items():
            ptr = self.alloc("float64", [str(len(names))])
            for kk, nm in enumerate(names):
                self.emit(f"{ptr}[unsafe_offset={kk}] = {nm}.reduce_add() + {nm}_t")
            self.vals[sq.outputs[0]] = Buf(ptr, "float64", [str(len(names))], ["1"])
        self.scatter_acc = getattr(self, "scatter_acc", {})
        for (k, c), (ptr, _) in scat.items():
            self.scatter_acc[c] = ptr
        for k, o in enumerate(node.outputs):
            if k not in outbufs:
                self.vals[o] = Virtual()

    def nd_elemwise(self, node):
        """Unfused N-d Elemwise with broadcasting: nested loops, SIMD innermost dim."""
        o0 = node.outputs[0]
        nd = o0.type.ndim
        shape = []
        for d in range(nd):
            e = str(o0.type.shape[d]) if o0.type.shape[d] is not None else None
            if e is None:
                for x in node.inputs:
                    v = self.vals[x]
                    if isinstance(v, Buf) and x.type.shape[d] != 1:
                        e = v.shape[d]
                        break
            if e is None:
                e = "1"
            shape.append(e)
        outs = []
        for o in node.outputs:
            if o.type.dtype not in ("float64", "bool"):
                raise NotImplementedError(f"materialized {o.type.dtype} elemwise output")
            ptr = self.alloc("float64", shape)   # bool outputs stored as float64 0/1 masks
            b = Buf(ptr, "float64", shape, self.c_strides(shape), mask=o.type.dtype == "bool")
            self.vals[o] = b
            outs.append(b)
        outer = [self.fresh("j") for _ in range(nd - 1)]
        for ix, n in zip(outer, shape[:-1]):
            self.emit(f"for {ix} in range({n}):")
            self.ind += 1

        def sub(val):
            if isinstance(val, Buf):
                st = val.strides
                off = val.offset
                for ix, s in zip(outer, st[:-1]):
                    off = _add(off, _mul(ix, s))
                return Buf(val.ptr, val.dtype, [val.shape[-1]], [st[-1]], off, mask=val.mask)
            return val

        ins = [sub(self.vals[x]) for x in node.inputs]
        osub = [sub(b) for b in outs]
        n = shape[-1]

        def body(w, i):
            args = [self.loader(v, i, w) for v in ins]
            res = self.scalar_expr(node.op.scalar_op, args, w, [x.type.dtype for x in node.inputs], node.outputs[0].type.dtype)
            for b, e in zip(osub, res):
                if b.mask:
                    e = f"{e}.select(SIMD[DType.float64, {w}](1), SIMD[DType.float64, {w}](0))"
                if w == "1":
                    self.emit(f"{b.ptr}[unsafe_offset={b.index([i])}] = {e}")
                else:
                    self.emit(f"{b.ptr}.unsafe_offset({b.index([i])}).unsafe_store({e})")

        iv = self.fresh("i")
        self.emit(f"var {iv} = 0")
        self.emit(f"while {iv} + W <= {n}:")
        self.ind += 1
        body("W", iv)
        self.emit(f"{iv} += W")
        self.ind -= 1
        self.emit(f"while {iv} < {n}:")
        self.ind += 1
        body("1", iv)
        self.emit(f"{iv} += 1")
        self.ind -= 1
        self.ind -= len(outer)

    # ------------------------------------------------------------ other ops
    def reduce_sum(self, node):
        (x,), (o,) = node.inputs, node.outputs
        val = self.vals[x]
        axis = node.op.axis
        if isinstance(val, (Scal, Bcast)):
            if o.type.ndim != 0:
                raise NotImplementedError("Sum over Bcast to array")
            self.vals[o] = val if isinstance(val, Scal) else val.scal
            return
        if isinstance(val, Virtual):
            raise RuntimeError("Sum of virtual value not fused")
        nd = val.ndim
        axes = tuple(range(nd)) if axis is None else tuple(a % nd for a in axis)
        if len(axes) == nd:
            s = self.fresh("s")
            self.emit(f"var {s} = Float64(0)")
            idx = [self.fresh("j") for _ in range(nd)]
            for ix, n in zip(idx, val.shape):
                self.emit(f"for {ix} in range({n}):")
                self.ind += 1
            self.emit(f"{s} += {self.scalar_of(val, idx)}")
            self.ind -= nd
            self.vals[o] = Scal(s, "float64")
            return
        keep = [d for d in range(nd) if d not in axes]
        oshape = [val.shape[d] for d in keep]
        ptr = self.alloc("float64", oshape)
        ob = Buf(ptr, "float64", oshape, self.c_strides(oshape))
        oidx = [self.fresh("j") for _ in keep]
        for ix, n in zip(oidx, oshape):
            self.emit(f"for {ix} in range({n}):")
            self.ind += 1
        self.emit(f"{ptr}[unsafe_offset={ob.index(oidx)}] = 0.0")
        self.ind -= len(oidx)
        idx = [self.fresh("j") for _ in range(nd)]
        for ix, n in zip(idx, val.shape):
            self.emit(f"for {ix} in range({n}):")
            self.ind += 1
        self.emit(f"{ptr}[unsafe_offset={ob.index([idx[d] for d in keep])}] += {self.scalar_of(val, idx)}")
        self.ind -= nd
        self.vals[o] = ob

    def reduce_all(self, node):
        """All over a bool value: scalars/broadcasts/MakeVector lists only (arrays are fused)."""
        (x,), (o,) = node.inputs, node.outputs
        val = self.vals[x]
        if isinstance(val, Scal):
            self.vals[o] = val
        elif isinstance(val, Bcast):
            self.vals[o] = val.scal
        elif isinstance(val, ScalList):
            s = self.fresh("s")
            e = " and ".join(f"Bool({v.expr})" for v in val.items) or "True"
            self.emit(f"var {s} = SIMD[DType.bool, 1]({e})")
            self.vals[o] = Scal(s, "bool")
        else:
            raise NotImplementedError("MojoND: All over a materialized bool array")

    def make_vector(self, node):
        o = node.outputs[0]
        items = [self.as_scal(self.vals[x], x.type.dtype) for x in node.inputs]
        if o.type.dtype == "bool":
            self.vals[o] = ScalList(items, "bool")
            return
        if o.type.dtype != "float64":
            raise NotImplementedError(f"MakeVector of {o.type.dtype}")
        n = str(len(items))
        ptr = self.alloc("float64", [n])
        for k, it in enumerate(items):
            self.emit(f"{ptr}[unsafe_offset={k}] = Float64({it.expr})")
        self.vals[o] = Buf(ptr, "float64", [n], ["1"])

    def dot(self, node):
        a, b = node.inputs
        A, B = self.vals[a], self.vals[b]

        def as2(v, var, is_left):
            if isinstance(v, Buf) and v.ndim == 2:
                return v
            if isinstance(v, Buf) and v.ndim == 1:
                return Buf(v.ptr, v.dtype, ["1", v.shape[0]] if is_left else [v.shape[0], "1"],
                           ["0", v.strides[0]] if is_left else [v.strides[0], "0"], v.offset)
            raise NotImplementedError(f"Dot operand {var}")

        A2, B2 = as2(A, a, True), as2(B, b, False)
        M, K, P = A2.shape[0], A2.shape[1], B2.shape[1]
        o = node.outputs[0]
        oshape = [M, P] if o.type.ndim == 2 else ([M] if a.type.ndim == 2 else [P]) if o.type.ndim == 1 else []
        full = [M, P]
        ptr = self.alloc("float64", full)
        i, j, k = self.fresh("j"), self.fresh("j"), self.fresh("j")
        acc = self.fresh("s")
        self.emit(f"for {i} in range({M}):")
        self.emit(f"    for {j} in range({P}):")
        self.emit(f"        var {acc} = Float64(0)")
        self.emit(f"        for {k} in range({K}):")
        self.emit(f"            {acc} += {self.scalar_of(A2, [i, k])} * {self.scalar_of(B2, [k, j])}")
        self.emit(f"        {ptr}[unsafe_offset={_add(_mul(i, P), j)}] = {acc}")
        if o.type.ndim == 0:
            s = self.fresh("s")
            self.emit(f"var {s} = {ptr}[unsafe_offset=0]")
            self.vals[o] = Scal(s, "float64")
        else:
            self.vals[o] = Buf(ptr, "float64", oshape, self.c_strides(oshape))

    def gather(self, node):
        src, idx = node.inputs[0], node.inputs[1]
        S, I = self.vals[src], self.vals[idx]
        if not (isinstance(S, Buf) and S.ndim == 1 and isinstance(I, Buf) and I.ndim == 1):
            raise NotImplementedError("only 1-D gather src[idx]")
        n = I.shape[0]
        ptr = self.alloc("float64", [n])
        j = self.fresh("j")
        self.emit(f"for {j} in range({n}):")
        self.emit(f"    {ptr}[unsafe_offset={j}] = {S.ptr}[unsafe_offset={S.index([f'Int({I.ptr}[unsafe_offset={I.index([j])}])'])}]")
        self.vals[node.outputs[0]] = Buf(ptr, "float64", [n], ["1"])

    def scatter(self, node):
        base, vals, idx = node.inputs
        B = self.vals[base]
        L = self.shape_of(base)
        if len(L) != 1:
            raise NotImplementedError("only 1-D scatter-add")
        ptr = self.alloc("float64", L)
        bb = Buf(ptr, "float64", L, ["1"])
        self.copy_into(bb, B, L)
        acc = getattr(self, "scatter_acc", {}).get(node)
        j = self.fresh("j")
        if acc is not None:
            self.emit(f"for {j} in range({L[0]}):")
            self.emit(f"    {ptr}[unsafe_offset={j}] += {acc}[unsafe_offset={j}]")
        else:
            if node.op.set_instead_of_inc:
                raise NotImplementedError("AdvancedSetSubtensor")
            V, I = self.vals[vals], self.vals[idx]
            n = I.shape[0]
            self.emit(f"for {j} in range({n}):")
            self.emit(f"    {ptr}[unsafe_offset=Int({I.ptr}[unsafe_offset={I.index([j])}])] += {self.scalar_of(V, [j]) if isinstance(V, Buf) else self.scalar_of(V)}")
        self.vals[node.outputs[0]] = bb

    def join(self, node):
        if hasattr(node.op, "axis") and isinstance(node.op.axis, (int, np.integer)):
            ax, parts = int(node.op.axis), node.inputs      # PyTensor 3: Join{axis=k}
        else:
            ax, parts = self.const_int(node.inputs[0]), node.inputs[1:]
        if ax != 0 or any(p.type.ndim != 1 for p in parts):
            raise NotImplementedError("Join only along axis 0 of 1-D")
        lens = [self.shape_of(p)[0] for p in parts]
        total = " + ".join(f"({x})" for x in lens)
        ptr = self.alloc("float64", [total])
        off = "0"
        for p, L in zip(parts, lens):
            self.copy_into(Buf(ptr, "float64", [L], ["1"], off), self.vals[p], [L])
            off = _add(off, f"({L})")
        o = node.outputs[0]
        shape = [str(o.type.shape[0]) if o.type.shape[0] is not None else total]
        self.vals[o] = Buf(ptr, "float64", shape, ["1"])

    def alloc_op(self, node):
        v, *dims = node.inputs
        shape = []
        for d, s in zip(dims, node.outputs[0].type.shape):
            if s is not None:
                shape.append(str(s))
            else:
                c = self.const_int(d)
                if c is None:
                    dv = self.vals[d]
                    shape.append(f"Int({self.scalar_of(dv)})")
                else:
                    shape.append(str(c))
        ptr = self.alloc("float64", shape)
        b = Buf(ptr, "float64", shape, self.c_strides(shape))
        self.copy_into(b, self.vals[v] if not isinstance(self.vals[v], Buf) else self.vals[v], shape) \
            if not isinstance(self.vals[v], Buf) else self._bcast_copy(b, self.vals[v], node.inputs[0], shape)
        self.vals[node.outputs[0]] = b

    def _bcast_copy(self, dst, src, var, shape):
        pad = len(shape) - src.ndim
        st = ["0"] * pad + ["0" if var.type.shape[d] == 1 else s for d, s in enumerate(src.strides)]
        sb = Buf(src.ptr, src.dtype, list(shape), st, src.offset)
        self.copy_into(dst, sb, shape)


# ---------------------------------------------------------------- building
def build_library(fgraph, name, flat_input_dim=None, simd=W):
    """Generate Mojo source for a FunctionGraph and compile it to a shared library.

    Parameters
    ----------
    fgraph : FunctionGraph
        Optimized graph (see ``ndlinker.QUERY`` for the rewrites used).
    name : str
        Symbol prefix of the exported functions.
    flat_input_dim : int, optional
        When given, the graph must have one flat 1-D input and two outputs (logp, grad), and
        the library also exports the sampler ABI ``<name>_dim`` / ``<name>_logp_grad``.
    simd : int
        SIMD width (float64 lanes) of the main loops.

    Returns
    -------
    so_path : pathlib.Path
        Library path, cached by a hash of the source.
    info : dict
        ``consts`` (constant, layout, dtype), output shape expressions, ``source``, ``src_path``.
    """
    cg = Codegen(fgraph, name, simd)
    body = cg.generate()
    nin = len(fgraph.inputs)
    src = HEADER.replace("{W}", str(simd))
    if STDEXP:
        src = src.replace("def vexp[w: Int](x: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:",
                          "def vexp[w: Int](x: SIMD[DType.float64, w]) -> SIMD[DType.float64, w]:\n    return std_exp(x)", 1)
        src = src.replace("from std.math import log, log1p, sqrt", "from std.math import log, log1p, sqrt\nfrom std.math import exp as std_exp", 1)
    src += f"\ndef _run_{name}(tab: Pointer[Int64, MutAnyOrigin], shp: Pointer[Int64, MutAnyOrigin]) -> Int32:\n" + body
    src += f"""

@export
def {name}_run(tab: Pointer[Int64, MutAnyOrigin], shp: Pointer[Int64, MutAnyOrigin]) abi("C") -> Int32:
    return _run_{name}(tab, shp)
"""
    nconst = len(cg.consts)
    nout = len(fgraph.outputs)
    if flat_input_dim is not None:
        # sampler ABI: one flat 1-D input x of static length, outputs (scalar logp, grad vector)
        assert nin == 1 and nout == 2
        src += f"""

@export
def {name}_dim(data: Pointer[Float64, MutAnyOrigin]) abi("C") -> Int32:
    return Int32({flat_input_dim})


@export
def {name}_logp_grad(
    data: Pointer[Float64, MutAnyOrigin],
    x: Pointer[Float64, MutAnyOrigin],
    grad: Pointer[Float64, MutAnyOrigin],
    lp: Pointer[Float64, MutAnyOrigin],
) abi("C") -> Int32:
    var tab_s = Array[Int64, {1 + nconst + 2}](fill=0)
    var shp_s = Array[Int64, 1](fill=Int64({flat_input_dim}))
    var tab = tab_s.unsafe_ptr()
    var shp = shp_s.unsafe_ptr()
    tab[unsafe_offset=0] = Int64(Int(x))
    var hdr = data.unsafe_bitcast[Int64]()
    for c in range({nconst}):
        tab[unsafe_offset=1 + c] = Int64(Int(data.unsafe_offset(Int(hdr[unsafe_offset=1 + c]))))
    tab[unsafe_offset={1 + nconst}] = Int64(Int(lp))
    tab[unsafe_offset={2 + nconst}] = Int64(Int(grad))
    var rc = _run_{name}(
        Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(tab)),
        Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(shp)),
    )
    if rc == 0 and not (lp[unsafe_offset=0] == lp[unsafe_offset=0] and abs(lp[unsafe_offset=0]) < POS_INF):
        rc = Int32(1)
    return rc
"""
    digest = hashlib.sha256(src.encode()).hexdigest()[:12]
    CACHE.mkdir(parents=True, exist_ok=True)
    so = CACHE / f"lib{name}_{digest}.dylib"
    srcf = CACHE / f"{name}_{digest}.mojo"
    if not so.exists():
        srcf.write_text(src)
        r = subprocess.run(["mojo", "build", str(srcf), "--emit", "shared-lib", "-o", str(so)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not so.exists():
            raise RuntimeError(f"mojo build failed for {srcf}:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    consts = [(c, cg.const_order.get(c, "C"), cg.const_dtype[c]) for c in cg.consts]
    info = dict(consts=consts, out_shapes=cg.output_shape_exprs, in_shape_names=cg.input_shape_names,
                source=src, src_path=srcf)
    return so, info


def const_array(c, order, dtype="float64"):
    # "bool" constants are packed as float64 0/1 and read back with .ne(0)
    a = np.asarray(c.data).astype(np.int64 if dtype == "int64" else np.float64)
    return np.asfortranarray(a) if order == "F" else np.ascontiguousarray(a)


def pack_data(consts):
    """Pack array constants into the float64 ``data`` buffer of the sampler ABI.

    Layout: ``[n, offset_1 .. offset_n, payload_1 .. payload_n]``. The count, the offsets and
    int64 payloads are stored bitwise in the float64 slots.
    """
    n = len(consts)
    head = 1 + n
    arrs = [const_array(c, o, dt) for c, o, dt in consts]
    sizes = [a.size for a in arrs]
    offs = np.cumsum([head] + sizes[:-1]).astype(np.int64)
    buf = np.zeros(head + sum(sizes), dtype=np.float64)
    iv = buf.view(np.int64)
    iv[0] = n
    iv[1:head] = offs
    for a, off in zip(arrs, offs):
        flat = a.ravel(order="K")
        if a.dtype == np.int64:
            iv[off:off + a.size] = flat
        else:
            buf[off:off + a.size] = flat
    return buf


# ---------------------------------------------------------------- static op census
_TENSOR_OK = (Elemwise, DimShuffle, Subtensor, CheckAndRaise, ScalarFromTensor, TensorFromScalar, MakeVector,
              Dot, AdvancedSubtensor, AdvancedSubtensor1, AdvancedIncSubtensor, AdvancedIncSubtensor1,
              Join, Alloc)


def _scalar_ops_of(sop, acc):
    if isinstance(sop, ps.Composite):
        for nd in sop.fgraph.apply_nodes:
            _scalar_ops_of(nd.op, acc)
    else:
        acc.add(type(sop))


def missing_ops(fgraph):
    """Return (missing tensor op names, missing scalar op names) that Codegen cannot emit.

    Mirrors the dispatch in Codegen.node and the scalar table, without generating code, so a
    census can list every gap of a graph in one pass instead of stopping at the first error.
    """
    known_scalar = set(_sop_table("1"))
    miss_t, miss_s = set(), set()
    for node in fgraph.apply_nodes:
        op = node.op
        if isinstance(op, Sum) or (isinstance(op, CAReduce) and isinstance(op.scalar_op, (ps.Add, ps.AND))):
            continue
        if isinstance(op, CAReduce):
            miss_t.add(f"CAReduce[{type(op.scalar_op).__name__}]")
            continue
        if not isinstance(op, _TENSOR_OK):
            miss_t.add(type(op).__name__)
            continue
        if isinstance(op, Elemwise):
            acc = set()
            _scalar_ops_of(op.scalar_op, acc)
            miss_s |= {t.__name__ for t in acc - known_scalar}
    return sorted(miss_t), sorted(miss_s)
