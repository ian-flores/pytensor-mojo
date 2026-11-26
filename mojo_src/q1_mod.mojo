# Q1: Python -> Mojo call overhead and throughput microbenchmarks.
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python.numpy import from_numpy_array
from std.os import abort
from std.math import exp
from std.ffi import external_call


@export
def PyInit_q1_mod() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("q1_mod")
        m.def_function[noop0]("noop0")
        m.def_function[noop1]("noop1")
        m.def_function[first_span]("first_span")
        m.def_function[first_ptr]("first_ptr")
        m.def_function[sum_span]("sum_span")
        m.def_function[axpy_span]("axpy_span")
        m.def_function[sum_ptr]("sum_ptr")
        m.def_function[exp_std]("exp_std")
        m.def_function[exp_libm]("exp_libm")
        m.def_function[axpy_ptr]("axpy_ptr")
        return m.finalize()
    except e:
        abort(String("error creating q1_mod:", e))


def noop0() raises -> PythonObject:
    return PythonObject(0.0)


def noop1(x: PythonObject) raises -> PythonObject:
    return x


def first_span(arr: PythonObject) raises -> PythonObject:
    # Zero-copy borrow via std.python.numpy (validates dtype/contiguity).
    var s = from_numpy_array[DType.float64](arr)
    return PythonObject(s[0])


def first_ptr(addr: PythonObject) raises -> PythonObject:
    # Caller passes the raw buffer address as a Python int (precomputed once).
    var p = addr.unsafe_get_as_pointer[DType.float64]()
    return PythonObject(p[unsafe_offset=0])


def sum_span(arr: PythonObject) raises -> PythonObject:
    var s = from_numpy_array[DType.float64](arr)
    var acc = Float64(0)
    for v in s:
        acc += v
    return PythonObject(acc)


def axpy_span(a: PythonObject, x: PythonObject, y: PythonObject, out_arr: PythonObject) raises -> PythonObject:
    # out = a * x + y, written in place into a preallocated NumPy array.
    var av = Float64(py=a)
    var xs = from_numpy_array[DType.float64](x)
    var ys = from_numpy_array[DType.float64](y)
    var o = out_arr.ctypes.data.unsafe_get_as_pointer[DType.float64]()
    var n = len(xs)
    for i in range(n):
        o[unsafe_offset=i] = av * xs[i] + ys[i]
    return PythonObject(None)


comptime W = 8


def sum_ptr(addr: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    # Raw-address variant with explicit SIMD (4 independent accumulators).
    var p = addr.unsafe_get_as_pointer[DType.float64]()
    var n = Int(py=n_obj)
    var a0 = SIMD[DType.float64, W](0)
    var a1 = SIMD[DType.float64, W](0)
    var i = 0
    while i + 2 * W <= n:
        a0 += p.unsafe_offset(i).unsafe_load[width=W]()
        a1 += p.unsafe_offset(i + W).unsafe_load[width=W]()
        i += 2 * W
    var acc = (a0 + a1).reduce_add()
    while i < n:
        acc += p[unsafe_offset=i]
        i += 1
    return PythonObject(acc)


def axpy_ptr(a: PythonObject, xa: PythonObject, ya: PythonObject, oa: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    var av = Float64(py=a)
    var x = xa.unsafe_get_as_pointer[DType.float64]()
    var y = ya.unsafe_get_as_pointer[DType.float64]()
    var o = oa.unsafe_get_as_pointer[DType.float64]()
    var n = Int(py=n_obj)
    var i = 0
    while i + W <= n:
        o.unsafe_offset(i).unsafe_store(av * x.unsafe_offset(i).unsafe_load[width=W]() + y.unsafe_offset(i).unsafe_load[width=W]())
        i += W
    while i < n:
        o[unsafe_offset=i] = av * x[unsafe_offset=i] + y[unsafe_offset=i]
        i += 1
    return PythonObject(None)


def exp_std(x: PythonObject) raises -> PythonObject:
    return PythonObject(exp(Float64(py=x)))


def exp_libm(x: PythonObject) raises -> PythonObject:
    return PythonObject(external_call["exp", Float64](Float64(py=x)))
