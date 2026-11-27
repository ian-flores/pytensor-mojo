"""MojoNDLinker (pytensor.function backend) and PyMC model -> sampler C-ABI library."""
from __future__ import annotations

import ctypes
import hashlib

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.basic import JITLinker

from .ndcodegen import build_library, const_array, pack_data

QUERY = RewriteDatabaseQuery(include=["fast_run"], exclude=["cxx_only", "BlasOpt", "inplace", "numba"])


def _name_for(fgraph):
    return "m" + hashlib.sha256(str(fgraph).encode() + str(id(fgraph)).encode()).hexdigest()[:10]


class _Runner:
    """Python callable around a generated library's <m>_run."""

    def __init__(self, fgraph, name=None):
        self.name = name or _name_for(fgraph)
        self.so, self.info = build_library(fgraph, self.name)
        self.lib = ctypes.CDLL(str(self.so))
        self.fn = getattr(self.lib, f"{self.name}_run")
        self.fn.restype = ctypes.c_int32
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.consts = [const_array(c, o, dt) for c, o, dt in self.info["consts"]]
        self.nin = len(fgraph.inputs)
        self.in_types = [v.type for v in fgraph.inputs]
        self.out_dtypes = [o.type.dtype for o in fgraph.outputs]

    def __call__(self, *inputs):
        arrs = []
        env = {}
        shp = []
        for k, (x, t) in enumerate(zip(inputs, self.in_types)):
            a = np.ascontiguousarray(x, dtype=t.dtype)
            if a.ndim != t.ndim:
                a = a.reshape(a.shape if a.ndim else ())
            arrs.append(a)
            for d, nm in enumerate(self.info["in_shape_names"][k]):
                env[nm] = a.shape[d]
                shp.append(a.shape[d])
        outs = []
        for e in self.info["out_shapes"]:
            shape = tuple(int(eval(s, {"max": max, "Int": int}, env)) for s in e)  # noqa: S307
            outs.append(np.empty(shape, dtype=np.float64))
        bufs = arrs + self.consts + outs
        tab = np.array([b.ctypes.data for b in bufs], dtype=np.int64)
        shp = np.array(shp + [0], dtype=np.int64)
        rc = self.fn(tab.ctypes.data, shp.ctypes.data)
        if rc != 0:
            raise ValueError(f"MojoND kernel returned status {rc} (parameter check failed)")
        return [o if o.ndim else np.array(o[()]) for o in outs]


class MojoNDLinker(JITLinker):
    """PyTensor ``JITLinker`` that compiles a FunctionGraph with ``ndcodegen``.

    Each compiled function becomes one generated Mojo library, called through ``ctypes``.
    Use it through the :data:`MOJO_ND` mode: ``pytensor.function(..., mode=MOJO_ND)``.
    """

    required_rewrites = ("minimum_compile",)
    incompatible_rewrites = ("cxx_only", "BlasOpt", "inplace", "numba")

    def fgraph_convert(self, fgraph, **kwargs):
        r = _Runner(fgraph)
        r.mojo_source = r.info["source"]
        return r

    def jit_compile(self, fn):
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]


MOJO_ND = Mode(MojoNDLinker(), QUERY)
"""PyTensor ``Mode``: ``MojoNDLinker`` plus backend-neutral ``fast_run`` rewrites (no C/BLAS/inplace)."""


# ----------------------------------------------------------- sampler path
def flat_logp_grad(model):
    """Build logp and its gradient over one flat unconstrained vector.

    Parameters
    ----------
    model : pymc.Model
        Model whose value variables all have static shapes.

    Returns
    -------
    q : TensorVariable
        Flat float64 vector of length ``dim``, the value variables in ``model.value_vars`` order.
    logp : TensorVariable
        ``model.logp(jacobian=True)`` with the value variables replaced by slices of ``q``.
    grad : TensorVariable
        Gradient of ``logp`` with respect to ``q``.
    dim : int
        Length of ``q``.
    """
    vv = model.value_vars
    sizes = [int(np.prod(v.type.shape)) if v.ndim else 1 for v in vv]
    if any(s is None for v in vv for s in v.type.shape):
        raise NotImplementedError("value vars need static shapes")
    dim = int(sum(sizes))
    q = pt.dvector("q", shape=(dim,))
    parts, i = [], 0
    for v, n in zip(vv, sizes):
        parts.append(q[i] if v.ndim == 0 else q[i:i + n].reshape(v.type.shape))
        i += n
    lp = model.logp(jacobian=True)
    outs = graph_replace([lp, *pt.grad(lp, vv)], dict(zip(vv, parts)), strict=False)
    g = pt.concatenate([pt.atleast_1d(o).ravel() for o in outs[1:]])
    return q, outs[0], g, dim


class _Capture(JITLinker):
    """Linker that only records the optimized FunctionGraph (pytensor clones linkers)."""
    required_rewrites = MojoNDLinker.required_rewrites
    incompatible_rewrites = MojoNDLinker.incompatible_rewrites
    sink: list = []

    def fgraph_convert(self, fgraph, **kwargs):
        _Capture.sink.append(fgraph)
        return lambda *a: [np.zeros(()) for _ in fgraph.outputs]

    def jit_compile(self, fn):
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]


def compile_model_capi(model, name):
    """Compile a PyMC model's logp and gradient into a C-ABI shared library.

    Parameters
    ----------
    model : pymc.Model
        Model to compile. Free variables need static shapes, and every op of the optimized
        graph must be supported (see ``missing_ops``).
    name : str
        Symbol prefix: the library exports ``<name>_dim``, ``<name>_logp_grad`` and
        ``<name>_run``.

    Returns
    -------
    so_path : pathlib.Path
        The compiled library, cached by a hash of the generated source.
    data : numpy.ndarray
        float64 buffer of packed constants to pass as the ``data`` argument.
    dim : int
        Length of the unconstrained parameter vector.
    info : dict
        ``source`` (generated Mojo), ``src_path``, ``consts`` and shape metadata.
    """
    q, lp, g, dim = flat_logp_grad(model)
    _Capture.sink.clear()
    pytensor.function([q], [lp, g], mode=Mode(_Capture(), QUERY))
    so, info = build_library(_Capture.sink[-1], name, flat_input_dim=dim)
    data = pack_data(info["consts"])
    return so, data, dim, info


class CapiModel:
    """Call a library's ``<name>_logp_grad`` from Python through ``ctypes``.

    Calling the object with a float64 vector ``x`` returns ``(status, logp, grad)``.

    Parameters
    ----------
    so : path-like
        Library from :func:`compile_model_capi`.
    name : str
        Symbol prefix used at compile time.
    data : numpy.ndarray
        The packed data buffer returned by :func:`compile_model_capi`.
    """

    def __init__(self, so, name, data):
        self.lib = ctypes.CDLL(str(so))
        self.f = getattr(self.lib, f"{name}_logp_grad")
        self.f.restype = ctypes.c_int32
        self.dim = getattr(self.lib, f"{name}_dim")(None)
        self.data = np.ascontiguousarray(data)

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        g = np.empty(self.dim)
        lp = np.empty(1)
        P = ctypes.c_void_p
        rc = self.f(P(self.data.ctypes.data), P(x.ctypes.data), P(g.ctypes.data), P(lp.ctypes.data))
        return rc, lp[0], g
