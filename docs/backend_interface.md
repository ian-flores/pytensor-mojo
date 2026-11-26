# What a PyTensor backend needs (Q3)

Read against PyTensor 3.3.8 (`pytensor/link/basic.py`, `link/numba/`, `link/mlx/`, `compile/mode.py`).

## The contract

A backend is a `JITLinker` subclass plus a `Mode` that pairs it with a rewrite query.

```python
class JITLinker(PerformLinker):
    def fgraph_convert(self, fgraph, order, input_storage, output_storage, storage_map, **kw) -> Callable
    def jit_compile(self, fn) -> Callable          # compile what fgraph_convert returned
    def create_thunk_inputs(self, storage_map) -> list   # usually [storage_map[v] for v in fgraph.inputs]
    # optional: input_filter / output_filter
    required_rewrites = ("minimum_compile", ...)   # rewrite tags that must run
    incompatible_rewrites = ("cxx_only", "BlasOpt", "inplace", ...)  # tags to drop
```

`JITLinker.make_all` (basic.py:708) does the rest. It calls `create_jitable_thunk`, which calls
`fgraph_convert`, then `jit_compile`, and wraps the result in a single thunk:
`outputs = jit_fn(*(x[0] for x in thunk_inputs)); out_storage[i][0] = outputs[i]`.
That means the backend function is called with Python values and has to return a sequence of
output values. The raw compiled function is exposed as `f.vm.jit_fn`.

Registration is one `Mode(...)` object (mode.py:514 NUMBA, :532 MLX), optionally added to
`predefined_modes`. Users then write `pytensor.function(..., mode=MOJO)`.

## How existing backends convert the graph

- **Numba** (32-line linker, about 8.4k lines of dispatch). A `@singledispatch numba_funcify(op, node, **kw)`
  returns a Python/numba function for each Op. `fgraph_to_python` (link/utils.py:779) then writes a
  Python source string that calls those per-node functions in topological order, and the whole thing
  is `numba.njit`-ed. Elemwise and Composite are turned into generated scalar loop code
  (`vectorize_codegen.py`). Numba also has its own rewrite passes (`numba` tag: FusedElemwise,
  ScalarizedElemwise, ScalarCAReduce).
- **MLX** (143-line linker, about 5.3k lines of dispatch). Same `singledispatch` plus
  `fgraph_to_python` design, and each Op maps to an `mx.*` call. Then `mx.compile`.
- **JAX / PyTorch**: same pattern.

## Implications for Mojo

The numba/MLX design assumes the target is a Python-callable JIT that can compose Python
closures. Mojo cannot do that: you emit source, build a shared lib, and import it. So a Mojo
backend should **generate one Mojo source file per FunctionGraph** instead of composing per-Op
callables. That is what `pytensor_mojo/linker.py` does:

- `mojo_funcify(fgraph)` walks `fgraph.toposort()` and emits one `run(tab_addr)` function.
  0-d values become Mojo locals, 1-d values become buffers, and Composite inner graphs are
  inlined per element.
- Buffers (inputs, array constants, intermediates, outputs) are NumPy arrays allocated once in
  Python. Their addresses go into one int64 table, so each call passes **one** argument.
- `mojo build --emit shared-lib` runs into a content-hashed cache. Cold compile is about 0.9 s and a
  warm cache hit is about 0.04 s.
- The singledispatch layer still works well as the *codegen* registry (Op -> Mojo snippet emitter).

**Autodiff does not belong in the backend.** PyTensor differentiates symbolically, so `pt.grad`
output is just more forward Ops. The backend never needs reverse-mode support. MAX 26.7 has no
autodiff API anyway, and that does not matter here.

## Op census (bench/q3_op_census.py)

These are the logp+grad graphs of six typical PyMC models after backend-neutral `fast_run`
rewrites (fusion on; C-only, BLAS, inplace and numba-specific passes off):

| model | nodes | notable ops |
|---|---|---|
| eight_schools | 9 | Elemwise(Composite) x3, Sum x4, DimShuffle x2 |
| linreg (Normal, dot) | 21 | Dot x2, CheckParameterValue |
| logistic (Bernoulli logit) | 18 | Dot, All[AND], MakeVector, Softplus/Sigmoid |
| hier_poisson (indexing) | 15 | AdvancedSubtensor, AdvancedIncSubtensor |
| robust StudentT/Gamma/Beta | 25 | GammaLn, Psi scalars |
| MvNormal + LKJCholeskyCov | 99 | SolveTriangular x4, AllocDiag, ExtractDiag, CumOp, (Adv)IncSubtensor |

Union across those models: **17 tensor-level Op kinds and 22 scalar Op kinds.**

- Tensor: Elemwise, DimShuffle, Sum/CAReduce, All, Dot, Subtensor, IncSubtensor,
  AdvancedSubtensor, AdvancedIncSubtensor, Alloc, AllocDiag, ExtractDiag, CumOp,
  SolveTriangular, MakeVector, ScalarFromTensor, CheckParameterValue.
- Scalar: Add Sub Mul TrueDiv Neg Sqr Sqrt Exp Log Log1p Reciprocal Sigmoid Softplus Sign
  GammaLn Psi Switch AND EQ GE GT LE.

## Estimate

- **Tier 1, about 70% of everyday PyMC models** (GLMs, hierarchical models with indexing,
  common distributions): N-d Elemwise with broadcasting and strides, about 40 scalar ops
  (adding Pow, Log1mexp, Erf/Erfc/Erfcx, Expm1, Abs, Floor, Maximum/Minimum, IsNan/IsInf,
  Cast, Gammaln, Psi, the Betaln family), CAReduce (Sum/Prod/Max/Min/All/Any along axes),
  DimShuffle, Reshape, Dot/Gemv/Gemm (call Accelerate/BLAS), Subtensor, IncSubtensor,
  Advanced(Inc)Subtensor, Alloc, Join/Split, MakeVector, Shape ops, CheckParameterValue/Assert,
  Cast. That comes to **about 25 tensor Ops plus about 40 scalar Ops**, plus a real buffer model
  (dynamic shapes, strides, intermediate allocation).
- **Tier 2**: Cholesky, SolveTriangular, Solve, Det/SlogDet, CumOp, Sort, Softmax/LogSoftmax,
  Blockwise (batched linalg).
- **Tier 3, hard**: Scan (time series, ODE loops), RandomVariable ops (only needed for prior
  or posterior predictive, not for NUTS logp), ScalarLoop (Hyp2F1, incomplete gamma/beta).

Rough effort, by analogy with numba's dispatch size (about 8k lines): **Tier 1 is 4-8 weeks** for one
experienced dev, and Tier 2 adds about 3-4 weeks. Scan is its own project.

Special-function accuracy needs care (see the `std.math.exp` finding in the README). Plan to bind
libm or Cephes-style implementations instead of trusting the stdlib transcendental functions.
