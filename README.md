# pytensor-mojo

A Mojo backend for PyTensor that compiles PyMC models' logp and gradient into fused SIMD
kernels, sampled with NUTS through a C ABI with no Python in the gradient loop.

> **Status: experimental research prototype.** Not on PyPI, no stable API. It runs two models
> end to end (eight schools and a hierarchical linear regression) and a set of unit-test graphs.
> It has only been developed and benchmarked on macOS / Apple Silicon (M1 Pro).

## What it is

- **A Mojo code generator for PyTensor graphs** (`pytensor_mojo/ndcodegen.py`). It takes the
  optimized FunctionGraph that PyMC builds for logp + gradient and emits one Mojo source file.
  That source is compiled to a shared library.
- **A small C ABI** that every generated library exports:

  ```c
  int32_t <m>_dim(const double* data);
  int32_t <m>_logp_grad(const double* data, const double* x, double* grad, double* logp);
  ```

  It also exports a generic `<m>_run` entry point, which the `MOJO_ND` mode uses through
  `pytensor.function`.
- **A Rust driver** (`sampler/`, crate `mojo-nuts`). It loads any such library and samples it
  with [nuts-rs](https://github.com/pymc-devs/nuts-rs), the same NUTS implementation and the
  same default settings that nutpie uses.

## Headline results

Sampling time only, compile time excluded. Median of 5 runs on an Apple M1 Pro, 1000 tune +
1000 draws per chain, with nuts-rs 0.18.3 and identical settings on both sides. nutpie is timed
without its Python-side trace conversion (`return_raw_trace=True`).

| model | chains | generated Mojo + nuts-rs | nutpie (numba backend) | speedup | min ESS bulk (Mojo / nutpie) |
|---|---|---|---|---|---|
| hierarchical linear regression, N=10k, 20 groups, 5 predictors | 4 | 0.317 s | 1.753 s | **5.5x** | 7557 / 7880 |
| hierarchical linear regression, N=10k, 20 groups, 5 predictors | 1 | 0.335 s | 1.788 s | **5.3x** | 1708 / 1700 |
| eight schools (non-centered) | 4 | 0.0167 s | 0.0165 s | 0.99x | 2013 / 2192 |
| eight schools (non-centered) | 1 | 0.0095 s | 0.0127 s | 1.34x | 461 / 577 |

Eight schools ties because its logp + gradient takes about 50 ns, while each NUTS leapfrog
step costs about 700 ns of sampler work (identical on both sides), so the kernel speed barely
shows.

Per-evaluation logp + gradient on the regression model:

| implementation | time |
|---|---|
| generated Mojo | 19.9 us |
| hand-written Mojo | 18.0 us |
| PyTensor numba graph | 122 us |

Correctness:

- Generated kernels match PyMC's `compile_logp` / `compile_dlogp` at rtol 1e-10. The observed
  max error is 3e-13 for logp and 2e-14 for the gradient.
- Posteriors match nutpie within Monte Carlo error.

The full measurements, controls and caveats are in [docs/FINDINGS.md](docs/FINDINGS.md).

## How it works

```
PyMC model
  -> PyTensor graph: logp + symbolic gradient on a flat unconstrained vector
  -> PyTensor rewrites (fast_run, FusionOptimizer Composites; no C/BLAS/inplace passes)
  -> ndcodegen: optimized FunctionGraph -> one Mojo source file
  -> mojo build --emit shared-lib -> lib<m>.dylib exporting <m>_dim / <m>_logp_grad
  -> sampler/ (Rust): libloading + nuts-rs CpuLogpFunc -> draws
```

Key ideas:

- **No autodiff in the backend.** PyTensor differentiates symbolically, so the gradient is just
  more forward ops.
- **Buffer model.** Every array is a pointer plus shape, strides and offset. Dims known
  statically become literals; others are read at runtime. Views such as transposes, broadcasts,
  constant slices and indexing never copy.
- **Loop fusion.** Every 1-D elementwise loop over the data axis absorbs its producers and
  consumers:
  - Producers are inlined per element: gathers like `a[g[i]]`, and row mat-vecs like `X @ beta`.
  - Consumers become in-loop accumulators: sums, the transposed mat-vec `X.T @ w`, and the
    scatter-add `inc_subtensor(a[g], w)` that the gradient of a gather produces.
  - For the regression model, the entire N=10,000 pass, both logp and all gradient terms, is
    one loop with no N-length intermediate arrays.
- **Explicit SIMD.** The fused loop is emitted 8-wide (float64), with a scalar tail.
  Transcendentals use libm per lane, because Mojo's `std.math.exp` is not accurate enough (see
  below).
- **Run-length scatter.** The index lanes are loaded once per chunk. When all lanes hit the
  same group (sorted data), the partial sum stays in a register and is flushed only when the
  group changes. Otherwise it falls back to per-lane updates, so it is correct for any index
  order. This single change took the regression kernel from 36.7 us to 23.0 us.
- **No allocation per gradient.** The workspace is one arena, carved at compile time, and kept
  on the stack when it is small.

## Quickstart

Requirements: macOS on Apple Silicon (the only tested platform),
[pixi](https://pixi.sh), and a Rust toolchain (`cargo`) for the sampler. pixi installs Mojo
1.2, Python 3.12, PyMC, PyTensor, nutpie and the rest from conda-forge and the Modular channel.

```bash
pixi install                  # create the environment from pixi.toml / pixi.lock
pixi run build                # build the Mojo modules and hand-written C-ABI libraries
pixi run build-sampler        # build the Rust nuts-rs driver (sampler/)
pixi run test                 # full pytest suite (builds first)
pixi run test-rust            # cargo fmt --check, clippy -D warnings, cargo test
pixi run bench-phase3         # generated vs hand-written Mojo vs nutpie, 5 reps
```

Other benchmark tasks: `bench-q1` (Python->Mojo call overhead), `bench-q2` (eight-schools
kernel timings), `census` (op census over typical PyMC models) and `bench-phase2`.

Compiling a PyMC model to a C-ABI library and evaluating it from Python. Run this from the repo
root, for example with `PYTHONPATH=. pixi run python`; the package is not installed:

```python
import numpy as np
from pytensor_mojo import hlr_ref
from pytensor_mojo.ndlinker import compile_model_capi, CapiModel

d = hlr_ref.make_data()                       # synthetic data: N=10k rows, 20 groups, 5 predictors
model = hlr_ref.build_model(d)                # an ordinary PyMC model
so_path, data, dim, info = compile_model_capi(model, "hlrgen")   # about 1.2 s cold, cached after
status, logp, grad = CapiModel(so_path, "hlrgen", data)(np.zeros(dim))
data.astype("<f8").tofile("data.f64")         # model data for the Rust driver
```

To sample it, run the driver on the library and the data file:

```bash
sampler/target/release/mojo-nuts --lib <so_path> --model hlrgen --data data.f64 \
    --chains 4 --tune 1000 --draws 1000 --seed 1 --out trace.json
```

## Supported ops and limitations

Supported by the generated backend (`ndcodegen`):

- **Elemwise** with N-d broadcasting, including PyTensor Composites.
- **Scalar ops:** add, sub, mul, true_div, neg, sqr, sqrt, exp, log, log1p, reciprocal, abs,
  maximum, minimum, sigmoid, softplus, comparisons, and/or/invert, switch, identity.
- **Reductions:** Sum over all axes or given axes.
- **Linear algebra and indexing:** Dot (2-D/1-D combinations), 1-D AdvancedSubtensor (gather)
  and AdvancedIncSubtensor (scatter-add).
- **Shape and structure:** DimShuffle (transpose, expand, squeeze), Subtensor with constant
  indices or slices, Join (axis 0), Alloc, ScalarFromTensor / TensorFromScalar, and
  CheckParameterValue (reported as status 1 instead of raising).
- **Unsupported ops** raise `NotImplementedError("MojoND: unsupported op ...")`.

Current limitations:

- **Fusion specializes on static shapes.** The fast fused loops need dims known from the graph
  types, which holds for PyMC models with constant data. Graphs with runtime shapes work
  through a generic path, but dots and scatters are not fused there.
- **No BLAS.** Unfused Dot and axis reductions are plain scalar loops.
- **Indexing is 1-D only.** Gather and scatter support only `a[idx]`; there is no `set_subtensor`.
- **Missing ops:** no special functions beyond exp/log/log1p/sqrt (no gammaln, digamma, erf),
  no Cholesky / linalg, no Scan, no Sort, and no random variables. Many PyMC distributions
  therefore do not compile yet.
- **Data types:** float64 only, with int64 for index arrays and bool for conditions.
- **Coverage:** only eight schools, the hierarchical regression and unit-test graphs have been
  run end to end.
- **Python call path:** calling generated code through `pytensor.function` (`MOJO_ND` mode) pays
  PyTensor's per-call Python overhead of about 2 us. The fast path is the C ABI driven from
  Rust.
- **Platform:** macOS / Apple Silicon only so far. Linux should be possible but is untested,
  and the build scripts assume `.dylib`.

## Repo layout

```
pytensor_mojo/ndcodegen.py    FunctionGraph -> Mojo source (buffer model, fusion, SIMD, C ABI)
pytensor_mojo/ndlinker.py     MOJO_ND pytensor mode; compile_model_capi(PyMC model); ctypes wrapper
pytensor_mojo/linker.py       first prototype linker (ndim <= 1, scalar loops), kept for its tests
pytensor_mojo/*_ref.py        reference PyMC models (eight schools, hierarchical regression)
mojo_capi/                    hand-written C-ABI kernels used as performance baselines
mojo_src/                     Python-extension Mojo modules (call-overhead and kernel experiments)
sampler/                      Rust crate mojo-nuts: nuts-rs driver for any C-ABI model library
nuts_bench/                   helpers to run Mojo+nuts-rs and nutpie, posterior summaries
bench/                        benchmarks, with recorded results (phase2/phase3_results.json)
gpu/                          Apple GPU (Metal, float32) experiments
tests/                        pytest suite (62 tests); sampler/tests has the Rust tests
docs/FINDINGS.md              full findings log: every measurement, control and caveat
docs/backend_interface.md     how a PyTensor backend plugs in; op census; effort estimate
build.sh                      builds mojo_src/*.mojo and mojo_capi/*_capi.mojo
```

## Roadmap

1. **Special functions:** gammaln, digamma, erf/erfc, log1mexp and others, each with an
   accuracy test suite against SciPy / PyTensor.
2. **posteriordb model sweep:** compile and sample a broad set of reference models, and track
   coverage and speed against nutpie.
3. **Linear algebra:** Cholesky, triangular solves, log-determinants (MvNormal, LKJ).
4. **Scan:** time-series and recurrent models.
5. **Apple GPU many-chain sampler:** float32 batched logp + gradient across many chains. A
   first check showed about 1e-7 typical error, so the GPU suits massively parallel chains
   rather than single-chain NUTS.

## Upstream issues found

- **Mojo 1.2 `std.math.exp(Float64)`** is not correctly rounded. It has up to about 8e-12
  relative error on [-20, 20]. Generated code calls libm `exp` instead.
  `tests/test_q1_mod.py::test_exp_accuracy_std_vs_libm` pins the behaviour.
- **nuts-rs 0.18.3 `HashMapConfig` storage** panics on the first divergence ("Type mismatch
  when combining stats for divergence_message"). `ArrowConfig` works, and it is what nutpie
  uses.
- **PyMC / PyTensor literal parameters become float32 constants.** For example,
  `pm.Normal("mu", 0, 5.0)` computes `log(5)` in float32, and HalfCauchy's `log(2)` is an int8
  constant upcast to float32. This shifts logp by about 1e-9 relative. Pass `np.float64(...)` when
  comparing against an exact implementation.
  `tests/test_eight_schools.py::test_pymc_literal_params_float32_offset_documented` pins the
  behaviour.

## Acknowledgements

This builds directly on:

- [PyMC](https://github.com/pymc-devs/pymc) and [PyTensor](https://github.com/pymc-devs/pytensor)
  for the modelling language, graph rewrites and symbolic gradients.
- [nuts-rs](https://github.com/pymc-devs/nuts-rs) and [nutpie](https://github.com/pymc-devs/nutpie)
  for the NUTS implementation, its adaptation, and the baseline this project measures against.
- [Modular](https://www.modular.com/) for [Mojo](https://mojolang.org/) and MAX.

## License

Apache-2.0. See [LICENSE](LICENSE).
