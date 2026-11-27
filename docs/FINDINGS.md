# pytensor-mojo: findings log

This is the full record of the feasibility spike (Phase 1) and the two follow-up phases. It
covers the questions asked, every measurement, the surprises and blockers, and the caveats.
The project summary is in the top-level `README.md`. The backend interface write-up is in
`docs/backend_interface.md`.

Contents:

- Environment
- Phase 1: feasibility spike (Q1 call overhead, Q2 hand-written eight schools, Q3 backend
  interface and v1 prototype, Apple GPU check, other checks, go / no-go)
- Phase 2: NUTS with no Python in the gradient loop (Mojo C ABI driven by nuts-rs)
- Phase 3: generated N-d, fused, SIMD code through the same C ABI
- Phase 4: special functions and common likelihoods (census, accuracy, sampling)
- Phase 7: SIMD transcendentals (vectorized exp, log, log1p, expm1, lgamma)
- Caveats across all phases
- Test record

All timings are from one Apple M1 Pro (8 performance + 2 efficiency cores, 16-core GPU),
single machine, with no CPU pinning.

## Environment (pixi.lock is authoritative)

| package | version |
|---|---|
| Mojo | 1.2.0 (2016d0cd), conda `mojo`/`mojo-compiler` 1.2.0 |
| Python | 3.12.16 (3.12 pinned; 3.13 not tried) |
| NumPy | 2.5.5 |
| PyTensor | 3.3.8 |
| PyMC | 6.4.0 |
| Numba | 0.67.0 |
| nutpie | 0.16.11 |
| pytest | 9.2.0 |
| Rust / cargo | 1.92.0 |
| nuts-rs | 0.18.3 (the version bundled in nutpie 0.16.11), features `arrow` |
| MAX (conda `max`, needed for `max.gpu`) | 26.7.0 |
| Machine | Apple M1 Pro, 16-core GPU, Metal 4, macOS 27.2.1 (26B131) |
| Xcode / CLT / Metal | Xcode 16.4 (16F6), CLT 27.0, `metal` 32023.620 already installed |

The project was created with
`pixi init . -c https://conda.modular.com/max/ -c conda-forge` and then
`pixi add "mojo>=1.2,<1.3" python=3.12 numpy pytensor pymc nutpie pytest numba`.
This follows the current install page at mojolang.org.

---

# Phase 1: feasibility spike

Phase 1 asked three questions (Q1-Q3) about whether Mojo is a viable compile target for
PyTensor, plus two short side checks.

All numbers come from an Apple M1 Pro and are the minimum over timeit repeats, so expect a few
percent of run-to-run noise.

## Q1. Python to Mojo call overhead: good, if you avoid the convenient path

Mojo functions are exposed with `PythonModuleBuilder.def_function` and built with
`mojo build --emit shared-lib`. The result is a normal CPython extension module.

| call | ns/call |
|---|---|
| Python `def f(x): return x` | 23 |
| C builtin `math.sqrt(2.0)` | 23 |
| Mojo `noop0()` | 23 |
| Mojo `noop1(x)` | 21 |
| Mojo `first_ptr(addr)`: raw buffer address as Python int, read one element | **37** |
| Mojo `first_span(arr)`: `std.python.numpy.from_numpy_array` zero-copy borrow | **2729** |
| Python `x.ctypes.data` (address lookup) | 572 |
| NumPy `x.sum()`, n=10 | 375 |

Throughput (us per call):

| n | np.sum | Mojo sum, span loop | Mojo sum, ptr + SIMD | np axpy (`out=`) | Mojo axpy, span | Mojo axpy, ptr + SIMD |
|---|---|---|---|---|---|---|
| 10 | 0.38 | 2.50 | **0.06** | 0.64 | 6.36 | **0.09** |
| 1,000 | 0.57 | 3.54 | **0.10** | 0.94 | 6.66 | **0.21** |
| 100,000 | 19.3 | 98.7 | **9.4** | 37.5 | 91.6 | **20.2** |
| 10,000,000 | 2269 | 9541 | **1498** | 5541 | 8904 | **3307** |

What this shows:

- **Raw call overhead is the same as a C extension, about 20-40 ns.** That is negligible next to
  NUTS needs. Each extra `PythonObject -> Int/Float64` argument conversion adds roughly
  15-30 ns.
- **The documented zero-copy NumPy helper (`from_numpy_array`) costs about 2.7 us per array.** It
  probably validates through Python attribute lookups (`ctypes`, `flags`, `dtype`). A naive
  binding per gradient call would spend most of its time there. The fix is to precompute buffer
  addresses once (`arr.ctypes.data`, about 0.57 us, also too slow per call) and pass ints, or
  pack them into one address table. With that, a full call that reads data stays under 40 ns.
- With explicit SIMD (`ptr.unsafe_offset(i).unsafe_load[width=8]()`), Mojo beats NumPy at every
  size: about 6x for small n, where NumPy's dispatch dominates, and 1.5-2x at 10M. A scalar
  `for v in span` reduction does **not** auto-vectorize. That is expected for strict FP, but
  generated code has to emit SIMD explicitly.
- Mojo 1.2 is mid-migration. `ptr[i]`, `+` and `load` all warn as deprecated in favour of
  `unsafe_offset=` / `unsafe_load`, so expect API churn.

## Q2. Hand-written eight schools: correct to 1e-10 and much faster per eval

`mojo_src/eight_schools.mojo` implements the non-centered model's logp and gradient
w.r.t. `q = [mu, tau_log__, theta_tilde[0..7]]`, including the log-transform Jacobian.

Correctness:

- 25 random points, scale 0.1 to 3, match `model.compile_logp()` and `model.compile_dlogp()` at
  rtol 1e-10.
- They also match the flat-vector `pytensor.function` in both the default C mode and NUMBA mode.
- logp matches scipy at rtol 1e-12, and the gradient matches central finite differences.

Timing, logp and grad together (d=10), us per eval:

| implementation | us/eval | vs C backend |
|---|---|---|
| PyTensor C backend, `f(q)`, `trust_input=True` | 1.646 | 1.0x |
| PyTensor Numba backend, `f(q)`, `trust_input=True` | 1.648 | 1.0x |
| Numba-jitted graph called directly (`f.vm.jit_fn`) | 0.800 | 2.1x |
| **Mojo `logp_dlogp(addrs...)` from Python** | **0.203** | **8.1x** |
| Mojo batched, per point (compute only, no Python) | 0.074 | 22x |
| *Stretch:* generated Mojo, raw `run(tab_addr)` | 0.124 | 13x |
| *Stretch:* generated Mojo via `pytensor.function` (value_vars signature) | 3.93 | 0.4x |
| PyTensor C backend, same value_vars signature | 2.40 | 0.7x |
| nutpie reference: compile 1.8 s; 1 chain, 1000 tune + 2000 draws in 0.078 s (13k post-warmup leapfrog steps) | | |

How to read this:

- The Mojo compute kernel takes about 70 ns. Everything above that is glue. PyTensor's own
  Function and VM wrapper costs about 1.5-2.5 us per call, whatever the backend. The generated
  Mojo backend called *through* `pytensor.function` is slower than C, because the v1 wrapper
  copies inputs into buffers and copies outputs out. The same generated code called raw takes
  0.124 us.
- **Conclusion: the speedup is real only if the sampler loop does not go through Python.** That is
  the nutpie architecture (Rust NUTS calling a numba cfunc). A Mojo NUTS sampler that calls the
  generated logp in-process, or a C-ABI export that nutpie or another sampler can call, is the
  design that pays off.
- The hand-written kernel is not a like-for-like comparison with the PyTensor graph. PyTensor's
  graph is a little less fused (4 separate Sum loops, a buffer write per intermediate). The
  generated-code result (0.124 us) is the fairer backend comparison.

### Surprises and blockers found in Q2

1. **`std.math.exp(Float64)` in Mojo 1.2 is not accurate to double precision.** It has up to
   8e-12 relative error on [-20, 20], which is thousands of ulps. libm, called through
   `external_call["exp", Float64]`, matches NumPy exactly. This first showed up as a 1e-9
   logp mismatch at large `tau_log__`. `tests/test_q1_mod.py::test_exp_accuracy_std_vs_libm`
   pins the behaviour and will fail once Mojo fixes it. Both the hand-written kernel and the
   generated code use libm `exp`. `log` and `log1p` looked correctly rounded at the few points
   checked. **Any serious backend needs an accuracy test suite for every transcendental it emits.**
2. **PyMC/PyTensor's own logp has float32-precision constants when parameters are Python
   literals.** `pt.as_tensor_variable(5.0)` autocasts to float32, and `log(2)` inside HalfCauchy's
   logp is an int8 constant, so it upcasts to float32. As a result, `pm.Normal("mu", 0, 5.0)`
   computes `log(5)` in float32. On eight schools the logp is off by a constant -5.8e-8
   (1.3e-9 relative), which fails rtol 1e-10 against an exact implementation.
   `0.6931471824645996` appears constant-folded in the optimized graph. The reference model
   therefore passes `np.float64(5.0)`. The remaining HalfCauchy `log(2)` error (1.9e-9 absolute)
   is inside tolerance. `test_pymc_literal_params_float32_offset_documented` proves the offset
   is exactly the float32 rounding of those logs. This does not affect gradients or sampling,
   but a backend test suite has to compare against PyTensor's own graph, not against scipy,
   at 1e-10.

## Q3. What a minimal PyTensor backend needs

See `docs/backend_interface.md` for the full write-up. In short:

- The interface is small. Subclass `JITLinker` and implement `fgraph_convert` (FunctionGraph to
  callable), `jit_compile`, and `create_thunk_inputs`. Declare `required_rewrites` and
  `incompatible_rewrites`, then wrap it in a `Mode`. The Numba linker is 32 lines and the MLX
  linker 143. All the work is in the per-Op dispatch: numba has about 8.4k lines, MLX about 5.3k.
- Numba and MLX compose per-Op Python closures through `fgraph_to_python`. Mojo cannot JIT
  Python closures, so a Mojo backend should **emit one Mojo source per FunctionGraph** and build
  it as a shared lib. Use singledispatch as the per-Op codegen registry.
- **No autodiff is needed in the backend.** `pt.grad` produces forward Ops.
- Op census: across logp+grad of 6 typical PyMC models (eight schools, linear regression,
  logistic, hierarchical Poisson with indexing, StudentT/Gamma/Beta, MvNormal+LKJ) after
  backend-neutral rewrites there are **17 tensor Op kinds and 22 scalar Op kinds**. Tier 1
  (most GLM and hierarchical models) is about 25 tensor plus 40 scalar Ops plus a real N-d
  strided buffer model. The estimate is 4-8 weeks for one developer. Linalg (Cholesky,
  SolveTriangular, SlogDet) adds about 3-4 weeks. Scan is a separate project.

### Stretch prototype: done (`pytensor_mojo/linker.py`)

`MojoLinker` plus the `MOJO` mode turn the optimized eight-schools logp+grad FunctionGraph into
one generated Mojo function. It supports Elemwise (with inlined nested Composite), Sum over all
axes, and 0-d <-> (1,) DimShuffle, for float64 tensors of ndim 0 or 1 with static shapes. It
builds through `mojo build`, caches by content hash, and imports the `.so`.

- `pytensor.function(model.value_vars, [logp, *grads], mode=MOJO)` matches the C backend at
  rtol 1e-12 on 25 random points, plus a small hand-built graph. Unsupported Ops raise
  `NotImplementedError`.
- Compile latency is 0.96 s cold (mojo build) and 0.04 s warm (cached .so). NUMBA takes 0.38 s
  on a numba cache hit. Mojo build time for the hand-written module is about 0.9 s.
- The generated code is a set of plain scalar loops. It has no SIMD and does not fuse
  reductions, and it still runs in 0.124 us.

## Apple GPU stretch (Metal, float32)

Setup notes:
- The Metal toolchain was **already installed** (`xcrun -f metal` resolves; Xcode 16.4), so
  `xcodebuild -downloadComponent MetalToolchain` was not run, and nothing needed sudo or
  user interaction.
- GPU APIs live in the **`max` package, not `std`**. With only the `mojo` package installed,
  `from max.gpu import ...` fails to resolve, and `std.gpu` does not exist (the compiler hints
  that "many stdlib items recently moved to the `max` package"). `pixi add max` (26.7.0) fixes it.
- Kernel arguments must be fixed-width. `Int` is not `DevicePassable`, so pass `Int32`.

Results (`gpu/`, `tests/test_gpu.py`):

- **Vector add:** float32, 1M elements on `Apple M1 Pro` through `DeviceContext` /
  `enqueue_function`. **0 mismatches.**
- **Batched float32 eight-schools logp + grad:**
  - Each GPU thread evaluates one point, with data baked in as SIMD constants and `std.math`
    exp/log in float32.
  - The reference is the float64 CPU Mojo kernel evaluated at the same float32-rounded points.
  - The grad error metric is max|dg| / max|g| per point. Point scale is the standard deviation
    of the random unconstrained points; at scale 3, tau reaches about e^9.

| n | point scale | GPU ns/point (kernel only) | CPU f64 ns/point (1 thread) | logp max rel err | logp median rel err | grad max norm-rel err |
|---|---|---|---|---|---|---|
| 1,000 | 1.0 | 99 (launch-latency bound, about 0.1 ms/launch) | 157 | 2.5e-7 | 4.6e-8 | 3.6e-7 |
| 20,000 | 1.0 | 7.2 | 91 | 2.5e-7 | 4.5e-8 | 6.3e-7 |
| 1,000,000 | 0.5 | 2.66 | 110 | 3.0e-7 | 4.6e-8 | 4.6e-7 |
| 1,000,000 | 1.0 | 2.61 | 113 | 3.8e-7 | 4.5e-8 | 8.4e-7 |
| 1,000,000 | 3.0 | 2.98 | 95 | **1.7e-6** | 5.3e-8 | **1.7e-6** |

How to read this:

- float32 errors are about 1e-7 typical and up to about 2e-6 worst case. That is fine for
  many-chain or pathfinder-style batched logp evaluation and for warmup heuristics. It is
  **not** fine for the NUTS energy/Hamiltonian accept test on high-dimensional models, where
  float32 roundoff in a summed logp grows with the number of terms. Apple GPUs have no float64,
  so on a Mac the GPU fits massively parallel chains or batched evaluation, not a
  single-chain NUTS.
- At 1M points the GPU is about 40x faster per point than one CPU thread running float64. It
  only wins with large batches: at n=1000, launch latency of about 0.1 ms dominates, and one
  launch is about 1 ms on a cold pipeline.
- Timing is kernel-only, using 20 back-to-back launches after a warm-up. It excludes
  host-to-device copies. Unified memory helps, but the copies were not measured.
- The comparison is unfair to the CPU: CPU is 1 thread and float64, GPU is float32.

## Other checks (short, time-boxed)

- **nabla (github.com/nabla-ml/nabla):** it is a *Python* framework on MAX's Python graph API.
  The repo is 99.9% Python with about 1 KB of Mojo, and the last push was 2026-03. It is not a
  Mojo autodiff library. `pip install nabla-ml modular` resolves to MAX 26.7.0 / Mojo 1.2.0,
  but `import nabla` hard-fails: "Nabla requires Modular nightly builds". With that gate bypassed
  in a scratch venv, `nb.grad` crashes inside the graph engine
  (`'ModuleOp' object has no attribute 'operation'`), which is MAX API drift. **It does not work on
  stable Mojo 1.2 / MAX 26.7, and it is not useful for this project.**
- **MAX autodiff:** the MAX 26.7 Python package (`max.graph.ops`, `max.experimental`) has no
  grad, vjp or jvp API. The only "gradient" mentions are conv-transpose docs. This is not a
  blocker, because PyTensor supplies symbolic gradients.

## Phase 1 go / no-go

**GO for a scoped backend, but only as part of a design where the sampler loop never re-enters Python.**

For:
- The Mojo kernel is fast: 70-125 ns per logp+grad versus 800 ns for numba's jitted graph
  fn on this model, and SIMD beats NumPy at every size.
- The PyTensor backend interface is small and well understood.
- Codegen from a FunctionGraph worked on the first pass for the eight-schools graph.
- Compile latency is acceptable at about 1 s cold.

Against, or risks:
- Mojo is still beta and churning. The Python-binding docs label it "early development", and
  1.2 deprecates the basic pointer syntax.
- The stdlib numerics are not trustworthy by default (`exp` has 1e-12 error).
- The convenient NumPy interop path costs 2.7 us per array.
- The PyTensor Function wrapper adds 1.5-2.5 us per call, which erases the kernel's advantage
  for NUTS driven from Python.
- The real work is the N-d strided buffer model plus about 65 Ops, roughly 1-2 months.

Suggested next steps:
1. Build the N-d Elemwise/CAReduce/Subtensor codegen against the tier-1 op list, with an
   accuracy test harness that compares to PyTensor's C backend Op by Op.
2. Export the generated logp+grad through a C ABI (`@export abi("C")`, pointer in and pointer
   out) and call it from nutpie's Rust side or from a Mojo NUTS. Measure end-to-end ESS/s
   against nutpie+numba.
3. Track the exp accuracy issue upstream, and bind libm for all transcendentals until it is fixed.

---

# Phase 2: NUTS with no Python in the gradient loop

## Setup

- **Mojo side.** The model is exported with `@export def es8_logp_grad(...) abi("C") -> Int32`
  and built with `mojo build --emit shared-lib` into `libes8.dylib`.
  - The library links `@rpath/libKGENCompilerRTShared.dylib` and `libAsyncRTMojoBindings.dylib`.
    The rpath into `.pixi/envs/default/lib` is baked in, so it loads with no `DYLD_*` variables.
  - Both models share one C ABI, so a single driver can run any of them:
    `int32 <m>_dim(const double* data)` and
    `int32 <m>_logp_grad(const double* data, const double* x, double* grad, double* logp)`.
    `data` is an opaque model-data buffer (NULL for eight schools), and status 0 means OK.
- **Rust side.** `sampler/` is the `mojo-nuts` crate.
  - It loads the library with `libloading` and implements nuts-rs's `CpuLogpFunc` and `Model`.
  - It runs `nuts_rs::Sampler` with Arrow trace storage on `cores = chains` threads.
- **Same sampler as nutpie.** nutpie 0.16.11 bundles **nuts-rs 0.18.3** (found with `strings` on
  its `_lib.so`), so the crate pins `=0.18.3`.
  - `nutpie_like_settings()` reproduces nutpie's `PyNutsSettings.Diag` defaults: diag
    grad-based mass matrix, maxdepth 10, target_accept 0.8, max_energy_error 1000, the same
    window schedule.
  - A test asserts that the serialized settings JSON is identical to nutpie's `as_dict()`, apart
    from chains, tune, draws and seed.
  - Both sides ran 1000 tune + 1000 draws.
- **Remaining differences.**
  - The logp+grad implementation: hand-written Mojo vs PyTensor's numba-compiled graph (nutpie
    calls a numba cfunc from Rust, so there is no Python per gradient on either side).
  - Init: uniform(-2, 2) in Mojo vs nutpie's jittered PyMC initial point.
  - nutpie's Python-side trace conversion. "sampler-only" below uses `return_raw_trace=True`
    to exclude it.
- **Larger model.** A hierarchical linear regression (HLR) was added:
  - N = 10,000 rows, G = 20 groups, K = 5 predictors, dim = 28.
  - **Centered** parameterization (`a_g ~ N(mu_a, sigma_a)`). The non-centered version mixed
    badly in *both* samplers (min ESS about 170, R-hat 1.03), because there are about 500
    rows per group.
  - Hand-written Mojo with 8-wide SIMD over rows, rows sorted by group, one fused pass.
  - Validated against PyMC `compile_logp`/`compile_dlogp` at rtol 1e-10, including a
    ragged-group test that exercises the scalar tail. The observed error is about 2e-13 for
    logp and under 4e-14 for the gradient.

## Correctness

| check | eight schools | HLR |
|---|---|---|
| posterior means (all params) vs nutpie | within 4 x sqrt(mcse_mojo^2 + mcse_nutpie^2) | same |
| posterior sd vs nutpie | within 20% (tau is heavy-tailed) | same |
| R-hat (4 chains) | < 1.01 (max 1.0025) | < 1.01 (max 1.0045) |
| ESS bulk | > 400 per param (min 2218 for tau) | min 7952 |
| divergences, 4000 draws (median of 5) | 2 Mojo vs 3 nutpie | 0 vs 0 |
| vs known values | mu about 4.4, tau about 3.6 | sigma = 0.80 (true 0.8) |

Example run, seed 7: mu 4.381 vs 4.431, tau 3.577 vs 3.597, theta[0] 6.226 vs 6.208
(Mojo vs nutpie). These are in `tests/test_nuts.py`, with fixed seeds.

## Speed

Apple M1 Pro, median of 5 runs. Sampling time only; compile time is excluded on both sides.
"sampler-only" for nutpie is `return_raw_trace=True`, i.e. without building the xarray
DataTree. Mojo's time runs from `Sampler::new` to finished traces, inside the Rust process. Grad
evals are total leapfrog steps (warmup + draws, from the `n_steps` stat). ESS/s and grad evals/s
below use sampler-only seconds.

| model | chains | impl | sampler-only s | full call s | grad evals | grad evals/s | min ESS bulk | ESS/s | speedup (sampler-only) |
|---|---|---|---|---|---|---|---|---|---|
| eight schools | 4 | Mojo + nuts-rs | 0.0182 | 0.0182 | 55,136 | 3.0e6 | 2218 | 1.2e5 | **0.9x** |
| eight schools | 4 | nutpie (numba) | 0.0163 | 0.0367 | 54,706 | 3.4e6 | 2192 | 1.3e5 | 1x |
| eight schools | 1 | Mojo + nuts-rs | 0.0096 | 0.0096 | 13,748 | 1.4e6 | 571 | 5.9e4 | **1.3x** |
| eight schools | 1 | nutpie (numba) | 0.0128 | 0.0291 | 13,472 | 1.1e6 | 577 | 4.5e4 | 1x |
| HLR N=10k | 4 | Mojo + nuts-rs | 0.286 | 0.286 | 55,623 | 1.9e5 | 7952 | 2.8e4 | **6.0x** |
| HLR N=10k | 4 | nutpie (numba) | 1.727 | 1.791 | 55,536 | 3.2e4 | 7880 | 4.6e3 | 1x |
| HLR N=10k | 1 | Mojo + nuts-rs | 0.261 | 0.261 | 13,828 | 5.3e4 | 1708 | 6.6e3 | **6.4x** |
| HLR N=10k | 1 | nutpie (numba) | 1.665 | 1.713 | 13,992 | 8.4e3 | 1700 | 1.0e3 | 1x |

Kernel-only logp+grad per eval:

| | eight schools | HLR N=10k |
|---|---|---|
| Mojo, timed from Rust | 0.033 us | 18.9 us |
| PyTensor numba graph (`jit_fn`, called from Python) | 0.95 us (mostly Python call overhead) | 125 us |
| same hand-fused HLR algorithm in numba (`bench/hlr_numba_hand.py`), fastmath on or off | n/a | 96 us |

nutpie compile time is 0.6-2.1 s; the Mojo .dylib build is about 1 s.

## Reading the numbers

- **Eight schools is bound by sampler overhead, not by logp.**
  - Mojo's kernel takes 33 ns, but a single chain costs about 0.7 us per leapfrog step:
    9.6 ms for 13.7k steps. The rest is nuts-rs itself (tree building, mass-matrix
    adaptation, Arrow storage) and is identical on both sides.
  - The result is a tie: 1.3x faster on 1 chain and 0.9x on 4 chains.
  - In both tools, 4 chains take about 1.7x as long as 1 chain on this tiny model, from
    thread start-up and storage. A shared atomic gradient counter was suspected first, but
    removing it changed 4-chain time only from about 17 ms to about 16 ms, so it was not the
    cause. This was not investigated further.
  - nutpie's full Python call adds about 15-20 ms of DataTree conversion, which is more than
    the sampling itself.
- **When compute dominates (HLR, 10k rows), Mojo is 6.0-6.4x faster end to end.** ESS per
  draw is the same, so ESS/s improves by the same factor. The two controls split that speedup:
  - Against the **same scalar algorithm hand-written in numba (96 us)**, Mojo (18.9 us) is
    about 5x faster. The gain comes from explicit 8-wide SIMD over rows, which numba does not
    produce from the scalar loop, even with fastmath. A numba kernel restructured for
    auto-vectorization was not tried, and might close part of that gap.
  - PyTensor's generated numba graph (125 us) is about 1.3x slower again than hand-fused
    numba. It pays for `a[g]` gather plus `AdvancedIncSubtensor` scatter, and for unfused
    intermediates.
  - **So the 6x is "hand-written SIMD Mojo" vs "PyTensor-generated numba". A generated Mojo
    backend would only capture it if the codegen emits SIMD and fuses reductions, which the
    Phase 1 prototype does not do.**
- The C ABI + nuts-rs route works and adds nothing measurable per call. Eight schools'
  per-step overhead equals nutpie's, which is evidence that the FFI hop is negligible.

## Surprises and blockers

1. **nuts-rs 0.18.3 `HashMapConfig` storage panics on the first divergence**
   ("Type mismatch when combining stats for divergence_message"): its finalize step does not
   handle String-typed stats. The workaround is to use `ArrowConfig`, which is what nutpie
   uses anyway. This is worth an upstream issue.
2. In 0.18.3 the posterior draw column is named `value` when `ExpandedVector = Vec<f64>`, not
   `unconstrained_parameter`.
3. Non-centered HLR with 500 rows per group mixes badly in both samplers; the centered
   parameterization fixed it. This is a modelling issue, not a sampler issue.
4. Mojo 1.2: `InlineArray` is now `std.collections.Array`, and pointer `bitcast`, `load` and
   `store` are deprecated in favour of `unsafe_*`.

## Phase 2 verdict

- The architecture works: Mojo C ABI to Rust nuts-rs, with no Python per gradient. It is
  correct against nutpie within MC error.
- It is **6x faster when logp compute dominates** and **at parity when sampler overhead
  dominates**.
- To get the 6x for real PyMC models, the Mojo backend's codegen must produce SIMD and fused
  reductions, not the scalar loops of the Phase 1 prototype.

---

# Phase 3: generated N-d, fused, SIMD code through the same C ABI

## What was built

**`pytensor_mojo/ndcodegen.py` + `ndlinker.py`** take PyTensor's optimized FunctionGraph and
generate a single Mojo source file. The graph is the same flat-vector logp+grad graph used as
the `compile_logp`/`compile_dlogp` equivalent, after backend-neutral `fast_run` rewrites with
PyTensor's FusionOptimizer Composites. The source is built with `mojo build --emit shared-lib`
into a `.dylib` that exports:

- `<m>_run(tab, shp)`: a generic entry point. `MOJO_ND`, a `pytensor.function` mode, uses it
  for arbitrary graphs.
- `<m>_dim` / `<m>_logp_grad`: the Phase 2 sampler ABI. **The Rust nuts-rs driver samples
  generated libraries with no code changes.**
  - Array constants (X, y, g, ...) go into the opaque `data` buffer through `pack_data`, with an
    offset header and int64 payloads stored bitwise. They are not baked into the source.

## Increments, each verified before the next

1. **N-d buffer model.**
   - Each array is a pointer plus shape, strides and offset, held as Mojo expressions. Dims
     that are static in the PyTensor type become literals; runtime dims are read from the shape
     table. Broadcast dims get stride 0.
   - Views never copy: general DimShuffle (transpose, expand, squeeze), Subtensor with constant
     index or slice, ScalarFromTensor, CheckParameterValue.
   - Ops: N-d Elemwise with broadcasting; Sum over all axes or given axes; Dot (2-D/1-D
     combinations); 1-D AdvancedSubtensor gather and AdvancedIncSubtensor scatter-add; Join;
     Alloc; and 26 scalar ops.
   - Unsupported ops raise `NotImplementedError("MojoND: unsupported op ...")`.
   - The op census was run on the HLR flat graph first:
     - Tensor ops: Subtensor{i} x3, Subtensor{a:b} x2, ExpandDims x10, Squeeze x2, Dot x2,
       AdvancedSubtensor x1, AdvancedIncSubtensor x1, Elemwise x10, Sum x6,
       CheckParameterValue x2, ScalarFromTensor x2, Join x1.
     - Scalar ops: Add, Sub, Mul, TrueDiv, Neg, Sqr, Exp, Log, GE, GT, Switch.
2. **Fusion.** For every 1-D Elemwise (including PyTensor Composites) over length n, one loop
   absorbs:
   - **Producers**, inlined per element: the gather `a[g[i]]`, and the row mat-vec
     `Squeeze(Dot(X, ExpandDims(beta)))`. X is packed column-major, so its column loads are
     contiguous.
   - **Consumers**, turned into in-loop accumulators: `Sum`, seen through `Check`; the
     transposed mat-vec `Squeeze(Dot(X^T, ExpandDims(w)))` as K SIMD accumulators; and the
     `AdvancedIncSubtensor` scatter-add, accumulated into a zeroed temp, with `base + acc` added
     when the node is reached.
   - Nodes already claimed by an earlier loop cannot be inlined again. One linreg-gradient case
     needed this: the same Dot was both a consumer of loop 1 and a producer for loop 2.
   - For HLR, the whole N=10,000 pass, i.e. logp likelihood, d/dsigma, d/dbeta and d/da, is
     **one loop with no N-length intermediates**. A test asserts that the workspace is a small
     static stack array.
3. **SIMD.**
   - The fused loop and the innermost dim of the N-d Elemwise loops are emitted at width
     W = 8 (`simd_width_of[float64]` is 2 on M1, so this is a 4x NEON unroll), plus a width-1
     scalar tail.
   - `exp` is libm per lane (`vexp[w]`), keeping the Phase 1 accuracy workaround. `log`/`log1p`
     come from std.math.
   - Gather and scatter load the int64 index lanes once per chunk. If all lanes share one index
     (sorted groups), the gather is a splat, and the scatter keeps a **run-length register
     accumulator** that is flushed only when the index changes. Otherwise both fall back to
     per-lane code. This stays correct for any index pattern; it is tested with sorted and
     unsorted indices, runs that cross chunk boundaries, and ragged tails.
4. **C-ABI emission + sampling.**
   - Per-call workspace is one arena. It is carved at compile time, and becomes a stack
     `Array` when small and static, so there is no malloc per gradient.
   - The sampler wrapper's address table is on the stack.

## Correctness

- **Generated vs PyMC** (`compile_logp`/`compile_dlogp`), 24 points at rtol 1e-10:
  - eight schools: max relative error 1.6e-16 (logp), 2.3e-14 (grad).
  - HLR: 2.9e-13 (logp), 1.1e-14 (grad).
  - HLR with ragged groups (N=1003, G=7, K=3): also passes.
- **Unit tests vs the C backend at rtol 1e-12:**
  - N-d broadcasting (matrix + row + scalar, transpose, tensor3, axis sums), run with 4
    different runtime shapes through one compiled library, including n < W.
  - Gather/scatter, sorted and unsorted, fused and unfused.
  - Fused rowdot + coldot.
  - CheckParameterValue failure returns status 1.
- **Sampling:** the generated es8 and HLR posteriors match nutpie within 4x combined MCSE,
  with sds within 20%, R-hat < 1.01, ESS > 400 and comparable divergences
  (`tests/test_nuts.py`).

## Speed

Apple M1 Pro, median of 5 runs, sampling only (compile excluded). nutpie is timed sampler-only
(`return_raw_trace=True`).

| model | chains | impl | sampler-only s | grad evals | min ESS bulk | ESS/s | speedup vs nutpie | time / hand-written time |
|---|---|---|---|---|---|---|---|---|
| eight schools | 4 | hand-written Mojo | 0.0177 | 55,136 | 2218 | 1.25e5 | 0.93x | 1x |
| eight schools | 4 | **generated Mojo** | **0.0167** | 54,876 | 2013 | 1.21e5 | 0.99x | 0.94x |
| eight schools | 4 | nutpie (numba) | 0.0165 | 54,706 | 2192 | 1.33e5 | 1x | |
| eight schools | 1 | hand-written Mojo | 0.0094 | 13,748 | 571 | 6.1e4 | 1.35x | 1x |
| eight schools | 1 | **generated Mojo** | **0.0095** | 13,706 | 461 | 4.9e4 | 1.34x | 1.01x |
| eight schools | 1 | nutpie (numba) | 0.0127 | 13,472 | 577 | 4.5e4 | 1x | |
| HLR N=10k | 4 | hand-written Mojo | 0.284 | 55,623 | 7952 | 2.8e4 | 6.2x | 1x |
| HLR N=10k | 4 | **generated Mojo** | **0.317** | 55,599 | 7557 | 2.4e4 | **5.5x** | **1.12x** |
| HLR N=10k | 4 | nutpie (numba) | 1.753 | 55,536 | 7880 | 4.5e3 | 1x | |
| HLR N=10k | 1 | hand-written Mojo | 0.295 | 13,828 | 1708 | 5.8e3 | 6.1x | 1x |
| HLR N=10k | 1 | **generated Mojo** | **0.335** | 13,828 | 1708 | 5.1e3 | **5.3x** | **1.14x** |
| HLR N=10k | 1 | nutpie (numba) | 1.788 | 13,992 | 1700 | 9.5e2 | 1x | |

For single-chain HLR, the generated and hand-written kernels take **identical** trajectories
(13,828 steps, same ESS), as you'd expect when the numerics agree to about 1e-13.

Kernel-only logp+grad per eval, timed from Rust, median of 3:

| | hand-written | generated (default) | generated, `PYTENSOR_MOJO_RECIP=1` | PyTensor numba graph |
|---|---|---|---|---|
| eight schools | 33 ns | **50 ns** (1.5x) | n/a | 0.9 us (mostly Python call) |
| HLR N=10k | 18.0 us | **19.9 us (1.10x)** | 18.6 us (1.03x) | 122 us |

Success criteria: kernel within 1.5x is **met** (1.10x HLR, 1.5x es8). End-to-end sampling
within 1.5x is **met** (1.12-1.14x HLR, 0.94-1.01x es8). The posterior matches nutpie, which is
**met**.

Build time for a generated library is 1.2 s cold (`mojo build`) and about 0.1 s from cache.
nutpie compiles in 0.7-2.0 s.

## How the gap closed, and what is left

Measured on the HLR kernel, from first working version to final:

| step | HLR us/eval | ratio vs hand-written |
|---|---|---|
| first correct fused + SIMD version | 36.7 | 2.04x |
| + index lanes loaded once per chunk, splat gather, **run-length scatter accumulator** | 23.0 | 1.28x |
| + one hoisted workspace arena instead of a malloc per buffer; stack address table | 22.0 | 1.22x |
| + stack arena when small and static | 19.9-21.9 | 1.10-1.2x |
| (opt-in) loop-invariant `x / s` as `x * (1/s)` | 18.6 | 1.03x |

- **The big one was the scatter-add.** Sorted groups made consecutive lanes read-modify-write
  the same address, which is a serial store-to-load dependency chain through memory. The
  register run accumulator removes it without assuming the data is sorted.
- **The remaining about 1.9 us (10%) breaks down as:**
  - About 1.3 us: two vector divisions per element by the loop-invariant sigma. PyTensor's graph
    contains `(r / sigma) / sigma`, and the hand-written kernel multiplies by `1/sigma^2`.
    Rewriting division as multiply-by-reciprocal changes rounding by 1-2 ulp, so it is opt-in
    (`PYTENSOR_MOJO_RECIP=1`); tests still pass at 1e-10.
  - About 0.6 us: small unfused pieces: the 20- and 5-length Elemwise loops, `base + acc` for
    the scatter, the Join copy into grad, and the wrapper. Also, PyTensor constant-folds X and
    X^T into **two separate 400 KB constants**; the hand-written kernel reads one X twice. So
    the generated code's cache footprint is 800 KB instead of 400 KB.
- **Eight schools: 50 ns vs 33 ns.** That is per-call overhead from the same small unfused
  ops. It is invisible in sampling, where nuts-rs spends about 700 ns per leapfrog step.

## Limits / NOT done

- **Fused loops specialize on static dims** (N, K, G come from the PyMC constants'
  types). Fused rowdot/coldot need a static K, and fused scatter needs a static base length.
  Runtime-shaped graphs go through the generic N-d path (tested), but they are not fused
  through dots or scatters.
- The generic N-d Elemwise vectorizes only the innermost dim. Strided inner dims use per-lane
  loads.
- Generic `Dot`, `Sum(axis)` and N-d `Sum` are plain scalar loops; there is no BLAS.
- Gather and scatter are 1-D only (`a[idx]`). There is no multi-dim advanced indexing and no
  `set_subtensor`.
- There is no Cholesky/linalg, Scan, RandomVariable, Sort, or most special functions (gammaln,
  psi, erf). Those are tier 1/2 in `docs/backend_interface.md`. Only the HLR, eight-schools and
  unit-test graphs have been run.
- `MOJO_ND` through `pytensor.function` still pays PyTensor's per-call Python wrapper (Phase 1
  finding). The fast path is the C ABI driven from Rust.
- The bool and int64 support is minimal: index arrays and scalar conditions only.

---

# Phase 4: special functions and common likelihoods

Goal: make common PyMC likelihoods and priors compile and sample through `ndcodegen`. Each
increment was verified before the next.

## 1. Census

`bench/phase4_census.py` builds each distribution twice:

- as an observed likelihood, with free parameters under simple priors;
- as a free prior with PyMC's default transform.

`ndcodegen.missing_ops(fgraph)` lists every tensor and scalar op the codegen cannot emit for
the optimized logp + gradient graph, in one pass.

The census before Phase 4:

| distribution | role | transforms | missing tensor ops | missing scalar ops |
|---|---|---|---|---|
| StudentT | observed | Log | CAReduce[AND], MakeVector | GammaLn, Psi, Sign |
| Gamma | observed | Log | CAReduce[AND], MakeVector | GammaLn, Psi |
| Beta | observed | Log | CAReduce[AND], MakeVector | GammaLn, Psi |
| Poisson (log link) | observed | - | CAReduce[AND] | - |
| Binomial (Uniform p) | observed | Interval | CAReduce[AND], MakeVector | - |
| NegativeBinomial | observed | Log | CAReduce[AND], MakeVector | GammaLn, Psi |
| Bernoulli (logit_p) | observed | - | CAReduce[AND], MakeVector | - |
| LogNormal | observed | Log | - | - |
| Exponential | observed | Log | - | - |
| HalfNormal | observed | Log | - | - |
| InverseGamma | observed | Log | CAReduce[AND], MakeVector | GammaLn, Psi |
| Weibull | observed | Log | CAReduce[AND], MakeVector | Pow |
| Laplace | observed | Log | - | Sign |
| Cauchy | observed | Log | - | - |
| StudentT | prior | - | - | - |
| Gamma | prior | Log | - | - |
| Beta | prior | LogOdds | - | - |
| LogNormal | prior | Log | - | - |
| Exponential | prior | Log | - | - |
| HalfNormal | prior | Log | - | - |
| InverseGamma | prior | Log | - | - |
| Weibull | prior | Log | - | - |
| Laplace | prior | - | - | Sign |
| Cauchy | prior | - | - | - |
| Uniform (interval) | prior | Interval | - | - |
| HalfCauchy | prior | Log | - | - |

The union was **2 tensor ops** (`All`, i.e. CAReduce[AND] over parameter checks, and
`MakeVector`) and **4 scalar ops** (GammaLn, Psi, Sign, Pow). Two observations:

- Priors with constant parameters need no special functions at all, because PyTensor
  constant-folds terms like `gammaln(alpha)`.
- erf/erfc are not needed by any of these, since they appear only in CDFs, i.e. censored or
  truncated likelihoods.

After Phase 4 the census reports nothing missing.

## 2. Implementation

| op | implementation | gradient |
|---|---|---|
| gammaln | macOS libm `lgamma_r` (reentrant; plain `lgamma` writes the global `signgam`, a data race across sampler threads) | psi |
| psi (digamma) | own: reflection with exact `__sinpi`/`__cospi` for x <= 0; degree-25 Taylor series about the positive root 1.4616... for \|x - root\| < 0.25 (coefficients `polygamma(n, root)/n!`); otherwise recurrence to y >= 10 in double-double (TwoSum plus FMA residual of each 1/y) and the asymptotic series through B_14 | trigamma: **not implemented**, raises clearly (NUTS needs first order only) |
| erf, erfc, expm1, pow, floor, ceil, log2, log10, tanh, sin, cos | libm per lane | via existing ops |
| log1mexp | Maechler (2012): `log(-expm1(x))` for x > -log 2, else `log1p(-exp(x))` | via existing ops |
| sign, isinf, isnan | SIMD compare/select | zero / n.a. |
| All (CAReduce AND) | fused into the loop as a SIMD bool AND accumulator; scalar/MakeVector forms folded | n.a. |
| MakeVector | bool lists folded into All; float vectors materialized | n.a. |

Codegen fixes the new graphs needed:

- **Integer and bool data in arithmetic.** Integer observations (counts, Bernoulli y) and
  constant-folded bool masks (e.g. `y == 0` in Poisson) are now packed as float64. Integer
  arrays used only as gather/scatter indices stay int64.
- **PyTensor semantics reproduced:**
  - a nonzero numeric Switch condition is true;
  - bool operands of float ops are upcast;
  - Mul/Add of bools are logical and/or.
- **Bool arrays** that must be materialized are stored as float64 0/1 masks, and views keep
  the mask flag.
- **0-d integer constants** are rendered as Float64.

## Accuracy

`bench/phase4_accuracy.py` (also asserted in `tests/test_special.py`):

- Each op is compiled through the `MOJO_ND` backend, so both the SIMD lanes and the scalar
  tail run.
- References: a 40-digit `Decimal` implementation (`bench/hiprec_ref.py`; recurrence plus
  Stirling/asymptotic series with exact Bernoulli numbers, reflection with Decimal sin/cos,
  erfc continued fraction / Taylor series) for gammaln, psi and erfc; numpy/scipy elsewhere.
- The last column shows scipy's own error against the high-precision reference.

| op | range | n | reference | max ulp | median ulp | max rel err | scipy vs hi-prec (max rel) |
|---|---|---|---|---|---|---|---|
| gammaln | tiny (1e-300..1e-3) | 1000 | 40-digit Decimal | 0 | 0 | 0.00e+00 | 2.21e-16 |
| gammaln | (1e-3, 50) | 1000 | 40-digit Decimal | 2 | 0 | 3.00e-16 | 1.52e-14 |
| gammaln | near 1 and 2 (zeros) | 1000 | 40-digit Decimal | 1 | 0 | 2.21e-16 | 1.59e-10 |
| gammaln | large (50..1e300) | 1000 | 40-digit Decimal | 1 | 0 | 1.67e-16 | 2.11e-16 |
| gammaln | negative non-integer | 1000 | 40-digit Decimal | 155 | 0 | 3.19e-14 | 4.36e-15 |
| psi | tiny (1e-300..1e-3) | 1000 | 40-digit Decimal | 1 | 0 | 2.02e-16 | 2.09e-16 |
| psi | (1e-3, 10) | 1000 | 40-digit Decimal | 15 | 1 | 1.80e-15 | 3.26e-16 |
| psi | near root 1.4616 | 1000 | 40-digit Decimal | 13 | 1 | 1.75e-15 | 2.21e-16 |
| psi | very near root (+-1e-8) | 1000 | 40-digit Decimal | 2 | 1 | 4.41e-16 | 2.29e-16 |
| psi | large (10..1e300) | 1000 | 40-digit Decimal | 0 | 0 | 0.00e+00 | 1.94e-16 |
| psi | negative, >=1e-3 from poles | 1000 | 40-digit Decimal | 131 | 1 | 2.08e-14 | 3.48e-14 |
| erf | (-6, 6) | 4000 | scipy.special | 3 | 0 | 4.35e-16 | - |
| erf | tiny | 4000 | scipy.special | 3 | 0 | 4.11e-16 | - |
| erfc | (-6, 1) | 1000 | 40-digit Decimal | 3 | 0 | 5.01e-16 | 9.13e-16 |
| erfc | (1, 5) | 1000 | 40-digit Decimal | 4 | 1 | 6.58e-16 | 2.22e-15 |
| erfc | deep tail (5, 26.5) | 1000 | 40-digit Decimal | 5 | 1 | 6.26e-16 | 5.68e-14 |
| expm1 | (-50, 50) | 4000 | numpy | 0 | 0 | 0.00e+00 | - |
| expm1 | tiny | 4000 | numpy | 0 | 0 | 0.00e+00 | - |
| log1mexp | (-50, -1e-3) | 4000 | numpy | 1 | 0 | 2.21e-16 | - |
| log1mexp | near 0 (-1e-3..-1e-300) | 4000 | numpy | 1 | 0 | 2.05e-16 | - |
| log1mexp | around -log 2 | 4000 | numpy | 0 | 0 | 0.00e+00 | - |
| pow | x in (1e-3, 100), y in (-10, 10) | 4000 | numpy | 0 | 0 | 0.00e+00 | - |
| log (std.math) | (1e-300, 1e300) | 4000 | numpy | 0 | 0 | 0.00e+00 | - |
| log (std.math) | near 1 | 4000 | numpy | 1 | 0 | 1.36e-16 | - |
| log1p (std.math) | (-0.999, 1e10) | 4000 | numpy | 1 | 0 | 2.20e-16 | - |
| log1p (std.math) | tiny | 4000 | numpy | 0 | 0 | 0.00e+00 | - |
| sqrt (std.math) | (1e-300, 1e300) | 4000 | numpy | 0 | 0 | 0.00e+00 | - |

Reading the table:

- **gammaln (libm `lgamma_r`) is correctly rounded within 1-2 ulp for x > 0, including at its
  zeros x = 1, 2.** SciPy is not: its relative error near the zeros is 1.6e-10 on this grid,
  about 3e-9 for \|x - 1\| < 1e-3 and 8.8e-4 for \|x - 1\| < 1e-9. The absolute error is
  tiny, but scipy cannot serve as the reference there. Negative x reaches 155 ulp (3e-14
  relative), libm's accuracy there, and it does not arise in these models.
- **psi is at most 15 ulp (1.8e-15) for x > 0.**
  - The worst points sit just outside the root window (x ~ 1.75, psi ~ 0.25). There the
    result is `log(y) - sum(1/k)` with \|result\| ~ 0.1 \|log y\|, so the half-ulp
    rounding of `log(y)` alone is about 8 ulp. Closing this would need a double-double log.
  - The compensated recurrence took the worst case from 25 to 13-15 ulp.
  - The root series keeps 2 ulp within 1e-8 of the root, where a plain recurrence loses all
    relative accuracy.
  - Negative x (reflection) gives 2e-14 relative at >= 1e-3 from the poles.
- **erfc (libm) is at most 5 ulp in the deep tail.** SciPy is about 90x worse there (5.7e-14)
  and flushes x > 26.55 to 0, where libm returns the correct subnormals.
- **expm1, pow, log1mexp and std.math log/log1p/sqrt are within 1 ulp.** Of Mojo's `std.math`
  functions, only `exp` is inaccurate (Phase 1).
- **Test tolerances** (`tests/test_special.py`): <= 4 ulp for libm-backed ops (erfc <= 6),
  <= 20 ulp for psi at x > 0, and <= 2e-13 relative for negative x.
- **Gradients** (grad of gammaln, erf, erfc, expm1, log1mexp, sign, pow with respect to both
  arguments) match the C backend at rtol 1e-12.

## 3. Census models vs PyMC

**All 26 census models** (14 likelihoods, 12 priors) compile and match PyMC's
`compile_logp` / `compile_dlogp` at **rtol 1e-10 on 24 points each**
(`tests/test_phase4_models.py`). No tolerance had to be loosened. Because the generated code
evaluates the same optimized graph, PyMC's float32-folded constants are reproduced exactly.

## 4. Sampling three representative models

`bench/phase4_sampling.py` (results in `bench/phase4_results.json`): 4 chains, 1000 tune + 1000
draws, median of 5 runs. nutpie is timed sampler-only. Both sides use nuts-rs 0.18.3 with
identical settings, and nothing is hand-written: all kernels are generated from the PyMC
model.

| model | dim | impl | sampler-only s | grad evals | min ESS bulk | ESS/s | max R-hat | divergences | speedup |
|---|---|---|---|---|---|---|---|---|---|
| robust StudentT regression, N=5000, K=3, nu free | 6 | generated Mojo + nuts-rs | 0.313 | 38,393 | 2713 | 8.7e3 | 1.0014 | 0 | **2.02x** |
| | 6 | nutpie (numba) | 0.631 | 38,075 | 2650 | 4.2e3 | 1.0019 | 0 | 1x |
| Poisson GLM, log link, N=10,000, K=5 | 6 | generated Mojo + nuts-rs | 0.390 | 32,520 | 5005 | 1.3e4 | 1.0023 | 0 | **2.51x** |
| | 6 | nutpie (numba) | 0.977 | 32,081 | 4993 | 5.1e3 | 1.0029 | 0 | 1x |
| hierarchical Beta-Binomial, G=200 | 202 | generated Mojo + nuts-rs | 0.131 | 96,777 | 2152 | 1.6e4 | 1.0059 | 0 | **1.65x** |
| | 202 | nutpie (numba) | 0.217 | 94,551 | 2191 | 1.0e4 | 1.0052 | 0 | 1x |

Kernel-only logp + gradient per evaluation:

| model | generated Mojo | PyTensor numba graph (`jit_fn`, from Python) |
|---|---|---|
| robust StudentT, N=5000 | 28.6 us | 82.4 us |
| Poisson GLM, N=10,000 | 46.4 us | 127 us |
| Beta-Binomial, G=200 | 3.85 us | 5.25 us (includes about 0.5 us of Python call) |

Posterior checks (`tests/test_phase4_sampling.py`), in the unconstrained space using nutpie's
`store_unconstrained=True`:

- means within 4.5 combined MCSE (the Beta-Binomial compares 202 coordinates);
- sds within 20%;
- R-hat < 1.01 and ESS > 400 for every coordinate;
- divergence counts comparable.

All three models also match PyMC at rtol 1e-10.

**Why the speedup is 1.6-2.5x here and not the regression's 5.5x.** The per-row work is now a
transcendental (`exp` in Poisson, `log1p` in StudentT, `lgamma` and `psi` in Beta-Binomial),
and the generated code evaluates libm one SIMD lane at a time. Measured trade-off on the
Poisson kernel:

| `exp` implementation | kernel us/eval | max logp rel. error vs PyMC (24 points) |
|---|---|---|
| libm per lane (default) | 47.4 | 1.7e-16 |
| Mojo `std.math.exp`, vectorized (`PYTENSOR_MOJO_STDEXP=1`) | 32.3 | 5.0e-13 |

The vectorized std exp is 1.47x faster but about 1e-12 off per call (Phase 1). It is opt-in;
an accurate vectorized exp/log1p/lgamma is the next step for these models.

## Surprises and fixes

1. **SciPy `gammaln` near its zeros** is not accurate in relative terms; see the accuracy
   section. That is why a Decimal reference was needed.
2. **Plain `lgamma` is not thread-safe** (`signgam`); `lgamma_r` is used instead.
3. **`signbit` is a macro in libm**, so it cannot go through `external_call`. Signed zero is
   detected with `1/x`.
4. **Mojo 1.2 details:**
   - A bool SIMD splat needs `SIMD[DType.bool, w](fill=True)`.
   - A comptime list element must be bound with `comptime c = LIST[k]` before runtime use.
5. **nutpie omits the `unconstrained_posterior` group** when no variable is transformed (as in
   the Poisson GLM).
6. **Integer observations and constant-folded bool masks** exercised the Switch, bool-upcast
   and bool-Mul semantics above. These are silent correctness traps for any new backend, and
   are now covered by the census tests.

## NOT done in Phase 4

- trigamma / polygamma (only needed for second-order methods), incomplete gamma/beta (CDFs,
  censoring, truncation), Bessel functions, erfcx.
- An accurate vectorized exp/log1p/lgamma.
- Distributions outside the census list (e.g. ZeroInflated*, mixtures, Dirichlet,
  multivariate, which need linalg).

---

# Phase 7: SIMD transcendentals

Phase 4 left the Poisson, StudentT and Beta-Binomial models at 1.6-2.5x against nutpie,
because their per-row `exp`, `log1p` and `lgamma` ran as one libm call per SIMD lane. Phase 7
adds accurate vectorized implementations and makes them the default.

## Implementation (`pytensor_mojo/mojo_math.py`)

The code is a Mojo prelude spliced into every generated file. Its hooks in `ndcodegen.py` are
small: five op-table entries, plus sigmoid and softplus, route through `mojo_math.call()`. All
functions are branch-free over `SIMD[DType.float64, w]`, with special cases handled by
`select`.

| Function | Algorithm |
|---|---|
| exp | fdlibm e_exp. Cody-Waite reduction by ln 2; the high part of the split constant has trailing zero bits, so `k ln2_hi` is exact. Remez rational kernel `1 - ((lo - r c / (2 - c)) - hi)`. `2^k` is built from the exponent bits and split into two exact factors, so subnormal results round once. |
| log | fdlibm e_log. `x = 2^k (1 + f)` with 1 + f in [sqrt(2)/2, sqrt(2)); subnormal inputs are pre-scaled by 2^54. `s = f / (2 + f)` with the Lg1..Lg7 polynomial. |
| log1p | fdlibm s_log1p. The same kernel on `u = 1 + x`, plus the correction `c / u` for the rounding of u. As in fdlibm, the choice of `c` uses u's raw exponent, from before the sqrt(2) renormalization. The first version used the renormalized exponent and measured 2.05 ulp. |
| expm1 | fdlibm s_expm1, with its k = 0, +-1, 2..19, 20..56 and far branches all evaluated and selected. |
| lgamma, x > 0 | **[0.75, 3]:** `(x - 1)(x - 2) g(x)`, where g is a degree-38 Chebyshev interpolant fitted in 40-digit Decimal (`bench/fit_lgamma.py`; coefficients decay to 1e-20). The factors are exact near the zeros, which keeps relative accuracy there. **Below 0.75:** `lgamma(x + 1) - log x`. **(3, 12):** downward recurrence as a product, then one log. **12 and above:** Stirling in the `(x - 1/2)(log x - 1)` form, through B14. x <= 0 lanes call libm `lgamma_r`, and only when some lane needs it. |

Two variants were rejected by measurement:

- **A division-free degree-13 Taylor `exp` kernel.** It had the same accuracy (max 0.86 ulp)
  but was 13% slower than fdlibm's rational form. The f64 vector divide is not the bottleneck
  on M1. Interleaved microbenchmark, ns per element: fdlibm rational 1.50, polynomial 1.69,
  libm per lane 6.45, Mojo `std.math.exp` 1.13 (inaccurate).
- **Mojo `std.math.exp`.** It has 8e-12 relative error (Phase 1). `PYTENSOR_MOJO_STDEXP` is
  retired; `PYTENSOR_MOJO_LIBM=1` restores per-lane libm for all five functions.

## Accuracy (10M points per function)

`bench/phase7_accuracy.py` runs each function through the generated SIMD code against the
40-digit Decimal reference, using all cores. For tiny |x|, log1p and expm1 use Decimal series,
because 1 + x would need hundreds of digits. The lgamma reference now uses one log of the
recurrence product, which is about 10x faster; the zeros x = 1, 2 are compared as exact zeros.

| Function | Points | Max ulp | Mean ulp | Correctly rounded |
|---|---|---|---|---|
| exp | 10,000,000 | 0.905 | 0.26 | 81.7-100% by set |
| log | 10,000,000 | 0.841 | 0.01-0.27 | 88.2-100% |
| log1p | 10,000,000 | 0.852 | 0.01-0.27 | 88.2-99.97% |
| expm1 | 10,000,000 | 0.855 | 0.01-0.27 | 87.6-100% |
| lgamma, x > 0 | 10,002,000 | 3.707 | 0.13-0.53 | 55.9-99.7% |

- **Where the worst cases are.** exp peaks at the reduction boundaries ((k + 1/2) ln 2); log
  and log1p near sqrt(2)/2; lgamma below x = 1, inside [0.75, 1), where x - 2 is not exact.
- **Special values.** inf, NaN, signed zero, subnormal inputs and the overflow and underflow
  thresholds are tested in `tests/test_mojo_math.py::test_special_values`.
- **Where this differs from SciPy.** `gammaln(-inf)` is +inf, as in C99 and libm, where SciPy
  gives -inf. `gammaln` of subnormal inputs is the correct `-log x`, where SciPy gives inf.

## Found along the way: SIMD `.ne()` is ordered in Mojo 1.2

`x.ne(x)` is False for NaN, unlike IEEE `!=`. The first NaN tests passed NaN through as a finite
number, and the Phase 4 `IsNan` op (`a.ne(a)`) had been silently wrong. NaN tests now use the
bit pattern (`|bits| > 0x7ff0...0`); this is pinned by `test_isnan_op_is_bitwise`, and listed on
the upstream-issues page.

## Performance

Kernel time per logp + gradient evaluation, timed from Rust. The libm and vectorized builds
are alternated in the same window, 7 rounds, minimum; the medians agree within 2%.

| Model | libm per lane (us) | Vectorized (us) | Speedup |
|---|---|---|---|
| Poisson GLM, N=10k | 46.5 | 36.2 | 1.28x |
| StudentT regression, N=5k | 31.9 | 23.9 | 1.33x |
| Beta-Binomial, G=200 | 3.98 | 2.53 | 1.57x |
| HLR, N=10k | 19.0 | 19.2 | 0.99x |
| Eight schools | 0.040 | 0.052 | 0.76x |

- **The exp portion itself is about 4x faster;** the whole kernel gains less because the rest
  of each row is unchanged. In the Poisson kernel, about 20 us of the per-evaluation time is
  the HLR-like mat-vecs, sums and gradient terms.
- **Eight schools is 12 ns slower per evaluation,** from the select-heavy code at width 1 on 8
  elements.

Sampling speedup against nutpie: 4 chains, 1000 tune + 1000 draws, median of 5, median of the
per-run ratios. Measured with no other benchmark jobs running; load average (1, 5, 15 min)
5.94 / 7.46 / 11.06 before and 4.04 / 5.39 / 8.79 after.

| Model | Phase 4, libm per lane | Phase 7, vectorized | Mojo s / nutpie s (Phase 7) |
|---|---|---|---|
| Poisson GLM, N=10k | 2.51x | **3.23x** | 0.330 / 1.058 |
| StudentT regression, N=5k | 2.02x | **2.44x** | 0.260 / 0.636 |
| Beta-Binomial, G=200 | 1.66x | **2.12x** | 0.100 / 0.213 |
| HLR, N=10k | 5.54x | 5.46x | 0.294 / 1.611 |
| Eight schools | 1.00x | **0.86x** | 0.0176 / 0.0148 |

- **Eight schools is a small regression at 4 chains.** The 1-chain figure is 1.28x (9.4 ms vs
  12.1 ms). The 12 ns per evaluation explains about 0.7 ms of a 17.6 ms 4-chain run; the rest
  is thread and sampler overhead on a tiny model, which was already 0.93x with the
  hand-written kernel in rev 0.1.0.
- **Earlier session.** Under load average 10-14 the same runs gave 3.14x / 2.34x / 2.15x /
  5.62x / 0.78x. An interleaved A/B of eight schools' libm and vectorized builds there gave
  18.1 vs 17.9 ms.
- **HLR** is within run-to-run variation of Phase 4, as expected: it has no transcendentals in
  its per-row work.

## Not done in Phase 7

- **psi** is still per lane. It appears in the gradients of models with gammaln (for example
  Beta-Binomial).
- **erf, erfc, pow and trig** are still per-lane libm calls.
- **lgamma for x <= 0** goes through libm.

---

# Caveats across all phases

- Phase 3 samples generated code for two models only (eight schools, HLR); see the Phase 3
  limits above.
- Phase 2 benchmarks are 5 reps each on one machine, with no CPU pinning; the M1 Pro has
  8 performance and 2 efficiency cores.
- Python 3.13 was not tried (3.12 pinned up front).
- GPU coverage is only the two float32 stretch kernels; there is no GPU codegen from PyTensor graphs.
- Q1/Q2 benchmarks are single-machine and single-thread, with no CPU pinning.
- The Phase 1 linker (`pytensor_mojo/linker.py`, v1) supports only ndim <= 1 with static shapes
  and was only run on eight schools. It is superseded by the Phase 3 codegen
  (`pytensor_mojo/ndcodegen.py`) and kept for its tests.


---

## Test record at the end of Phase 3

The last line of `pixi run test`:

```
tests/test_ndcodegen.py::test_generated_hlr_matches_pymc_and_is_fused PASSED [ 79%]
tests/test_ndcodegen.py::test_generated_hlr_ragged_groups PASSED         [ 80%]
tests/test_ndcodegen.py::test_check_failure_returns_status PASSED        [ 82%]
tests/test_nuts.py::test_settings_match_nutpie_defaults PASSED           [ 83%]
tests/test_nuts.py::test_es8_posterior_matches_nutpie PASSED             [ 85%]
tests/test_nuts.py::test_es8_divergences_comparable PASSED               [ 87%]
tests/test_nuts.py::test_es8_known_posterior_ballpark PASSED             [ 88%]
tests/test_nuts.py::test_hlr_posterior_matches_nutpie_and_truth PASSED   [ 90%]
tests/test_nuts.py::test_rust_driver_rejects_bad_symbol PASSED           [ 91%]
tests/test_nuts.py::test_generated_es8_posterior_matches_nutpie PASSED   [ 93%]
tests/test_nuts.py::test_generated_hlr_posterior_matches_nutpie PASSED   [ 95%]
tests/test_q1_mod.py::test_noops PASSED                                  [ 96%]
tests/test_q1_mod.py::test_zero_copy_reads_and_writes PASSED             [ 98%]
tests/test_q1_mod.py::test_exp_accuracy_std_vs_libm PASSED               [100%]
============================= 62 passed in 36.72s ==============================
(tests/test_gpu.py: 4 GPU tests, skipped automatically if the Mojo GPU build is unavailable)

pixi run test-rust: cargo fmt --check OK, cargo clippy --release --all-targets -- -D warnings OK,
test result: ok. 2 passed; 0 failed
```
