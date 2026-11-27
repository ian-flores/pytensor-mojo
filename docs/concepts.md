# Concepts

## Architecture

<figure class="ds-figure" markdown>
--8<-- "fig-arch.svg"
<figcaption>Compile-time path in Python (left) and the native sampling path (right). After compilation, no Python runs per gradient.</figcaption>
</figure>

1. **PyMC to a graph.** PyMC builds `logp(jacobian=True)` over the model's value variables, and
   `pytensor.grad` adds the gradient as more graph. Both are written over one flat vector `q`.
2. **Rewrites.** PyTensor's backend-neutral rewrites run, including elementwise fusion into
   Composites.
3. **Code generation.** `ndcodegen` emits one Mojo source file for the whole graph, compiled
   into a shared library with a [C ABI](guide/c-abi.md).
4. **Sampling.** `mojo-nuts` (Rust, nuts-rs) calls `logp_grad` in-process from its worker threads.

## No autodiff in the backend

PyTensor differentiates symbolically, so the gradient of logp is ordinary forward ops
(`grad(gammaln)` is `psi`, `grad(gather)` is a scatter-add, and so on). The backend only
has to evaluate a graph; it never has to differentiate one. The only cost is that every op
appearing in a gradient must be supported too.

## Buffer model

- **Arrays.** An array value is a typed pointer plus `shape`, `strides` and an element
  `offset`, each held as a Mojo expression. Dims that are static in the PyTensor type become
  literals, so loops over them unroll. Other dims are read at runtime from a shape table.
  Broadcast dims get stride 0.
- **Views, not copies.** DimShuffle (transpose, expand, squeeze), Subtensor with constant
  indices, ScalarFromTensor and parameter checks are views.
- **Scalars and broadcasts.** 0-d values are Mojo locals, and arrays that are a broadcast
  scalar stay scalars.
- **Workspace.** It is one arena per call, laid out at compile time and placed on the stack
  when small.

## Fusion

For each 1-D elementwise node over the data axis (length n), the generator emits one loop.
That loop absorbs, when they have no other users:

| Absorbed | Kind | How |
|---|---|---|
| `a[g[i]]` (AdvancedSubtensor) | producer | gathered per element inside the loop |
| `X @ beta` (row mat-vec) | producer | `sum_k X[i,k] * beta[k]`, with X stored column-major |
| `sum(...)`, `all(...)` | consumer | SIMD accumulator, reduced after the loop |
| `X.T @ w` (column mat-vec) | consumer | K SIMD accumulators |
| `inc_subtensor(a[g], w)` (scatter-add) | consumer | run-length accumulator (below) |

For the hierarchical regression, the whole N=10,000 pass is one loop with no N-length
intermediates. That pass covers the likelihood, d/dsigma, d/dbeta and d/da.

## SIMD

Fused loops and the innermost dimension of N-d elementwise loops are emitted at width W = 8
float64 lanes, plus a width-1 tail. `simd_width_of[float64]` is 2 on Apple M1, so W = 8 is a 4x
unroll of NEON registers. Lane-wise operations use Mojo's `SIMD` type, with comparisons via
`.ge()`/`.lt()` and selects via `.select()`.

## Run-length scatter

The gradient of a gather `a[g]` is a scatter-add into a short vector. With rows sorted by
group, consecutive lanes add into the same address, which forms a chain of dependent
read-modify-writes through memory.

The generator loads W indices at once. When all lanes share one index, it keeps the partial
sum in a register and writes it out only when the index changes; otherwise it falls back to
per-lane adds. This is correct for any index order. It took the regression kernel from 36.7 us
to 23.0 us.

## libm and std.math

| Function | Source in generated code | Why |
|---|---|---|
| `exp` | libm, per lane | Mojo 1.2 `std.math.exp(Float64)` has up to 8e-12 relative error |
| `log`, `log1p`, `sqrt` | Mojo `std.math`, vectorized | Measured within 1 ulp |
| `lgamma` | libm `lgamma_r`, per lane | Correctly rounded; `lgamma_r` is reentrant |
| `psi` | own implementation | No libm digamma; see [accuracy](accuracy.md) |
| `erf`, `erfc`, `expm1`, `pow`, trig | libm, per lane | Within a few ulp |

Per-lane libm calls are the main remaining cost in models whose per-row work is
transcendental (Poisson, StudentT, Beta-Binomial). The opt-in `PYTENSOR_MOJO_STDEXP` flag shows
the trade-off ([Environment flags](guide/flags.md)).
