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

## Transcendentals

| Function | Source in generated code | Accuracy |
|---|---|---|
| `exp`, `log`, `log1p`, `expm1` | vectorized, `pytensor_mojo.mojo_math` (fdlibm algorithms) | <= 1 ulp, measured over 10M points each |
| `lgamma`, x > 0 | vectorized, `mojo_math`: Chebyshev kernel of `lgamma(x) / ((x-1)(x-2))` on [0.75, 3], recurrence, Stirling | <= 4 ulp (measured about 3) |
| `lgamma`, x <= 0 | libm `lgamma_r`, only for the lanes that need it | libm |
| `sqrt` | Mojo `std.math`, vectorized | Within 1 ulp |
| `psi` | own implementation, per lane | See [accuracy](accuracy.md) |
| `erf`, `erfc`, `pow`, trig | libm, per lane | Within a few ulp |

The vectorized functions port the fdlibm algorithms to Mojo `SIMD` with no branches:

- **exp** uses Cody-Waite reduction by ln 2, with a split constant whose high part has
  trailing zeros so the product is exact, then fdlibm's Remez rational kernel. The result is
  rebuilt with a power of two made from the exponent bits, split into two exact factors near
  the subnormal and overflow edges.
- **log and log1p** split the argument into 2^k (1 + f) with 1 + f in [sqrt(2)/2, sqrt(2)),
  and use the `s = f / (2 + f)` polynomial kernel. log1p adds fdlibm's correction for the
  rounding of `1 + x`.
- **Special cases** (inf, NaN, signed zero, subnormal inputs and results) are handled with
  `select`.

Mojo 1.2's own `std.math.exp(Float64)` is off by up to 8e-12 relative, so it is not used.
`PYTENSOR_MOJO_LIBM=1` switches all five functions back to per-lane libm
([Environment flags](guide/flags.md)).
