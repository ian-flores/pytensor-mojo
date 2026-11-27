# Limitations and roadmap

## Current limitations

| Area | Limitation |
|---|---|
| Platform | Built and tested only on macOS / Apple Silicon. Linux should be possible, but the build assumes `.dylib`. |
| Shapes | Fused loops specialize on static dims, which PyMC models with constant data have. Runtime-shaped graphs use a generic path that does not fuse dots or scatters. |
| Linear algebra | No BLAS. Unfused `Dot` and axis reductions are scalar loops. No Cholesky, solves or determinants. |
| Indexing | Gather and scatter are 1-D (`a[idx]`). No multi-dimensional advanced indexing. `set_subtensor` / `inc_subtensor` only with constant indices and slices. |
| Special functions | No trigamma or polygamma, incomplete gamma or beta, Bessel functions or erfcx. |
| Transcendentals | exp, log, log1p, expm1 and lgamma (x > 0) are vectorized. psi, erf, erfc, pow and trig functions are still per-lane calls; psi appears in gradients that involve gammaln. |
| Control flow | No `Scan`, no `IfElse`. |
| Data types | Computation is float64. Integer and bool data are packed as float64, which is exact for these values. |
| Python path | `MOJO_ND` through `pytensor.function` pays PyTensor's per-call overhead. Sampling should go through the C ABI. |
| Coverage | 27 single-distribution census models, five benchmarked models and a 28-posterior posteriordb sweep ([results](benchmarks/posteriordb.md)). |

## Roadmap

1. **Special functions, done in rev 0.2.** gammaln, digamma, erf/erfc, log1mexp, pow, sign and
   others, with accuracy tests. Accurate vectorized exp, log, log1p, expm1 and lgamma
   followed. Next in this area:
   - a vectorized psi;
   - trigamma;
   - incomplete gamma and beta.
2. **posteriordb model sweep, first pass done.** 28 posteriors with reference draws; see
   [posteriordb sweep](benchmarks/posteriordb.md) for coverage, the missing-op ranking and
   speed against nutpie. Next: more hierarchical models (most posteriordb hierarchical
   posteriors have no reference draws).
3. **Linear algebra.** Cholesky, triangular solves and log-determinants (MvNormal, LKJ).
4. **Scan.** Time-series and recurrent models.
5. **Apple GPU many-chain sampler.** float32 batched logp + gradient across many chains. A first
   check gave about 1e-7 typical error, so the GPU suits massively parallel chains rather than
   single-chain NUTS.
