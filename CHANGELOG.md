# Changelog

All notable changes. Revisions follow the datasheet numbering used in the documentation.

## Unreleased (rev 0.2-dev)

### Added

- Vectorized float64 `exp`, `log`, `log1p`, `expm1` and `lgamma` (x > 0) in generated code
  (`pytensor_mojo/mojo_math.py`, fdlibm algorithms plus a Chebyshev lgamma kernel). These are
  the default now, within 0.91 ulp (lgamma 3.71 ulp), measured over 10M points each with
  `bench/phase7_accuracy.py`. The kernel speedup is 1.3-1.6x on transcendental-heavy models,
  and sampling against nutpie improves from 1.7-2.5x to 2.1-3.2x on those models. Eight schools
  regresses slightly at 4 chains (0.86x, +12 ns per evaluation).
- `PYTENSOR_MOJO_LIBM=1`, which restores per-lane libm calls.
- Special functions in generated code:
  - gammaln (libm `lgamma_r`) and psi / digamma (own implementation);
  - erf, erfc, expm1, log1mexp, pow, sign, isinf, isnan, floor, ceil, log2, log10, tanh, sin,
    cos;
  - accuracy tests against 40-digit references.
- `All` (fused SIMD AND accumulator) and `MakeVector`.
- Op census over 14 likelihoods and 12 priors, plus `ndcodegen.missing_ops()`.
- Three more sampled models (robust StudentT regression, Poisson GLM, hierarchical
  Beta-Binomial), with posterior-match tests against nutpie.
- `pytensor_mojo.sample.sample()`: PyMC model to ArviZ `DataTree` in one call.
- `mojo-nuts`: `--help`, and per-draw `diverging` and `n_steps` in the output.
- Documentation site (MkDocs + Material) with generated reference tables and figures.
- `CONTRIBUTING.md` and this changelog.

### Changed

- `PYTENSOR_MOJO_STDEXP` (added earlier in this cycle) is removed. Mojo's `std.math.exp` is
  inaccurate (8e-12), and the new vectorized `exp` is accurate and nearly as fast.
- Integer data used in arithmetic and bool masks are packed as float64; index arrays stay
  int64.
- PyTensor semantics are reproduced for numeric Switch conditions, bool upcasts and bool
  Mul / Add.

### Fixed

- `isnan` in generated code: Mojo's SIMD `.ne()` is an ordered comparison, so `x != x` was
  False for NaN. NaN tests now use the bit pattern.

## 0.1.0 (2026-11-26)

First public release: phases 1-3 of the research log.

### Added

- **Phase 1:** feasibility spike, covering Python-to-Mojo call overhead, a hand-written eight
  schools kernel matching PyMC at 1e-10, a backend-interface study, an op census, and a first
  prototype linker.
- **Phase 2:** C-ABI kernel libraries driven by nuts-rs 0.18.3 (`sampler/`, crate
  `mojo-nuts`), matching nutpie's posterior; a hand-written hierarchical regression.
- **Phase 3:** `ndcodegen`, an N-d, fused, SIMD code generator with a run-length scatter and a
  stack workspace arena. Generated kernels come within 1.10x of hand-written ones and sample
  5.3-5.5x faster than nutpie on the regression model.
- Apple GPU (Metal, float32) experiments.
