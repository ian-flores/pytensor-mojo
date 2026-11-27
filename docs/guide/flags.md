# Environment flags

Both opt-in flags trade accuracy for speed. They are read when `pytensor_mojo.ndcodegen` is
imported and change the generated source, so they also change the cache key.

| Flag | Default | Effect |
|---|---|---|
| `PYTENSOR_MOJO_RECIP=1` | off | Inside SIMD loops, `x / s` with a loop-invariant scalar `s` becomes `x * (1 / s)` |
| `PYTENSOR_MOJO_STDEXP=1` | off | `exp` uses Mojo's vectorized `std.math.exp` instead of libm `exp` per lane |
| `PYTENSOR_MOJO_SIMD=N` | 8 | SIMD width of the main loops (float64 lanes) |
| `PYTENSOR_MOJO_CACHE=DIR` | `pytensor_mojo/__mojocache__` | Location of generated sources and libraries |
| `PYTENSOR_MOJO_SAMPLER=PATH` | auto | `mojo-nuts` binary used by `pytensor_mojo.sample` |

## Measured trade-offs

| Parameter | Conditions | Default | Flag on | Unit |
|---|---|---:|---:|---|
| HLR logp + gradient time, `PYTENSOR_MOJO_RECIP` | N=10k, from Rust | 19.9 | 18.6 | us |
| HLR rounding change, `PYTENSOR_MOJO_RECIP` | per division | 0 | 1-2 | ulp |
| Poisson GLM logp + gradient time, `PYTENSOR_MOJO_STDEXP` | N=10k, from Rust | 47.4 | 32.3 | us |
| Poisson GLM logp max rel. error vs PyMC, `PYTENSOR_MOJO_STDEXP` | 24 random points | 1.7e-16 | 5.0e-13 | - |
| `std.math.exp(Float64)` max rel. error | x in [-20, 20] | - | 8e-12 | - |

<div class="ds-conditions" markdown>
Test conditions: Apple M1 Pro, Mojo 1.2.0. The error figures are measured only on the models
named. Neither flag is covered by the default test suite.
</div>
