# Environment flags

Flags are read when `pytensor_mojo` is imported and change the generated source, so they
also change the cache key.

| Flag | Default | Effect |
|---|---|---|
| `PYTENSOR_MOJO_LIBM=1` | off | `exp`, `log`, `log1p`, `expm1` and `gammaln` call libm one SIMD lane at a time, instead of the vectorized implementations in `pytensor_mojo.mojo_math` |
| `PYTENSOR_MOJO_RECIP=1` | off | Inside SIMD loops, `x / s` with a loop-invariant scalar `s` becomes `x * (1 / s)` |
| `PYTENSOR_MOJO_SIMD=N` | 8 | SIMD width of the main loops (float64 lanes) |
| `PYTENSOR_MOJO_CACHE=DIR` | `pytensor_mojo/__mojocache__` | Location of generated sources and libraries |
| `PYTENSOR_MOJO_SAMPLER=PATH` | auto | `mojo-nuts` binary used by `pytensor_mojo.sample` |
| `PYTENSOR_MOJO_NO_EXT=1` | off | Disables the extension ops in `pytensor_mojo/ndcodegen_ext.py` (baseline census) |

`PYTENSOR_MOJO_STDEXP` (rev 0.2-dev, Phase 4) has been retired. It selected Mojo's
`std.math.exp`, which is fast but has up to 8e-12 relative error. The vectorized `exp` that is
now the default is within 1 ulp and nearly as fast.

## Measured trade-offs

### Vectorized transcendentals (default) vs `PYTENSOR_MOJO_LIBM=1`

<div class="ds-spec" markdown>

--8<-- "kernels7_spec.md"

</div>

Accuracy: the vectorized functions are within 0.91 ulp (exp, log, log1p, expm1) and 3.71 ulp
(lgamma, x > 0), measured over 10M points each; see
[Special-function accuracy](../accuracy.md). libm is within about 1 ulp. Models with no
transcendentals in their per-row work (the regression, eight schools) are unaffected or a few
nanoseconds slower per evaluation, which does not show in sampling time.

### `PYTENSOR_MOJO_RECIP`

| Parameter | Conditions | Default | Flag on | Unit |
|---|---|---:|---:|---|
| HLR logp + gradient time | N=10k, from Rust | 19.9 | 18.6 | us |
| Rounding change | per division | 0 | 1-2 | ulp |

<div class="ds-conditions" markdown>
Test conditions: Apple M1 Pro, Mojo 1.2.0. The kernel timings alternate the two builds in the
same window (7 rounds, minimum). The `PYTENSOR_MOJO_RECIP` error figures are measured only on
the model named, and that flag is not covered by the default test suite.
`PYTENSOR_MOJO_LIBM=1` is covered by `tests/test_mojo_math.py`.
</div>
