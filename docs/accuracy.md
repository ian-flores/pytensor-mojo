# Special-function accuracy

## Method

<div class="ds-conditions" markdown>
Each function is compiled through the `MOJO_ND` backend, so both the SIMD lanes and the scalar
tail run, and evaluated on the grids listed. Error in ulp = |a - b| / spacing(b). Typ = median,
Max = maximum over the grid.

References:

- A 40-digit Python `Decimal` implementation (`bench/hiprec_ref.py`) for gammaln, psi and erfc.
  It uses recurrence plus Stirling / asymptotic series with exact Bernoulli numbers,
  reflection with Decimal sin/cos, and the erfc continued fraction.
- numpy or SciPy elsewhere.

The last column gives SciPy's own error against the high-precision reference, where one exists.
Command: `pixi run accuracy`; raw data in `bench/phase4_accuracy.json`.
</div>

## Vectorized exp, log, log1p, expm1, lgamma

These are the default implementations in generated code (`pytensor_mojo.mojo_math`; see
[Concepts](concepts.md#transcendentals)). Each function was evaluated on 10M points from wide
random and edge-case sets, through the generated SIMD code, against the 40-digit `Decimal`
reference. Command: `pixi run python bench/phase7_accuracy.py`; raw data in
`bench/phase7_accuracy.json`.

<div class="ds-spec" markdown>

--8<-- "accuracy7_spec.md"

</div>

<div class="ds-conditions" markdown>
Mean and Max are in ulp of the correctly rounded result. "Within 0.5 ulp" is the fraction of
points that are correctly rounded. The edge-case sets are:

- reduction boundaries (exp at (k + 1/2) ln 2);
- arguments near sqrt(2) and sqrt(2)/2 for log and log1p;
- expm1's branch boundaries;
- subnormal inputs;
- the overflow and underflow thresholds;
- the zeros of lgamma at 1 and 2;
- the boundaries 0.75, 3 and 12 between lgamma's pieces.
</div>

## All special functions

<div class="ds-spec" markdown>

--8<-- "accuracy_spec.md"

</div>

## Tolerances used by the tests

| Function | Test limit | Reason |
|---|---|---|
| exp, log, log1p, expm1 (vectorized) | 1 ulp | fdlibm algorithms, documented below 1 ulp; measured max 0.91 ulp over 10M points |
| lgamma, x > 0 (vectorized) | 4 ulp | Measured max 3.71 ulp over 10M points; the worst case is below x = 1, where x - 2 is not exact |
| libm-backed functions | 4 ulp | Documented libm accuracy |
| erfc | 6 ulp | Measured 5 ulp in the tail, (5, 26.5) |
| psi, x > 0 | 20 ulp | Near x = 1.75 the result is `log(y) - sum(1/k)` with \|result\| about 0.1 \|log y\|, so the rounding of `log(y)` alone costs about 8 ulp |
| Negative arguments | 2e-13 relative | Condition number grows near the poles; grids stay at least 1e-3 from them |

## Notes

- **SciPy's `gammaln` is not a usable reference near its zeros** (x = 1, 2). Its relative error
  there reaches about 3e-9 at |x - 1| < 1e-3 and 8.8e-4 at |x - 1| < 1e-9. libm `lgamma_r`,
  which PyTensor's C backend also uses, is correctly rounded there.
- **SciPy's `erfc` flushes x > 26.55 to 0**, where libm returns the correct subnormal values.
- **Gradients.** The gradients of gammaln, erf, erfc, expm1, log1mexp, sign and pow match the C
  backend at rtol 1e-12. `grad(psi)` needs trigamma, which is not implemented and raises a
  clear error.
