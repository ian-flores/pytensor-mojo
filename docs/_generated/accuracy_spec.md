| Parameter | Conditions | Reference | Typ | Max | Unit | Max rel. error | SciPy max rel. error |
|---|---|---|---:|---:|---|---:|---:|
| gammaln error | tiny (1e-300..1e-3), n=1000 | 40-digit Decimal | 0 | 0 | ulp | 0.0e+00 | 2.2e-16 |
| gammaln error | (1e-3, 50), n=1000 | 40-digit Decimal | 0 | 2 | ulp | 2.8e-16 | 1.5e-14 |
| gammaln error | near 1 and 2 (zeros), n=1000 | 40-digit Decimal | 1 | 3 | ulp | 4.1e-16 | 1.6e-10 |
| gammaln error | large (50..1e300), n=1000 | 40-digit Decimal | 0 | 1 | ulp | 1.7e-16 | 2.1e-16 |
| gammaln error | negative non-integer, n=1000 | 40-digit Decimal | 0 | 155 | ulp | 3.2e-14 | 4.4e-15 |
| psi error | tiny (1e-300..1e-3), n=1000 | 40-digit Decimal | 0 | 1 | ulp | 2.0e-16 | 2.1e-16 |
| psi error | (1e-3, 10), n=1000 | 40-digit Decimal | 1 | 15 | ulp | 1.8e-15 | 3.3e-16 |
| psi error | near root 1.4616, n=1000 | 40-digit Decimal | 1 | 13 | ulp | 1.8e-15 | 2.2e-16 |
| psi error | very near root (+-1e-8), n=1000 | 40-digit Decimal | 1 | 2 | ulp | 4.4e-16 | 2.3e-16 |
| psi error | large (10..1e300), n=1000 | 40-digit Decimal | 0 | 0 | ulp | 0.0e+00 | 1.9e-16 |
| psi error | negative, >=1e-3 from poles, n=1000 | 40-digit Decimal | 1 | 131 | ulp | 2.1e-14 | 3.5e-14 |
| erf error | (-6, 6), n=4000 | scipy.special | 0 | 3 | ulp | 4.3e-16 | - |
| erf error | tiny, n=4000 | scipy.special | 0 | 3 | ulp | 4.1e-16 | - |
| erfc error | (-6, 1), n=1000 | 40-digit Decimal | 0 | 3 | ulp | 5.0e-16 | 9.1e-16 |
| erfc error | (1, 5), n=1000 | 40-digit Decimal | 1 | 4 | ulp | 6.6e-16 | 2.2e-15 |
| erfc error | deep tail (5, 26.5), n=1000 | 40-digit Decimal | 1 | 5 | ulp | 6.3e-16 | 5.7e-14 |
| expm1 error | (-50, 50), n=4000 | numpy | 0 | 1 | ulp | 2.2e-16 | - |
| expm1 error | tiny, n=4000 | numpy | 0 | 0 | ulp | 0.0e+00 | - |
| log1mexp error | (-50, -1e-3), n=4000 | numpy | 0 | 1 | ulp | 2.2e-16 | - |
| log1mexp error | near 0 (-1e-3..-1e-300), n=4000 | numpy | 0 | 1 | ulp | 2.0e-16 | - |
| log1mexp error | around -log 2, n=4000 | numpy | 0 | 0 | ulp | 0.0e+00 | - |
| pow error | x in (1e-3, 100), y in (-10, 10), n=4000 | numpy | 0 | 0 | ulp | 0.0e+00 | - |
| log error | (1e-300, 1e300), n=4000 | numpy | 0 | 0 | ulp | 0.0e+00 | - |
| log error | near 1, n=4000 | numpy | 0 | 1 | ulp | 1.4e-16 | - |
| log1p error | (-0.999, 1e10), n=4000 | numpy | 0 | 1 | ulp | 2.2e-16 | - |
| log1p error | tiny, n=4000 | numpy | 0 | 0 | ulp | 0.0e+00 | - |
| sqrt (std.math) error | (1e-300, 1e300), n=4000 | numpy | 0 | 0 | ulp | 0.0e+00 | - |
