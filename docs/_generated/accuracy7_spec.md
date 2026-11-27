| Parameter | Conditions | n | Mean | Max | Unit | Within 0.5 ulp |
|---|---|---:|---:|---:|---|---:|
| exp error | uniform [-745, 709.78] | 4,000,000 | 0.264 | 0.871 | ulp | 90.47% |
| exp error | uniform [-1, 1] | 2,000,000 | 0.263 | 0.866 | ulp | 90.56% |
| exp error | tiny |x| (1e-300..1e-3), both signs | 2,000,000 | 0.012 | 0.500 | ulp | 100.00% |
| exp error | reduction boundaries (k +- 1/2) ln2 | 1,000,000 | 0.288 | 0.905 | ulp | 81.66% |
| exp error | near overflow / underflow | 1,000,000 | 0.353 | 0.824 | ulp | 99.08% |
| log error | log-uniform [5e-324, 1.8e308] (incl. subnormal) | 5,000,000 | 0.008 | 0.743 | ulp | 99.99% |
| log error | near 1 (1 +- 1e-3, log-spaced offsets) | 2,000,000 | 0.239 | 0.500 | ulp | 100.00% |
| log error | near sqrt(2)/2 and sqrt(2) | 1,000,000 | 0.269 | 0.841 | ulp | 88.15% |
| log error | uniform (0, 10) | 2,000,000 | 0.258 | 0.795 | ulp | 93.35% |
| log1p error | uniform (-0.999, 1) | 3,000,000 | 0.260 | 0.832 | ulp | 92.39% |
| log1p error | log-uniform (1, 1e300) | 2,000,000 | 0.014 | 0.742 | ulp | 99.93% |
| log1p error | near -1 (-1 + 1e-16..1) | 1,000,000 | 0.252 | 0.794 | ulp | 98.75% |
| log1p error | tiny |x| (1e-300..1e-3), both signs | 3,000,000 | 0.012 | 0.660 | ulp | 99.97% |
| log1p error | near sqrt(2) - 1 and sqrt(2)/2 - 1 | 1,000,000 | 0.269 | 0.852 | ulp | 88.22% |
| expm1 error | uniform [-40, 709.78] | 3,000,000 | 0.262 | 0.780 | ulp | 90.67% |
| expm1 error | uniform [-1, 1] | 3,000,000 | 0.263 | 0.840 | ulp | 90.99% |
| expm1 error | tiny |x| (1e-300..1e-3), both signs | 2,000,000 | 0.012 | 0.500 | ulp | 100.00% |
| expm1 error | branch boundaries +-0.5 ln2, +-1.5 ln2, k = 20, 56 | 2,000,000 | 0.269 | 0.855 | ulp | 87.57% |
| gammaln error | log-uniform [1e-300, 1e300] | 3,000,000 | 0.133 | 2.923 | ulp | 99.66% |
| gammaln error | uniform (0, 15) | 5,000,000 | 0.392 | 3.707 | ulp | 68.86% |
| gammaln error | near zeros 1, 2 (log-spaced offsets 1e-16..1e-2) | 1,000,000 | 0.522 | 3.518 | ulp | 56.32% |
| gammaln error | piece boundaries 0.75, 3, 12 | 1,002,000 | 0.530 | 3.689 | ulp | 57.85% |
