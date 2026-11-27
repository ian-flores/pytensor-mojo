| Tensor op | Notes |
|---|---|
| `AdvancedIncSubtensor` | 1-D scatter-add inc_subtensor(a[idx], v); fused as a run-length register accumulator |
| `AdvancedSubtensor` | 1-D gather a[idx]; inlined into fused loops (splat when all lanes share an index) |
| `All` | CAReduce AND; fused as a SIMD bool accumulator |
| `Alloc` | broadcast fill |
| `CheckAndRaise` | CheckParameterValue: failed condition returns status 1 |
| `DimShuffle` | view (transpose, expand, squeeze); no copy |
| `Dot` | 2-D/1-D combinations; mat-vec fused into loops (row dot inlined, X^T w accumulated); otherwise scalar loops, no BLAS |
| `Elemwise` | N-d broadcasting; 1-D loops fused with producers/consumers and emitted as SIMD (W=8) + scalar tail |
| `Join` | axis 0 of 1-D inputs |
| `MakeVector` | bool vectors folded into All; float vectors materialized |
| `ScalarFromTensor` | view |
| `Subtensor` | view, constant integer index or slice (step 1) |
| `Sum` | all axes or given axes; fused as a SIMD accumulator when it consumes a 1-D loop |
| `TensorFromScalar` | view |

| Scalar op | Implementation |
|---|---|
| `AND` | inline SIMD |
| `Abs` | inline SIMD |
| `Add` | inline SIMD |
| `Ceil` | libm `ceil` per lane |
| `Cos` | libm `cos` per lane |
| `EQ` | inline SIMD |
| `Erf` | libm `erf` per lane |
| `Erfc` | libm `erfc` per lane |
| `Exp` | vectorized fdlibm-style exp (`mojo_math`, <= 1 ulp) |
| `Expm1` | vectorized fdlibm-style expm1 (`mojo_math`, <= 1 ulp) |
| `Floor` | libm `floor` per lane |
| `GE` | inline SIMD |
| `GT` | inline SIMD |
| `GammaLn` | vectorized: Chebyshev kernel at the zeros, recurrence, Stirling (`mojo_math`); libm for x <= 0 |
| `Identity` | inline SIMD |
| `Invert` | inline SIMD |
| `IsInf` | inline SIMD |
| `IsNan` | inline SIMD |
| `LE` | inline SIMD |
| `LT` | inline SIMD |
| `Log` | vectorized fdlibm-style log (`mojo_math`, <= 1 ulp) |
| `Log10` | libm `log10` per lane |
| `Log1mexp` | Maechler (2012) piecewise, libm |
| `Log1p` | vectorized fdlibm-style log1p (`mojo_math`, <= 1 ulp) |
| `Log2` | libm `log2` per lane |
| `Maximum` | inline SIMD |
| `Minimum` | inline SIMD |
| `Mul` | inline SIMD |
| `NEQ` | inline SIMD |
| `Neg` | inline SIMD |
| `OR` | inline SIMD |
| `Pow` | libm `pow` per lane |
| `Psi` | own implementation (reflection, root Taylor series, compensated recurrence + asymptotic) |
| `Reciprocal` | inline SIMD |
| `Sigmoid` | vectorized fdlibm-style exp (`mojo_math`, <= 1 ulp) (inside formula) |
| `Sign` | inline SIMD |
| `Sin` | libm `sin` per lane |
| `Softplus` | vectorized fdlibm-style log1p (`mojo_math`, <= 1 ulp) (inside formula) |
| `Sqr` | inline SIMD |
| `Sqrt` | Mojo `std.math`, SIMD (accuracy-tested, <= 1 ulp) |
| `Sub` | inline SIMD |
| `Switch` | inline SIMD |
| `Tanh` | libm `tanh` per lane |
| `TrueDiv` | inline SIMD |
