# Upstream issues

Issues found in upstream libraries while building this project. Each one is pinned by a test,
so a fix upstream shows up as a test failure.

| Library | Issue | Effect here | Pinned by |
|---|---|---|---|
| Mojo 1.2 | `std.math.exp(Float64)` is not correctly rounded: up to about 8e-12 relative error on [-20, 20] | Generated code calls libm `exp` per lane; vectorized `std.math.exp` is opt-in | `tests/test_q1_mod.py::test_exp_accuracy_std_vs_libm` |
| nuts-rs 0.18.3 | `HashMapConfig` storage panics on the first divergence ("Type mismatch when combining stats for divergence_message") | `mojo-nuts` uses `ArrowConfig`, as nutpie does | none; worked around |
| PyMC / PyTensor | Python float literal parameters become float32 constants; `pm.Normal("mu", 0, 5.0)` computes `log(5)` in float32, and HalfCauchy's `log(2)` is an int8 constant upcast to float32 | logp is shifted by about 1e-9 relative against an exact implementation. Generated code reproduces PyTensor's graph, so it agrees with PyMC | `tests/test_eight_schools.py::test_pymc_literal_params_float32_offset_documented` |
| SciPy | `gammaln` loses relative accuracy near its zeros x = 1, 2 (about 3e-9 at \|x - 1\| < 1e-3, 8.8e-4 at \|x - 1\| < 1e-9); `erfc` flushes x > 26.55 to 0 | SciPy is not used as the reference there; tests use a 40-digit Decimal implementation | `tests/test_special.py` (accuracy table) |
