# Compiling models

## From a PyMC model to a shared library

```python
from pytensor_mojo.ndlinker import compile_model_capi
so_path, data, dim, info = compile_model_capi(model, "mymodel")
```

| Return value | Meaning |
|---|---|
| `so_path` | Path of `libmymodel_<hash>.dylib` in the cache directory |
| `data` | float64 array to pass as the `data` argument of the C ABI (constants, see below) |
| `dim` | Length of the flat unconstrained parameter vector |
| `info` | Dict with the generated Mojo `source`, its path, the constant list and output shape expressions |

The steps are:

1. **Flat logp + gradient graph.** The model's value variables (the unconstrained parameters,
   in `model.value_vars` order) are replaced by slices of one flat vector `q` of static length
   `dim`. The graph computes `model.logp(jacobian=True)` and `pytensor.grad` of it, concatenated
   in the same order. This is the graph behind PyMC's `compile_logp` / `compile_dlogp`.
2. **PyTensor rewrites.** The graph is optimized with the `fast_run` rewrites, including the
   FusionOptimizer that builds elementwise Composites. The C-only, BLAS, inplace and
   numba-specific passes are excluded.
3. **Code generation.** `ndcodegen` emits one Mojo source file for the optimized graph (see
   [Concepts](../concepts.md)).
4. **Build.** `mojo build --emit shared-lib` produces the library. Builds are cached by a hash
   of the source: about 1.2 s cold, near zero on a cache hit.

## Constants and the `data` buffer

Array constants such as observed data, design matrices and group indices are not written into
the source. They are packed into `data`:

- **Header:** `data[0]` holds the number of constants and `data[1..n]` their offsets, stored
  bitwise as int64.
- **Payloads** follow, one per constant.
- **Integer arrays used only as gather/scatter indices** stay int64, stored bitwise.
- **All other integer arrays** (counts, Bernoulli outcomes) and **bool masks** are converted to
  float64, which is exact for these values.
- **2-D constants** used as the left operand of a fused mat-vec are stored column-major, so
  that SIMD loads along the data axis are contiguous.

## Requirements on the model

- **Static shapes.** Every free variable needs a static shape. Data constants have static
  shapes automatically.
- **Supported ops only.** Every op in the optimized graph must be supported; check with
  `missing_ops` ([Troubleshooting](troubleshooting.md)).
- **float64.** Computation is float64 (`pytensor.config.floatX = "float64"`, the PyMC default
  on CPU).
