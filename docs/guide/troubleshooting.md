# Troubleshooting

## Find unsupported ops before compiling

```python
import pytensor
from pytensor.compile.mode import Mode
from pytensor_mojo.ndcodegen import missing_ops
from pytensor_mojo.ndlinker import QUERY, _Capture, flat_logp_grad

q, logp, grad, dim = flat_logp_grad(model)
_Capture.sink.clear()
pytensor.function([q], [logp, grad], mode=Mode(_Capture(), QUERY))
tensor_missing, scalar_missing = missing_ops(_Capture.sink[-1])
```

`missing_ops` lists, in one pass, every tensor op and scalar op that the code generator cannot
emit for the optimized graph. `bench/phase4_census.py` uses it for the table in
[Supported distributions and ops](../reference/supported.md).

## Error messages

| Message | Cause | What to do |
|---|---|---|
| `NotImplementedError: MojoND: unsupported op X` | A tensor op without codegen, e.g. `Cholesky`, `Scan`, `Sort` | Check the roadmap, or [add the op](../contributing.md) |
| `NotImplementedError: MojoND: unsupported scalar op X` | A scalar op without an entry, e.g. `TriGamma` | Same |
| `NotImplementedError: value vars need static shapes` | A free variable with unknown shape | Give `shape=` explicitly |
| `NotImplementedError: runtime K in fused rowdot` | A fused mat-vec whose inner dimension is not static | Use constant data (static shapes) |
| `RuntimeError: mojo build failed for ...` | The generated source did not compile | Please report it, with the `.mojo` path from the message |
| `mojo-nuts not found` | The sampler is not built | `pixi run build-sampler`, or set `PYTENSOR_MOJO_SAMPLER` |
| `ValueError: ... status 1` | A parameter check failed in `MOJO_ND` mode | Inputs out of the distribution's support |

## logp differs from PyMC by about 1e-9

Python float literals such as `pm.Normal("mu", 0, 5.0)` become float32 constants in PyTensor.
The generated code reproduces PyTensor's graph exactly, so it agrees with PyMC. A comparison
against SciPy will differ, and that is expected. Use `np.float64(5.0)` when you need exact
constants. See [Upstream issues](../upstream-issues.md).
