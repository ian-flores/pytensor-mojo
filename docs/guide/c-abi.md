# The C ABI

Every library built by `compile_model_capi(model, "m")` exports three C functions. The
hand-written baselines in `mojo_capi/` export the first two.

```c
int32_t m_dim(const double* data);
int32_t m_logp_grad(const double* data, const double* x, double* grad, double* logp);
int32_t m_run(int64_t* tab, int64_t* shp);
```

| Function | Contract |
|---|---|
| `m_dim` | Returns the length of `x` and `grad` |
| `m_logp_grad` | Reads `x[0..dim)`, writes `grad[0..dim)` and `*logp`. Returns 0 on success, or 1 if a parameter check failed or logp is not finite. A non-zero status is reported to nuts-rs as a recoverable error, so the point is rejected. |
| `m_run` | Generic entry point. `tab` holds buffer addresses (inputs, array constants, outputs) and `shp` the input dims. Used by the [MOJO_ND mode](pytensor-mode.md). |

Properties of `m_logp_grad`:

- **Reentrant.** The per-call workspace is a stack array, or one heap arena when it is large.
  There is no global state; `lgamma_r` is used instead of `lgamma`. Several sampler threads can
  share one `data` buffer.
- **No allocation** when the workspace is small and static.
- **Libraries are self-contained** apart from the Mojo runtime libraries. Their rpath points to
  the pixi environment, so no `DYLD_*` variables are needed.

Calling it from Python with `ctypes`:

```python
from pytensor_mojo.ndlinker import CapiModel
cm = CapiModel(so_path, "m", data)
status, logp, grad = cm(x)
```
