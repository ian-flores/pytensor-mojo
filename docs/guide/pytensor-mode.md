# The MOJO_ND PyTensor mode

The code generator also works as an ordinary PyTensor backend:

```python
import pytensor, pytensor.tensor as pt
from pytensor_mojo.ndlinker import MOJO_ND

x = pt.dmatrix("x"); v = pt.dvector("v")
f = pytensor.function([x, v], [pt.exp(x) * v[None, :], (x ** 2).sum(axis=1)], mode=MOJO_ND)
```

`MOJO_ND` is a `Mode` with the `MojoNDLinker` (a PyTensor `JITLinker`) and the same rewrites
used for models. On first use it compiles one library per function and calls it through
`ctypes`. Runtime shapes are supported: one compiled function accepts inputs of any size that
matches the graph's static dimensions.

## When to use it

| Use | Fit |
|---|---|
| Testing generated code against PyTensor's C backend on small graphs | Good. This is how the unit tests use it. |
| Inner loops called from Python | Poor. Each call pays PyTensor's Function overhead of about 2 us plus `ctypes` marshalling. |
| Sampling | Use the C ABI with `mojo-nuts` instead |

A failing `CheckParameterValue` raises `ValueError("... status 1 ...")`.
