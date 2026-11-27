# Getting started

## Requirements

| Requirement | Version tested | Notes |
|---|---|---|
| macOS on Apple Silicon | macOS 27.2, M1 Pro | The only platform built and tested so far |
| [pixi](https://pixi.sh) | 0.85 | Installs Mojo, Python and the scientific stack from conda channels |
| Rust toolchain (`cargo`) | 1.92 | Builds the `mojo-nuts` sampler driver |
| Xcode command-line tools | 16 or later | Provide the system linker used by `mojo build` |

pixi installs Mojo 1.2 (Modular conda channel), Python 3.12, PyMC 6.4, PyTensor 3.3, nutpie
and pytest.

## Install and build

```bash
git clone https://github.com/ian-flores/pytensor-mojo.git
cd pytensor-mojo
pixi install               # environment from pixi.toml / pixi.lock
pixi run build             # hand-written baseline kernels (optional for users)
pixi run build-sampler     # cargo build --release of sampler/ -> sampler/target/release/mojo-nuts
pixi run test              # full pytest suite
```

Generated kernels are compiled on demand the first time a model is used, in about 1.2 s. They
are cached under `pytensor_mojo/__mojocache__/` by a hash of the generated source; the
`PYTENSOR_MOJO_CACHE` variable moves the cache.

The package is not installed into the environment. Run Python from the repository root with
`PYTHONPATH=.`, for example `PYTHONPATH=. pixi run python my_script.py`.

## A first model, end to end

### Define the model in PyMC

```python
import numpy as np
import pymc as pm

rng = np.random.default_rng(0)
X = rng.normal(scale=0.3, size=(10_000, 5))
y = rng.poisson(np.exp(1.0 + X @ rng.normal(scale=0.5, size=5)))

with pm.Model() as model:
    a = pm.Normal("a", np.float64(0), np.float64(2))
    b = pm.Normal("b", np.float64(0), np.float64(1), shape=5)
    pm.Poisson("y", mu=pm.math.exp(a + X @ b), observed=y)
```

Pass scale parameters as `np.float64` when you compare against an exact reference. PyTensor
stores Python float literals such as `5.0` as float32 constants; see
[Upstream issues](upstream-issues.md).

### Sample it and get an ArviZ object

```python
from pytensor_mojo.sample import sample

idata = sample(model, chains=4, tune=1000, draws=1000, seed=1)
idata["posterior"]          # constrained variables and deterministics, (chain, draw, ...)
idata["sample_stats"]       # diverging, n_steps per draw
idata.attrs["sampling_time"]
```

`sample` does the steps described in the next section and returns an ArviZ `DataTree`.
`arviz_stats` and `arviz_plots` work on it directly.

### What `sample` does, step by step

```python
from pytensor_mojo.ndlinker import compile_model_capi

so_path, data, dim, info = compile_model_capi(model, "poisson")   # -> lib<name>_<hash>.dylib
data.astype("<f8").tofile("poisson.f64")                           # model data for the driver
```

```bash
sampler/target/release/mojo-nuts --lib <so_path> --model poisson --data poisson.f64 \
    --chains 4 --tune 1000 --draws 1000 --seed 1 --out trace.json
```

```python
import json
from pytensor_mojo.sample import raw_to_datatree

idata = raw_to_datatree(model, json.load(open("trace.json")))
```

`trace.json` holds the unconstrained draws, per-draw `diverging` and `n_steps`, and the
sampling time. Its format is described in [The mojo-nuts sampler](guide/sampler-cli.md).

## Next

- [Compiling models](guide/compiling.md): what `compile_model_capi` produces.
- [Supported distributions and ops](reference/supported.md): check that your model compiles.
- [Troubleshooting](guide/troubleshooting.md): what to do when an op is missing.
