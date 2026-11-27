# pytensor-mojo

A Mojo backend for PyTensor that compiles PyMC models' logp and gradient into fused SIMD
kernels. The kernels are sampled with NUTS through a C ABI, with no Python in the gradient
loop.

**Documentation: <https://ian-flores.github.io/pytensor-mojo/>**

> **Status: experimental research prototype.** Not on PyPI, no stable API. Built and tested
> only on macOS / Apple Silicon.

## What it is

- **A code generator** (`pytensor_mojo/ndcodegen.py`). It turns the optimized PyTensor graph of
  a PyMC model's logp and gradient into one Mojo source file. Every pass over the data is
  fused into a single 8-wide SIMD loop, including gathers, mat-vecs, reductions and the
  gradient's scatter-add.
- **A four-function C ABI** exported by every generated library
  (`<m>_dim`, `<m>_logp_grad`, `<m>_run`).
- **A Rust driver**, `mojo-nuts` (`sampler/`). It samples any such library with
  [nuts-rs](https://github.com/pymc-devs/nuts-rs), the NUTS core of nutpie, using nutpie's
  default settings.
- **A one-call helper**, `pytensor_mojo.sample.sample(model)`, that goes from a PyMC model to an
  ArviZ `DataTree`.

Common likelihoods and priors compile, each matching PyMC's `compile_logp` / `compile_dlogp`
at rtol 1e-10: StudentT, Gamma, Beta, Poisson, Binomial, NegativeBinomial, Bernoulli,
LogNormal, Exponential, HalfNormal, HalfCauchy, InverseGamma, Weibull, Laplace, Cauchy,
Uniform and Normal, with log, logodds and interval transforms. See
[Supported distributions and ops](https://ian-flores.github.io/pytensor-mojo/reference/supported/).

## Results

Sampling time against nutpie (numba backend). Apple M1 Pro, nuts-rs 0.18.3 and identical
settings on both sides, 4 chains, 1000 tune + 1000 draws, sampling only, median of 5 runs.

| Model | dim | Mojo + nuts-rs (s) | nutpie (s) | Speedup (median of runs) | Min ESS bulk, Mojo / nutpie |
|---|---:|---:|---:|---:|---:|
| Hierarchical linear regression, N=10k, G=20, K=5 | 28 | 0.294 | 1.611 | 5.46x | 7557 / 7880 |
| Poisson GLM, log link, N=10k, K=5 | 6 | 0.330 | 1.058 | 3.23x | 5005 / 4993 |
| Robust StudentT regression, N=5000, nu free | 6 | 0.260 | 0.636 | 2.44x | 2773 / 2650 |
| Hierarchical Beta-Binomial, G=200 | 202 | 0.1000 | 0.213 | 2.12x | 2152 / 2191 |
| Eight schools, non-centered | 10 | 0.0176 | 0.0148 | 0.86x | 2026 / 2192 |
The data-bound regression gains the most. Models whose per-row work is `exp`, `log1p` or
`lgamma` gain less, because those run as libm calls per SIMD lane. Eight schools ties because
its logp + gradient (about 50 ns) is small next to the sampler's own work per step (about
700 ns). Methodology and Min / Typ / Max spec tables are on the
[Benchmarks](https://ian-flores.github.io/pytensor-mojo/benchmarks/) page.

## Quickstart

Requires macOS on Apple Silicon, [pixi](https://pixi.sh) and a Rust toolchain.

```bash
git clone https://github.com/ian-flores/pytensor-mojo.git && cd pytensor-mojo
pixi install
pixi run build-sampler        # Rust nuts-rs driver
pixi run test                 # full test suite
```

```python
# run from the repo root, e.g. PYTHONPATH=. pixi run python example.py
import numpy as np
import pymc as pm
from pytensor_mojo.sample import sample

y = np.random.default_rng(0).normal(3.0, 2.0, size=200)
with pm.Model() as model:
    mu = pm.Normal("mu", np.float64(0), np.float64(10))
    sigma = pm.HalfNormal("sigma", np.float64(5))
    pm.Normal("y", mu, sigma, observed=y)

idata = sample(model, chains=4, tune=1000, draws=1000, seed=1)   # ArviZ DataTree
```

Other tasks:

- `pixi run bench-phase3` and `pixi run bench-phase4`: benchmarks.
- `pixi run accuracy`: special-function accuracy.
- `pixi run test-rust`: fmt, clippy and Rust tests.
- `pixi run -e docs docs-serve 8000`: the documentation site.

## Learn more

- [Getting started](https://ian-flores.github.io/pytensor-mojo/getting-started/)
- [Concepts](https://ian-flores.github.io/pytensor-mojo/concepts/): architecture, fusion,
  SIMD, run-length scatter
- [Limitations and roadmap](https://ian-flores.github.io/pytensor-mojo/limitations/)
- [Upstream issues found](https://ian-flores.github.io/pytensor-mojo/upstream-issues/): Mojo
  `exp` accuracy, a nuts-rs storage panic, PyMC float32 literals, SciPy `gammaln` near its
  zeros
- [Research log](docs/FINDINGS.md): every measurement, phase by phase
- [Contributing](CONTRIBUTING.md) and [Changelog](CHANGELOG.md)

## Acknowledgements

- [PyMC](https://github.com/pymc-devs/pymc) and [PyTensor](https://github.com/pymc-devs/pytensor),
  for the modelling language, the graph rewrites and the symbolic gradients.
- [nuts-rs](https://github.com/pymc-devs/nuts-rs) and [nutpie](https://github.com/pymc-devs/nutpie),
  for the NUTS implementation and the baseline this project is measured against.
- [Modular](https://www.modular.com/), for [Mojo](https://mojolang.org/).

## License

Apache-2.0. See [LICENSE](LICENSE).
