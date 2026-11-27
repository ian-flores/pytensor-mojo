# pytensor-mojo

[![tests](https://github.com/ian-flores/pytensor-mojo/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ian-flores/pytensor-mojo/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-ian--flores.github.io%2Fpytensor--mojo-3f5d72)](https://ian-flores.github.io/pytensor-mojo/)
[![license](https://img.shields.io/badge/license-Apache--2.0-7c8a4e)](LICENSE)

**Sample unmodified PyMC models with NUTS, using logp and gradient code compiled to Mojo.**
On data-heavy models it is 2.1x to 5.5x faster than nutpie, with the same sampler and the
same settings, and it matches PyMC's own logp and gradient to rtol 1e-10.

```python
from pytensor_mojo.sample import sample

idata = sample(model)   # any supported PyMC model -> ArviZ DataTree
```

> **Status: experimental research prototype.** Not on PyPI, and the API may change. Built and
> tested only on macOS with Apple Silicon. Read
> [Limitations](https://ian-flores.github.io/pytensor-mojo/limitations/) before relying on it.

**Documentation: <https://ian-flores.github.io/pytensor-mojo/>**

## Results

Apple M1 Pro. Both sides run nuts-rs 0.18.3 with identical settings: 4 chains, 1000 tune +
1000 draws, sampling time only, median of 5 runs. nutpie uses its numba backend.

| Model | dim | Mojo + nuts-rs (s) | nutpie (s) | Speedup (median of runs) | Min ESS bulk, Mojo / nutpie |
|---|---:|---:|---:|---:|---:|
| Hierarchical linear regression, N=10k, G=20, K=5 | 28 | 0.294 | 1.611 | 5.46x | 7557 / 7880 |
| Poisson GLM, log link, N=10k, K=5 | 6 | 0.330 | 1.058 | 3.23x | 5005 / 4993 |
| Robust StudentT regression, N=5000, nu free | 6 | 0.260 | 0.636 | 2.44x | 2773 / 2650 |
| Hierarchical Beta-Binomial, G=200 | 202 | 0.1000 | 0.213 | 2.12x | 2152 / 2191 |
| Eight schools, non-centered | 10 | 0.0176 | 0.0148 | 0.86x | 2026 / 2192 |

Effective sample size per draw is the same on both sides, so ESS per second improves by the
same factor. Full methodology and Min / Typ / Max tables are on the
[Benchmarks](https://ian-flores.github.io/pytensor-mojo/benchmarks/) page.

### When it helps

- **It helps most when the gradient cost scales with the data,** for example a regression or
  GLM over thousands of rows. The whole pass over the data becomes one fused SIMD loop.
- **It helps less when each row's work is dominated by a transcendental** (`exp`, `log1p`,
  `lgamma`). Those are vectorized and accurate to under 1 ulp (`lgamma`: under 4 ulp), but they
  still cost several times more than the arithmetic around them. Some special functions
  (`digamma`, `erf`, `pow`) still run as one libm call per SIMD lane.
- **It doesn't help small models** like eight schools, and at 4 chains it is slightly slower
  (0.86x). Their logp + gradient takes about 50 ns, while the sampler spends about 700 ns per
  leapfrog step on its own work, and that part is identical on both sides.
- **It doesn't run models it can't compile yet**, such as those that use linear algebra
  (MvNormal, LKJ, exact GPs), Scan (time-series loops and ODEs) or CDFs. `missing_ops()`
  lists what a given model is missing. See
  [Supported distributions and ops](https://ian-flores.github.io/pytensor-mojo/reference/supported/).

## How it works

```
PyMC model
  -> PyTensor graph of logp + symbolic gradient    (PyTensor does the autodiff)
  -> ndcodegen: one fused, SIMD Mojo source file   (this project)
  -> shared library with a small C ABI             (mojo build)
  -> mojo-nuts: nuts-rs NUTS in Rust               (no Python per gradient)
  -> ArviZ DataTree
```

- **No autodiff in the backend.** PyTensor already expresses the gradient as ordinary forward
  ops. The backend only has to generate fast code for them.
- **Fusion.** Each pass over the data absorbs everything around it into one loop: gathers like
  `a[g[i]]`, row mat-vecs, sums, the transposed mat-vec, and the scatter-add from the
  gradient of a gather. No data-length temporary arrays are created.
- **Explicit SIMD.** The fused loops are emitted 8 lanes wide, with a scalar tail.
  Run-length accumulation keeps grouped scatter-adds in registers.
- **A C ABI, not Python calls.** Each model exports `<m>_dim` and `<m>_logp_grad`, plus a
  generic `<m>_run`. The Rust driver calls them directly, with the same nuts-rs version and
  defaults as nutpie, so comparisons measure only the logp + gradient code.

Details are on the [Concepts](https://ian-flores.github.io/pytensor-mojo/concepts/) page.

## Quickstart

Requires macOS on Apple Silicon (macOS 15+, Xcode Command Line Tools 16+),
[pixi](https://pixi.sh) and a Rust toolchain.

```bash
git clone https://github.com/ian-flores/pytensor-mojo.git
cd pytensor-mojo
pixi install                # Mojo 1.2, PyMC, PyTensor, nutpie, ...
pixi run build-sampler      # builds the Rust mojo-nuts driver
pixi run test               # optional: full test suite
```

Save this as `example.py` in the repo root. The package isn't installed, so it runs from
there.

```python
import numpy as np
import pymc as pm
from pytensor_mojo.sample import sample

y = np.random.default_rng(0).normal(3.0, 2.0, size=200)

with pm.Model() as model:
    mu = pm.Normal("mu", np.float64(0), np.float64(10))
    sigma = pm.HalfNormal("sigma", np.float64(5))
    pm.Normal("y", mu, sigma, observed=y)

idata = sample(model, chains=4, tune=1000, draws=1000, seed=1)
print(idata["posterior"]["mu"].mean().item())
```

```bash
PYTHONPATH=. pixi run python example.py
```

The first call compiles the model in about a second, and later calls are served from a
cache. Pass parameters as `np.float64(...)` rather than Python floats, because PyTensor
otherwise stores them as float32 constants (see
[Upstream issues](https://ian-flores.github.io/pytensor-mojo/upstream-issues/)).

## Correctness

- **Models nobody wrote for this backend:** of 28
  [posteriordb](https://github.com/stan-dev/posteriordb) reference posteriors ported to PyMC,
  21 compile, all 21 match PyMC's logp and gradient at rtol 1e-10, and 19 match the published
  reference draws (the other 2 are sampler mixing issues). The remaining 7 need Scan or linear
  algebra. See the [posteriordb coverage](https://ian-flores.github.io/pytensor-mojo/benchmarks/posteriordb/)
  page; speed on those mostly small models is roughly at parity with nutpie.
- **Generated logp and gradient** match PyMC's `compile_logp` / `compile_dlogp` at rtol 1e-10
  for every supported model in the test suite.
- **Sampled posteriors** match nutpie within Monte Carlo error. The tests check means against
  combined MCSE, standard deviations, R-hat and ESS.
- **Special functions** (`gammaln`, `digamma`, `erf`, `erfc`, ...) are accuracy-tested against
  high-precision references, with per-function ulp bounds. See
  [Special-function accuracy](https://ian-flores.github.io/pytensor-mojo/accuracy/).
- **CI** runs the full suite on a macOS arm64 runner for every pull request.

## Other tasks

| Command | What it does |
|---|---|
| `pixi run test-rust` | cargo fmt, clippy and the Rust tests |
| `pixi run bench-phase3`, `pixi run bench-phase4` | sampling benchmarks against nutpie |
| `pixi run accuracy` | special-function accuracy tables |
| `pixi run -e docs docs-serve 8000` | the documentation site, locally |

## Learn more

- [Getting started](https://ian-flores.github.io/pytensor-mojo/getting-started/): one model
  end to end
- [User guide](https://ian-flores.github.io/pytensor-mojo/guide/compiling/): compiling models,
  the C ABI, the `mojo-nuts` CLI, environment flags, troubleshooting
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
