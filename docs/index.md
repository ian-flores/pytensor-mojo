---
template: home.html
title: Home
---

<div class="ds-front" markdown>

<div markdown>

## Features

- Compiles a PyMC model's logp and gradient, taken from PyTensor's own optimized graph, into one Mojo source file and a shared library
- Fuses every pass over the data into a single 8-wide SIMD loop, including gathers, mat-vecs, reductions and the scatter-add of the gradient
- Exports a four-function C ABI; the bundled `mojo-nuts` driver samples it with nuts-rs, the NUTS core of nutpie, with nutpie's default settings
- Has no Python in the per-gradient loop
- Supports common likelihoods and priors with their default transforms; see [Supported distributions and ops](reference/supported.md)
- Has special functions accuracy-tested against 40-digit references
- Matches PyMC's `compile_logp` / `compile_dlogp` at rtol 1e-10 in every test

## Applications

- Faster NUTS for PyMC models whose gradient cost is dominated by data size
- A study of Mojo as a compile target for probabilistic programming

</div>

<div markdown>

## Typical performance characteristics

<figure class="ds-figure" markdown>
--8<-- "fig-speedup.svg"
<figcaption>Sampling speedup over nutpie (numba backend) at equal settings, 4 chains, median of 5 runs.</figcaption>
</figure>

</div>

</div>

## Typical performance

--8<-- "results_home.md"

<div class="ds-conditions" markdown>
Test conditions: Apple M1 Pro, macOS 27.2; Mojo 1.2.0, PyMC 6.4.0, PyTensor 3.3.8, nutpie 0.16.11,
nuts-rs 0.18.3 on both sides. 1000 tune + 1000 draws per chain, sampling time only (compile
excluded); nutpie timed without its Python trace conversion. Median of 5 runs. Eight schools ties
because its logp + gradient costs about 50 ns against about 700 ns of sampler work per leapfrog
step, which is identical on both sides. Full methodology: [Benchmarks](benchmarks.md).
</div>

## Example

```python
import numpy as np
import pymc as pm
from pytensor_mojo.sample import sample

y = np.random.default_rng(0).normal(3.0, 2.0, size=200)
with pm.Model() as model:
    mu = pm.Normal("mu", np.float64(0), np.float64(10))
    sigma = pm.HalfNormal("sigma", np.float64(5))
    pm.Normal("y", mu, sigma, observed=y)

idata = sample(model, chains=4, tune=1000, draws=1000, seed=1)   # an ArviZ DataTree
print(idata["posterior"]["mu"].mean().item())
```

`sample` compiles the model with the Mojo code generator, writes the model data, runs `mojo-nuts`,
and maps the unconstrained draws back through PyMC's transforms.

## Status

This is an experimental research prototype. It is not on PyPI and has no stable API. It has
been built and benchmarked only on macOS with Apple Silicon. Read
[Limitations and roadmap](limitations.md) before relying on it.

## Where to go next

- [Getting started](getting-started.md): install, build, and sample a first model.
- [Concepts](concepts.md): how a PyMC model becomes a fused SIMD kernel.
- [Benchmarks](benchmarks.md) and [Special-function accuracy](accuracy.md): spec tables with Min / Typ / Max.
- [Research log](FINDINGS.md): every measurement, control and dead end, phase by phase.
