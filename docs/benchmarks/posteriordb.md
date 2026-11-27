# posteriordb sweep

A coverage sweep over [posteriordb](https://github.com/stan-dev/posteriordb). It uses 28
posteriors that have reference posterior draws, covering GLMs, linear regressions,
hierarchical models, a mixture, time series, a hidden Markov model, Gaussian processes and
ODEs. Each Stan program was ported to PyMC by hand in `bench/posteriordb/models.py`, keeping
Stan's parameterization. The deviations are listed per model in that file and in
`bench/posteriordb_results.json`.

## Method

<div class="ds-conditions" markdown>
For each posterior, `bench/posteriordb/sweep.py`:

1. builds the PyMC model and runs `ndcodegen.missing_ops` on the optimized logp + gradient
   graph;
2. if nothing is missing, compiles the model with `ndcodegen` and checks logp and gradient
   against PyMC's `compile_logp` / `compile_dlogp` at **rtol 1e-10** on 20 points around the
   initial point;
3. samples with `pytensor_mojo.sample` and with nutpie (numba backend): 4 chains,
   1000 tune + 1000 draws, same seed;
4. compares every parameter mean with the posteriordb reference draws (10 chains x 1000
   draws from Stan). A model matches when every \|mean - ref mean\| is under 4.5 combined MCSEs
   (`sqrt(mcse^2 + mcse_ref^2)`), for all of the reference's parameters, and the max R-hat is
   below 1.05.

Timing is sampling only, the median of 3 runs per sampler. For Mojo it is the `mojo-nuts`
sampling time; for nutpie it is `nutpie.sample(..., return_raw_trace=True)`. The kernel column
is the logp + gradient time per evaluation: the Mojo library is timed from Rust, and the
PyTensor numba function from Python, as in [Sampling performance](../benchmarks.md).
Hardware and software are the same as on that page.
</div>

Commands:

```bash
pixi run python bench/posteriordb/sweep.py --fetch       # sparse shallow clone into .cache/posteriordb
pixi run python bench/posteriordb/sweep.py               # full sweep, about 25 minutes
pixi run python bench/posteriordb/sweep.py --no-sample   # coverage and PyMC checks only
```

## Coverage

Status levels:

- `unsupported`: `missing_ops` lists ops the codegen cannot emit.
- `compiles`: the model compiles.
- `matches-PyMC`: logp and gradient also match PyMC at rtol 1e-10.
- `matches-reference`: the Mojo posterior also matches the reference draws.
- `matches-PyMC (means match reference, R-hat >= 1.05)`: the means agree within their (large)
  MCSEs, but the chains did not mix.

--8<-- "posteriordb_coverage.md"

## Missing ops

The ops are ranked first by how many models each one unblocks on its own, then by how many
models need it. The first table is the census before this sweep added any ops; it is computed
with `ndcodegen_ext` switched off.

--8<-- "posteriordb_ranking.md"

The sweep added the cheap ops at the top of the first table. They live in
`pytensor_mojo/ndcodegen_ext.py`:

- `IncSubtensor` / `SetSubtensor` with constant indices and slices. These appear in the
  gradient of every `beta[i]` term.
- `AllocEmpty`, `CumOp`, `Second` and a stepped `Subtensor`, which PyMC's ordered transform
  needs.
- `SpecifyShape`, `max` / `min` reductions, and `All` over N-d bool arrays.

`Scan` (time series, HMM, ODEs) and the linear algebra ops (`Cholesky`, `SolveTriangular`,
`AllocDiag`, `ExtractDiag`, for exact GPs) are left for later. Each is a larger piece of
work. See [Limitations and roadmap](../limitations.md).

`missing_ops` had blind spots. On three models it reported nothing missing at baseline, but
codegen still failed or gave wrong answers:

- `dogs` and `dogs_hierarchical`: `All` over a 2-D bool array failed in codegen.
- `nes_logit_model`: a length-1 view `q[1:]` was read with stride 1 inside a fused loop. The
  logp was wrong by up to 66% with no error raised. This is fixed, and a regression test is in
  `tests/test_ndcodegen_ext.py`.

## Result

**28 posteriors: 21 compile, 21 match PyMC at rtol 1e-10, and 19 match the reference
posterior.** The two that do not match are sampling problems, not codegen problems:

- `eight_schools_centered` is the funnel. Mojo had R-hat 1.06 and 26 divergences; nutpie
  had R-hat 1.03 and 36 divergences with the same seed (1.13 in an earlier run of the sweep).
- `low_dim_gauss_mix`: with seed 1, one of the four Mojo chains stayed in the degenerate mode
  where the two components merge (135 divergences, R-hat 1.58). Seeds 2 and 3 mix on all
  four chains. The logp and gradient along the stuck chain match PyMC.

nutpie matches the reference on 20 of the 21. It fails on `accel_gp`, with 3087 divergences
and R-hat 4.1; Mojo has 235 divergences and matches. All seven unsupported models are blocked
by `Scan` or by linear algebra.

## Speed

The median speedup is 0.91x (range 0.21x to 1.53x, 21 models). Most of these posteriors are
small (N <= 1200, dim <= 10). There the Mojo kernel is as fast as numba or faster
(0.03-2 us per evaluation), but the whole run takes only 0.02-0.3 s, and the differences come
from sampler-side costs and step counts, not from the kernel. Step counts were not recorded
by this sweep. Larger logistic models gain 1.1-1.5x (`wells_dist`, `nes_logit_model`, `dogs`).
Two larger models are slower than nutpie:

- `diamonds` (N=5000, K=24), 0.21x, with a kernel of 169 us against numba's 47 us: the
  gradient term `Xc^T ((y - mu) / sigma^2)` is not fused.
  PyTensor places the division after the `ExpandDims`, so the coldot pattern does not match.
  The kernel then falls back to a 2-D elemwise plus a scalar `Dot` loop (no BLAS).
- `accel_gp`, 0.44x (11 us against 5.9 us): the same unfused scalar `Dot` loops, on the
  133 x 40 and 133 x 20 basis-function matrices.

The data-bound regressions in [Sampling performance](../benchmarks.md) are fused and gain
2-5x. Extending the fusion planner to these graph shapes is the next speed item.
