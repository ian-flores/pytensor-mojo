# Sampling performance

## Methodology

<div class="ds-conditions" markdown>
Hardware: Apple M1 Pro (8 performance + 2 efficiency cores), macOS 27.2, no CPU pinning.

Software: Mojo 1.2.0, Python 3.12.16, PyMC 6.4.0, PyTensor 3.3.8, nutpie 0.16.11 (numba
backend), nuts-rs 0.18.3 on both sides, Rust 1.92.

Timed: sampling only, with compile time excluded on both sides. For Mojo, the wall time from
`Sampler::new` to finished traces inside `mojo-nuts`. For nutpie, `nutpie.sample(...,
return_raw_trace=True)`, i.e. without building the xarray DataTree, which adds 15-20 ms per
call.

Load: the rev 0.2-dev results were measured with no other benchmark jobs running. The load
average (1, 5, 15 min) was 5.94 / 7.46 / 11.06 before the runs and 4.04 / 5.39 / 8.79 after;
the remaining load came from unrelated desktop processes. Each repetition runs Mojo and nutpie
back to back.

Settings: 4 chains on 4 threads, 1000 tune + 1000 draws, nutpie's default NUTS settings on both
sides (asserted by test). Repetitions: 5 per configuration with seeds 100-104 or 200-204; Typ =
median, Min / Max = extremes over the repetitions. ESS is the bulk ESS minimum over all
unconstrained coordinates (arviz-stats).
</div>

Kernels are generated from the PyMC model in every case; nothing is hand-written. Commands:
`pixi run bench-phase3` (regression, eight schools) and `pixi run bench-phase4` (the other
three). The raw results are in `bench/phase3_results.json` and `bench/phase4_results.json`.

<figure class="ds-figure" markdown>
--8<-- "fig-speedup.svg"
<figcaption>Sampling speedup over nutpie per model. Bar: median of 5; whisker: min to max of the per-repetition ratio.</figcaption>
</figure>

<figure class="ds-figure" markdown>
--8<-- "fig-kernel.svg"
<figcaption>logp + gradient time per evaluation, log scale. The numba timing includes about 0.5 us of Python call overhead, which dominates for eight schools.</figcaption>
</figure>

## Results

<div class="ds-spec" markdown>

--8<-- "bench_spec.md"

</div>

## Reading the results

- **Data-bound models** gain the most. In the regression model the per-row work is
  multiply-adds, and fusion plus explicit SIMD give 5.5x.
- **Transcendental-bound models** (Poisson, StudentT, Beta-Binomial) gain 2.1-3.1x. Their
  per-row work is `exp`, `log1p` or `lgamma`, which since rev 0.2-dev run as vectorized SIMD
  code instead of a libm call per lane (they gained 1.6-2.5x with libm); see
  [Concepts](concepts.md#transcendentals). The remaining gap to the regression is the rest of
  the row: mat-vecs, sums and the gradient.
- **Eight schools** is 1.28x faster with one chain and 0.86x with four chains. Its logp +
  gradient takes about 50 ns, against about 700 ns per leapfrog step inside nuts-rs that is the
  same on both sides, so the result is decided by sampler and thread overhead. That overhead
  was already 0.93x in rev 0.1.0. The vectorized math adds 13 ns per evaluation here, which is
  about 0.7 ms of a 17.6 ms run, so rev 0.2-dev has a small regression on this model.
- **The full log**, with controls such as the same algorithm hand-written in numba and the
  hand-written Mojo kernels, is in the [research log](FINDINGS.md).
