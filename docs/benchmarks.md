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
- **Transcendental-bound models** gain 1.6-2.5x. Their per-row work is a libm call per lane
  (`exp`, `log1p`, `lgamma`); see [Concepts](concepts.md#libm-and-stdmath).
- **Eight schools** ties. Its logp + gradient takes about 50 ns, against about 700 ns per
  leapfrog step inside nuts-rs that is the same on both sides.
- **The full log**, with controls such as the same algorithm hand-written in numba and the
  hand-written Mojo kernels, is in the [research log](FINDINGS.md).
