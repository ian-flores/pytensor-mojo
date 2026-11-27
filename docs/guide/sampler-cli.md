# The mojo-nuts sampler

`sampler/` is a Rust crate, `mojo-nuts`. It loads a C-ABI model library with `libloading` and
samples it with [nuts-rs](https://github.com/pymc-devs/nuts-rs) 0.18.3, the version bundled in
nutpie 0.16.11.

## Settings

The settings are nutpie's `PyNutsSettings.Diag` defaults: a diagonal, gradient-based mass
matrix adaptation, maxdepth 10, target accept 0.8, max energy error 1000, and the same
adaptation windows. A test (`tests/test_nuts.py::test_settings_match_nutpie_defaults`) asserts
that the serialized settings equal nutpie's, apart from chains, tune, draws and seed.

The initial point is uniform(-2, 2) in the unconstrained space.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--lib PATH` | required | Shared library exporting `<NAME>_dim` and `<NAME>_logp_grad` |
| `--model NAME` | required | Symbol prefix in the library |
| `--data FILE` | none | Little-endian float64 data buffer (from `compile_model_capi`) |
| `--chains N` | 4 | Number of chains |
| `--tune N` | 1000 | Warmup draws per chain |
| `--draws N` | 1000 | Post-warmup draws per chain |
| `--seed N` | 1 | Base random seed |
| `--cores N` | chains | Worker threads |
| `--out FILE` | stdout | Where to write the JSON result |
| `--repeat N` | 1 | Sample N times in-process (seeds `seed .. seed+N-1`); all timings go to `repeat_seconds` |
| `--bench-evals N` | none | Time N logp + gradient evaluations at `x = 0.1` and exit |
| `--help` | | Print usage |

## Output

| Field | Meaning |
|---|---|
| `dim` | Parameter dimension |
| `chains[i].draws` | Post-warmup draws, row-major `(n_draws, dim)`, unconstrained space |
| `chains[i].diverging`, `chains[i].n_steps` | Per post-warmup draw |
| `chains[i].divergences`, `n_steps_draws`, `n_steps_total`, `step_size` | Per-chain summaries |
| `sample_seconds` | Wall time from sampler start to finished traces |
| `grad_evals` | Total leapfrog steps over all chains, warmup + draws |
| `settings` | The nuts-rs settings used, serialized |

`pytensor_mojo.sample.raw_to_datatree(model, result)` maps the draws back to the constrained
space and returns an ArviZ `DataTree`.
