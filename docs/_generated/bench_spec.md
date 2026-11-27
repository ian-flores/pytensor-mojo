### Hierarchical linear regression, N=10k, G=20, K=5

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.310 | 0.317 | 0.320 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 1.661 | 1.753 | 2.067 | s |
| Speedup vs nutpie | per repetition, same seed | 5.18 | 5.54 | 6.66 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 7110 | 7557 | 7952 | - |
| Min ESS bulk, nutpie | same | 7408 | 7880 | 8238 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 55198 | 55599 | 55920 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0045 | 1.0085 | - |
| Divergences, Mojo | 4000 draws | 0 | 0 | 0 | - |
| Divergences, nutpie | 4000 draws | 0 | 0 | 0 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 19.9 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 122 | - | us |

### Poisson GLM, log link, N=10k, K=5

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.384 | 0.390 | 0.393 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 0.965 | 0.977 | 0.998 | s |
| Speedup vs nutpie | per repetition, same seed | 2.47 | 2.51 | 2.55 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 4662 | 5005 | 5023 | - |
| Min ESS bulk, nutpie | same | 4815 | 4993 | 5142 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 31881 | 32520 | 32815 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0023 | 1.0028 | - |
| Divergences, Mojo | 4000 draws | 0 | 0 | 0 | - |
| Divergences, nutpie | 4000 draws | 0 | 0 | 0 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 46.4 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 127 | - | us |

### Robust StudentT regression, N=5000, nu free

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.311 | 0.313 | 0.316 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 0.627 | 0.631 | 0.641 | s |
| Speedup vs nutpie | per repetition, same seed | 1.98 | 2.02 | 2.06 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 2597 | 2713 | 2840 | - |
| Min ESS bulk, nutpie | same | 2496 | 2650 | 2695 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 38341 | 38393 | 39577 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0014 | 1.0019 | - |
| Divergences, Mojo | 4000 draws | 0 | 0 | 0 | - |
| Divergences, nutpie | 4000 draws | 0 | 0 | 0 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 28.6 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 82.4 | - | us |

### Hierarchical Beta-Binomial, G=200

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.124 | 0.131 | 0.140 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 0.209 | 0.217 | 0.233 | s |
| Speedup vs nutpie | per repetition, same seed | 1.53 | 1.66 | 1.78 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 1976 | 2152 | 2559 | - |
| Min ESS bulk, nutpie | same | 1993 | 2191 | 2367 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 92255 | 96777 | 97750 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0059 | 1.0103 | - |
| Divergences, Mojo | 4000 draws | 0 | 0 | 0 | - |
| Divergences, nutpie | 4000 draws | 0 | 0 | 0 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 3.85 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 5.25 | - | us |

### Eight schools, non-centered

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.0161 | 0.0167 | 0.0208 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 0.0152 | 0.0165 | 0.0169 | s |
| Speedup vs nutpie | per repetition, same seed | 0.74 | 1.00 | 1.02 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 1756 | 2013 | 2381 | - |
| Min ESS bulk, nutpie | same | 1923 | 2192 | 2310 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 54699 | 54876 | 55179 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0021 | 1.0036 | - |
| Divergences, Mojo | 4000 draws | 0 | 2 | 7 | - |
| Divergences, nutpie | 4000 draws | 2 | 3 | 6 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 0.05 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 0.902 | - | us |
