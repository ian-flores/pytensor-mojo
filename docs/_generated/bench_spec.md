### Hierarchical linear regression, N=10k, G=20, K=5

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.291 | 0.294 | 0.300 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 1.586 | 1.611 | 1.625 | s |
| Speedup vs nutpie | per repetition, same seed | 5.36 | 5.46 | 5.53 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 7110 | 7557 | 7952 | - |
| Min ESS bulk, nutpie | same | 7408 | 7880 | 8238 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 55198 | 55599 | 55920 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0045 | 1.0085 | - |
| Divergences, Mojo | 4000 draws | 0 | 0 | 0 | - |
| Divergences, nutpie | 4000 draws | 0 | 0 | 0 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 19.3 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 118 | - | us |

### Poisson GLM, log link, N=10k, K=5

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.325 | 0.330 | 0.340 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 1.046 | 1.058 | 1.099 | s |
| Speedup vs nutpie | per repetition, same seed | 3.17 | 3.23 | 3.37 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 4662 | 5005 | 5023 | - |
| Min ESS bulk, nutpie | same | 4815 | 4993 | 5142 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 31881 | 32520 | 32815 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0023 | 1.0028 | - |
| Divergences, Mojo | 4000 draws | 0 | 0 | 0 | - |
| Divergences, nutpie | 4000 draws | 0 | 0 | 0 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 37.7 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 129 | - | us |

### Robust StudentT regression, N=5000, nu free

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.258 | 0.260 | 0.262 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 0.625 | 0.636 | 0.643 | s |
| Speedup vs nutpie | per repetition, same seed | 2.40 | 2.44 | 2.48 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 2322 | 2773 | 2840 | - |
| Min ESS bulk, nutpie | same | 2496 | 2650 | 2695 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 38031 | 38331 | 38679 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0018 | 1.0027 | - |
| Divergences, Mojo | 4000 draws | 0 | 0 | 0 | - |
| Divergences, nutpie | 4000 draws | 0 | 0 | 0 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 23.8 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 83 | - | us |

### Hierarchical Beta-Binomial, G=200

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.0956 | 0.1000 | 0.1043 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 0.2046 | 0.2134 | 0.2154 | s |
| Speedup vs nutpie | per repetition, same seed | 2.06 | 2.12 | 2.24 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 1976 | 2152 | 2559 | - |
| Min ESS bulk, nutpie | same | 1993 | 2191 | 2367 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 92255 | 96777 | 97750 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0059 | 1.0103 | - |
| Divergences, Mojo | 4000 draws | 0 | 0 | 0 | - |
| Divergences, nutpie | 4000 draws | 0 | 0 | 0 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 2.5 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 5.26 | - | us |

### Eight schools, non-centered

| Parameter | Conditions | Min | Typ | Max | Unit |
|---|---|---:|---:|---:|---|
| Sampling time, generated Mojo + nuts-rs | 4 chains, 1000 tune + 1000 draws | 0.0165 | 0.0176 | 0.0195 | s |
| Sampling time, nutpie (numba) | same; sampler only (`return_raw_trace`) | 0.0142 | 0.0148 | 0.0156 | s |
| Speedup vs nutpie | per repetition, same seed | 0.76 | 0.86 | 0.91 | x |
| Min ESS bulk, Mojo | over all unconstrained coordinates | 2003 | 2026 | 2163 | - |
| Min ESS bulk, nutpie | same | 1923 | 2192 | 2310 | - |
| Gradient evaluations, Mojo | warmup + draws, all chains | 54641 | 55094 | 55536 | - |
| Max R-hat, Mojo | 4 chains | - | 1.0018 | 1.0035 | - |
| Divergences, Mojo | 4000 draws | 2 | 4 | 7 | - |
| Divergences, nutpie | 4000 draws | 2 | 3 | 6 | - |
| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | 0.0523 | - | us |
| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | 0.895 | - | us |
