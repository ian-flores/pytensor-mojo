| Model | dim | Mojo + nuts-rs (s) | nutpie (s) | Speedup (median of runs) | Min ESS bulk, Mojo / nutpie |
|---|---:|---:|---:|---:|---:|
| Hierarchical linear regression, N=10k, G=20, K=5 | 28 | 0.317 | 1.753 | 5.54x | 7557 / 7880 |
| Poisson GLM, log link, N=10k, K=5 | 6 | 0.390 | 0.977 | 2.51x | 5005 / 4993 |
| Robust StudentT regression, N=5000, nu free | 6 | 0.313 | 0.631 | 2.02x | 2713 / 2650 |
| Hierarchical Beta-Binomial, G=200 | 202 | 0.131 | 0.217 | 1.66x | 2152 / 2191 |
| Eight schools, non-centered | 10 | 0.0167 | 0.0165 | 1.00x | 2013 / 2192 |
