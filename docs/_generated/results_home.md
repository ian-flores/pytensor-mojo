| Model | dim | Mojo + nuts-rs (s) | nutpie (s) | Speedup (median of runs) | Min ESS bulk, Mojo / nutpie |
|---|---:|---:|---:|---:|---:|
| Hierarchical linear regression, N=10k, G=20, K=5 | 28 | 0.294 | 1.611 | 5.46x | 7557 / 7880 |
| Poisson GLM, log link, N=10k, K=5 | 6 | 0.330 | 1.058 | 3.23x | 5005 / 4993 |
| Robust StudentT regression, N=5000, nu free | 6 | 0.260 | 0.636 | 2.44x | 2773 / 2650 |
| Hierarchical Beta-Binomial, G=200 | 202 | 0.1000 | 0.213 | 2.12x | 2152 / 2191 |
| Eight schools, non-centered | 10 | 0.0176 | 0.0148 | 0.86x | 2026 / 2192 |
