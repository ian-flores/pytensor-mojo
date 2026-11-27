| posterior | family | dim | status | missing ops | max z vs ref (Mojo / nutpie) | max R-hat | min ESS bulk | div. (Mojo / nutpie) | kernel us, Mojo / numba | Mojo s | nutpie s | speedup |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Rate_1_data-Rate_1_model | GLM (binomial) | 1 | matches-reference | - | 0.72 / 0.42 | 1.002 | 1887 | 0 / 0 | 0.03 / 0.80 | 0.019 | 0.012 | 0.66x |
| Rate_3_data-Rate_3_model | GLM (binomial) | 1 | matches-reference | - | 0.27 / 0.91 | 1.002 | 1836 | 0 / 0 | 0.03 / 0.85 | 0.019 | 0.011 | 0.55x |
| wells_data-wells_dist | GLM (logistic) | 2 | matches-reference | - | 0.89 / 0.92 | 1.002 | 1000 | 0 / 0 | 51.04 / 59.63 | 0.472 | 0.703 | 1.49x |
| nes_logit_data-nes_logit_model | GLM (logistic) | 2 | matches-reference | - | 1.12 / 0.65 | 1.016 | 597 | 0 / 0 | 20.25 / 21.54 | 0.262 | 0.401 | 1.53x |
| dogs-dogs | GLM (logistic) | 3 | matches-reference | - | 1.13 / 0.70 | 1.007 | 1267 | 0 / 0 | 15.96 / 15.78 | 0.241 | 0.273 | 1.13x |
| dogs-dogs_hierarchical | GLM (Bernoulli, power link) | 2 | matches-reference | - | 1.53 / 2.04 | 1.003 | 2065 | 0 / 0 | 31.39 / 45.06 | 0.236 | 0.230 | 0.98x |
| earnings-logearn_interaction | linear regression | 5 | matches-reference | - | 1.87 / 0.64 | 1.007 | 478 | 0 / 0 | 1.73 / 1.85 | 0.300 | 0.192 | 0.64x |
| earnings-logearn_logheight_male | linear regression | 4 | matches-reference | - | 0.59 / 1.15 | 1.016 | 360 | 0 / 0 | 0.92 / 1.68 | 0.240 | 0.207 | 0.86x |
| kidiq-kidscore_momhsiq | linear regression | 4 | matches-reference | - | 1.39 / 1.17 | 1.005 | 681 | 0 / 0 | 0.36 / 1.28 | 0.034 | 0.031 | 0.91x |
| kidiq_with_mom_work-kidscore_interaction_c | linear regression | 5 | matches-reference | - | 1.94 / 1.22 | 1.001 | 3010 | 0 / 0 | 0.58 / 1.43 | 0.027 | 0.019 | 0.71x |
| kilpisjarvi_mod-kilpisjarvi | linear regression | 3 | matches-reference | - | 0.09 / 2.16 | 1.014 | 348 | 0 / 0 | 0.09 / 0.90 | 0.089 | 0.120 | 1.36x |
| mesquite-logmesquite_logvas | linear regression | 8 | matches-reference | - | 2.08 / 2.23 | 1.006 | 715 | 0 / 0 | 0.17 / 1.28 | 0.042 | 0.044 | 1.03x |
| nes2000-nes | linear regression | 10 | matches-reference | - | 1.71 / 1.58 | 1.003 | 1348 | 0 / 0 | 1.55 / 1.92 | 0.116 | 0.092 | 0.79x |
| radon_mn-radon_pooled | linear regression | 3 | matches-reference | - | 0.95 / 1.19 | 1.002 | 4038 | 0 / 0 | 0.54 / 1.24 | 0.020 | 0.014 | 0.69x |
| sblrc-blr | linear regression | 6 | matches-reference | - | 1.32 / 1.65 | 1.003 | 848 | 0 / 0 | 0.61 / 1.73 | 0.023 | 0.023 | 1.00x |
| diamonds-diamonds | linear regression (K=24, N=5000) | 26 | matches-reference | - | 2.51 / 2.26 | 1.014 | 393 | 0 / 0 | 168.76 / 47.06 | 91.214 | 19.375 | 0.21x |
| eight_schools-eight_schools_centered | hierarchical | 10 | matches-PyMC (means match reference, R-hat >= 1.05) | - | 1.88 / 0.75 | 1.059 | 70 | 26 / 36 | 0.05 / 0.92 | 0.021 | 0.019 | 0.91x |
| eight_schools-eight_schools_noncentered | hierarchical | 10 | matches-reference | - | 1.71 / 2.06 | 1.002 | 2296 | 7 / 8 | 0.05 / 0.90 | 0.019 | 0.015 | 0.83x |
| low_dim_gauss_mix-low_dim_gauss_mix | mixture | 5 | matches-PyMC (means match reference, R-hat >= 1.05) | - | 1.35 / 1.54 | 1.581 | 7 | 135 / 0 | 10.30 / 19.54 | 0.130 | 0.193 | 1.48x |
| arK-arK | time series (AR(5)) | 7 | matches-reference | - | 2.46 / 2.12 | 1.003 | 1790 | 0 / 0 | 1.20 / 2.33 | 0.069 | 0.079 | 1.14x |
| arma-arma11 | time series (ARMA(1,1)) | 4 | unsupported | `Scan`, `Split` | - | - | - | - | - | - | - | - |
| garch-garch11 | time series (GARCH(1,1)) | 4 | unsupported | `Scan`, `Split` | - | - | - | - | - | - | - | - |
| hmm_example-hmm_example | hidden Markov model | 4 | unsupported | `CAReduce[OR]`, `Scan`, `Split` | - | - | - | - | - | - | - | - |
| mcycle_gp-accel_gp | Gaussian process (basis-function approx.) | 66 | matches-reference | - | 2.05 / 5.02 | 1.011 | 670 | 235 / 3087 | 10.97 / 5.88 | 9.971 | 4.398 | 0.44x |
| gp_pois_regr-gp_regr | Gaussian process (exact) | 3 | unsupported | `AllocDiag`, `Cholesky`, `ExtractDiag`, `SolveTriangular` | - | - | - | - | - | - | - | - |
| gp_pois_regr-gp_pois_regr | Gaussian process (latent, Poisson) | 13 | unsupported | `AllocDiag`, `Cholesky`, `ExtractDiag`, `SolveTriangular` | - | - | - | - | - | - | - | - |
| hudson_lynx_hare-lotka_volterra | ODE (Lotka-Volterra) | 8 | unsupported | `Scan` | - | - | - | - | - | - | - | - |
| one_comp_mm_elim_abs-one_comp_mm_elim_abs | ODE (pharmacokinetic) | 4 | unsupported | `Scan` | - | - | - | - | - | - | - | - |
