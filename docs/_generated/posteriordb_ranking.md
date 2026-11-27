**Before the sweep's ops** (`ndcodegen_ext` disabled):

| op | models needing it | models it unblocks alone | posteriors |
|---|---:|---:|---|
| `IncSubtensor` | 14 | 8 | wells_data-wells_dist, dogs-dogs, earnings-logearn_interaction, earnings-logearn_logheight_male, kidiq-kidscore_momhsiq, kidiq_with_mom_work-kidscore_interaction_c, mesquite-logmesquite_logvas, nes2000-nes, low_dim_gauss_mix-low_dim_gauss_mix, arma-arma11, garch-garch11, hmm_example-hmm_example, hudson_lynx_hare-lotka_volterra, one_comp_mm_elim_abs-one_comp_mm_elim_abs |
| `AllocEmpty` | 6 | 0 | low_dim_gauss_mix-low_dim_gauss_mix, arma-arma11, garch-garch11, hmm_example-hmm_example, hudson_lynx_hare-lotka_volterra, one_comp_mm_elim_abs-one_comp_mm_elim_abs |
| `Scan` | 5 | 0 | arma-arma11, garch-garch11, hmm_example-hmm_example, hudson_lynx_hare-lotka_volterra, one_comp_mm_elim_abs-one_comp_mm_elim_abs |
| `Split` | 3 | 0 | arma-arma11, garch-garch11, hmm_example-hmm_example |
| `AllocDiag` | 2 | 0 | gp_pois_regr-gp_regr, gp_pois_regr-gp_pois_regr |
| `Cholesky` | 2 | 0 | gp_pois_regr-gp_regr, gp_pois_regr-gp_pois_regr |
| `CumOp` | 2 | 0 | low_dim_gauss_mix-low_dim_gauss_mix, hmm_example-hmm_example |
| `ExtractDiag` | 2 | 0 | gp_pois_regr-gp_regr, gp_pois_regr-gp_pois_regr |
| `Second` | 2 | 0 | low_dim_gauss_mix-low_dim_gauss_mix, hmm_example-hmm_example |
| `SolveTriangular` | 2 | 0 | gp_pois_regr-gp_regr, gp_pois_regr-gp_pois_regr |
| `CAReduce[Maximum]` | 1 | 0 | hmm_example-hmm_example |
| `CAReduce[OR]` | 1 | 0 | hmm_example-hmm_example |
| `SpecifyShape` | 1 | 0 | garch-garch11 |

**Still missing:**

| op | models needing it | models it unblocks alone | posteriors |
|---|---:|---:|---|
| `Scan` | 5 | 2 | arma-arma11, garch-garch11, hmm_example-hmm_example, hudson_lynx_hare-lotka_volterra, one_comp_mm_elim_abs-one_comp_mm_elim_abs |
| `Split` | 3 | 0 | arma-arma11, garch-garch11, hmm_example-hmm_example |
| `AllocDiag` | 2 | 0 | gp_pois_regr-gp_regr, gp_pois_regr-gp_pois_regr |
| `Cholesky` | 2 | 0 | gp_pois_regr-gp_regr, gp_pois_regr-gp_pois_regr |
| `ExtractDiag` | 2 | 0 | gp_pois_regr-gp_regr, gp_pois_regr-gp_pois_regr |
| `SolveTriangular` | 2 | 0 | gp_pois_regr-gp_regr, gp_pois_regr-gp_pois_regr |
| `CAReduce[OR]` | 1 | 0 | hmm_example-hmm_example |
