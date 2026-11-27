| Parameter | Conditions | libm per lane | Vectorized | Unit | Speedup |
|---|---|---:|---:|---|---:|
| logp + gradient per evaluation, Poisson GLM N=10k | from Rust, min of 7 alternating rounds | 46.5 | 36.2 | us | 1.28x |
| logp + gradient per evaluation, StudentT reg. N=5k | from Rust, min of 7 alternating rounds | 31.9 | 23.9 | us | 1.33x |
| logp + gradient per evaluation, Beta-Binomial G=200 | from Rust, min of 7 alternating rounds | 3.98 | 2.53 | us | 1.57x |
| logp + gradient per evaluation, HLR N=10k | from Rust, min of 7 alternating rounds | 19 | 19.2 | us | 0.99x |
| logp + gradient per evaluation, Eight schools | from Rust, min of 7 alternating rounds | 0.0398 | 0.0522 | us | 0.76x |
