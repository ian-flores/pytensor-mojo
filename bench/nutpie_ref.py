"""Reference only: nutpie (Rust NUTS + numba-compiled logp, no Python in the loop) on eight schools."""
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import nutpie  # noqa: E402
from pytensor_mojo.eight_schools_ref import build_model  # noqa: E402

t0 = time.perf_counter()
cm = nutpie.compile_pymc_model(build_model())
t1 = time.perf_counter()
tr = nutpie.sample(cm, draws=2000, tune=1000, chains=1, seed=1, progress_bar=False, store_unconstrained=False)
t2 = time.perf_counter()
steps = int(tr.sample_stats["n_steps"].sum()) if "n_steps" in tr.sample_stats else None
print(f"nutpie compile {t1-t0:.2f}s, sample 1 chain 3000 iters {t2-t1:.3f}s, post-warmup leapfrog steps={steps}")
