"""Phase 4: sample three representative models with generated Mojo + nuts-rs vs nutpie (numba).

Models (all compiled from the PyMC model by ndcodegen; nothing hand-written):
  * robust StudentT regression, N=5000, K=3, nu free       (gammaln/psi in nu, log1p per row)
  * Poisson GLM with log link, N=10,000, K=5                (exp per row)
  * hierarchical Beta-Binomial, G=200 groups               (Beta logp with free a, b: gammaln/psi)

Posteriors are compared in the unconstrained space (nutpie store_unconstrained=True), so no
per-model back-transforms are needed. Usage: python bench/phase4_sampling.py [--reps 5]
"""
import argparse, json, pathlib, sys, time
import numpy as np
import pymc as pm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
from nuts_bench.runners import compile_nutpie, run_mojo, summarize  # noqa: E402
from phase2_nuts import mojo_kernel_us, numba_kernel_us  # noqa: E402
from pytensor_mojo.ndlinker import compile_model_capi  # noqa: E402

f64 = np.float64


def robust_regression(N=5000, K=3, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(N, K)); beta = r.normal(size=K)
    y = 0.5 + X @ beta + 0.7 * r.standard_t(3, size=N)
    with pm.Model() as m:
        a = pm.Normal("a", f64(0), f64(5))
        b = pm.Normal("b", f64(0), f64(2), shape=K)
        s = pm.HalfNormal("sigma", f64(2))
        nu = pm.Gamma("nu", f64(2), f64(0.1))
        pm.StudentT("y", nu=nu, mu=a + X @ b, sigma=s, observed=y)
    return m


def poisson_glm(N=10_000, K=5, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(scale=0.3, size=(N, K)); beta = r.normal(scale=0.5, size=K)
    y = r.poisson(np.exp(1.0 + X @ beta))
    with pm.Model() as m:
        a = pm.Normal("a", f64(0), f64(2))
        b = pm.Normal("b", f64(0), f64(1), shape=K)
        pm.Poisson("y", mu=pm.math.exp(a + X @ b), observed=y)
    return m


def beta_binomial(G=200, seed=0):
    r = np.random.default_rng(seed)
    n = r.integers(5, 60, size=G)
    p = r.beta(2.0, 5.0, size=G)
    y = r.binomial(n, p)
    with pm.Model() as m:
        mu = pm.Beta("mu", f64(2), f64(2))
        kappa = pm.Gamma("kappa", f64(2), f64(0.05))
        theta = pm.Beta("theta", alpha=mu * kappa, beta=(1 - mu) * kappa, shape=G)
        pm.Binomial("y", n=n, p=theta, observed=y)
    return m


MODELS = {"robust_studentt_N5000": robust_regression, "poisson_glm_N10000": poisson_glm,
          "beta_binomial_G200": beta_binomial}


def nutpie_unconstrained(tr, model):
    """(chains, draws, dim) unconstrained draws in model.value_vars order (C-order ravel)."""
    post = tr["posterior"]
    # nutpie only creates this group when some variable is transformed
    up = tr["unconstrained_posterior"] if "unconstrained_posterior" in tr.children else None
    parts = []
    for v in model.value_vars:
        src = up if up is not None and v.name in up.data_vars else post
        a = np.asarray(src[v.name].values)
        parts.append(a.reshape(a.shape[0], a.shape[1], -1))
    return np.concatenate(parts, axis=2)


def as_post(u):
    return {"x": u}


def run_nutpie_u(cm, model, chains, seed, raw=False):
    import nutpie
    t = time.perf_counter()
    tr = nutpie.sample(cm, chains=chains, tune=1000, draws=1000, seed=seed, cores=chains,
                       progress_bar=False, return_raw_trace=raw, store_unconstrained=not raw)
    sec = time.perf_counter() - t
    return tr, sec


def bench_one(name, reps, chains):
    model = MODELS[name]()
    t = time.perf_counter()
    so, data, dim, _ = compile_model_capi(model, "p4s_" + name.split("_")[0])
    gen_build = time.perf_counter() - t
    sym = "p4s_" + name.split("_")[0]
    cm, nut_compile = compile_nutpie(model)
    res = dict(model=name, dim=dim, gen_build_s=gen_build, nutpie_compile_s=nut_compile,
               gen_kernel_us=mojo_kernel_us(so, sym, data), numba_kernel_us=numba_kernel_us(model), rows=[])
    for rep in range(reps):
        seed = 200 + rep
        r = run_mojo(so, sym, chains=chains, seed=seed, data=data)
        sm = summarize(as_post(r["draws"]))
        res["rows"].append(dict(impl="generated-mojo+nuts-rs", seconds=r["sample_seconds"], grad_evals=r["n_steps_total"],
                                min_ess=min(v["ess_bulk"] for v in sm.values()), max_rhat=max(v["rhat"] for v in sm.values()),
                                divergences=r["divergences"]))
        tr, sec = run_nutpie_u(cm, model, chains, seed)
        sn = summarize(as_post(nutpie_unconstrained(tr, model)))
        ss = tr["sample_stats"]; ws = tr["warmup_sample_stats"]
        _, sec_raw = run_nutpie_u(cm, model, chains, seed, raw=True)
        res["rows"].append(dict(impl="nutpie-numba", seconds=sec_raw, seconds_full=sec,
                                grad_evals=int(ss["n_steps"].values.sum() + ws["n_steps"].values.sum()),
                                min_ess=min(v["ess_bulk"] for v in sn.values()), max_rhat=max(v["rhat"] for v in sn.values()),
                                divergences=int(ss["diverging"].values.sum())))
    return res


def table(allres):
    out = ["| model | dim | impl | sampler-only s | grad evals | min ESS bulk | ESS/s | max R-hat | divergences | speedup |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for res in allres:
        med = {}
        for impl in ("generated-mojo+nuts-rs", "nutpie-numba"):
            rs = [r for r in res["rows"] if r["impl"] == impl]
            med[impl] = {k: float(np.median([r[k] for r in rs])) for k in ("seconds", "grad_evals", "min_ess", "max_rhat", "divergences")}
        for impl, m in med.items():
            sp = med["nutpie-numba"]["seconds"] / m["seconds"]
            out.append(f"| {res['model']} | {res['dim']} | {impl} | {m['seconds']:.3f} | {m['grad_evals']:.0f} | {m['min_ess']:.0f} | "
                       f"{m['min_ess'] / m['seconds']:.3g} | {m['max_rhat']:.4f} | {m['divergences']:.0f} | {sp:.2f}x |")
    out.append("")
    for res in allres:
        out.append(f"{res['model']}: kernel us/eval generated-mojo={res['gen_kernel_us']:.3f} numba(jit_fn from Python)="
                   f"{res['numba_kernel_us']:.3f}; build: mojo {res['gen_build_s']:.2f}s, nutpie {res['nutpie_compile_s']:.2f}s")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default=str(ROOT / "bench" / "phase4_results.json"))
    a = ap.parse_args()
    allres = [bench_one(m, a.reps, a.chains) for m in a.models.split(",")]
    pathlib.Path(a.out).write_text(json.dumps(allres, indent=1))
    print(table(allres))


if __name__ == "__main__":
    main()
