"""Phase 2/3: NUTS with Mojo logp+grad driven by nuts-rs (no Python per gradient) vs nutpie (numba).

Implementations: hand-written Mojo (mojo_capi/), Phase 3 generated Mojo (pytensor_mojo.ndcodegen,
built from the PyMC model's optimized logp+grad graph), nutpie numba.

Usage: python bench/phase2_nuts.py [--reps 5] [--models es8,hlr] [--out results.json]
"""
import argparse, json, pathlib, sys, tempfile, time, timeit
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nuts_bench.runners import (BIN, CAPI, compile_nutpie, es8_constrained, es8_from_nutpie,  # noqa: E402
                                nutpie_posterior, run_mojo, run_nutpie, summarize)
from pytensor_mojo import eight_schools_ref as es8ref, hlr_ref  # noqa: E402
import subprocess  # noqa: E402


def hlr_constrained(u, G, K):
    return {"mu_a": u[..., 0], "sigma_a": np.exp(u[..., 1]), "a": u[..., 2:2 + G],
            "beta": u[..., 2 + G:2 + G + K], "sigma": np.exp(u[..., 2 + G + K])}


def hlr_from_nutpie(post):
    return {k: post[k] for k in ["mu_a", "sigma_a", "a", "beta", "sigma"]}


def numba_kernel_us(model):
    """Per-eval time of PyTensor's numba-compiled flat logp+grad (Python call included)."""
    import pytensor, pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace
    vv = model.value_vars
    q = pt.dvector("q")
    sizes = [int(np.prod(v.type.shape)) if v.ndim else 1 for v in vv]
    parts, i = [], 0
    for v, n in zip(vv, sizes):
        parts.append(q[i] if v.ndim == 0 else q[i:i + n].reshape(v.type.shape))
        i += n
    lp = model.logp(jacobian=True)
    outs = graph_replace([lp, *pt.grad(lp, vv)], dict(zip(vv, parts)), strict=False)
    f = pytensor.function([q], [outs[0], pt.concatenate([pt.atleast_1d(o).ravel() for o in outs[1:]])],
                          mode="NUMBA")
    jf = f.vm.jit_fn
    x = np.full(i, 0.1)
    jf(x)
    n = 2000 if i > 20 else 200_000
    return min(timeit.repeat(lambda: jf(x), number=n, repeat=5)) / n * 1e6


def mojo_kernel_us(lib, model, data=None):
    big = data is not None and len(data) > 1000
    cmd = [str(BIN), "--lib", str(lib), "--model", model, "--bench-evals", "20000" if big else "2000000"]
    with tempfile.TemporaryDirectory() as d:
        if data is not None:
            p = pathlib.Path(d) / "d.f64"; data.astype("<f8").tofile(p); cmd += ["--data", str(p)]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)["ns_per_eval"] / 1e3


def one(rows, label, seconds, grad_evals, summ, div):
    ess = min(r["ess_bulk"] for r in summ.values())
    rhat = max(r["rhat"] for r in summ.values())
    rows.append(dict(impl=label, seconds=seconds, grad_evals=grad_evals, grad_per_s=grad_evals / seconds,
                     min_ess_bulk=ess, ess_per_s=ess / seconds, max_rhat=rhat, divergences=div))


def bench_model(name, reps, chains_list):
    if name == "es8":
        model = es8ref.build_model(); lib, sym, data = CAPI / "libes8.dylib", "es8", None
        conv_m, conv_n = es8_constrained, es8_from_nutpie
    else:
        d = hlr_ref.make_data(); model = hlr_ref.build_model(d)
        lib, sym, data = CAPI / "libhlr.dylib", "hlr", hlr_ref.pack(d)
        conv_m = lambda u: hlr_constrained(u, d["G"], d["K"])  # noqa: E731
        conv_n = hlr_from_nutpie
    from pytensor_mojo.ndlinker import compile_model_capi
    gsym = sym + "gen"
    t = time.perf_counter()
    gso, gdata, _, _ = compile_model_capi(model, gsym)
    gen_compile_s = time.perf_counter() - t
    cm, compile_s = compile_nutpie(model)
    res = dict(model=name, nutpie_compile_s=compile_s, gen_compile_s=gen_compile_s,
               numba_kernel_us=numba_kernel_us(model),
               mojo_kernel_us=mojo_kernel_us(lib, sym, data),
               gen_kernel_us=mojo_kernel_us(gso, gsym, gdata), runs={})
    for chains in chains_list:
        rows = []
        for rep in range(reps):
            seed = 100 + rep
            r = run_mojo(lib, sym, chains=chains, seed=seed, data=data)
            one(rows, "mojo+nuts-rs", r["sample_seconds"], r["n_steps_total"], summarize(conv_m(r["draws"])), r["divergences"])
            r = run_mojo(gso, gsym, chains=chains, seed=seed, data=gdata)
            one(rows, "generated-mojo+nuts-rs", r["sample_seconds"], r["n_steps_total"], summarize(conv_m(r["draws"])), r["divergences"])
            tr, sec = run_nutpie(cm, chains=chains, seed=seed)
            post, div, ns = nutpie_posterior(tr)
            one(rows, "nutpie-numba (full call)", sec, ns, summarize(conv_n(post)), div)
            _, sec_raw = run_nutpie(cm, chains=chains, seed=seed, raw=True)
            rows[-1]["seconds_raw_trace"] = sec_raw
        res["runs"][chains] = rows
    return res


def median_table(res):
    lines = []
    for chains, rows in res["runs"].items():
        for impl in dict.fromkeys(r["impl"] for r in rows):
            rs = [r for r in rows if r["impl"] == impl]
            med = {k: float(np.median([r[k] for r in rs])) for k in rs[0] if k != "impl"}
            lines.append((res["model"], chains, impl, med, len(rs)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--models", default="es8,hlr")
    ap.add_argument("--out", default=str(ROOT / "bench" / "phase2_results.json"))
    a = ap.parse_args()
    allres = [bench_model(m, a.reps, [4, 1]) for m in a.models.split(",")]
    pathlib.Path(a.out).write_text(json.dumps(allres, indent=1))
    print("| model | chains | impl | sample s (median) | sampler-only s | grad evals | grad evals/s | min ESS bulk | ESS/s | max R-hat | divergences |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, chains, impl, m, n in sum((median_table(r) for r in allres), []):
        raw = f"{m['seconds_raw_trace']:.4f}" if "seconds_raw_trace" in m else f"{m['seconds']:.4f}"
        print(f"| {model} | {chains} | {impl} | {m['seconds']:.4f} | {raw} | {m['grad_evals']:.0f} | "
              f"{m['grad_per_s']:.3g} | {m['min_ess_bulk']:.0f} | {m['ess_per_s']:.3g} | {m['max_rhat']:.4f} | {m['divergences']:.0f} |")
    for r in allres:
        print(f"{r['model']}: kernel us/eval hand-mojo={r['mojo_kernel_us']:.3f} generated-mojo={r['gen_kernel_us']:.3f} "
              f"numba(jit_fn from Python)={r['numba_kernel_us']:.3f}; nutpie compile {r['nutpie_compile_s']:.2f}s, "
              f"generated build {r['gen_compile_s']:.2f}s (cached .so if unchanged)")


if __name__ == "__main__":
    main()
