"""Phase 7: generated logp+grad kernel time per evaluation, libm per lane vs vectorized math.

Each mode is compiled in a fresh process (PYTENSOR_MOJO_LIBM is read at import). The two
libraries are then timed from Rust (mojo-nuts --bench-evals) in alternation, 7 rounds, so
both see the same machine load; min and median are reported. Writes bench/phase7_kernels.json.
"""
import json, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ["hlr", "es8", "poisson_glm_N10000", "robust_studentt_N5000", "beta_binomial_G200"]

CHILD = r'''
import sys, json, statistics
sys.path[:0] = [{root!r}, {root!r} + "/bench"]
from pytensor_mojo.ndlinker import compile_model_capi
from phase2_nuts import mojo_kernel_us
from pytensor_mojo import hlr_ref, eight_schools_ref
import phase4_sampling as p4
out = {{}}
for key in {models!r}:
    model = {{"hlr": lambda: hlr_ref.build_model(hlr_ref.make_data()), "es8": eight_schools_ref.build_model}}.get(
        key, lambda: p4.MODELS[key]())()
    sym = "k7_" + key.split("_")[0] + ("_libm" if {libm} else "_simd")
    so, data, dim, _ = compile_model_capi(model, sym)
    dpath = {scratch!r} + "/" + sym + ".f64"
    data.astype("<f8").tofile(dpath)
    out[key] = [str(so), sym, dpath, int(len(data) > 1000)]
print("RESULT " + json.dumps(out))
'''


def build(libm: bool, scratch: str):
    env = dict(os.environ, PYTENSOR_MOJO_LIBM="1" if libm else "0")
    code = CHILD.format(root=str(ROOT), models=MODELS, libm=libm, scratch=scratch)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        raise SystemExit(r.stdout[-2000:] + r.stderr[-3000:])
    return json.loads(line[0][7:])


def bench_evals(so, sym, dpath, big):
    binp = ROOT / "sampler" / "target" / "release" / "mojo-nuts"
    n = "20000" if big else "2000000"
    out = subprocess.run([str(binp), "--lib", so, "--model", sym, "--data", dpath, "--bench-evals", n],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["ns_per_eval"] / 1e3


def main():
    import statistics, tempfile
    with tempfile.TemporaryDirectory() as scratch:
        libm, simd = build(True, scratch), build(False, scratch)
        t = {k: {"libm": [], "simd": []} for k in MODELS}
        for _ in range(7):
            for k in MODELS:
                t[k]["libm"].append(bench_evals(*libm[k]))
                t[k]["simd"].append(bench_evals(*simd[k]))
    res = {}
    for k in MODELS:
        lm, sm = min(t[k]["libm"]), min(t[k]["simd"])
        res[k] = dict(libm_us=lm, simd_us=sm, speedup=lm / sm,
                      libm_median_us=statistics.median(t[k]["libm"]), simd_median_us=statistics.median(t[k]["simd"]))
    (ROOT / "bench" / "phase7_kernels.json").write_text(json.dumps(res, indent=1))
    print("| model | libm per lane, min (median) us | vectorized, min (median) us | kernel speedup (min) |")
    print("|---|---|---|---|")
    for k, v in res.items():
        print(f"| {k} | {v['libm_us']:.3f} ({v['libm_median_us']:.3f}) | {v['simd_us']:.3f} ({v['simd_median_us']:.3f}) | {v['speedup']:.2f}x |")


if __name__ == "__main__":
    main()
