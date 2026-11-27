"""Generate data-derived documentation assets so the site cannot drift from code and results.

Writes into docs/_generated/:
  results_home.md       headline results table (all sampled models, 4 chains)
  bench_spec.md         datasheet spec tables (Min / Typ / Max over the 5 benchmark reps)
  accuracy_spec.md      special-function accuracy spec table
  accuracy7_spec.md     vectorized exp/log/log1p/expm1/lgamma accuracy (10M points each)
  kernels7_spec.md      kernel time, libm per lane vs vectorized transcendentals
  supported_ops.md      tensor and scalar op tables, read from ndcodegen's dispatch and op table
  distributions.md      distribution / transform table, from the Phase 4 census (missing_ops)
  fig-speedup.svg       Figure: sampling speedup vs nutpie, per model (min..max whiskers)
  fig-kernel.svg        Figure: logp+grad time per evaluation, generated Mojo vs numba graph
  fig-arch.svg          Figure: architecture

Run from the repo root in the default pixi env:  pixi run docs-assets
Check mode (used by tests/test_docs_generated.py):  python docs/gen/build_assets.py --check
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "_generated"
BENCH = ROOT / "bench"

PAPER, PAPER2, INK, RULE = "#f4ecdc", "#fbf6ec", "#2b1f12", "#dcc8a4"
CLAY, SAGE, OCHRE, DUSK = "#bf5a36", "#7c8a4e", "#d4a03c", "#3f5d72"
MONO = "Space Mono, ui-monospace, Menlo, monospace"

MODEL_LABELS = {
    "hlr": "Hierarchical linear regression, N=10k, G=20, K=5",
    "es8": "Eight schools, non-centered",
    "poisson_glm_N10000": "Poisson GLM, log link, N=10k, K=5",
    "robust_studentt_N5000": "Robust StudentT regression, N=5000, nu free",
    "beta_binomial_G200": "Hierarchical Beta-Binomial, G=200",
}
SHORT = {"hlr": "HLR N=10k", "es8": "Eight schools", "poisson_glm_N10000": "Poisson GLM N=10k",
         "robust_studentt_N5000": "StudentT reg. N=5k", "beta_binomial_G200": "Beta-Binomial G=200"}
DIMS = {"hlr": 28, "es8": 10}


# ------------------------------------------------------------------ results
def load_runs():
    """Per model: per-rep (mojo_s, nutpie_s, mojo_ess, nutpie_ess, ...) for 4 chains."""
    models = []
    p3 = json.loads((BENCH / "phase3_results.json").read_text())
    for r in sorted(p3, key=lambda r: 0 if r["model"] == "hlr" else 1):
        rows = r["runs"]["4"]
        gen = [x for x in rows if x["impl"] == "generated-mojo+nuts-rs"]
        nut = [x for x in rows if x["impl"].startswith("nutpie")]
        models.append(dict(key=r["model"], dim=DIMS[r["model"]],
                           mojo_s=[x["seconds"] for x in gen], nut_s=[x["seconds_raw_trace"] for x in nut],
                           mojo_ess=[x["min_ess_bulk"] for x in gen], nut_ess=[x["min_ess_bulk"] for x in nut],
                           mojo_grad=[x["grad_evals"] for x in gen], nut_grad=[x["grad_evals"] for x in nut],
                           mojo_rhat=[x["max_rhat"] for x in gen], mojo_div=[x["divergences"] for x in gen],
                           nut_div=[x["divergences"] for x in nut],
                           gen_kernel_us=r["gen_kernel_us"], numba_kernel_us=r["numba_kernel_us"]))
    p4 = json.loads((BENCH / "phase4_results.json").read_text())
    for r in p4:
        gen = [x for x in r["rows"] if x["impl"] == "generated-mojo+nuts-rs"]
        nut = [x for x in r["rows"] if x["impl"] == "nutpie-numba"]
        models.append(dict(key=r["model"], dim=r["dim"],
                           mojo_s=[x["seconds"] for x in gen], nut_s=[x["seconds"] for x in nut],
                           mojo_ess=[x["min_ess"] for x in gen], nut_ess=[x["min_ess"] for x in nut],
                           mojo_grad=[x["grad_evals"] for x in gen], nut_grad=[x["grad_evals"] for x in nut],
                           mojo_rhat=[x["max_rhat"] for x in gen], mojo_div=[x["divergences"] for x in gen],
                           nut_div=[x["divergences"] for x in nut],
                           gen_kernel_us=r["gen_kernel_us"], numba_kernel_us=r["numba_kernel_us"]))
    for m in models:
        m["speedups"] = [n / g for g, n in zip(m["mojo_s"], m["nut_s"])]
    order = ["hlr", "poisson_glm_N10000", "robust_studentt_N5000", "beta_binomial_G200", "es8"]
    return sorted(models, key=lambda m: order.index(m["key"]))


def fmt_s(x):
    return f"{x:.3f}" if x >= 0.1 else f"{x:.4f}"


def results_home(models):
    lines = ["| Model | dim | Mojo + nuts-rs (s) | nutpie (s) | Speedup (median of runs) | Min ESS bulk, Mojo / nutpie |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        ms, ns = np.median(m["mojo_s"]), np.median(m["nut_s"])
        sp = np.median(m["speedups"])  # median of per-repetition ratios (same as the figures)
        lines.append(f"| {MODEL_LABELS[m['key']]} | {m['dim']} | {fmt_s(ms)} | {fmt_s(ns)} | "
                     f"{sp:.2f}x | {np.median(m['mojo_ess']):.0f} / {np.median(m['nut_ess']):.0f} |")
    return "\n".join(lines) + "\n"


def bench_spec(models):
    out = []
    for k, m in enumerate(models, 1):
        def row(param, cond, vals, unit, fmt="{:.3f}", mn=True, mx=True):
            v = np.asarray(vals, float)
            lo = fmt.format(v.min()) if mn else "-"
            hi = fmt.format(v.max()) if mx else "-"
            return f"| {param} | {cond} | {lo} | {fmt.format(np.median(v))} | {hi} | {unit} |"
        f3 = "{:.4f}" if np.median(m["mojo_s"]) < 0.1 else "{:.3f}"
        out += [f"### {MODEL_LABELS[m['key']]}", "",
                "| Parameter | Conditions | Min | Typ | Max | Unit |", "|---|---|---:|---:|---:|---|",
                row("Sampling time, generated Mojo + nuts-rs", "4 chains, 1000 tune + 1000 draws", m["mojo_s"], "s", f3),
                row("Sampling time, nutpie (numba)", "same; sampler only (`return_raw_trace`)", m["nut_s"], "s", f3),
                row("Speedup vs nutpie", "per repetition, same seed", m["speedups"], "x", "{:.2f}"),
                row("Min ESS bulk, Mojo", "over all unconstrained coordinates", m["mojo_ess"], "-", "{:.0f}"),
                row("Min ESS bulk, nutpie", "same", m["nut_ess"], "-", "{:.0f}"),
                row("Gradient evaluations, Mojo", "warmup + draws, all chains", m["mojo_grad"], "-", "{:.0f}"),
                row("Max R-hat, Mojo", "4 chains", m["mojo_rhat"], "-", "{:.4f}", mn=False),
                row("Divergences, Mojo", "4000 draws", m["mojo_div"], "-", "{:.0f}"),
                row("Divergences, nutpie", "4000 draws", m["nut_div"], "-", "{:.0f}"),
                f"| logp + gradient per evaluation, generated Mojo | timed from Rust, 20k-2M evals | - | {m['gen_kernel_us']:.3g} | - | us |",
                f"| logp + gradient per evaluation, PyTensor numba graph | `jit_fn` called from Python | - | {m['numba_kernel_us']:.3g} | - | us |",
                ""]
    return "\n".join(out)


def accuracy7_spec():
    p = BENCH / "phase7_accuracy.json"
    rows = json.loads(p.read_text())
    lines = ["| Parameter | Conditions | n | Mean | Max | Unit | Within 0.5 ulp |",
             "|---|---|---:|---:|---:|---|---:|"]
    for r in rows:
        lines.append(f"| {r['func']} error | {r['set']} | {r['n']:,} | {r['mean_ulp']:.3f} | {r['max_ulp']:.3f} | ulp | "
                     f"{100 * r['frac_le_half']:.2f}% |")
    return "\n".join(lines) + "\n"


def kernels7_spec():
    res = json.loads((BENCH / "phase7_kernels.json").read_text())
    order = ["poisson_glm_N10000", "robust_studentt_N5000", "beta_binomial_G200", "hlr", "es8"]
    lines = ["| Parameter | Conditions | libm per lane | Vectorized | Unit | Speedup |",
             "|---|---|---:|---:|---|---:|"]
    for k in order:
        v = res[k]
        lines.append(f"| logp + gradient per evaluation, {SHORT[k]} | from Rust, min of 7 alternating rounds | "
                     f"{v['libm_us']:.3g} | {v['simd_us']:.3g} | us | {v['speedup']:.2f}x |")
    return "\n".join(lines) + "\n"


def accuracy_spec():
    rows = json.loads((BENCH / "phase4_accuracy.json").read_text())
    ref = {"gammaln": "40-digit Decimal", "psi": "40-digit Decimal", "erfc": "40-digit Decimal"}
    lines = ["| Parameter | Conditions | Reference | Typ | Max | Unit | Max rel. error | SciPy max rel. error |",
             "|---|---|---|---:|---:|---|---:|---:|"]
    for r in rows:
        op = r["op"]
        refn = ref.get(op, "numpy" if op in ("expm1", "pow", "log1mexp", "log", "log1p") or "std.math" in op else "scipy.special")
        sr = f"{r['scipy_rel']:.1e}" if r["scipy_rel"] is not None else "-"
        lines.append(f"| {op} error | {r['range']}, n={r['n']} | {refn} | {r['median_ulp']:.0f} | {r['max_ulp']:.0f} | ulp | "
                     f"{r['max_rel']:.1e} | {sr} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ ops / distributions
TENSOR_NOTES = {
    "Elemwise": "N-d broadcasting; 1-D loops fused with producers/consumers and emitted as SIMD (W=8) + scalar tail",
    "DimShuffle": "view (transpose, expand, squeeze); no copy",
    "Subtensor": "view, constant integer index or slice (step 1)",
    "CheckAndRaise": "CheckParameterValue: failed condition returns status 1",
    "ScalarFromTensor": "view",
    "TensorFromScalar": "view",
    "MakeVector": "bool vectors folded into All; float vectors materialized",
    "Dot": "2-D/1-D combinations; mat-vec fused into loops (row dot inlined, X^T w accumulated); otherwise scalar loops, no BLAS",
    "AdvancedSubtensor": "1-D gather a[idx]; inlined into fused loops (splat when all lanes share an index)",
    "AdvancedSubtensor1": "alias of AdvancedSubtensor in PyTensor 3",
    "AdvancedIncSubtensor": "1-D scatter-add inc_subtensor(a[idx], v); fused as a run-length register accumulator",
    "AdvancedIncSubtensor1": "alias of AdvancedIncSubtensor in PyTensor 3",
    "Join": "axis 0 of 1-D inputs",
    "Alloc": "broadcast fill",
    "Sum": "all axes or given axes; fused as a SIMD accumulator when it consumes a 1-D loop",
    "All": "CAReduce AND; fused as a SIMD bool accumulator",
}


def _impl_of(fn, arity):
    e = fn(*["x", "y", "z"][:arity])
    if "slgamma" in e:
        return "vectorized: Chebyshev kernel at the zeros, recurrence, Stirling (`mojo_math`); libm for x <= 0"
    for k, nm in (("sexpm1", "expm1"), ("slog1p", "log1p"), ("sexp", "exp"), ("slog", "log")):
        if k in e:
            return f"vectorized fdlibm-style {nm} (`mojo_math`, <= 1 ulp)" + (" (inside formula)" if not e.startswith(k) else "")
    if "vpsi" in e:
        return "own implementation (reflection, root Taylor series, compensated recurrence + asymptotic)"
    if "vlgamma" in e:
        return "libm `lgamma_r` per lane"
    if "vexp" in e:
        return "libm `exp` per lane" + (" (inside formula)" if not e.startswith("vexp") else "")
    if "vlibm1" in e:
        name = e.split('"')[1]
        return f"libm `{name}` per lane"
    if "vpow" in e:
        return "libm `pow` per lane"
    if "vlog1mexp" in e:
        return "Maechler (2012) piecewise, libm"
    if "log1p(" in e or e.startswith("log(") or "sqrt(" in e:
        return "Mojo `std.math`, SIMD (accuracy-tested, <= 1 ulp)"
    return "inline SIMD"


def supported_ops():
    sys.path.insert(0, str(ROOT))
    from pytensor_mojo import ndcodegen as nd
    tbl = nd._sop_table("1")
    rows = []
    for cls, fn in tbl.items():
        ar = fn.__code__.co_argcount if fn.__code__.co_argcount else 2
        if fn.__code__.co_flags & 0x04:  # *args
            ar = 2
        rows.append((cls.__name__, _impl_of(fn, ar)))
    rows.sort()
    tensor = sorted({c.__name__ for c in nd._TENSOR_OK} | {"Sum", "All"})
    missing = [t for t in tensor if t not in TENSOR_NOTES]
    if missing:
        raise SystemExit(f"add TENSOR_NOTES for: {missing}")
    out = ["| Tensor op | Notes |", "|---|---|"] + [f"| `{t}` | {TENSOR_NOTES[t]} |" for t in tensor]
    out += ["", "| Scalar op | Implementation |", "|---|---|"] + [f"| `{n}` | {i} |" for n, i in rows]
    return "\n".join(out) + "\n"


def distributions():
    sys.path[:0] = [str(ROOT), str(BENCH)]
    import phase4_census as pc
    from pytensor_mojo.ndcodegen import missing_ops
    out = ["| Distribution | Role | Default transform | Status |", "|---|---|---|---|"]
    for role, builders in (("likelihood (observed)", pc.obs_models()), ("prior", pc.prior_models())):
        for name, b in builders.items():
            mod = b()
            mt, ms = missing_ops(pc.census_graph(mod))
            status = "supported" if not (mt or ms) else "missing: " + ", ".join(mt + ms)
            out.append(f"| {name} | {role} | {pc.transforms(mod)} | {status} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ figures
def _svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{title}" '
            f'class="ds-figure-svg" font-family="{MONO}">\n<title>{title}</title>\n'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{PAPER2}"/>\n{body}\n</svg>\n')


def fig_speedup(models):
    w, left, top, bh, gap = 580, 172, 34, 22, 14
    xmax = 7.0
    h = top + len(models) * (bh + gap) + 30
    sx = lambda v: left + (w - left - 30) * v / xmax  # noqa: E731
    b = []
    for t in range(0, 8):
        x = sx(t)
        b.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{h - 40}" stroke="{RULE}" stroke-width="1"/>')
        b.append(f'<text x="{x:.1f}" y="{h - 24}" font-size="12" fill="{INK}" text-anchor="middle">{t}x</text>')
    b.append(f'<line x1="{sx(1):.1f}" y1="{top - 12}" x2="{sx(1):.1f}" y2="{h - 40}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="3 3"/>')
    b.append(f'<text x="{sx(1) + 4:.1f}" y="{top - 16}" font-size="11" fill="{INK}">nutpie = 1x</text>')
    for i, m in enumerate(models):
        y = top + i * (bh + gap)
        s = np.asarray(m["speedups"])
        typ, lo, hi = np.median(s), s.min(), s.max()
        b.append(f'<text x="{left - 10}" y="{y + bh * 0.7:.1f}" font-size="12.5" fill="{INK}" text-anchor="end">{SHORT[m["key"]]}</text>')
        b.append(f'<rect x="{left}" y="{y}" width="{sx(typ) - left:.1f}" height="{bh}" fill="{CLAY}"/>')
        b.append(f'<line x1="{sx(lo):.1f}" y1="{y + bh / 2}" x2="{sx(hi):.1f}" y2="{y + bh / 2}" stroke="{INK}" stroke-width="1.4"/>')
        for xv in (lo, hi):
            b.append(f'<line x1="{sx(xv):.1f}" y1="{y + 5}" x2="{sx(xv):.1f}" y2="{y + bh - 5}" stroke="{INK}" stroke-width="1.4"/>')
        b.append(f'<text x="{sx(hi) + 8:.1f}" y="{y + bh * 0.7:.1f}" font-size="12.5" fill="{INK}" font-weight="700">{typ:.2f}x</text>')
    return _svg(w, h, "\n".join(b), "Sampling speedup vs nutpie per model")


def fig_kernel(models):
    w, left, top, bh = 720, 190, 30, 12
    rows = len(models)
    h = top + rows * 38 + 56
    lo_e, hi_e = -2, 3  # 0.01 us .. 1000 us
    sx = lambda v: left + (w - left - 40) * (np.log10(v) - lo_e) / (hi_e - lo_e)  # noqa: E731
    b = []
    for e in range(lo_e, hi_e + 1):
        x = sx(10.0 ** e)
        b.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{h - 46}" stroke="{RULE}"/>')
        lab = {-2: "0.01", -1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1000"}[e]
        b.append(f'<text x="{x:.1f}" y="{h - 30}" font-size="11" fill="{INK}" text-anchor="middle">{lab}</text>')
    for i, m in enumerate(models):
        y = top + i * 38
        b.append(f'<text x="{left - 10}" y="{y + 15}" font-size="11.5" fill="{INK}" text-anchor="end">{SHORT[m["key"]]}</text>')
        for j, (v, col, lab) in enumerate(((m["gen_kernel_us"], CLAY, "Mojo"), (m["numba_kernel_us"], DUSK, "numba"))):
            yy = y + j * (bh + 2)
            b.append(f'<rect x="{left}" y="{yy}" width="{sx(v) - left:.1f}" height="{bh}" fill="{col}"/>')
            b.append(f'<text x="{sx(v) + 6:.1f}" y="{yy + 10}" font-size="10" fill="{INK}">{v:.3g} us</text>')
    ly = h - 14
    b.append(f'<rect x="{left}" y="{ly - 9}" width="10" height="10" fill="{CLAY}"/><text x="{left + 14}" y="{ly}" font-size="10.5" fill="{INK}">generated Mojo (from Rust)</text>')
    b.append(f'<rect x="{left + 210}" y="{ly - 9}" width="10" height="10" fill="{DUSK}"/><text x="{left + 224}" y="{ly}" font-size="10.5" fill="{INK}">PyTensor numba graph (from Python)</text>')
    b.append(f'<text x="{w - 40}" y="16" font-size="10.5" fill="{INK}" text-anchor="end">us / evaluation, log scale</text>')
    return _svg(w, h, "\n".join(b), "logp and gradient time per evaluation")


def fig_arch():
    w, h = 720, 250
    boxes = [("PyMC model", "pm.Model"), ("PyTensor graph", "logp + grad,\nflat vector"),
             ("ndcodegen", "fuse, SIMD,\nC ABI"), ("lib<m>.dylib", "mojo build"), ("mojo-nuts", "nuts-rs NUTS")]
    bw, bh, y = 118, 74, 60
    gapx = (w - 40 - len(boxes) * bw) / (len(boxes) - 1)
    b = []
    for i, (t, sub) in enumerate(boxes):
        x = 20 + i * (bw + gapx)
        stroke = CLAY if t == "ndcodegen" else INK
        b.append(f'<rect x="{x:.1f}" y="{y}" width="{bw}" height="{bh}" fill="{PAPER}" stroke="{stroke}" stroke-width="{1.8 if t == "ndcodegen" else 1.1}"/>')
        b.append(f'<text x="{x + bw / 2:.1f}" y="{y + 24}" font-size="12.5" fill="{INK}" text-anchor="middle" font-weight="700">{t}</text>')
        for k, line in enumerate(sub.split("\n")):
            b.append(f'<text x="{x + bw / 2:.1f}" y="{y + 44 + k * 14}" font-size="10.5" fill="{INK}" text-anchor="middle">{line}</text>')
        if i < len(boxes) - 1:
            x1, x2 = x + bw, x + bw + gapx
            b.append(f'<line x1="{x1:.1f}" y1="{y + bh / 2}" x2="{x2 - 6:.1f}" y2="{y + bh / 2}" stroke="{INK}" stroke-width="1.2"/>')
            b.append(f'<path d="M{x2 - 7:.1f},{y + bh / 2 - 4} L{x2:.1f},{y + bh / 2} L{x2 - 7:.1f},{y + bh / 2 + 4} Z" fill="{INK}"/>')
    # data path + draws path
    x3 = 20 + 2 * (bw + gapx)
    x5 = 20 + 4 * (bw + gapx)
    b.append(f'<path d="M{x3 + bw / 2:.1f},{y + bh} V{y + bh + 40} H{x5 + bw / 2:.1f} V{y + bh + 6}" fill="none" stroke="{DUSK}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    b.append(f'<path d="M{x5 + bw / 2 - 4:.1f},{y + bh + 12} L{x5 + bw / 2:.1f},{y + bh + 4} L{x5 + bw / 2 + 4:.1f},{y + bh + 12} Z" fill="{DUSK}"/>')
    b.append(f'<text x="{(x3 + x5) / 2 + bw / 2:.1f}" y="{y + bh + 34}" font-size="10.5" fill="{DUSK}" text-anchor="middle">packed model data (X, y, indices) as float64 / int64</text>')
    b.append(f'<text x="20" y="30" font-size="10.5" fill="{INK}">Python, compile time</text>')
    b.append(f'<line x1="{20 + 3 * (bw + gapx) - gapx / 2:.1f}" y1="18" x2="{20 + 3 * (bw + gapx) - gapx / 2:.1f}" y2="{h - 20}" stroke="{RULE}" stroke-width="1"/>')
    b.append(f'<text x="{20 + 3 * (bw + gapx) - gapx / 2 + 8:.1f}" y="30" font-size="10.5" fill="{INK}">native, per gradient: no Python</text>')
    b.append(f'<text x="20" y="{h - 8}" font-size="10.5" fill="{INK}">int32 &lt;m&gt;_logp_grad(const double* data, const double* x, double* grad, double* logp)</text>')
    return _svg(w, h, "\n".join(b), "Architecture")


# ------------------------------------------------------------------ main
def build():
    models = load_runs()
    return {
        "results_home.md": results_home(models),
        "bench_spec.md": bench_spec(models),
        "accuracy_spec.md": accuracy_spec(),
        "accuracy7_spec.md": accuracy7_spec(),
        "kernels7_spec.md": kernels7_spec(),
        "supported_ops.md": supported_ops(),
        "distributions.md": distributions(),
        "fig-speedup.svg": fig_speedup(models),
        "fig-kernel.svg": fig_kernel(models),
        "fig-arch.svg": fig_arch(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if committed assets differ")
    a = ap.parse_args()
    files = build()
    stale = []
    for name, text in files.items():
        p = OUT / name
        if a.check:
            if not p.exists() or p.read_text() != text:
                stale.append(name)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    if a.check and stale:
        raise SystemExit(f"stale generated docs (run `pixi run docs-assets`): {stale}")
    print("ok" if a.check else f"wrote {len(files)} files to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
