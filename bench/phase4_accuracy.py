"""Phase 4: accuracy of generated special functions vs scipy.special (float64).

Each op is compiled through the MOJO_ND backend (so the SIMD lanes and the scalar tail both
run) and evaluated on wide grids, including tails and points near poles and roots.
Reports max error in ulps (|a - b| / spacing(b)) and max relative error.

Reference: a 40-digit Decimal implementation (bench/hiprec_ref.py) for gammaln and psi (all
x, negative x via reflection) and erfc (continued fraction / Taylor series), because scipy.special.gammaln loses relative accuracy near its zeros x = 1, 2 (shown in
the "scipy vs hi-prec" column). scipy.special / numpy for everything else.
"""
import pathlib, sys
import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.special as sc

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from hiprec_ref import erfc_dec_any, lgamma_dec_any, psi_dec_any  # noqa: E402
from pytensor_mojo.ndlinker import MOJO_ND  # noqa: E402

PSI_ROOT = 1.4616321449683622


def grids():
    g = np.random.default_rng(0)
    lg = lambda lo, hi, n=4000: np.exp(g.uniform(np.log(lo), np.log(hi), n))  # noqa: E731
    return {
        "gammaln": [("tiny (1e-300..1e-3)", lg(1e-300, 1e-3)),
                    ("(1e-3, 50)", g.uniform(1e-3, 50, 4000)),
                    ("near 1 and 2 (zeros)", np.concatenate([1 + g.uniform(-1e-3, 1e-3, 2000), 2 + g.uniform(-1e-3, 1e-3, 2000)])),
                    ("large (50..1e300)", lg(50, 1e300)),
                    ("negative non-integer", -g.uniform(0, 30, 4000) - 0.5 * (g.random(4000) > 2))],
        "psi": [("tiny (1e-300..1e-3)", lg(1e-300, 1e-3)),
                ("(1e-3, 10)", g.uniform(1e-3, 10, 4000)),
                ("near root 1.4616", PSI_ROOT + g.uniform(-0.3, 0.3, 4000)),
                ("very near root (+-1e-8)", PSI_ROOT + g.uniform(-1e-8, 1e-8, 2000)),
                ("large (10..1e300)", lg(10, 1e300)),
                ("negative, >=1e-3 from poles", -(np.floor(g.uniform(0, 20, 4000)) + g.uniform(1e-3, 1 - 1e-3, 4000)))],
        "erf": [("(-6, 6)", g.uniform(-6, 6, 4000)), ("tiny", lg(1e-300, 1e-3) * np.sign(g.normal(size=4000)))],
        "erfc": [("(-6, 1)", g.uniform(-6, 1, 4000)), ("(1, 5)", g.uniform(1, 5, 4000)),
                 ("deep tail (5, 26.5)", g.uniform(5, 26.5, 4000))],
        "expm1": [("(-50, 50)", g.uniform(-50, 50, 4000)), ("tiny", lg(1e-300, 1e-5) * np.sign(g.normal(size=4000)))],
        "log1mexp": [("(-50, -1e-3)", -lg(1e-3, 50)), ("near 0 (-1e-3..-1e-300)", -lg(1e-300, 1e-3)),
                     ("around -log 2", -np.log(2) + g.uniform(-1e-3, 1e-3, 4000))],
        "pow": [("x in (1e-3, 100), y in (-10, 10)", None)],
        "log (std.math)": [("(1e-300, 1e300)", lg(1e-300, 1e300)), ("near 1", 1 + g.uniform(-1e-3, 1e-3, 4000))],
        "log1p (std.math)": [("(-0.999, 1e10)", np.concatenate([g.uniform(-0.999, 1, 2000), lg(1, 1e10, 2000)])),
                             ("tiny", lg(1e-300, 1e-5) * np.sign(g.normal(size=4000)))],
        "sqrt (std.math)": [("(1e-300, 1e300)", lg(1e-300, 1e300))],
    }


def ref(name, x, y=None):
    return {"gammaln": lambda: sc.gammaln(x), "psi": lambda: sc.psi(x), "erf": lambda: sc.erf(x),
            "erfc": lambda: sc.erfc(x), "expm1": lambda: np.expm1(x),
            "log1mexp": lambda: np.log(-np.expm1(x)) if False else _log1mexp_ref(x),
            "pow": lambda: np.power(x, y), "log (std.math)": lambda: np.log(x),
            "log1p (std.math)": lambda: np.log1p(x), "sqrt (std.math)": lambda: np.sqrt(x)}[name]()


def _log1mexp_ref(x):
    # extended-precision-free reference: same piecewise formula in numpy (correctly rounded pieces)
    with np.errstate(divide="ignore", invalid="ignore"):  # np.where evaluates both branches
        return np.where(x > -np.log(2), np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def sym(name, x, y=None):
    return {"gammaln": pt.gammaln, "psi": pt.psi, "erf": pt.erf, "erfc": pt.erfc, "expm1": pt.expm1,
            "log1mexp": pt.log1mexp, "log (std.math)": pt.log, "log1p (std.math)": pt.log1p,
            "sqrt (std.math)": pt.sqrt}[name](x) if name != "pow" else x ** y


def ulps(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b) / np.spacing(np.abs(b))
    d[both_inf | both_nan] = 0
    return d


def relerr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.abs(a - b) / np.abs(b)
    r[(a == b) | (np.isnan(a) & np.isnan(b))] = 0
    return r


def run():
    x = pt.dvector("x"); y = pt.dvector("y")
    fns = {n: pytensor.function([x], sym(n, x), mode=MOJO_ND) for n in grids() if n != "pow"}
    fns["pow"] = pytensor.function([x, y], x ** y, mode=MOJO_ND)
    rows = []
    g = np.random.default_rng(1)
    for name, sets in grids().items():
        for label, xs in sets:
            if name == "pow":
                xs = np.exp(g.uniform(np.log(1e-3), np.log(100), 4000)); ys = g.uniform(-10, 10, 4000)
                a, b = fns[name](xs, ys), ref(name, xs, ys)
            else:
                a = fns[name](xs)
                if name in ("gammaln", "psi", "erfc"):
                    xs = xs[:1000]; a = a[:1000]  # Decimal reference is slow
                    fref = {"gammaln": lgamma_dec_any, "psi": psi_dec_any, "erfc": erfc_dec_any}[name]
                    b = np.array([fref(float(v)) for v in xs])
                    sp = ref(name, xs)
                    scipy_rel = float(np.max(relerr(sp, b)))
                else:
                    b = ref(name, xs)
                    scipy_rel = None
                if name == "erfc":  # scipy flushes the subnormal tail (x > 26.55) to 0; compare normals
                    m = np.abs(b) >= np.finfo(float).tiny
                    xs, a, b = xs[m], a[m], b[m]
            u, r = ulps(a, b), relerr(a, b)
            rows.append((name, label, len(xs), float(np.max(u)), float(np.median(u)), float(np.max(r)),
                         scipy_rel if name != "pow" else None))
    return rows


def main():
    import json
    rows = run()
    keys = ["op", "range", "n", "max_ulp", "median_ulp", "max_rel", "scipy_rel"]
    (ROOT / "bench" / "phase4_accuracy.json").write_text(json.dumps([dict(zip(keys, r)) for r in rows], indent=1))
    print("| op | range | n | reference | max ulp | median ulp | max rel err | scipy vs hi-prec (max rel) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, label, n, mu, md, mr, sr in rows:
        refname = "40-digit Decimal" if sr is not None else ("numpy" if name in ("expm1", "pow", "log1mexp") or "std.math" in name else "scipy.special")
        srs = f"{sr:.2e}" if sr is not None else "-"
        print(f"| {name} | {label} | {n} | {refname} | {mu:.3g} | {md:.3g} | {mr:.2e} | {srs} |")


if __name__ == "__main__":
    main()
