"""Phase 7: accuracy of the vectorized exp, log, log1p, expm1, lgamma (pytensor_mojo.mojo_math).

Each function is compiled through the MOJO_ND backend (SIMD lanes + scalar tail), evaluated
on N points (default 10M per function) drawn from wide random and edge-case sets, and compared
in ulps against a 40-digit Decimal reference (bench/hiprec_ref.py), using all CPU cores.
ulp error = |result - ref| / ulp(ref), with ulp(ref) = spacing of the double nearest ref.

Usage: python bench/phase7_accuracy.py [--n 10000000] [--funcs exp,log,log1p,expm1,gammaln]
                                       [--out bench/phase7_accuracy.json]
"""
from __future__ import annotations

import argparse, json, math, os, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
import hiprec_ref as hp  # noqa: E402

LN2 = math.log(2.0)
REF = {"exp": hp.exp_D, "log": hp.log_D, "log1p": hp.log1p_D, "expm1": hp.expm1_D, "gammaln": hp.lgamma_D}


def _logu(g, lo, hi, n):
    return np.exp(g.uniform(np.log(lo), np.log(hi), n))


def point_sets(name, n, seed=0):
    """List of (label, x) with sizes summing to n."""
    g = np.random.default_rng(seed)
    q = lambda f: max(1, int(n * f))  # noqa: E731
    sgn = lambda k: np.where(g.random(k) < 0.5, -1.0, 1.0)  # noqa: E731
    if name == "exp":
        k = np.round(g.uniform(-1070, 1020, q(0.1)))
        return [("uniform [-745, 709.78]", g.uniform(-745.13, 709.78, q(0.4))),
                ("uniform [-1, 1]", g.uniform(-1, 1, q(0.2))),
                ("tiny |x| (1e-300..1e-3), both signs", _logu(g, 1e-300, 1e-3, q(0.2)) * sgn(q(0.2))),
                ("reduction boundaries (k +- 1/2) ln2", (k + 0.5 * sgn(q(0.1))) * LN2 + g.uniform(-1e-9, 1e-9, q(0.1))),
                ("near overflow / underflow", np.concatenate([709.78 - _logu(g, 1e-12, 5, q(0.05)),
                                                              -745.13 + _logu(g, 1e-12, 30, q(0.05))]))]
    if name == "log":
        return [("log-uniform [5e-324, 1.8e308] (incl. subnormal)", _logu(g, 5e-324, 1.7e308, q(0.5))),
                ("near 1 (1 +- 1e-3, log-spaced offsets)", 1 + _logu(g, 1e-16, 1e-3, q(0.2)) * sgn(q(0.2))),
                ("near sqrt(2)/2 and sqrt(2)", np.concatenate([math.sqrt(2) * (1 + g.uniform(-1e-6, 1e-6, q(0.05))),
                                                               math.sqrt(0.5) * (1 + g.uniform(-1e-6, 1e-6, q(0.05)))])),
                ("uniform (0, 10)", g.uniform(1e-12, 10, q(0.2)))]
    if name == "log1p":
        return [("uniform (-0.999, 1)", g.uniform(-0.999, 1, q(0.3))),
                ("log-uniform (1, 1e300)", _logu(g, 1, 1e300, q(0.2))),
                ("near -1 (-1 + 1e-16..1)", -1 + _logu(g, 1.2e-16, 1, q(0.1))),
                ("tiny |x| (1e-300..1e-3), both signs", _logu(g, 1e-300, 1e-3, q(0.3)) * sgn(q(0.3))),
                ("near sqrt(2) - 1 and sqrt(2)/2 - 1", np.concatenate([(math.sqrt(2) - 1) + g.uniform(-1e-6, 1e-6, q(0.05)),
                                                                     (math.sqrt(0.5) - 1) + g.uniform(-1e-6, 1e-6, q(0.05))]))]
    if name == "expm1":
        return [("uniform [-40, 709.78]", g.uniform(-40, 709.78, q(0.3))),
                ("uniform [-1, 1]", g.uniform(-1, 1, q(0.3))),
                ("tiny |x| (1e-300..1e-3), both signs", _logu(g, 1e-300, 1e-3, q(0.2)) * sgn(q(0.2))),
                ("branch boundaries +-0.5 ln2, +-1.5 ln2, k = 20, 56", np.concatenate(
                    [c * LN2 + g.uniform(-1e-6, 1e-6, q(0.025)) for c in (0.5, -0.5, 1.5, -1.5)]
                    + [k * LN2 + g.uniform(-0.5, 0.5, q(0.05)) for k in (20, 56)]))]
    if name == "gammaln":
        return [("log-uniform [1e-300, 1e300]", _logu(g, 1e-300, 1e300, q(0.3))),
                ("uniform (0, 15)", g.uniform(1e-9, 15, q(0.5))),
                ("near zeros 1, 2 (log-spaced offsets 1e-16..1e-2)", np.concatenate(
                    [c + _logu(g, 1e-16, 1e-2, q(0.05)) * sgn(q(0.05)) for c in (1.0, 2.0)])),
                ("piece boundaries 0.75, 3, 12", np.concatenate(
                    [c + g.uniform(-1e-3, 1e-3, q(0.0334)) for c in (0.75, 3.0, 12.0)]))]
    raise KeyError(name)


def _ulps_chunk(args):
    name, xs, ys = args
    f = REF[name]
    out = np.empty(len(xs))
    for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist())):
        if name == "gammaln" and x in (1.0, 2.0):   # exact zeros; the series gives ~1e-41
            out[i] = 0.0 if y == 0.0 else math.inf
            continue
        r = f(x)
        if r.is_infinite() or r == 0:
            rf = float(r)
            out[i] = 0.0 if (y == rf) else math.inf
            continue
        rf = float(r)
        if math.isinf(rf):            # reference overflows double: result must be inf
            out[i] = 0.0 if math.isinf(y) and (y > 0) == (rf > 0) else math.inf
            continue
        if math.isnan(y) or math.isinf(y):
            out[i] = math.inf
            continue
        sp = math.ulp(abs(rf))
        out[i] = float(abs(Decimal(y) - r) / Decimal(sp))
    return out


def compiled(name):
    import pytensor, pytensor.tensor as pt
    from pytensor_mojo.ndlinker import MOJO_ND
    x = pt.dvector("x")
    op = {"exp": pt.exp, "log": pt.log, "log1p": pt.log1p, "expm1": pt.expm1, "gammaln": pt.gammaln}[name]
    return pytensor.function([x], op(x), mode=MOJO_ND)


def measure(name, n, workers=None, chunk=20000):
    f = compiled(name)
    rows = []
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        for label, xs in point_sets(name, n):
            xs = np.ascontiguousarray(xs, dtype=np.float64)
            ys = f(xs)
            parts = [(name, xs[i:i + chunk], ys[i:i + chunk]) for i in range(0, len(xs), chunk)]
            u = np.concatenate(list(ex.map(_ulps_chunk, parts)))
            j = int(np.argmax(u))
            rows.append(dict(func=name, set=label, n=len(xs), max_ulp=float(u[j]), mean_ulp=float(u.mean()),
                             frac_le_half=float(np.mean(u <= 0.5)), frac_le_1=float(np.mean(u <= 1.0)),
                             worst_x=float(xs[j])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--funcs", default="exp,log,log1p,expm1,gammaln")
    ap.add_argument("--out", default=str(ROOT / "bench" / "phase7_accuracy.json"))
    a = ap.parse_args()
    allrows = []
    for name in a.funcs.split(","):
        t = time.time()
        rows = measure(name, a.n)
        allrows += rows
        print(f"{name}: {sum(r['n'] for r in rows):,} points in {time.time() - t:.0f}s, "
              f"max {max(r['max_ulp'] for r in rows):.3f} ulp", flush=True)
    pathlib.Path(a.out).write_text(json.dumps(allrows, indent=1))
    print("| function | set | n | max ulp | mean ulp | <= 0.5 ulp | worst x |")
    print("|---|---|---|---|---|---|---|")
    for r in allrows:
        print(f"| {r['func']} | {r['set']} | {r['n']:,} | {r['max_ulp']:.3f} | {r['mean_ulp']:.3f} | "
              f"{100 * r['frac_le_half']:.2f}% | {r['worst_x']:.17g} |")


if __name__ == "__main__":
    main()
