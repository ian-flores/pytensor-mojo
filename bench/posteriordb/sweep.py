"""posteriordb coverage sweep for the ndcodegen backend.

For each hand-ported posterior in ``models.MODELS``:

1. build the PyMC model and list the ops ``ndcodegen`` cannot emit (``missing_ops``);
2. if nothing is missing, compile it through ``ndcodegen`` and check logp / dlogp against
   PyMC at rtol 1e-10 on 20 points;
3. sample it with ``pytensor_mojo.sample`` and with nutpie (4 chains, 1000 tune + 1000 draws,
   same seed), time both, and compare each with the posteriordb reference draws: every
   parameter mean must lie within ``Z_TOL`` combined MCSEs of the reference mean and the max
   R-hat must be below ``RHAT_TOL``. Bulk ESS and divergences are reported.

Outputs: ``bench/posteriordb_results.json`` and the generated tables
``docs/_generated/posteriordb_coverage.md`` and ``docs/_generated/posteriordb_ranking.md``.

Usage::

    python bench/posteriordb/sweep.py --fetch          # shallow sparse clone into .cache/posteriordb
    python bench/posteriordb/sweep.py                  # full sweep (about 25 minutes; diamonds alone is about 10)
    python bench/posteriordb/sweep.py --no-sample      # coverage and PyMC checks only
    python bench/posteriordb/sweep.py --tables-only    # regenerate tables from the JSON
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import traceback
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
import models as M  # noqa: E402

OUT_JSON = ROOT / "bench" / "posteriordb_results.json"
GEN = ROOT / "docs" / "_generated"
Z_TOL = 4.5          # |mean - ref mean| / sqrt(mcse^2 + mcse_ref^2), per parameter
RHAT_TOL = 1.05      # and every chain must have mixed: max R-hat below this
RTOL = 1e-10
REPO = "https://github.com/stan-dev/posteriordb.git"


# ------------------------------------------------------------------ fetching
def fetch(dest: pathlib.Path):
    """Shallow, sparse, blob-less clone of posteriordb with only the files the sweep reads."""
    dest = dest.resolve()
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(dest)], check=True)
    pats = ["/posterior_database/posteriors/", "/posterior_database/models/stan/",
            "/posterior_database/models/info/", "/posterior_database/data/info/"]
    for name in M.MODELS:
        info = json.loads((dest / "posterior_database" / "posteriors" / f"{name}.json").read_text()) \
            if (dest / "posterior_database" / "posteriors" / f"{name}.json").exists() else None
        if info is None:
            subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *pats], check=True)
            info = json.loads((dest / "posterior_database" / "posteriors" / f"{name}.json").read_text())
        ref = info["reference_posterior_name"]
        pats += [f"/posterior_database/data/data/{info['data_name']}.json.zip",
                 f"/posterior_database/reference_posteriors/draws/draws/{ref}.json.zip",
                 f"/posterior_database/reference_posteriors/draws/info/{ref}.info.json"]
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", "--stdin"],
                   input="\n".join(pats) + "\n", text=True, check=True)


# ------------------------------------------------------------------ pieces
def symbol(name: str) -> str:
    return "pdb_" + re.sub(r"[^a-z0-9]", "", name.split("-", 1)[1].lower())[:24]


def optimized_graph(model):
    import pytensor
    from pytensor.compile.mode import Mode
    from pytensor_mojo.ndlinker import QUERY, _Capture, flat_logp_grad
    q, lp, g, dim = flat_logp_grad(model)
    _Capture.sink.clear()
    pytensor.function([q], [lp, g], mode=Mode(_Capture(), QUERY))
    return _Capture.sink[-1], dim


def check_vs_pymc(model, so, sym, data, dim, n_points=20, seed=42):
    """Generated logp / dlogp vs PyMC's compile_logp / compile_dlogp on ``n_points`` points.

    Points are drawn around the initial point in the unconstrained space at four scales.
    Returns a dict with the worst relative errors and whether every point matched at RTOL.
    """
    from pymc.blocking import DictToArrayBijection, RaveledVars
    from pytensor_mojo.ndlinker import CapiModel
    cm = CapiModel(so, sym, data)
    lpf, dlf = model.compile_logp(), model.compile_dlogp()
    ip = model.initial_point()
    rv = DictToArrayBijection.map({v.name: ip[v.name] for v in model.value_vars})
    rng = np.random.default_rng(seed)
    scales = np.repeat([0.1, 0.3, 0.6, 1.0], n_points // 4)
    worst_lp = worst_g = 0.0
    ok = True
    n_finite = 0
    for s in scales:
        x = rv.data + rng.normal(scale=s, size=dim)
        pt = DictToArrayBijection.rmap(RaveledVars(x, rv.point_map_info))
        rc, lp, g = cm(x)
        lp_ref, g_ref = float(lpf(pt)), np.asarray(dlf(pt))
        if not np.isfinite(lp_ref):
            ok &= (rc != 0) or not np.isfinite(lp)
            continue
        n_finite += 1
        if rc != 0:
            ok = False
            continue
        e_lp = abs(lp - lp_ref) / max(abs(lp_ref), 1e-300)
        scale = max(1.0, float(np.max(np.abs(g_ref))))
        # relative error per coordinate, floored at 1e-3 of the largest gradient entry
        e_g = float(np.max(np.abs(g - g_ref) / np.maximum(np.abs(g_ref), 1e-3 * scale)))
        # same criterion as tests/test_phase4_models.py
        g_ok = np.allclose(g, g_ref, rtol=RTOL, atol=1e-9 * scale)
        worst_lp, worst_g = max(worst_lp, e_lp), max(worst_g, e_g if np.isfinite(e_g) else 0.0)
        ok &= bool(e_lp <= RTOL and g_ok)
    return dict(match=bool(ok and n_finite > 0), n_points=len(scales), n_finite=n_finite,
                max_rel_err_logp=worst_lp, max_rel_err_grad=worst_g)


def _summ(post: dict) -> dict:
    """{stan name: (chains, draws)} -> per-parameter mean, mcse, ess_bulk, rhat."""
    import arviz_stats as azs
    out = {}
    for k, x in post.items():
        x = np.asarray(x, dtype=float)
        out[k] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), mcse=float(azs.mcse(x)),
                      ess_bulk=float(azs.ess(x, method="bulk")), rhat=float(azs.rhat(x)))
    return out


def compare_reference(post: dict, ref_summ: dict) -> dict:
    s = _summ({k: v for k, v in post.items() if k in ref_summ})
    zs = {}
    for k, r in ref_summ.items():
        if k not in s:
            continue
        den = np.hypot(s[k]["mcse"], r["mcse"])
        zs[k] = float(abs(s[k]["mean"] - r["mean"]) / den) if den > 0 else float("inf")
    worst = max(zs, key=zs.get) if zs else None
    st = dict(n_params=len(zs), missing_params=sorted(set(ref_summ) - set(s)),
              max_z=zs[worst] if worst else None, worst_param=worst,
              max_rhat=max(v["rhat"] for v in s.values()), min_ess_bulk=min(v["ess_bulk"] for v in s.values()))
    return set_match(st)


def set_match(st):
    """Reference match: all reference parameters present, every mean within Z_TOL MCSEs, R-hat < RHAT_TOL."""
    st["means_match"] = bool(st["n_params"]) and not st["missing_params"] and st["max_z"] < Z_TOL
    st["match"] = st["means_match"] and st["max_rhat"] < RHAT_TOL
    return st


def _post_from_tree(tree) -> dict:
    p = tree["posterior"]
    return M.stan_names({k: np.asarray(p[k].values) for k in p.data_vars})


def sample_mojo(model, sym, chains, tune, draws, seed, reps):
    """pytensor_mojo.sample once for the posterior, plus ``reps - 1`` timing-only runs."""
    from pytensor_mojo.sample import sample
    t = time.perf_counter()
    dt = sample(model, chains=chains, tune=tune, draws=draws, seed=seed, name=sym)
    wall = time.perf_counter() - t
    secs = [float(dt.attrs["sampling_time"])]
    for k in range(1, reps):
        raw = sample(model, chains=chains, tune=tune, draws=draws, seed=seed + k, name=sym, return_raw=True)
        secs.append(float(raw["sample_seconds"]))
    div = int(np.asarray(dt["sample_stats"]["diverging"].values).sum())
    return _post_from_tree(dt), dict(sampler_s=float(np.median(secs)), sampler_s_runs=secs, wall_s=wall,
                                     divergences=div)


def sample_nutpie(model, chains, tune, draws, seed, reps):
    """nutpie (numba) once for the posterior, then ``reps`` sampler-only (raw trace) timing runs."""
    import nutpie
    t = time.perf_counter()
    cm = nutpie.compile_pymc_model(model, backend="numba")
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    tr = nutpie.sample(cm, chains=chains, tune=tune, draws=draws, seed=seed, cores=chains, progress_bar=False)
    full = time.perf_counter() - t
    secs = []
    for k in range(reps):
        t = time.perf_counter()
        nutpie.sample(cm, chains=chains, tune=tune, draws=draws, seed=seed + k, cores=chains, progress_bar=False,
                      return_raw_trace=True)
        secs.append(time.perf_counter() - t)
    div = int(np.asarray(tr["sample_stats"]["diverging"].values).sum())
    return _post_from_tree(tr), dict(sampler_s=float(np.median(secs)), sampler_s_runs=secs, wall_s=full,
                                     compile_s=compile_s, divergences=div)


# ------------------------------------------------------------------ one model
def run_one(name, do_sample=True, chains=4, tune=1000, draws=1000, seed=1, reps=3):
    from pytensor_mojo.ndcodegen import missing_ops
    from pytensor_mojo.ndlinker import compile_model_capi
    port = M.MODELS[name]
    res = dict(posterior=name, family=port.family, notes=port.notes, symbol=symbol(name))
    model = port.build(M.load_data(name))
    fg, dim = optimized_graph(model)
    mt, ms = missing_ops(fg)
    # baseline: the same census with the sweep's extension ops (ndcodegen_ext) switched off
    from pytensor_mojo import ndcodegen_ext
    ndcodegen_ext.DISABLED, was = True, ndcodegen_ext.DISABLED
    try:
        bt, bs = missing_ops(fg)
    finally:
        ndcodegen_ext.DISABLED = was
    res.update(dim=dim, missing_tensor=mt, missing_scalar=ms, baseline_missing=bt + bs, compiles=False,
               matches_pymc=False, matches_reference=None)
    if mt or ms:
        return res
    sym = res["symbol"]
    try:
        t = time.perf_counter()
        so, data, dim2, _ = compile_model_capi(model, sym)
        res["build_s"] = time.perf_counter() - t
    except Exception as e:     # codegen gap not visible to missing_ops
        res["compile_error"] = f"{type(e).__name__}: {e}"[:400]
        return res
    res["compiles"] = True
    res["pymc_check"] = check_vs_pymc(model, so, sym, data, dim2)
    res["matches_pymc"] = res["pymc_check"]["match"]
    if not (do_sample and res["matches_pymc"]):
        return res
    # per-evaluation logp + gradient cost (explains the speedups): Mojo kernel via the Rust
    # driver's --bench-evals, numba via PyTensor's jit_fn from Python (Phase 2 methodology)
    sys.path.insert(0, str(ROOT / "bench"))
    from phase2_nuts import mojo_kernel_us, numba_kernel_us
    res["kernel_us"] = dict(mojo=mojo_kernel_us(so, sym, data), numba=numba_kernel_us(model))
    ref = M.load_reference(name)
    ref_summ = _summ(ref)
    try:
        post, st = sample_mojo(model, sym, chains, tune, draws, seed, reps)
        st.update(compare_reference(post, ref_summ))
        res["mojo"] = st
    except Exception as e:
        res["mojo"] = dict(error=f"{type(e).__name__}: {e}"[:400])
    try:
        post, st = sample_nutpie(model, chains, tune, draws, seed, reps)
        st.update(compare_reference(post, ref_summ))
        res["nutpie"] = st
    except Exception as e:
        res["nutpie"] = dict(error=f"{type(e).__name__}: {e}"[:400])
    if "sampler_s" in res.get("mojo", {}) and "sampler_s" in res.get("nutpie", {}):
        res["speedup"] = res["nutpie"]["sampler_s"] / res["mojo"]["sampler_s"]
        res["matches_reference"] = bool(res["mojo"]["match"])
    return res


# ------------------------------------------------------------------ tables
def status_of(r):
    if r.get("matches_reference"):
        return "matches-reference"
    if r.get("mojo", {}).get("means_match"):
        return "matches-PyMC (means match reference, R-hat >= 1.05)"
    if r.get("matches_pymc"):
        return "matches-PyMC"
    if r.get("compiles"):
        return "compiles"
    return "unsupported"


def coverage_table(results):
    out = ["| posterior | family | dim | status | missing ops | max z vs ref (Mojo / nutpie) | max R-hat | min ESS bulk | div. (Mojo / nutpie) | kernel us, Mojo / numba | Mojo s | nutpie s | speedup |",
           "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        miss = ", ".join(f"`{o}`" for o in r["missing_tensor"] + r["missing_scalar"]) or "-"
        if r.get("compile_error"):
            miss = "codegen error"
        mj, nu = r.get("mojo", {}), r.get("nutpie", {})
        if "max_z" in mj and "max_z" in nu:
            z = f"{mj['max_z']:.2f} / {nu['max_z']:.2f}"
            rh, ess, dv = f"{mj['max_rhat']:.3f}", f"{mj['min_ess_bulk']:.0f}", f"{mj['divergences']} / {nu['divergences']}"
            ts, tn, sp = f"{mj['sampler_s']:.3f}", f"{nu['sampler_s']:.3f}", f"{r['speedup']:.2f}x"
        else:
            z = rh = ess = dv = ts = tn = sp = "-"
        ku = r.get("kernel_us")
        k = f"{ku['mojo']:.2f} / {ku['numba']:.2f}" if ku else "-"
        out.append(f"| {r['posterior']} | {r['family']} | {r['dim']} | {status_of(r)} | {miss} | {z} | {rh} | {ess} | {dv} | {k} | {ts} | {tn} | {sp} |")
    return "\n".join(out) + "\n"


def _rank(results, key):
    miss = {r["posterior"]: set(key(r)) for r in results}
    blocked = {n: ops for n, ops in miss.items() if ops}
    need, alone = Counter(), Counter()
    for ops in blocked.values():
        need.update(ops)
        if len(ops) == 1:
            alone.update(ops)
    rows = sorted(need, key=lambda o: (-alone[o], -need[o], o))
    out = ["| op | models needing it | models it unblocks alone | posteriors |", "|---|---:|---:|---|"]
    for o in rows:
        names = [n for n, ops in blocked.items() if o in ops]
        out.append(f"| `{o}` | {need[o]} | {alone[o]} | {', '.join(names)} |")
    return "\n".join(out) + "\n"


def ranking(results):
    """Missing ops ranked by how many models each unblocks alone, then by how many need it.

    Two tables: the census before the sweep's new ops (``baseline_missing``) and what is
    still missing now.
    """
    base = _rank(results, lambda r: r.get("baseline_missing", []))
    now = _rank(results, lambda r: r["missing_tensor"] + r["missing_scalar"])
    return ("**Before the sweep's ops** (`ndcodegen_ext` disabled):\n\n" + base
            + "\n**Still missing:**\n\n" + now)


def summary(results):
    n = len(results)
    c = sum(r["compiles"] for r in results)
    p = sum(bool(r["matches_pymc"]) for r in results)
    f = sum(bool(r.get("matches_reference")) for r in results)
    sp = [r["speedup"] for r in results if "speedup" in r]
    s = f"{n} posteriors: {c} compile, {p} match PyMC at rtol 1e-10, {f} match the reference posterior"
    if sp:
        s += f"; speedup vs nutpie median {np.median(sp):.2f}x (min {min(sp):.2f}x, max {max(sp):.2f}x, n={len(sp)})"
    return s


def write_tables(results):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "posteriordb_coverage.md").write_text(coverage_table(results))
    (GEN / "posteriordb_ranking.md").write_text(ranking(results))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="clone posteriordb into .cache/posteriordb and exit")
    ap.add_argument("--models", default="", help="comma-separated posterior names (default: all)")
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3, help="timing runs per sampler (median reported)")
    ap.add_argument("--out", default=str(OUT_JSON))
    a = ap.parse_args()
    if a.fetch:
        fetch(ROOT / ".cache" / "posteriordb")
        return
    if a.tables_only:
        doc = json.loads(pathlib.Path(a.out).read_text())
        results = doc["results"]
        for r in results:      # re-apply the current match rule to the stored statistics
            for side in ("mojo", "nutpie"):
                if "max_z" in r.get(side, {}):
                    set_match(r[side])
            if "max_z" in r.get("mojo", {}):
                r["matches_reference"] = r["mojo"]["match"]
        doc["meta"]["rhat_tol"] = RHAT_TOL
        pathlib.Path(a.out).write_text(json.dumps(doc, indent=1))
        write_tables(results)
        print(summary(results))
        return
    names = [n for n in (a.models.split(",") if a.models else M.MODELS)]
    missing = [n for n in names if not M.available(n)]
    if missing:
        raise SystemExit(f"posteriordb data not found for {missing}; run with --fetch first")
    results = []
    for n in names:
        t = time.perf_counter()
        try:
            r = run_one(n, do_sample=not a.no_sample, seed=a.seed, reps=a.reps)
        except Exception as e:
            traceback.print_exc()
            r = dict(posterior=n, family=M.MODELS[n].family, dim=0, missing_tensor=[], missing_scalar=[],
                     compiles=False, matches_pymc=False, error=f"{type(e).__name__}: {e}"[:400])
        r["sweep_s"] = time.perf_counter() - t
        results.append(r)
        print(f"{n}: {status_of(r)} {r.get('missing_tensor', [])} {r.get('missing_scalar', [])} "
              f"{r.get('compile_error', '')} speedup={r.get('speedup', '-')} ({r['sweep_s']:.1f}s)", flush=True)
    meta = dict(chains=4, tune=1000, draws=1000, seed=a.seed, z_tol=Z_TOL, rhat_tol=RHAT_TOL, rtol=RTOL, n_points=20,
                reps=a.reps, timing="sampler only, median of reps: mojo-nuts sample_seconds vs nutpie.sample(return_raw_trace=True) wall time",
                sampled=not a.no_sample)
    pathlib.Path(a.out).write_text(json.dumps(dict(meta=meta, results=results), indent=1))
    write_tables(results)
    print(summary(results))


if __name__ == "__main__":
    main()
