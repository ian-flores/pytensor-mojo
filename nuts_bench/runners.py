"""Run eight-schools (and other models) with Mojo+nuts-rs and with nutpie (numba backend).

Both use nuts-rs 0.18.3 with identical settings (diag mass matrix, maxdepth 10,
target_accept 0.8); the only difference is the logp+grad implementation and the
storage/Python glue around the sampler.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "sampler" / "target" / "release" / "mojo-nuts"
CAPI = ROOT / "mojo_capi"


def run_mojo(lib, model, chains=4, tune=1000, draws=1000, seed=1, cores=None, data=None):
    """Return dict with draws (chains, draws, dim) unconstrained + stats from the Rust driver."""
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "out.json"
        cmd = [str(BIN), "--lib", str(lib), "--model", model, "--chains", str(chains),
               "--tune", str(tune), "--draws", str(draws), "--seed", str(seed),
               "--cores", str(cores or chains), "--out", str(out)]
        if data is not None:
            dp = pathlib.Path(d) / "data.f64"
            np.ascontiguousarray(data, dtype="<f8").tofile(dp)
            cmd += ["--data", str(dp)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        r = json.loads(out.read_text())
    dim = r["dim"]
    r["draws"] = np.stack([np.asarray(c["draws"]).reshape(c["n_draws"], dim) for c in r["chains"]])
    r["divergences"] = sum(c["divergences"] for c in r["chains"])
    r["n_steps_total"] = sum(c["n_steps_total"] for c in r["chains"])
    return r


def es8_constrained(u):
    """u: (chains, draws, 10) unconstrained -> dict of (chains, draws[, 8]) arrays."""
    mu, tau = u[..., 0], np.exp(u[..., 1])
    return {"mu": mu, "tau": tau, "theta": mu[..., None] + tau[..., None] * u[..., 2:]}


def compile_nutpie(model):
    import nutpie
    t = time.perf_counter()
    cm = nutpie.compile_pymc_model(model, backend="numba")
    return cm, time.perf_counter() - t


def run_nutpie(cm, chains=4, tune=1000, draws=1000, seed=1, cores=None, raw=False):
    """Sample with nutpie. raw=True skips building the xarray DataTree (sampler time only)."""
    import nutpie
    t = time.perf_counter()
    tr = nutpie.sample(cm, chains=chains, tune=tune, draws=draws, seed=seed, cores=cores or chains,
                       progress_bar=False, return_raw_trace=raw, store_unconstrained=False)
    sec = time.perf_counter() - t
    return tr, sec


def nutpie_posterior(tr):
    post = tr["posterior"]
    ss = tr["sample_stats"]
    out = {k: np.asarray(post[k].values) for k in post.data_vars}
    nsteps = ss["n_steps"].values
    div = int(np.asarray(ss["diverging"].values).sum())
    warm = tr["warmup_sample_stats"]["n_steps"].values if "warmup_sample_stats" in tr.children else 0
    return out, div, int(np.sum(nsteps) + np.sum(warm))


def summarize(post):
    """Per-scalar mean, sd, mcse(mean), ess_bulk, rhat. post: name -> (chains, draws[, k])."""
    import arviz_stats as azs
    rows = {}
    for name, a in post.items():
        a = np.asarray(a)
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        for j in range(flat.shape[2]):
            x = flat[:, :, j]
            key = name if flat.shape[2] == 1 else f"{name}[{j}]"
            rows[key] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)),
                             mcse=float(azs.mcse(x)), ess_bulk=float(azs.ess(x, method="bulk")),
                             rhat=float(azs.rhat(x)))
    return rows


def es8_from_nutpie(post):
    mu, tau, tt = post["mu"], post["tau"], post["theta_tilde"]
    return {"mu": mu, "tau": tau, "theta": mu[..., None] + tau[..., None] * tt}
