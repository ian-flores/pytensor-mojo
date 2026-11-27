"""Sample a PyMC model end to end: compile to Mojo, run mojo-nuts, return an ArviZ DataTree.

```python
from pytensor_mojo.sample import sample
idata = sample(model, chains=4, tune=1000, draws=1000, seed=1)
```
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .ndlinker import compile_model_capi

__all__ = ["sample", "find_sampler", "raw_to_datatree"]

_REPO = pathlib.Path(__file__).resolve().parents[1]


def find_sampler() -> pathlib.Path:
    """Locate the mojo-nuts binary.

    Order: the ``PYTENSOR_MOJO_SAMPLER`` environment variable, ``mojo-nuts`` on ``PATH``,
    then ``sampler/target/release/mojo-nuts`` in this repository (``pixi run build-sampler``).
    """
    env = os.environ.get("PYTENSOR_MOJO_SAMPLER")
    if env:
        return pathlib.Path(env)
    on_path = shutil.which("mojo-nuts")
    if on_path:
        return pathlib.Path(on_path)
    local = _REPO / "sampler" / "target" / "release" / "mojo-nuts"
    if local.exists():
        return local
    raise FileNotFoundError("mojo-nuts not found: run `pixi run build-sampler` or set PYTENSOR_MOJO_SAMPLER")


def _symbol(model) -> str:
    base = re.sub(r"[^a-z0-9]", "", (model.name or "model").lower()) or "model"
    return f"pm_{base}"


def sample(model, *, chains: int = 4, tune: int = 1000, draws: int = 1000, seed: int = 1,
           cores: int | None = None, name: str | None = None, return_raw: bool = False):
    """Compile ``model`` with ``ndcodegen`` and sample it with ``mojo-nuts``.

    Parameters
    ----------
    model : pymc.Model
        Model whose free variables all have static shapes.
    chains : int
        Number of chains.
    tune : int
        Warmup draws per chain.
    draws : int
        Post-warmup draws per chain.
    seed : int
        Base random seed.
    cores : int, optional
        Worker threads; defaults to ``chains``. Other NUTS settings follow nutpie's defaults
        (diagonal mass matrix, maxdepth 10, target_accept 0.8).
    name : str, optional
        Symbol prefix for the generated library (default derived from the model name).
    return_raw : bool
        Return the decoded JSON result of ``mojo-nuts`` (unconstrained draws) instead.

    Returns
    -------
    xarray.DataTree
        ArviZ DataTree with ``posterior`` (constrained variables and deterministics) and
        ``sample_stats`` (``diverging``, ``n_steps``) groups, plus the sampling time in the
        ``sampling_time`` attribute.
    """
    sym = name or _symbol(model)
    so, data, dim, _ = compile_model_capi(model, sym)
    with tempfile.TemporaryDirectory() as d:
        dp = pathlib.Path(d) / "data.f64"
        out = pathlib.Path(d) / "out.json"
        np.ascontiguousarray(data, dtype="<f8").tofile(dp)
        cmd = [str(find_sampler()), "--lib", str(so), "--model", sym, "--data", str(dp),
               "--chains", str(chains), "--tune", str(tune), "--draws", str(draws),
               "--seed", str(seed), "--cores", str(cores or chains), "--out", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        raw = json.loads(out.read_text())
    if return_raw:
        return raw
    return raw_to_datatree(model, raw)


def raw_to_datatree(model, raw):
    """Convert a mojo-nuts JSON result (unconstrained draws) to an ArviZ DataTree.

    Unconstrained draws are mapped back to the constrained space, and deterministics are
    evaluated, with a compiled PyMC point function (``model.unobserved_value_vars``).
    """
    import arviz_base
    import pytensor

    dim = raw["dim"]
    U = np.stack([np.asarray(c["draws"]).reshape(c["n_draws"], dim) for c in raw["chains"]])
    vv = model.value_vars
    # unobserved_value_vars holds both the transformed value vars (e.g. "sigma_log__") and the
    # back-transformed variables and deterministics; keep the latter, named by output.
    outs = [o for o in model.unobserved_value_vars if not o.name.endswith("__")]
    names = [o.name for o in outs]
    f = pytensor.function(vv, outs, on_unused_input="ignore")
    sizes = [int(np.prod(v.type.shape)) if v.ndim else 1 for v in vv]
    shapes = [tuple(v.type.shape) for v in vv]
    nc, nd = U.shape[:2]
    acc = {n: [] for n in names}
    for c in range(nc):
        for t in range(nd):
            x, parts, i = U[c, t], [], 0
            for n_, shp in zip(sizes, shapes):
                parts.append(x[i:i + n_].reshape(shp))
                i += n_
            for n, val in zip(names, f(*parts)):
                acc[n].append(np.asarray(val))
    post = {n: np.stack(v).reshape((nc, nd) + np.asarray(v[0]).shape) for n, v in acc.items()}
    stats = {"diverging": np.array([c["diverging"] for c in raw["chains"]], dtype=bool),
             "n_steps": np.array([c["n_steps"] for c in raw["chains"]], dtype=np.int64)}
    dt = arviz_base.from_dict({"posterior": post, "sample_stats": stats})
    dt.attrs["sampling_time"] = raw["sample_seconds"]
    dt.attrs["sampler"] = "mojo-nuts (nuts-rs 0.18.3)"
    return dt
