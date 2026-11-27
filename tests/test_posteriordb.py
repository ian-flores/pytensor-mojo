"""posteriordb ports (bench/posteriordb/models.py) through ndcodegen vs PyMC.

By default a fast subset of small models is compiled and its logp / dlogp checked against
PyMC at rtol 1e-10 on 20 points (no sampling). Set ``PYTENSOR_MOJO_POSTERIORDB_FULL=1`` to
run the whole sweep, including sampling and the comparison with the reference draws.

The posteriordb files are read from ``.cache/posteriordb`` (or ``$POSTERIORDB_PATH``); fetch
them with ``python bench/posteriordb/sweep.py --fetch``. Tests skip when they are absent, so
no network access is needed at test time.
"""
import importlib.util
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDB = ROOT / "bench" / "posteriordb"
sys.path[:0] = [str(ROOT), str(PDB)]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("models", PDB / "models.py")
sweep = _load("pdb_sweep", PDB / "sweep.py")

# small models that exercise the sweep's new ops and the broadcast fix
FAST = [
    "eight_schools-eight_schools_noncentered",   # baseline: already supported
    "wells_data-wells_dist",                     # IncSubtensor (gradient of beta[i])
    "low_dim_gauss_mix-low_dim_gauss_mix",       # ordered transform: CumOp, stepped Subtensor, Second
    "dogs-dogs",                                 # All over a 2-D bool array
    "nes_logit_data-nes_logit_model",            # length-1 view q[1:] in a fused loop
]
FULL = os.environ.get("PYTENSOR_MOJO_POSTERIORDB_FULL", "0") == "1"


def _need(name):
    if not M.available(name):
        pytest.skip(f"posteriordb data for {name} not cached (python bench/posteriordb/sweep.py --fetch)")


@pytest.mark.parametrize("name", FAST)
def test_fast_subset_matches_pymc(name):
    _need(name)
    r = sweep.run_one(name, do_sample=False)
    assert r["missing_tensor"] == [] and r["missing_scalar"] == [], r
    assert r["compiles"], r.get("compile_error")
    chk = r["pymc_check"]
    assert chk["n_points"] == 20 and chk["n_finite"] == 20, chk
    assert chk["match"], chk


@pytest.mark.skipif(not FULL, reason="set PYTENSOR_MOJO_POSTERIORDB_FULL=1 for the full sweep")
@pytest.mark.parametrize("name", list(M.MODELS))
def test_full_sweep(name):
    _need(name)
    r = sweep.run_one(name, do_sample=True)
    if r["missing_tensor"] or r["missing_scalar"]:
        pytest.skip(f"unsupported ops: {r['missing_tensor'] + r['missing_scalar']}")
    assert r["matches_pymc"], r.get("pymc_check")
    assert r["mojo"]["match"], r["mojo"]
