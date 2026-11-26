"""Phase 2: NUTS via nuts-rs + Mojo C ABI vs nutpie (numba), fixed seeds, MC-error tolerances."""
import pathlib, subprocess, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from nuts_bench.runners import (BIN, CAPI, compile_nutpie, es8_constrained, es8_from_nutpie,  # noqa: E402
                                nutpie_posterior, run_mojo, run_nutpie, summarize)
from pytensor_mojo import eight_schools_ref as es8, hlr_ref  # noqa: E402

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="build sampler: cargo build --release")


def compare(sm, sn, mean_z=4.0, sd_rtol=0.2):
    for k in sm:
        a, b = sm[k], sn[k]
        se = np.hypot(a["mcse"], b["mcse"])
        assert abs(a["mean"] - b["mean"]) < mean_z * se, (k, a, b)
        assert abs(a["sd"] - b["sd"]) < sd_rtol * b["sd"], (k, a, b)
        assert a["rhat"] < 1.01, (k, a)
        assert a["ess_bulk"] > 400, (k, a)


def test_settings_match_nutpie_defaults():
    """The Rust driver's nuts-rs settings equal nutpie's Diag settings (except chains/tune/draws/seed)."""
    import json
    from nutpie import _lib
    nut = _lib.PyNutsSettings.Diag(1).as_dict()["settings"]
    r = run_mojo(CAPI / "libes8.dylib", "es8", chains=1, tune=10, draws=10)
    ours = r["settings"]
    for k in ("num_chains", "num_tune", "num_draws", "seed"):
        nut.pop(k); ours.pop(k)
    assert json.dumps(ours, sort_keys=True) == json.dumps(nut, sort_keys=True)


@pytest.fixture(scope="module")
def es8_runs():
    r = run_mojo(CAPI / "libes8.dylib", "es8", chains=4, tune=1000, draws=1000, seed=7)
    cm, _ = compile_nutpie(es8.build_model())
    tr, _ = run_nutpie(cm, chains=4, tune=1000, draws=1000, seed=7)
    post, div, ns = nutpie_posterior(tr)
    return r, summarize(es8_constrained(r["draws"])), summarize(es8_from_nutpie(post)), div


def test_es8_posterior_matches_nutpie(es8_runs):
    r, sm, sn, div_n = es8_runs
    print({k: (round(sm[k]["mean"], 3), round(sn[k]["mean"], 3), round(sm[k]["ess_bulk"])) for k in sm})
    compare(sm, sn)


def test_es8_divergences_comparable(es8_runs):
    r, _, _, div_n = es8_runs
    assert r["divergences"] <= max(20, 3 * div_n), (r["divergences"], div_n)  # < 0.5% of 4000 draws


def test_es8_known_posterior_ballpark(es8_runs):
    _, sm, _, _ = es8_runs
    # Reference values for non-centered eight schools: mu ~ 4.4 (sd ~3.3), tau ~ 3.6 (sd ~3.2)
    assert abs(sm["mu"]["mean"] - 4.4) < 0.5 and abs(sm["tau"]["mean"] - 3.6) < 0.5


def test_hlr_posterior_matches_nutpie_and_truth():
    d = hlr_ref.make_data()
    from phase2_nuts import hlr_constrained, hlr_from_nutpie
    r = run_mojo(CAPI / "libhlr.dylib", "hlr", chains=4, tune=1000, draws=1000, seed=7, data=hlr_ref.pack(d))
    sm = summarize(hlr_constrained(r["draws"], d["G"], d["K"]))
    cm, _ = compile_nutpie(hlr_ref.build_model(d))
    tr, _ = run_nutpie(cm, chains=4, tune=1000, draws=1000, seed=7)
    post, div, _ = nutpie_posterior(tr)
    sn = summarize(hlr_from_nutpie(post))
    compare(sm, sn)
    assert r["divergences"] == 0 and div == 0
    assert abs(sm["sigma"]["mean"] - 0.8) < 0.03  # generating noise sd


def test_rust_driver_rejects_bad_symbol():
    p = subprocess.run([str(BIN), "--lib", str(CAPI / "libes8.dylib"), "--model", "nope"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "nope_dim" in p.stderr


# ---------------------------------------------------------------- Phase 3: generated kernels
def _gen(model, name):
    from pytensor_mojo.ndlinker import compile_model_capi
    so, data, dim, _ = compile_model_capi(model, name)
    return so, data


def test_generated_es8_posterior_matches_nutpie(es8_runs):
    _, _, sn, div_n = es8_runs
    so, data = _gen(es8.build_model(), "es8gen")
    r = run_mojo(so, "es8gen", chains=4, tune=1000, draws=1000, seed=7, data=data)
    sm = summarize(es8_constrained(r["draws"]))
    compare(sm, sn)
    assert r["divergences"] <= max(20, 3 * div_n)


def test_generated_hlr_posterior_matches_nutpie():
    d = hlr_ref.make_data()
    from phase2_nuts import hlr_constrained, hlr_from_nutpie
    model = hlr_ref.build_model(d)
    so, data = _gen(model, "hlrgen")
    r = run_mojo(so, "hlrgen", chains=4, tune=1000, draws=1000, seed=7, data=data)
    sm = summarize(hlr_constrained(r["draws"], d["G"], d["K"]))
    cm, _ = compile_nutpie(model)
    tr, _ = run_nutpie(cm, chains=4, tune=1000, draws=1000, seed=7)
    post, div, _ = nutpie_posterior(tr)
    compare(sm, summarize(hlr_from_nutpie(post)))
    assert r["divergences"] == 0
