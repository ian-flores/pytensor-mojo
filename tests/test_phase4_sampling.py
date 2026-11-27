"""Phase 4: three representative models compiled by ndcodegen, checked vs PyMC and nutpie."""
import pathlib, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
from nuts_bench.runners import BIN, compile_nutpie, run_mojo, summarize  # noqa: E402
from phase4_sampling import MODELS, nutpie_unconstrained, run_nutpie_u  # noqa: E402
from pytensor_mojo.ndlinker import CapiModel, compile_model_capi  # noqa: E402
from pymc.blocking import DictToArrayBijection, RaveledVars  # noqa: E402

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="build sampler: cargo build --release")


@pytest.fixture(scope="module", params=list(MODELS))
def compiled(request):
    name = request.param
    model = MODELS[name]()
    sym = "t4s_" + name.split("_")[0]
    so, data, dim, _ = compile_model_capi(model, sym)
    return name, model, sym, so, data, dim


def test_logp_grad_match_pymc(compiled):
    name, model, sym, so, data, dim = compiled
    cm = CapiModel(so, sym, data)
    lpf, dlf = model.compile_logp(), model.compile_dlogp()
    ip = model.initial_point()
    info = DictToArrayBijection.map({v.name: ip[v.name] for v in model.value_vars}).point_map_info
    r = np.random.default_rng(9)
    for s in np.repeat([0.1, 0.5, 1.0, 1.5], 6):
        x = r.normal(scale=s, size=dim)
        p = DictToArrayBijection.rmap(RaveledVars(x, info))
        rc, lp, g = cm(x)
        assert rc == 0
        np.testing.assert_allclose(lp, lpf(p), rtol=1e-10)
        gr = dlf(p)
        np.testing.assert_allclose(g, gr, rtol=1e-10, atol=1e-9 * max(1.0, np.max(np.abs(gr))))


def test_posterior_matches_nutpie(compiled):
    name, model, sym, so, data, dim = compiled
    r = run_mojo(so, sym, chains=4, tune=1000, draws=1000, seed=11, data=data)
    sm = summarize({"x": r["draws"]})
    cmn, _ = compile_nutpie(model)
    tr, _ = run_nutpie_u(cmn, model, 4, 11)
    sn = summarize({"x": nutpie_unconstrained(tr, model)})
    assert r["divergences"] <= max(20, 3 * int(tr["sample_stats"]["diverging"].values.sum()))
    for k in sm:
        a, b = sm[k], sn[k]
        se = np.hypot(a["mcse"], b["mcse"])
        # 4.5 combined MCSE: the Beta-Binomial model compares 202 coordinates (multiple comparisons)
        assert abs(a["mean"] - b["mean"]) < 4.5 * se, (name, k, a, b)
        assert abs(a["sd"] - b["sd"]) < 0.2 * b["sd"], (name, k, a, b)
        assert a["rhat"] < 1.01 and a["ess_bulk"] > 400, (name, k, a)
