"""Phase 4: every census model compiles through ndcodegen and matches PyMC at rtol 1e-10."""
import pathlib, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
from phase4_census import obs_models, prior_models  # noqa: E402
from pytensor_mojo.ndlinker import CapiModel, compile_model_capi  # noqa: E402
from pymc.blocking import DictToArrayBijection, RaveledVars  # noqa: E402

CASES = [("obs", n, b) for n, b in obs_models().items()] + [("prior", n, b) for n, b in prior_models().items()]


def _name(kind, n):
    return "p4_" + kind + "_" + "".join(ch for ch in n.lower() if ch.isalnum())


@pytest.mark.parametrize("kind,name,builder", CASES, ids=[f"{k}-{n}" for k, n, _ in CASES])
def test_census_model_matches_pymc(kind, name, builder):
    model = builder()
    so, data, dim, _ = compile_model_capi(model, _name(kind, name))
    cm = CapiModel(so, _name(kind, name), data)
    lpf, dlf = model.compile_logp(), model.compile_dlogp()
    ip = model.initial_point()
    info = DictToArrayBijection.map({v.name: ip[v.name] for v in model.value_vars}).point_map_info
    rng = np.random.default_rng(42)
    for s in np.repeat([0.1, 0.5, 1.0, 2.0], 6):  # 24 points
        x = rng.normal(scale=s, size=dim)
        pt = DictToArrayBijection.rmap(RaveledVars(x, info))
        rc, lp, g = cm(x)
        lp_ref, g_ref = lpf(pt), dlf(pt)
        assert rc == 0, (name, s)
        np.testing.assert_allclose(lp, lp_ref, rtol=1e-10, err_msg=f"{name} logp")
        np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-9 * max(1.0, np.max(np.abs(g_ref))),
                                   err_msg=f"{name} grad")
