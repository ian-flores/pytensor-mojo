"""pytensor_mojo.sample: PyMC model -> generated Mojo -> mojo-nuts -> ArviZ DataTree."""
import pathlib, sys
import numpy as np
import pymc as pm
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pytensor_mojo.sample import find_sampler, sample  # noqa: E402

try:
    find_sampler()
    HAVE = True
except FileNotFoundError:
    HAVE = False


@pytest.mark.skipif(not HAVE, reason="build sampler: pixi run build-sampler")
def test_sample_returns_datatree_with_constrained_values():
    rng = np.random.default_rng(0)
    y = rng.normal(3.0, 2.0, size=200)
    with pm.Model(name="quick") as m:
        mu = pm.Normal("mu", np.float64(0), np.float64(10))
        sigma = pm.HalfNormal("sigma", np.float64(5))
        pm.Deterministic("prec", 1 / sigma ** 2)
        pm.Normal("y", mu, sigma, observed=y)
    idata = sample(m, chains=2, tune=500, draws=400, seed=3)
    post = idata["posterior"]
    assert set(post.data_vars) >= {"quick::mu", "quick::sigma", "quick::prec"}
    assert post["quick::mu"].shape == (2, 400)
    assert (post["quick::sigma"].values > 0).all()           # constrained (back-transformed)
    np.testing.assert_allclose(post["quick::prec"].values, 1 / post["quick::sigma"].values ** 2)
    assert abs(float(post["quick::mu"].mean()) - y.mean()) < 0.5
    assert abs(float(post["quick::sigma"].mean()) - y.std()) < 0.4
    ss = idata["sample_stats"]
    assert ss["diverging"].shape == (2, 400) and ss["n_steps"].values.min() >= 1
    assert idata.attrs["sampling_time"] > 0
    import arviz_stats as azs
    assert float(azs.rhat(post["quick::mu"].values)) < 1.02
