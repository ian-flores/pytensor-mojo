"""Docs: generated tables and figures match the code and results; documented examples run."""
import pathlib, subprocess, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_generated_docs_are_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "docs" / "gen" / "build_assets.py"), "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_docs_requirements_match_pixi_docs_feature():
    pins = dict(l.split("==") for l in (ROOT / "docs" / "requirements.txt").read_text().splitlines()
                if "==" in l and not l.startswith("#"))
    toml = (ROOT / "pixi.toml").read_text()
    for pkg in ("mkdocs", "mkdocs-material", "mkdocstrings-python"):
        major_minor = ".".join(pins[pkg].split(".")[:2])
        assert f'{pkg} = "{major_minor}.*"' in toml, pkg


@pytest.mark.skipif(not (ROOT / "sampler" / "target" / "release" / "mojo-nuts").exists(), reason="sampler not built")
def test_home_page_example_runs():
    """The example on docs/index.md, verbatim apart from smaller tune/draws."""
    sys.path.insert(0, str(ROOT))
    import pymc as pm
    from pytensor_mojo.sample import sample
    y = np.random.default_rng(0).normal(3.0, 2.0, size=200)
    with pm.Model() as model:
        mu = pm.Normal("mu", np.float64(0), np.float64(10))
        sigma = pm.HalfNormal("sigma", np.float64(5))
        pm.Normal("y", mu, sigma, observed=y)
    idata = sample(model, chains=4, tune=500, draws=300, seed=1)
    assert abs(idata["posterior"]["mu"].mean().item() - y.mean()) < 0.5


def test_readme_results_table_matches_generated():
    table = (ROOT / "docs" / "_generated" / "results_home.md").read_text()
    assert table in (ROOT / "README.md").read_text(), "README results table is stale: copy docs/_generated/results_home.md"
