"""Phase 4 census: which ops does ndcodegen miss for common PyMC likelihoods and priors?

Each distribution is built twice: as an observed likelihood (parameters with simple priors)
and as a free prior with PyMC's default transform. Prints a markdown table.
"""
import pathlib, sys
import numpy as np
import pymc as pm
import pytensor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pytensor_mojo.ndcodegen import missing_ops  # noqa: E402
from pytensor_mojo.ndlinker import QUERY, _Capture, flat_logp_grad  # noqa: E402
from pytensor.compile.mode import Mode  # noqa: E402

f64 = np.float64
rng = np.random.default_rng(0)
N = 50
pos = rng.gamma(2.0, 1.0, size=N)            # positive data
unit = rng.beta(2.0, 3.0, size=N)            # (0, 1) data
real = rng.standard_t(4, size=N)             # real data
counts = rng.poisson(3.0, size=N)
binn = np.full(N, 10)
bink = rng.binomial(10, 0.3, size=N)
bern = rng.integers(0, 2, size=N)
X = rng.normal(size=(N, 2))


def obs_models():
    """name -> builder for observed-likelihood models."""
    def m(fn):
        def build():
            with pm.Model() as mod:
                fn()
            return mod
        return build
    return {
        "Normal": m(lambda: pm.Normal("y", mu=pm.Normal("mu", f64(0), f64(5)), sigma=pm.HalfNormal("s", f64(2)), observed=real)),
        "StudentT": m(lambda: pm.StudentT("y", nu=pm.Gamma("nu", f64(2), f64(0.1)), mu=pm.Normal("mu", f64(0), f64(5)),
                                          sigma=pm.HalfNormal("s", f64(2)), observed=real)),
        "Gamma": m(lambda: pm.Gamma("y", alpha=pm.HalfNormal("a", f64(5)), beta=pm.HalfNormal("b", f64(5)), observed=pos)),
        "Beta": m(lambda: pm.Beta("y", alpha=pm.HalfNormal("a", f64(5)), beta=pm.HalfNormal("b", f64(5)), observed=unit)),
        "Poisson (log link)": m(lambda: pm.Poisson("y", mu=pm.math.exp(pm.Normal("a", f64(0), f64(2)) + X @ pm.Normal("b", f64(0), f64(1), shape=2)), observed=counts)),
        "Binomial (Uniform p)": m(lambda: pm.Binomial("y", n=binn, p=pm.Uniform("p", f64(0), f64(1)), observed=bink)),
        "NegativeBinomial": m(lambda: pm.NegativeBinomial("y", mu=pm.Exponential("mu", f64(0.3)), alpha=pm.Exponential("al", f64(1)), observed=counts)),
        "Bernoulli (logit_p)": m(lambda: pm.Bernoulli("y", logit_p=pm.Normal("a", f64(0), f64(2)) + X @ pm.Normal("b", f64(0), f64(1), shape=2), observed=bern)),
        "LogNormal": m(lambda: pm.LogNormal("y", mu=pm.Normal("mu", f64(0), f64(2)), sigma=pm.HalfNormal("s", f64(1)), observed=pos)),
        "Exponential": m(lambda: pm.Exponential("y", lam=pm.HalfNormal("l", f64(2)), observed=pos)),
        "HalfNormal": m(lambda: pm.HalfNormal("y", sigma=pm.HalfNormal("s", f64(2)), observed=pos)),
        "InverseGamma": m(lambda: pm.InverseGamma("y", alpha=pm.HalfNormal("a", f64(5)), beta=pm.HalfNormal("b", f64(5)), observed=pos)),
        "Weibull": m(lambda: pm.Weibull("y", alpha=pm.HalfNormal("a", f64(3)), beta=pm.HalfNormal("b", f64(3)), observed=pos)),
        "Laplace": m(lambda: pm.Laplace("y", mu=pm.Normal("mu", f64(0), f64(2)), b=pm.HalfNormal("b", f64(2)), observed=real)),
        "Cauchy": m(lambda: pm.Cauchy("y", alpha=pm.Normal("mu", f64(0), f64(2)), beta=pm.HalfNormal("b", f64(2)), observed=real)),
    }


def prior_models():
    def m(fn):
        def build():
            with pm.Model() as mod:
                fn()
            return mod
        return build
    return {
        "StudentT": m(lambda: pm.StudentT("x", nu=f64(4), mu=f64(0), sigma=f64(1), shape=3)),
        "Gamma": m(lambda: pm.Gamma("x", alpha=f64(2), beta=f64(1), shape=3)),
        "Beta": m(lambda: pm.Beta("x", alpha=f64(2), beta=f64(3), shape=3)),
        "LogNormal": m(lambda: pm.LogNormal("x", mu=f64(0), sigma=f64(1), shape=3)),
        "Exponential": m(lambda: pm.Exponential("x", lam=f64(1), shape=3)),
        "HalfNormal": m(lambda: pm.HalfNormal("x", sigma=f64(1), shape=3)),
        "InverseGamma": m(lambda: pm.InverseGamma("x", alpha=f64(3), beta=f64(2), shape=3)),
        "Weibull": m(lambda: pm.Weibull("x", alpha=f64(2), beta=f64(1), shape=3)),
        "Laplace": m(lambda: pm.Laplace("x", mu=f64(0), b=f64(1), shape=3)),
        "Cauchy": m(lambda: pm.Cauchy("x", alpha=f64(0), beta=f64(1), shape=3)),
        "Uniform (interval)": m(lambda: pm.Uniform("x", lower=f64(-1), upper=f64(2), shape=3)),
        "HalfCauchy": m(lambda: pm.HalfCauchy("x", beta=f64(1), shape=3)),
    }


def census_graph(model):
    q, lp, g, dim = flat_logp_grad(model)
    _Capture.sink.clear()
    pytensor.function([q], [lp, g], mode=Mode(_Capture(), QUERY))
    return _Capture.sink[-1]


def transforms(model):
    return ", ".join(sorted({type(model.rvs_to_transforms[rv]).__name__.replace("Transform", "")
                             for rv in model.free_RVs if model.rvs_to_transforms.get(rv) is not None})) or "-"


def main():
    rows = []
    for kind, builders in [("observed", obs_models()), ("prior", prior_models())]:
        for name, b in builders.items():
            mod = b()
            mt, ms = missing_ops(census_graph(mod))
            rows.append((name, kind, transforms(mod), mt, ms))
    print("| distribution | role | transforms | missing tensor ops | missing scalar ops |")
    print("|---|---|---|---|---|")
    allt, alls = set(), set()
    for name, kind, tr, mt, ms in rows:
        allt |= set(mt); alls |= set(ms)
        print(f"| {name} | {kind} | {tr} | {', '.join(mt) or '-'} | {', '.join(ms) or '-'} |")
    print(f"\nUNION missing tensor ops: {sorted(allt)}")
    print(f"UNION missing scalar ops: {sorted(alls)}")


if __name__ == "__main__":
    main()
