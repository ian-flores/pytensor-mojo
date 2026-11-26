"""Q3: census of Op types in compiled logp+dlogp graphs of typical PyMC models.

Compiles each model's logp+grad with backend-neutral fast_run rewrites and counts
(a) tensor-level Op classes in the optimized FunctionGraph and
(b) scalar Ops inside Elemwise / Composite nodes (what a codegen must emit).
"""
import collections, pathlib, sys
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.scalar.basic import Composite
from pytensor.tensor.elemwise import Elemwise

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pytensor_mojo.eight_schools_ref import build_model as eight_schools  # noqa: E402

rng = np.random.default_rng(0)

# Backend-neutral rewrites: fast_run with fusion, minus C-only/BLAS/inplace and
# minus the numba-specific scalarization passes (what a new linker would see).
from pytensor.compile.mode import Mode  # noqa: E402
from pytensor.graph.rewriting.db import RewriteDatabaseQuery  # noqa: E402
MODE = Mode(linker="py", optimizer=RewriteDatabaseQuery(
    include=["fast_run"], exclude=["cxx_only", "BlasOpt", "inplace", "numba"]))


def linreg():
    X = rng.normal(size=(200, 5)); y = X @ rng.normal(size=5) + rng.normal(size=200)
    with pm.Model() as m:
        b = pm.Normal("b", 0, 1, shape=5); a = pm.Normal("a", 0, 5); s = pm.HalfNormal("s", 1)
        pm.Normal("y", a + pt.dot(X, b), s, observed=y)
    return m


def logistic():
    X = rng.normal(size=(300, 4)); y = rng.integers(0, 2, size=300)
    with pm.Model() as m:
        b = pm.Normal("b", 0, 2, shape=4)
        pm.Bernoulli("y", logit_p=pt.dot(X, b), observed=y)
    return m


def hier_poisson():
    g = rng.integers(0, 10, size=500); y = rng.poisson(3, size=500)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1); sd = pm.Exponential("sd", 1)
        z = pm.Normal("z", 0, 1, shape=10)
        pm.Poisson("y", pt.exp(mu + sd * z[g]), observed=y)
    return m


def robust_gamma_studentt():
    y = rng.standard_t(4, size=100)
    with pm.Model() as m:
        nu = pm.Gamma("nu", 2, 0.1); s = pm.InverseGamma("s", 2, 1); mu = pm.Cauchy("mu", 0, 2)
        w = pm.Beta("w", 2, 2)
        pm.StudentT("y", nu=nu, mu=mu * w, sigma=s, observed=y)
    return m


def mvn_lkj():
    y = rng.normal(size=(50, 3))
    with pm.Model() as m:
        chol, _, _ = pm.LKJCholeskyCov("c", n=3, eta=2, sd_dist=pm.Exponential.dist(1), compute_corr=True)
        pm.MvNormal("y", mu=np.zeros(3), chol=chol, observed=y)
    return m


MODELS = dict(eight_schools=eight_schools, linreg=linreg, logistic=logistic,
              hier_poisson=hier_poisson, robust_studentt=robust_gamma_studentt, mvn_lkj=mvn_lkj)


def scalar_ops(op, acc):
    sop = op.scalar_op
    if isinstance(sop, Composite):
        for n in sop.fgraph.apply_nodes:
            if isinstance(n.op, Composite):
                scalar_ops(Elemwise(n.op), acc)
            else:
                acc[type(n.op).__name__] += 1
    else:
        acc[type(sop).__name__] += 1


def census(model):
    logp = model.logp(jacobian=True)
    grads = pt.grad(logp, model.value_vars)
    f = pytensor.function(model.value_vars, [logp, *grads], mode=MODE)
    tensor_ops, sc = collections.Counter(), collections.Counter()
    for node in f.maker.fgraph.apply_nodes:
        op = node.op
        name = type(op).__name__
        if isinstance(op, Elemwise):
            scalar_ops(op, sc)
        elif hasattr(op, "scalar_op"):  # CAReduce etc.
            name = f"{name}[{type(op.scalar_op).__name__}]"
        tensor_ops[name] += 1
    return tensor_ops, sc


def main():
    all_t, all_s = set(), set()
    for name, mk in MODELS.items():
        t, s = census(mk())
        all_t |= set(t); all_s |= set(s)
        print(f"### {name}: {sum(t.values())} nodes")
        print("  tensor ops:", dict(sorted(t.items())))
        print("  scalar ops:", dict(sorted(s.items())))
    print(f"\nUNION tensor op kinds ({len(all_t)}):", sorted(all_t))
    print(f"UNION scalar op kinds ({len(all_s)}):", sorted(all_s))


if __name__ == "__main__":
    main()
