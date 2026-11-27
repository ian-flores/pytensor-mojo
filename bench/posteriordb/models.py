"""Hand ports of posteriordb Stan programs to PyMC, plus posteriordb data and reference loaders.

Each entry of ``MODELS`` maps a posteriordb posterior name (``<data>-<model>``) to a
:class:`Port`: a family label, a builder ``build(data) -> pm.Model`` and the per-model
deviations from the Stan program.

Porting conventions (they apply to every model unless its ``notes`` say otherwise):

- Parameter names, shapes and constraints follow the Stan program, so the posterior can be
  compared with the reference draws by name (``beta[1]`` is ``beta[0]`` here).
- Stan's unconstrained parameterization is kept: ``<lower=0>`` maps to PyMC's log transform,
  ``<lower=0, upper=1>`` to the logodds/interval transform and ``ordered`` to PyMC's ordered
  transform, which are the same maps Stan uses.
- A parameter with no prior statement gets ``pm.Flat`` (or ``pm.HalfFlat`` when ``<lower=0>``),
  i.e. Stan's improper uniform prior.
- A ``<lower=0>`` parameter with a prior centred at 0 (``normal(0, s)``, ``cauchy(0, s)``,
  ``student_t(nu, 0, s)``) uses the Half* distribution. This differs from Stan only by a
  constant (the truncation normalizer), which changes neither the posterior nor the gradient.
  A ``<lower=0>`` parameter with any other prior keeps that prior with ``transform=log``.
- ``transformed data`` is computed with NumPy at build time. Generated quantities are dropped
  (the reference draws only contain parameters).
- Constants are float64 to avoid PyMC's float32 folding of Python literals.

The data files are read as JSON from a local clone of posteriordb (see :func:`pdb_root`).
Nothing else from posteriordb is executed or imported.
"""
from __future__ import annotations

import json
import os
import pathlib
import zipfile
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

f64 = np.float64
ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG = pm.distributions.transforms.log
ORDERED = pm.distributions.transforms.ordered


# ------------------------------------------------------------------ posteriordb access
def pdb_root() -> pathlib.Path:
    """posterior_database directory: ``$POSTERIORDB_PATH`` or ``.cache/posteriordb`` in the repo."""
    env = os.environ.get("POSTERIORDB_PATH")
    base = pathlib.Path(env) if env else ROOT / ".cache" / "posteriordb"
    return base / "posterior_database" if (base / "posterior_database").is_dir() else base


def _read_json_zip(path: pathlib.Path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(z.namelist()[0]))


def posterior_info(name: str) -> dict:
    return json.loads((pdb_root() / "posteriors" / f"{name}.json").read_text())


def load_data(name: str) -> dict:
    """posteriordb data for a posterior, as a dict of NumPy arrays / Python scalars."""
    info = posterior_info(name)
    raw = _read_json_zip(pdb_root() / "data" / "data" / f"{info['data_name']}.json.zip")
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in raw.items()}


def load_reference(name: str) -> dict:
    """Reference draws: ``{param: (chains, draws) array}`` with Stan names (1-based)."""
    info = posterior_info(name)
    raw = _read_json_zip(pdb_root() / "reference_posteriors" / "draws" / "draws"
                         / f"{info['reference_posterior_name']}.json.zip")
    keys = raw[0].keys()
    return {k: np.stack([np.asarray(c[k], dtype=float) for c in raw]) for k in keys}


def available(name: str) -> bool:
    try:
        info = posterior_info(name)
    except FileNotFoundError:
        return False
    r = pdb_root()
    return ((r / "data" / "data" / f"{info['data_name']}.json.zip").exists()
            and (r / "reference_posteriors" / "draws" / "draws"
                 / f"{info['reference_posterior_name']}.json.zip").exists())


def stan_names(post: dict) -> dict:
    """``{var: (chains, draws, *shape)}`` -> ``{stan_name: (chains, draws)}`` (1-based, C order)."""
    out = {}
    for k, a in post.items():
        a = np.asarray(a)
        shp = a.shape[2:]
        if not shp:
            out[k] = a
            continue
        for idx in np.ndindex(*shp):
            out[f"{k}[{','.join(str(i + 1) for i in idx)}]"] = a[(slice(None), slice(None)) + idx]
    return out


# ------------------------------------------------------------------ helpers
def _normal_lpdf(y, mu, sigma):
    return -0.5 * ((y - mu) / sigma) ** 2 - pt.log(sigma) - f64(0.5 * np.log(2 * np.pi))


def _rk4(f, z0, ts, t0, theta, n_sub):
    """Fixed-step RK4 from t0 through every time in ``ts`` with ``n_sub`` steps per interval.

    ``f(t, z, theta)`` returns dz/dt. One Scan step covers one observation interval; the
    sub-steps are unrolled in the step graph. Returns the states at ``ts`` (len(ts), dim).
    """
    ts = np.asarray(ts, dtype=f64)
    starts = np.concatenate([[f64(t0)], ts[:-1]])
    dts = (ts - starts) / n_sub

    def step(t_start, dt, z, *th):
        t = t_start
        for _ in range(n_sub):
            k1 = f(t, z, th)
            k2 = f(t + dt / 2, z + dt / 2 * k1, th)
            k3 = f(t + dt / 2, z + dt / 2 * k2, th)
            k4 = f(t + dt, z + dt * k3, th)
            z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt
        return z

    zs = pytensor.scan(step, sequences=[pt.as_tensor(starts), pt.as_tensor(dts)], outputs_info=[z0],
                       non_sequences=list(theta), return_updates=False)
    return zs


def _scan(fn, **kw):
    return pytensor.scan(fn, return_updates=False, **kw)


# ------------------------------------------------------------------ models
def rate_1(d):
    with pm.Model() as m:
        theta = pm.Beta("theta", f64(1), f64(1))
        pm.Binomial("k", n=int(d["n"]), p=theta, observed=int(d["k"]))
    return m


def rate_3(d):
    with pm.Model() as m:
        theta = pm.Beta("theta", f64(1), f64(1))
        pm.Binomial("k1", n=int(d["n1"]), p=theta, observed=int(d["k1"]))
        pm.Binomial("k2", n=int(d["n2"]), p=theta, observed=int(d["k2"]))
    return m


def ark(d):
    K, y = int(d["K"]), np.asarray(d["y"], dtype=f64)
    T = len(y)
    lags = np.stack([y[K - k:T - k] for k in range(1, K + 1)], axis=1)   # (T-K, K): y[t-k]
    with pm.Model() as m:
        alpha = pm.Normal("alpha", f64(0), f64(10))
        beta = pm.Normal("beta", f64(0), f64(10), shape=K)
        sigma = pm.HalfCauchy("sigma", f64(2.5))
        pm.Normal("y", mu=alpha + pt.dot(lags, beta), sigma=sigma, observed=y[K:])
    return m


def arma11(d):
    y = np.asarray(d["y"], dtype=f64)
    with pm.Model() as m:
        mu = pm.Normal("mu", f64(0), f64(10))
        phi = pm.Normal("phi", f64(0), f64(2))
        theta = pm.Normal("theta", f64(0), f64(2))
        sigma = pm.HalfCauchy("sigma", f64(2.5))
        err1 = y[0] - (mu + phi * mu)

        def step(y_t, y_prev, err_prev, mu, phi, theta):
            return y_t - (mu + phi * y_prev + theta * err_prev)

        rest = _scan(step, sequences=[pt.as_tensor(y[1:]), pt.as_tensor(y[:-1])], outputs_info=[err1],
                     non_sequences=[mu, phi, theta])
        err = pt.concatenate([pt.atleast_1d(err1), rest])
        pm.Potential("err_lik", _normal_lpdf(err, f64(0), sigma).sum())
    return m


def garch11(d):
    y = np.asarray(d["y"], dtype=f64)
    with pm.Model() as m:
        mu = pm.Flat("mu")
        alpha0 = pm.HalfFlat("alpha0")
        alpha1 = pm.Uniform("alpha1", f64(0), f64(1))
        beta1 = pm.Uniform("beta1", f64(0), 1 - alpha1)
        # Uniform(0, 1 - alpha1) adds -log(1 - alpha1); Stan's bounded flat prior does not.
        pm.Potential("beta1_flat", pt.log(1 - alpha1))

        def step(y_prev, s_prev, mu, alpha0, alpha1, beta1):
            return pt.sqrt(alpha0 + alpha1 * (y_prev - mu) ** 2 + beta1 * s_prev ** 2)

        s1 = pt.as_tensor(f64(d["sigma1"]))
        rest = _scan(step, sequences=[pt.as_tensor(y[:-1])], outputs_info=[s1],
                     non_sequences=[mu, alpha0, alpha1, beta1])
        sigma = pt.concatenate([pt.atleast_1d(s1), rest])
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    return m


def diamonds(d):
    X = np.asarray(d["X"], dtype=f64)
    Xc = X[:, 1:] - X[:, 1:].mean(axis=0)
    means = X[:, 1:].mean(axis=0)
    with pm.Model() as m:
        b = pm.Normal("b", f64(0), f64(1), shape=Xc.shape[1])
        intercept = pm.StudentT("Intercept", nu=f64(3), mu=f64(8), sigma=f64(10))
        sigma = pm.HalfStudentT("sigma", nu=f64(3), sigma=f64(10))
        pm.Normal("Y", mu=intercept + pt.dot(Xc, b), sigma=sigma, observed=np.asarray(d["Y"], dtype=f64))
        pm.Deterministic("b_Intercept", intercept - pt.dot(means, b))
    return m


def _dogs_counts(y):
    y = np.asarray(y)
    J, T = y.shape
    avoid = np.zeros((J, T)); shock = np.zeros((J, T))
    for t in range(1, T):
        avoid[:, t] = avoid[:, t - 1] + 1 - y[:, t - 1]
        shock[:, t] = shock[:, t - 1] + y[:, t - 1]
    return avoid, shock


def dogs(d):
    y = np.asarray(d["y"])
    avoid, shock = _dogs_counts(y)
    with pm.Model() as m:
        beta = pm.Normal("beta", f64(0), f64(100), shape=3)
        p = beta[0] + beta[1] * avoid + beta[2] * shock
        pm.Bernoulli("y", logit_p=p, observed=y)
    return m


def dogs_hierarchical(d):
    y = np.asarray(d["y"])
    prev_avoid, prev_shock = _dogs_counts(y)
    with pm.Model() as m:
        a = pm.Uniform("a", f64(0), f64(1))
        b = pm.Uniform("b", f64(0), f64(1))
        p = a ** prev_shock * b ** prev_avoid
        pm.Bernoulli("y", p=p, observed=y)
    return m


def _earnings(d):
    return (np.log(np.asarray(d["earn"], dtype=f64)), np.asarray(d["height"], dtype=f64),
            np.asarray(d["male"], dtype=f64))


def logearn_interaction(d):
    le, h, male = _earnings(d)
    with pm.Model() as m:
        beta = pm.Flat("beta", shape=4)
        sigma = pm.HalfFlat("sigma")
        pm.Normal("log_earn", mu=beta[0] + beta[1] * h + beta[2] * male + beta[3] * (h * male), sigma=sigma,
                  observed=le)
    return m


def logearn_logheight_male(d):
    le, h, male = _earnings(d)
    with pm.Model() as m:
        beta = pm.Flat("beta", shape=3)
        sigma = pm.HalfFlat("sigma")
        pm.Normal("log_earn", mu=beta[0] + beta[1] * np.log(h) + beta[2] * male, sigma=sigma, observed=le)
    return m


def eight_schools_centered(d):
    with pm.Model() as m:
        tau = pm.HalfCauchy("tau", f64(5))
        mu = pm.Normal("mu", f64(0), f64(5))
        theta = pm.Normal("theta", mu, tau, shape=int(d["J"]))
        pm.Normal("y", theta, np.asarray(d["sigma"], dtype=f64), observed=np.asarray(d["y"], dtype=f64))
    return m


def eight_schools_noncentered(d):
    with pm.Model() as m:
        theta_trans = pm.Normal("theta_trans", f64(0), f64(1), shape=int(d["J"]))
        mu = pm.Normal("mu", f64(0), f64(5))
        tau = pm.HalfCauchy("tau", f64(5))
        theta = pm.Deterministic("theta", theta_trans * tau + mu)
        pm.Normal("y", theta, np.asarray(d["sigma"], dtype=f64), observed=np.asarray(d["y"], dtype=f64))
    return m


def _exp_quad(x, alpha, rho):
    x = np.asarray(x, dtype=f64)
    d2 = (x[:, None] - x[None, :]) ** 2
    return alpha ** 2 * pt.exp(-0.5 * d2 / rho ** 2)


def gp_regr(d):
    N = int(d["N"])
    with pm.Model() as m:
        rho = pm.Gamma("rho", alpha=f64(25), beta=f64(4))
        alpha = pm.HalfNormal("alpha", f64(2))
        sigma = pm.HalfNormal("sigma", f64(1))
        cov = _exp_quad(d["x"], alpha, rho) + sigma * pt.eye(N)   # Stan adds sigma, not sigma^2
        pm.MvNormal("y", mu=np.zeros(N), chol=pt.linalg.cholesky(cov), observed=np.asarray(d["y"], dtype=f64))
    return m


def gp_pois_regr(d):
    N = int(d["N"])
    with pm.Model() as m:
        rho = pm.Gamma("rho", alpha=f64(25), beta=f64(4))
        alpha = pm.HalfNormal("alpha", f64(2))
        f_tilde = pm.Normal("f_tilde", f64(0), f64(1), shape=N)
        L = pt.linalg.cholesky(_exp_quad(d["x"], alpha, rho) + f64(1e-10) * pt.eye(N))
        f = pm.Deterministic("f", pt.dot(L, f_tilde))
        pm.Poisson("k", mu=pt.exp(f), observed=np.asarray(d["k"]))
    return m


def hmm_example(d):
    y = np.asarray(d["y"], dtype=f64)
    K = int(d["K"])
    with pm.Model() as m:
        theta1 = pm.Dirichlet("theta1", a=np.ones(K))
        theta2 = pm.Dirichlet("theta2", a=np.ones(K))
        # positive_ordered[K] mu: Stan's transform written out (log of the first element, log
        # of the increments), with its log-Jacobian.
        mu_raw = pm.Flat("mu_raw__", shape=K)
        mu = pm.Deterministic("mu", pt.cumsum(pt.exp(mu_raw)))
        pm.Potential("mu_jacobian", mu_raw.sum())
        pm.Potential("mu_prior", _normal_lpdf(mu[0], f64(3), f64(1)) + _normal_lpdf(mu[1], f64(10), f64(1)))
        log_theta = pt.log(pt.stack([theta1, theta2]))             # [j, k]

        def step(y_t, g_prev, log_theta, mu):
            acc = g_prev[:, None] + log_theta                       # [j, k]
            return pt.logsumexp(acc, axis=0) + _normal_lpdf(y_t, mu, f64(1))

        g1 = _normal_lpdf(y[0], mu, f64(1))
        gs = _scan(step, sequences=[pt.as_tensor(y[1:])], outputs_info=[g1], non_sequences=[log_theta, mu])
        pm.Potential("forward", pt.logsumexp(gs[-1]))
    return m


def lotka_volterra(d):
    ts = np.asarray(d["ts"], dtype=f64)
    y = np.asarray(d["y"], dtype=f64)
    y_init = np.asarray(d["y_init"], dtype=f64)

    def dz(t, z, th):
        a, b, g, dl = th[0][0], th[0][1], th[0][2], th[0][3]
        u, v = z[0], z[1]
        return pt.stack([(a - b * v) * u, (-g + dl * u) * v])

    with pm.Model() as m:
        theta = pm.Normal("theta", mu=np.array([1.0, 0.05, 1.0, 0.05]), sigma=np.array([0.5, 0.05, 0.5, 0.05]),
                          shape=4, transform=LOG)
        z_init = pm.LogNormal("z_init", mu=f64(np.log(10)), sigma=f64(1), shape=2)
        sigma = pm.LogNormal("sigma", mu=f64(-1), sigma=f64(1), shape=2)
        z = _rk4(dz, z_init, ts, 0.0, [theta], n_sub=8)
        pm.LogNormal("y_init", mu=pt.log(z_init), sigma=sigma, observed=y_init)
        pm.LogNormal("y", mu=pt.log(z), sigma=sigma, observed=y)
    return m


def one_comp_mm_elim_abs(d):
    times = np.asarray(d["times"], dtype=f64)
    D, V = f64(d["D"]), f64(d["V"])

    def dy(t, y, th):
        k_a, K_m, V_m = th
        dose = pt.switch(pt.gt(t, 0), pt.exp(-k_a * t) * D * k_a / V, f64(0))
        return dose - (V_m / V) * y / (K_m + y)

    with pm.Model() as m:
        k_a = pm.HalfCauchy("k_a", f64(1))
        K_m = pm.HalfCauchy("K_m", f64(1))
        V_m = pm.HalfCauchy("V_m", f64(1))
        sigma = pm.HalfCauchy("sigma", f64(1))
        C = _rk4(dy, pt.as_tensor(f64(0.0)), times, f64(d["t0"]), [k_a, K_m, V_m], n_sub=8)
        pm.LogNormal("C_hat", mu=pt.log(C), sigma=sigma, observed=np.asarray(d["C_hat"], dtype=f64))
    return m


def kidscore_momhsiq(d):
    with pm.Model() as m:
        beta = pm.Flat("beta", shape=3)
        sigma = pm.HalfCauchy("sigma", f64(2.5))
        pm.Normal("kid_score", mu=beta[0] + beta[1] * np.asarray(d["mom_hs"], dtype=f64)
                  + beta[2] * np.asarray(d["mom_iq"], dtype=f64), sigma=sigma,
                  observed=np.asarray(d["kid_score"], dtype=f64))
    return m


def kidscore_interaction_c(d):
    hs = np.asarray(d["mom_hs"], dtype=f64); iq = np.asarray(d["mom_iq"], dtype=f64)
    c_hs, c_iq = hs - hs.mean(), iq - iq.mean()
    with pm.Model() as m:
        beta = pm.Flat("beta", shape=4)
        sigma = pm.HalfFlat("sigma")
        pm.Normal("kid_score", mu=beta[0] + beta[1] * c_hs + beta[2] * c_iq + beta[3] * (c_hs * c_iq), sigma=sigma,
                  observed=np.asarray(d["kid_score"], dtype=f64))
    return m


def kilpisjarvi(d):
    with pm.Model() as m:
        alpha = pm.Normal("alpha", f64(d["pmualpha"]), f64(d["psalpha"]))
        beta = pm.Normal("beta", f64(d["pmubeta"]), f64(d["psbeta"]))
        sigma = pm.HalfFlat("sigma")
        pm.Normal("y", mu=alpha + beta * np.asarray(d["x"], dtype=f64), sigma=sigma,
                  observed=np.asarray(d["y"], dtype=f64))
    return m


def low_dim_gauss_mix(d):
    y = np.asarray(d["y"], dtype=f64)
    with pm.Model() as m:
        mu = pm.Normal("mu", f64(0), f64(2), shape=2, transform=ORDERED, initval=np.array([-1.0, 1.0]))
        sigma = pm.HalfNormal("sigma", f64(2), shape=2)
        theta = pm.Beta("theta", f64(5), f64(5))
        lp1 = pt.log(theta) + _normal_lpdf(y, mu[0], sigma[0])
        lp2 = pt.log1p(-theta) + _normal_lpdf(y, mu[1], sigma[1])
        pm.Potential("log_mix", pt.logaddexp(lp1, lp2).sum())
    return m


def accel_gp(d):
    Y = np.asarray(d["Y"], dtype=f64)

    def gpa(X, sdgp, lscale, zgp, slambda):
        lam2 = (np.asarray(slambda, dtype=f64).reshape(len(slambda), -1) ** 2).sum(axis=1)
        spd = sdgp ** 2 * (f64(np.sqrt(2 * np.pi)) * lscale) * pt.exp(-0.5 * lscale ** 2 * lam2)
        return pt.dot(np.asarray(X, dtype=f64), pt.sqrt(spd) * zgp)

    with pm.Model() as m:
        intercept = pm.StudentT("Intercept", nu=f64(3), mu=f64(-13), sigma=f64(36))
        sdgp_1 = pm.HalfStudentT("sdgp_1", nu=f64(3), sigma=f64(36))
        lscale_1 = pm.InverseGamma("lscale_1", alpha=f64(1.124909), beta=f64(0.0177))
        zgp_1 = pm.Normal("zgp_1", f64(0), f64(1), shape=int(d["NBgp_1"]))
        intercept_s = pm.StudentT("Intercept_sigma", nu=f64(3), mu=f64(0), sigma=f64(10))
        sdgp_s = pm.HalfStudentT("sdgp_sigma_1", nu=f64(3), sigma=f64(36))
        lscale_s = pm.InverseGamma("lscale_sigma_1", alpha=f64(1.124909), beta=f64(0.0177))
        zgp_s = pm.Normal("zgp_sigma_1", f64(0), f64(1), shape=int(d["NBgp_sigma_1"]))
        mu = intercept + gpa(d["Xgp_1"], sdgp_1, lscale_1, zgp_1, d["slambda_1"])
        sig = pt.exp(intercept_s + gpa(d["Xgp_sigma_1"], sdgp_s, lscale_s, zgp_s, d["slambda_sigma_1"]))
        pm.Normal("Y", mu=mu, sigma=sig, observed=Y)
    return m


def logmesquite_logvas(d):
    g = {k: np.asarray(d[k], dtype=f64) for k in ("weight", "diam1", "diam2", "canopy_height", "total_height",
                                                    "density", "group")}
    cols = [np.log(g["diam1"] * g["diam2"] * g["canopy_height"]), np.log(g["diam1"] * g["diam2"]),
            np.log(g["diam1"] / g["diam2"]), np.log(g["total_height"]), np.log(g["density"]), g["group"]]
    with pm.Model() as m:
        beta = pm.Flat("beta", shape=7)
        sigma = pm.HalfFlat("sigma")
        mu = beta[0] + sum(beta[i + 1] * c for i, c in enumerate(cols))
        pm.Normal("log_weight", mu=mu, sigma=sigma, observed=np.log(g["weight"]))
    return m


def nes(d):
    age = np.asarray(d["age_discrete"])
    cols = [d["real_ideo"], d["race_adj"], age == 2, age == 3, age == 4, d["educ1"], d["gender"], d["income"]]
    cols = [np.asarray(c, dtype=f64) for c in cols]
    with pm.Model() as m:
        beta = pm.Flat("beta", shape=9)
        sigma = pm.HalfFlat("sigma")
        mu = beta[0] + sum(beta[i + 1] * c for i, c in enumerate(cols))
        pm.Normal("partyid7", mu=mu, sigma=sigma, observed=np.asarray(d["partyid7"], dtype=f64))
    return m


def radon_pooled(d):
    with pm.Model() as m:
        sigma_y = pm.HalfNormal("sigma_y", f64(1))
        alpha = pm.Normal("alpha", f64(0), f64(10))
        beta = pm.Normal("beta", f64(0), f64(10))
        pm.Normal("log_radon", mu=alpha + beta * np.asarray(d["floor_measure"], dtype=f64), sigma=sigma_y,
                  observed=np.asarray(d["log_radon"], dtype=f64))
    return m


def blr(d):
    X = np.asarray(d["X"], dtype=f64)
    with pm.Model() as m:
        beta = pm.Normal("beta", f64(0), f64(10), shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", f64(10))
        pm.Normal("y", mu=pt.dot(X, beta), sigma=sigma, observed=np.asarray(d["y"], dtype=f64))
    return m


def wells_dist(d):
    with pm.Model() as m:
        beta = pm.Flat("beta", shape=2)
        pm.Bernoulli("switched", logit_p=beta[0] + beta[1] * np.asarray(d["dist"], dtype=f64),
                     observed=np.asarray(d["switched"]))
    return m


def nes_logit_model(d):
    with pm.Model() as m:
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=1)
        x = np.asarray(d["income"], dtype=f64)[:, None]
        pm.Bernoulli("vote", logit_p=alpha + pt.dot(x, beta), observed=np.asarray(d["vote"]))
    return m


# ------------------------------------------------------------------ registry
@dataclass
class Port:
    family: str
    build: Callable
    notes: list = field(default_factory=list)


_HALF = "<lower=0> scale with a 0-centred prior: Half* distribution (differs by a constant)."
_FLAT = "No prior in Stan: pm.Flat / pm.HalfFlat (improper uniform)."

MODELS: dict[str, Port] = {
    "Rate_1_data-Rate_1_model": Port("GLM (binomial)", rate_1),
    "Rate_3_data-Rate_3_model": Port("GLM (binomial)", rate_3),
    "wells_data-wells_dist": Port("GLM (logistic)", wells_dist, [_FLAT]),
    "nes_logit_data-nes_logit_model": Port("GLM (logistic)", nes_logit_model,
                                           [_FLAT, "bernoulli_logit_glm(x, alpha, beta) written as Bernoulli(logit_p=alpha + x @ beta)."]),
    "dogs-dogs": Port("GLM (logistic)", dogs, ["n_avoid / n_shock depend only on data: computed in NumPy."]),
    "dogs-dogs_hierarchical": Port("GLM (Bernoulli, power link)", dogs_hierarchical,
                                   ["Uniform(0, 1) for <lower=0, upper=1> with no prior (same logit transform).",
                                    "prev_shock / prev_avoid computed in NumPy."]),
    "earnings-logearn_interaction": Port("linear regression", logearn_interaction, [_FLAT]),
    "earnings-logearn_logheight_male": Port("linear regression", logearn_logheight_male, [_FLAT]),
    "kidiq-kidscore_momhsiq": Port("linear regression", kidscore_momhsiq, [_FLAT, _HALF]),
    "kidiq_with_mom_work-kidscore_interaction_c": Port("linear regression", kidscore_interaction_c, [_FLAT]),
    "kilpisjarvi_mod-kilpisjarvi": Port("linear regression", kilpisjarvi, [_FLAT]),
    "mesquite-logmesquite_logvas": Port("linear regression", logmesquite_logvas, [_FLAT]),
    "nes2000-nes": Port("linear regression", nes, [_FLAT]),
    "radon_mn-radon_pooled": Port("linear regression", radon_pooled, [_HALF]),
    "sblrc-blr": Port("linear regression", blr, [_HALF]),
    "diamonds-diamonds": Port("linear regression (K=24, N=5000)", diamonds,
                              ["student_t_lpdf(sigma | 3, 0, 10) - student_t_lccdf(0 | 3, 0, 10): HalfStudentT(3, 10) (exact).",
                               "normal_id_glm_lpdf written as Normal(Intercept + Xc @ b, sigma)."]),
    "eight_schools-eight_schools_centered": Port("hierarchical", eight_schools_centered, [_HALF]),
    "eight_schools-eight_schools_noncentered": Port("hierarchical", eight_schools_noncentered, [_HALF]),
    "low_dim_gauss_mix-low_dim_gauss_mix": Port("mixture", low_dim_gauss_mix,
                                                [_HALF, "log_mix written as logaddexp over a Potential; ordered[2] mu uses PyMC's ordered transform (same as Stan's)."]),
    "arK-arK": Port("time series (AR(5))", ark, [_HALF, "Lagged design matrix built in NumPy (vectorized; same likelihood)."]),
    "arma-arma11": Port("time series (ARMA(1,1))", arma11,
                        [_HALF, "Error recursion written with pytensor.scan; err ~ normal(0, sigma) as a Potential."]),
    "garch-garch11": Port("time series (GARCH(1,1))", garch11,
                          [_FLAT, "beta1 <upper=1-alpha1>: Uniform(0, 1 - alpha1) plus Potential log(1 - alpha1) to cancel its density (same interval transform).",
                           "Volatility recursion written with pytensor.scan."]),
    "hmm_example-hmm_example": Port("hidden Markov model", hmm_example,
                                    ["simplex[2]: Dirichlet(1, 1) with PyMC's simplex transform (Stan uses stick-breaking; same posterior, different unconstrained space).",
                                     "positive_ordered mu written as cumsum(exp(mu_raw__)) with its Jacobian; normal_lpdf terms as Potentials.",
                                     "Forward algorithm written with pytensor.scan."]),
    "mcycle_gp-accel_gp": Port("Gaussian process (basis-function approx.)", accel_gp,
                               ["student_t_lpdf - student_t_lccdf(0 | ...) on sdgp: HalfStudentT (exact).",
                                "The length-1 vector wrappers vsdgp / vlscale are dropped."]),
    "gp_pois_regr-gp_regr": Port("Gaussian process (exact)", gp_regr,
                                 [_HALF, "multi_normal_cholesky as MvNormal(chol=cholesky(cov))."]),
    "gp_pois_regr-gp_pois_regr": Port("Gaussian process (latent, Poisson)", gp_pois_regr, [_HALF]),
    "hudson_lynx_hare-lotka_volterra": Port("ODE (Lotka-Volterra)", lotka_volterra,
                                            ["integrate_ode_rk45 (adaptive, rtol 1e-5) replaced by fixed-step RK4, 8 steps per year, in pytensor.scan.",
                                             "theta <lower=0> with normal priors: Normal(..., transform=log) (Stan's truncation constant omitted)."]),
    "one_comp_mm_elim_abs-one_comp_mm_elim_abs": Port("ODE (pharmacokinetic)", one_comp_mm_elim_abs,
                                                      [_HALF, "integrate_ode_bdf replaced by fixed-step RK4, 8 steps per 0.5-day interval, in pytensor.scan."]),
}
