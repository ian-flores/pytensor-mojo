"""PyMC eight-schools reference model and compiled logp/dlogp on a flat unconstrained vector."""
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import graph_replace

Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
J = len(Y)


def build_model(float64_params=True):
    """Eight schools, non-centered.

    float64_params=True passes the scale constants as np.float64. With plain
    Python literals (5.0) PyTensor autocasts them to float32/int8, so log(5)
    is evaluated in float32 and logp is off by ~3e-8 (see README FINDINGS).
    """
    c = np.float64 if float64_params else (lambda v: v)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, c(5.0))
        tau = pm.HalfCauchy("tau", c(5.0))
        theta_tilde = pm.Normal("theta_tilde", 0.0, 1.0, shape=J)
        pm.Normal("y", mu + tau * theta_tilde, sigma=SIGMA, observed=Y)
    return model


def flat_logp_dlogp_graph(model):
    """Return (q, logp, dlogp) symbolic graph with q = [mu, tau_log__, theta_tilde...]."""
    names = [v.name for v in model.value_vars]
    assert names == ["mu", "tau_log__", "theta_tilde"], names
    q = pt.dvector("q")
    logp = model.logp(jacobian=True)
    dlogp = pt.grad(logp, model.value_vars)
    sizes = [1, 1, J]
    parts, i = [], 0
    for v, n in zip(model.value_vars, sizes):
        parts.append(q[i] if v.ndim == 0 else q[i:i + n])
        i += n
    repl = dict(zip(model.value_vars, parts))
    logp_q, *dl = graph_replace([logp, *dlogp], repl, strict=False)
    dlogp_q = pt.concatenate([pt.atleast_1d(d) for d in dl])
    return q, logp_q, dlogp_q


def compile_flat(model, mode=None):
    q, lp, dlp = flat_logp_dlogp_graph(model)
    f = pytensor.function([q], [lp, dlp], mode=mode)
    f.trust_input = True
    return f


def point_to_dict(qv):
    return {"mu": qv[0], "tau_log__": qv[1], "theta_tilde": qv[2:]}
