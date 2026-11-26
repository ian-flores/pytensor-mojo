"""Hierarchical linear regression: data generator, PyMC reference, ctypes access to the Mojo C ABI."""
import ctypes
import pathlib

import numpy as np
import pymc as pm

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = ROOT / "mojo_capi" / "libhlr.dylib"
f64 = np.float64


def make_data(N=10_000, G=20, K=5, seed=0):
    """Rows sorted by group. Returns dict with X (N,K), y, g, offsets."""
    rng = np.random.default_rng(seed)
    g = np.sort(rng.integers(0, G, size=N))
    X = rng.normal(size=(N, K))
    a = 1.0 + 0.7 * rng.normal(size=G)
    beta = rng.normal(size=K)
    y = a[g] + X @ beta + 0.8 * rng.normal(size=N)
    offsets = np.searchsorted(g, np.arange(G + 1))
    return dict(N=N, G=G, K=K, X=X, y=y, g=g, offsets=offsets)


def pack(d):
    """Flat float64 buffer: [N, K, G, offsets(G+1), X col-major (K*N), y(N)]."""
    return np.concatenate([[d["N"], d["K"], d["G"]], d["offsets"].astype(f64),
                           np.asfortranarray(d["X"]).T.ravel(), d["y"]]).astype(f64)


def build_model(d):
    G, K = d["G"], d["K"]
    with pm.Model() as m:
        mu_a = pm.Normal("mu_a", f64(0), f64(5))
        sigma_a = pm.HalfNormal("sigma_a", f64(2.5))
        a = pm.Normal("a", mu_a, sigma_a, shape=G)  # centered
        beta = pm.Normal("beta", f64(0), f64(2), shape=K)
        sigma = pm.HalfNormal("sigma", f64(1))
        pm.Normal("y", a[d["g"]] + d["X"] @ beta, sigma, observed=d["y"])
    return m


def point_to_dict(x, G, K):
    return {"mu_a": x[0], "sigma_a_log__": x[1], "a": x[2:2 + G], "beta": x[2 + G:2 + G + K],
            "sigma_log__": x[2 + G + K]}


class MojoHLR:
    def __init__(self, packed):
        self.lib = ctypes.CDLL(str(LIB))
        self.lib.hlr_logp_grad.restype = ctypes.c_int32
        self.lib.hlr_dim.restype = ctypes.c_int32
        self.data = np.ascontiguousarray(packed)
        self.dp = ctypes.c_void_p(self.data.ctypes.data)
        self.dim = self.lib.hlr_dim(self.dp)

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=f64)
        g = np.empty(self.dim); lp = np.empty(1)
        rc = self.lib.hlr_logp_grad(self.dp, ctypes.c_void_p(x.ctypes.data),
                                    ctypes.c_void_p(g.ctypes.data), ctypes.c_void_p(lp.ctypes.data))
        return rc, lp[0], g
