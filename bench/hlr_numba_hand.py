"""Control: the same hand-fused HLR logp+grad algorithm as mojo_capi/hlr_capi.mojo, in numba.

Separates "Mojo vs numba" from "hand-written fused kernel vs PyTensor-generated graph".
"""
import pathlib, sys, timeit
import numba as nb
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pytensor_mojo import hlr_ref  # noqa: E402

HL2P = 0.91893853320467274178
LS2P = -0.22579135264472743236


@nb.njit(fastmath=False, cache=True)
def logp_grad(x, grad, Xc, y, offs, G, K):
    N = y.shape[0]
    mu_a, ls_a = x[0], x[1]
    sa = np.exp(ls_a); isa2 = 1.0 / (sa * sa)
    ls = x[2 + G + K]; s = np.exp(ls); is2 = 1.0 / (s * s)
    lp = -0.5 * (mu_a / 5) ** 2 - np.log(5.0) - HL2P
    g_mu = -mu_a / 25
    lp += LS2P - np.log(2.5) - 0.5 * (sa / 2.5) ** 2 + ls_a
    g_lsa = -(sa * sa) / 6.25 + 1
    lp += LS2P - 0.5 * s * s + ls
    g_ls = -(s * s) + 1
    for k in range(K):
        b = x[2 + G + k]
        lp += -0.5 * (b / 2) ** 2 - np.log(2.0) - HL2P
        grad[2 + G + k] = -b / 4
    gk = np.zeros(K)
    rr = 0.0
    for g in range(G):
        ag = x[2 + g]
        rs = 0.0
        for i in range(offs[g], offs[g + 1]):
            eta = ag
            for k in range(K):
                eta += Xc[k, i] * x[2 + G + k]
            r = y[i] - eta
            rr += r * r
            rs += r
            for k in range(K):
                gk[k] += r * Xc[k, i]
        z = (ag - mu_a) * isa2
        lp += -0.5 * (ag - mu_a) * z - ls_a - HL2P
        g_mu += z
        g_lsa += (ag - mu_a) * z - 1
        grad[2 + g] = rs * is2 - z
    lp += -0.5 * rr * is2 - N * (ls + HL2P)
    g_ls += rr * is2 - N
    for k in range(K):
        grad[2 + G + k] += gk[k] * is2
    grad[0] = g_mu; grad[1] = g_lsa; grad[2 + G + K] = g_ls
    return lp


def main():
    d = hlr_ref.make_data()
    Xc = np.ascontiguousarray(d["X"].T); y = d["y"]; offs = d["offsets"].astype(np.int64)
    G, K = d["G"], d["K"]
    mj = hlr_ref.MojoHLR(hlr_ref.pack(d))
    x = np.random.default_rng(0).normal(scale=0.5, size=G + K + 3)
    g = np.empty_like(x)
    lp = logp_grad(x, g, Xc, y, offs, G, K)
    _, lp_m, g_m = mj(x)
    assert abs(lp - lp_m) / abs(lp_m) < 1e-12 and np.allclose(g, g_m, rtol=1e-10)
    x1 = np.full_like(x, 0.1)
    t = min(timeit.repeat(lambda: logp_grad(x1, g, Xc, y, offs, G, K), number=2000, repeat=5)) / 2000
    print(f"hand-written numba HLR logp+grad: {t*1e6:.2f} us/eval (matches Mojo to 1e-12)")


if __name__ == "__main__":
    main()
