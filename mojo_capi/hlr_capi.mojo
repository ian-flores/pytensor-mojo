# Hierarchical linear regression, centered (about 500 rows per group, so centered mixes well),
# hand-written with SIMD, C ABI.
#
#   mu_a    ~ Normal(0, 5)
#   sigma_a ~ HalfNormal(2.5)      sampled as log(sigma_a), Jacobian +log
#   a_g     ~ Normal(mu_a, sigma_a)                                    (g = 0..G-1)
#   beta_k  ~ Normal(0, 2)                                             (k = 0..K-1)
#   sigma   ~ HalfNormal(1)        sampled as log(sigma), Jacobian +log
#   y_i     ~ Normal(a_{g(i)} + sum_k X_ik beta_k, sigma)
#
# Unconstrained x = [mu_a, log_sigma_a, a (G), beta (K), log_sigma]; dim = G + K + 3.
#
# data layout (float64): [N, K, G, offsets (G+1), X column-major (K*N), y (N)]
# Rows are sorted by group; offsets[g]..offsets[g+1] are the rows of group g.
#
#   int32_t hlr_dim(const double* data)
#   int32_t hlr_logp_grad(const double* data, const double* x, double* grad, double* logp)
from std.math import log, isfinite
from std.ffi import external_call
from std.collections import Array

comptime W = 8
comptime HALF_LOG_2PI = 0.91893853320467274178032973640561764
comptime LOG_SQRT_2_OVER_PI = -0.22579135264472743236
comptime MAXK = 16


@always_inline
def _exp(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


@export
def hlr_dim(data: Pointer[Float64, MutAnyOrigin]) abi("C") -> Int32:
    var K = Int(data[unsafe_offset=1])
    var G = Int(data[unsafe_offset=2])
    if K > MAXK:
        return Int32(-1)
    return Int32(G + K + 3)


@export
def hlr_logp_grad(
    data: Pointer[Float64, MutAnyOrigin],
    x: Pointer[Float64, MutAnyOrigin],
    grad: Pointer[Float64, MutAnyOrigin],
    lp_out: Pointer[Float64, MutAnyOrigin],
) abi("C") -> Int32:
    var N = Int(data[unsafe_offset=0])
    var K = Int(data[unsafe_offset=1])
    var G = Int(data[unsafe_offset=2])
    var offs = data.unsafe_offset(3)
    var X = data.unsafe_offset(3 + G + 1)
    var y = X.unsafe_offset(K * N)

    var mu_a = x[unsafe_offset=0]
    var ls_a = x[unsafe_offset=1]
    var sigma_a = _exp(ls_a)
    var a = x.unsafe_offset(2)
    var inv_sa2 = 1.0 / (sigma_a * sigma_a)
    var beta = x.unsafe_offset(2 + G)
    var ls = x[unsafe_offset=2 + G + K]
    var sigma = _exp(ls)
    var inv_s2 = 1.0 / (sigma * sigma)

    # priors
    var lp = -0.5 * (mu_a / 5.0) * (mu_a / 5.0) - log(5.0) - HALF_LOG_2PI
    var g_mu_a = -mu_a / 25.0
    lp += LOG_SQRT_2_OVER_PI - log(2.5) - 0.5 * (sigma_a / 2.5) * (sigma_a / 2.5) + ls_a
    var g_ls_a = -(sigma_a * sigma_a) / 6.25 + 1.0
    lp += LOG_SQRT_2_OVER_PI - 0.5 * sigma * sigma + ls
    var g_ls = -(sigma * sigma) + 1.0
    for k in range(K):
        var b = beta[unsafe_offset=k]
        lp += -0.5 * (b / 2.0) * (b / 2.0) - log(2.0) - HALF_LOG_2PI
        grad[unsafe_offset=2 + G + k] = -b / 4.0

    # likelihood: accumulate sum r^2, per-group sum r, per-k sum r * X_k
    var gk = Array[SIMD[DType.float64, W], MAXK](fill=SIMD[DType.float64, W](0))
    var bk = Array[Float64, MAXK](fill=0.0)
    for k in range(K):
        bk[k] = beta[unsafe_offset=k]
    var rr = SIMD[DType.float64, W](0)
    var g_mu_acc = 0.0
    var g_ls_a_acc = 0.0
    for g in range(G):
        var a_g = a[unsafe_offset=g]
        var lo = Int(offs[unsafe_offset=g])
        var hi = Int(offs[unsafe_offset=g + 1])
        var rsum = SIMD[DType.float64, W](0)
        var i = lo
        while i + W <= hi:
            var eta = SIMD[DType.float64, W](a_g)
            for k in range(K):
                eta += X.unsafe_offset(k * N + i).unsafe_load[width=W]() * bk[k]
            var r = y.unsafe_offset(i).unsafe_load[width=W]() - eta
            rr += r * r
            rsum += r
            for k in range(K):
                gk[k] += r * X.unsafe_offset(k * N + i).unsafe_load[width=W]()
            i += W
        var rs = rsum.reduce_add()
        while i < hi:
            var eta = a_g
            for k in range(K):
                eta += X[unsafe_offset=k * N + i] * bk[k]
            var r = y[unsafe_offset=i] - eta
            rr[0] += r * r
            rs += r
            for k in range(K):
                gk[k][0] += r * X[unsafe_offset=k * N + i]
            i += 1
        # a_g ~ Normal(mu_a, sigma_a)
        var z = (a_g - mu_a) * inv_sa2
        lp += -0.5 * (a_g - mu_a) * z - ls_a - HALF_LOG_2PI
        g_mu_acc += z
        g_ls_a_acc += (a_g - mu_a) * z - 1.0
        grad[unsafe_offset=2 + g] = rs * inv_s2 - z

    var rrs = rr.reduce_add()
    lp += -0.5 * rrs * inv_s2 - Float64(N) * (ls + HALF_LOG_2PI)
    g_ls += rrs * inv_s2 - Float64(N)
    for k in range(K):
        grad[unsafe_offset=2 + G + k] += gk[k].reduce_add() * inv_s2
    grad[unsafe_offset=0] = g_mu_a + g_mu_acc
    grad[unsafe_offset=1] = g_ls_a + g_ls_a_acc
    grad[unsafe_offset=2 + G + K] = g_ls
    lp_out[unsafe_offset=0] = lp
    return Int32(0) if isfinite(lp) else Int32(1)
