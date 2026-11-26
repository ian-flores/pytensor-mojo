# C-ABI export of the eight-schools logp + grad for non-Python callers (Rust NUTS).
# Common model ABI used by sampler/ (data may be NULL for models with baked-in data):
#   int32_t es8_dim(const double* data)
#   int32_t es8_logp_grad(const double* data, const double* x, double* grad, double* logp)
# Returns 0 on success, 1 if logp is not finite.
from std.math import log, pi, isfinite
from std.ffi import external_call

comptime J = 8
comptime HALF_LOG_2PI = 0.91893853320467274178032973640561764


@always_inline
def _exp(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


@export
def es8_dim(data: Pointer[Float64, MutAnyOrigin]) abi("C") -> Int32:
    return Int32(J + 2)


@export
def es8_logp_grad(
    data: Pointer[Float64, MutAnyOrigin],
    q: Pointer[Float64, MutAnyOrigin],
    g: Pointer[Float64, MutAnyOrigin],
    lp_out: Pointer[Float64, MutAnyOrigin],
) abi("C") -> Int32:
    var y = SIMD[DType.float64, 8](28, 8, -3, 7, -1, 1, 18, 12)
    var sigma = SIMD[DType.float64, 8](15, 10, 16, 11, 9, 11, 10, 18)
    comptime MU_SD = 5.0
    comptime BETA = 5.0
    var mu = q[unsafe_offset=0]
    var s = q[unsafe_offset=1]
    var tau = _exp(s)
    var lp = -0.5 * (mu / MU_SD) * (mu / MU_SD) - log(MU_SD) - HALF_LOG_2PI
    var g_mu = -mu / (MU_SD * MU_SD)
    var zt = tau / BETA
    lp += log(2.0 / pi) - log(BETA) - log(1.0 + zt * zt) + s
    var g_tau = -2.0 * tau / (BETA * BETA + tau * tau)
    comptime for j in range(J):
        var tt = q[unsafe_offset=2 + j]
        var sj = sigma[j]
        var r = y[j] - mu - tau * tt
        var w = r / (sj * sj)
        lp += -0.5 * tt * tt - HALF_LOG_2PI
        lp += -0.5 * r * w - log(sj) - HALF_LOG_2PI
        g_mu += w
        g_tau += w * tt
        g[unsafe_offset=2 + j] = -tt + tau * w
    g[unsafe_offset=0] = g_mu
    g[unsafe_offset=1] = g_tau * tau + 1.0
    lp_out[unsafe_offset=0] = lp
    return Int32(0) if isfinite(lp) else Int32(1)
