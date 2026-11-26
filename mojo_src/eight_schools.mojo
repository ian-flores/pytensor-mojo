# Q2: hand-translated eight-schools (non-centered) logp and gradient.
#
# Unconstrained vector q = [mu, log_tau, theta_tilde_0 .. theta_tilde_{J-1}].
#   mu          ~ Normal(0, 5)
#   tau         ~ HalfCauchy(5), sampled as log_tau with Jacobian +log_tau
#   theta_tilde ~ Normal(0, 1)
#   y_j         ~ Normal(mu + tau * theta_tilde_j, sigma_j)
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.math import log, pi
from std.ffi import external_call

comptime HALF_LOG_2PI = 0.91893853320467274178032973640561764


@always_inline
def _exp(x: Float64) -> Float64:
    # std.math.exp on Float64 has ~1e-12 relative error in Mojo 1.2 (see README);
    # call the platform libm instead, which matches NumPy bit-for-bit.
    return external_call["exp", Float64](x)


@export
def PyInit_eight_schools() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("eight_schools")
        m.def_function[logp_dlogp]("logp_dlogp")
        m.def_function[logp_dlogp_batch]("logp_dlogp_batch")
        return m.finalize()
    except e:
        abort(String("error creating eight_schools:", e))


@always_inline
def _logp_dlogp(
    q: Pointer[Float64, MutAnyOrigin],
    g: Pointer[Float64, MutAnyOrigin],
    y: Pointer[Float64, MutAnyOrigin],
    sigma: Pointer[Float64, MutAnyOrigin],
    J: Int,
) -> Float64:
    comptime MU_SD = 5.0
    comptime BETA = 5.0
    var mu = q[unsafe_offset=0]
    var s = q[unsafe_offset=1]
    var tau = _exp(s)

    # mu ~ Normal(0, 5)
    var lp = -0.5 * (mu / MU_SD) * (mu / MU_SD) - log(MU_SD) - HALF_LOG_2PI
    var g_mu = -mu / (MU_SD * MU_SD)

    # tau ~ HalfCauchy(5): log(2/pi) - log(beta) - log1p((tau/beta)^2), plus Jacobian s
    var zt = tau / BETA
    lp += log(2.0 / pi) - log(BETA) - log(1.0 + zt * zt) + s
    var g_tau = -2.0 * tau / (BETA * BETA + tau * tau)

    for j in range(J):
        var tt = q[unsafe_offset=2 + j]
        var sj = sigma[unsafe_offset=j]
        var r = y[unsafe_offset=j] - mu - tau * tt
        var w = r / (sj * sj)
        # theta_tilde ~ Normal(0, 1)
        lp += -0.5 * tt * tt - HALF_LOG_2PI
        # y ~ Normal(mu + tau * tt, sigma)
        lp += -0.5 * r * w - log(sj) - HALF_LOG_2PI
        g_mu += w
        g_tau += w * tt
        g[unsafe_offset=2 + j] = -tt + tau * w

    g[unsafe_offset=0] = g_mu
    # chain rule through tau = exp(s), plus d(Jacobian)/ds = 1
    g[unsafe_offset=1] = g_tau * tau + 1.0
    return lp


def logp_dlogp(qa: PythonObject, ga: PythonObject, ya: PythonObject, sa: PythonObject, j_obj: PythonObject) raises -> PythonObject:
    # Raw buffer addresses (Python ints) for q (in), grad (out), y, sigma.
    return PythonObject(_logp_dlogp(
        qa.unsafe_get_as_pointer[DType.float64](),
        ga.unsafe_get_as_pointer[DType.float64](),
        ya.unsafe_get_as_pointer[DType.float64](),
        sa.unsafe_get_as_pointer[DType.float64](),
        Int(py=j_obj),
    ))


def logp_dlogp_batch(qa: PythonObject, ga: PythonObject, la: PythonObject, ya: PythonObject, sa: PythonObject, j_obj: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    # Evaluate n points in one call: q is (n, J+2) row-major, g same shape, logp out is (n,).
    var q = qa.unsafe_get_as_pointer[DType.float64]()
    var g = ga.unsafe_get_as_pointer[DType.float64]()
    var lpo = la.unsafe_get_as_pointer[DType.float64]()
    var y = ya.unsafe_get_as_pointer[DType.float64]()
    var sg = sa.unsafe_get_as_pointer[DType.float64]()
    var J = Int(py=j_obj)
    var n = Int(py=n_obj)
    var d = J + 2
    for i in range(n):
        lpo[unsafe_offset=i] = _logp_dlogp(q.unsafe_offset(i * d), g.unsafe_offset(i * d), y, sg, J)
    return PythonObject(None)
