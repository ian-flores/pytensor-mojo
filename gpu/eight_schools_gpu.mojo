# Batched float32 eight-schools logp + grad on the Apple GPU (Metal).
# Usage: mojo run gpu/eight_schools_gpu.mojo <in_q.f32> <out_lp.f32> <out_g.f32> <n>
#   in_q:  n x 10 float32 row-major unconstrained points [mu, log_tau, theta_tilde x 8]
#   out_lp: n float32 logp; out_g: n x 10 float32 gradient
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.sys import has_accelerator, argv
from std.math import exp, log
from std.time import perf_counter_ns

comptime J = 8
comptime D = J + 2
comptime BLOCK = 256
comptime HALF_LOG_2PI = Float32(0.91893853320467274178)
comptime LOG_2_OVER_PI = Float32(-0.45158270528945486473)
comptime LOG5 = Float32(1.6094379124341003746)


def kernel(q: Pointer[Float32, MutAnyOrigin], lp_out: Pointer[Float32, MutAnyOrigin], g: Pointer[Float32, MutAnyOrigin], n: Int32):
    var p = Int(block_idx.x * block_dim.x + thread_idx.x)
    if p >= Int(n):
        return
    var y = SIMD[DType.float32, 8](28, 8, -3, 7, -1, 1, 18, 12)
    var sigma = SIMD[DType.float32, 8](15, 10, 16, 11, 9, 11, 10, 18)
    var base = p * D
    var mu = q[unsafe_offset=base]
    var s = q[unsafe_offset=base + 1]
    var tau = exp(s)
    var lp = Float32(-0.5) * (mu / 5) * (mu / 5) - LOG5 - HALF_LOG_2PI
    var g_mu = -mu / 25
    var zt = tau / 5
    lp += LOG_2_OVER_PI - LOG5 - log(Float32(1) + zt * zt) + s
    var g_tau = Float32(-2) * tau / (Float32(25) + tau * tau)
    comptime for j in range(J):
        var tt = q[unsafe_offset=base + 2 + j]
        var sj = sigma[j]
        var r = y[j] - mu - tau * tt
        var w = r / (sj * sj)
        lp += Float32(-0.5) * tt * tt - HALF_LOG_2PI
        lp += Float32(-0.5) * r * w - log(sj) - HALF_LOG_2PI
        g_mu += w
        g_tau += w * tt
        g[unsafe_offset=base + 2 + j] = -tt + tau * w
    g[unsafe_offset=base] = g_mu
    g[unsafe_offset=base + 1] = g_tau * tau + 1
    lp_out[unsafe_offset=p] = lp


def main() raises:
    comptime assert has_accelerator(), "no GPU accelerator"
    var args = argv()
    var n = Int(args[4])
    var ctx = DeviceContext()
    var qb = ctx.enqueue_create_buffer[DType.float32](n * D)
    var lb = ctx.enqueue_create_buffer[DType.float32](n)
    var gb = ctx.enqueue_create_buffer[DType.float32](n * D)
    var raw: List[UInt8]
    with open(String(args[1]), "r") as f:
        raw = f.read_bytes()
    if len(raw) != n * D * 4:
        raise Error("input size mismatch")
    with qb.map_to_host() as h:
        var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(n * D):
            h[i] = src[unsafe_offset=i]
    var grid = (n + BLOCK - 1) // BLOCK
    # warm-up launch (includes Metal pipeline compile), then timed launches
    ctx.enqueue_function[kernel](qb.unsafe_ptr(), lb.unsafe_ptr(), gb.unsafe_ptr(), Int32(n), grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()
    comptime REPS = 20
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        ctx.enqueue_function[kernel](qb.unsafe_ptr(), lb.unsafe_ptr(), gb.unsafe_ptr(), Int32(n), grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("device:", ctx.name(), " n:", n, " kernel ms/launch:", Float64(t1 - t0) / REPS / 1e6,
          " ns/point:", Float64(t1 - t0) / REPS / Float64(n))
    with lb.map_to_host() as h:
        with open(String(args[2]), "w") as f:
            f.write_bytes(Span(unsafe_ptr=h.unsafe_ptr().unsafe_bitcast[UInt8](), length=n * 4))
    with gb.map_to_host() as h:
        with open(String(args[3]), "w") as f:
            f.write_bytes(Span(unsafe_ptr=h.unsafe_ptr().unsafe_bitcast[UInt8](), length=n * D * 4))
