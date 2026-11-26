# Apple GPU smoke test: float32 vector add on the Metal device.
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.sys.info import CompilationTarget

comptime N = 1 << 20
comptime BLOCK = 256


def vadd(a: Pointer[Float32, MutAnyOrigin], b: Pointer[Float32, MutAnyOrigin], o: Pointer[Float32, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < Int(n):
        o[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def main() raises:
    comptime assert has_accelerator(), "no GPU accelerator"
    var ctx = DeviceContext()
    print("device:", ctx.name())
    var a = ctx.enqueue_create_buffer[DType.float32](N)
    var b = ctx.enqueue_create_buffer[DType.float32](N)
    var o = ctx.enqueue_create_buffer[DType.float32](N)
    with a.map_to_host() as h:
        for i in range(N):
            h[i] = Float32(i)
    with b.map_to_host() as h:
        for i in range(N):
            h[i] = Float32(2 * i) + 0.5
    ctx.enqueue_function[vadd](a.unsafe_ptr(), b.unsafe_ptr(), o.unsafe_ptr(), Int32(N), grid_dim=N // BLOCK, block_dim=BLOCK)
    ctx.synchronize()
    var bad = 0
    with o.map_to_host() as h:
        for i in range(N):
            if h[i] != Float32(3 * i) + 0.5:
                bad += 1
        print("o[0..3] =", h[0], h[1], h[2], h[3], " o[N-1] =", h[N - 1])
    print("mismatches:", bad, "of", N)
