"""Run the float32 Apple-GPU eight-schools kernel and compare to the float64 CPU Mojo reference.

Usage: python gpu/run_gpu_check.py [n ...]
"""
import pathlib, subprocess, sys, tempfile, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mojo_src")]
import eight_schools as es  # noqa: E402
from pytensor_mojo.eight_schools_ref import J, SIGMA, Y  # noqa: E402

D = J + 2
BIN = ROOT / "gpu" / "eight_schools_gpu"


def build():
    src = ROOT / "gpu" / "eight_schools_gpu.mojo"
    if not BIN.exists() or BIN.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["mojo", "build", str(src), "-o", str(BIN)], check=True, capture_output=True)


def cpu_ref(Q64):
    n = len(Q64)
    G = np.empty_like(Q64); L = np.empty(n)
    t = time.perf_counter()
    es.logp_dlogp_batch(Q64.ctypes.data, G.ctypes.data, L.ctypes.data, Y.ctypes.data, SIGMA.ctypes.data, J, n)
    return L, G, time.perf_counter() - t


def run(n, scale=1.0, seed=0):
    build()
    Q32 = (np.random.default_rng(seed).normal(size=(n, D)) * scale).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        Q32.tofile(d / "q"); 
        out = subprocess.run([str(BIN), str(d / "q"), str(d / "lp"), str(d / "g"), str(n)],
                             check=True, capture_output=True, text=True).stdout.strip()
        L32 = np.fromfile(d / "lp", dtype=np.float32)
        G32 = np.fromfile(d / "g", dtype=np.float32).reshape(n, D)
    # Reference evaluated at exactly the same (float32-rounded) points, in float64.
    L, G, cpu_s = cpu_ref(Q32.astype(np.float64))
    lp_rel = np.abs(L32 - L) / np.abs(L)
    g_rel = np.max(np.abs(G32 - G), axis=1) / np.maximum(np.max(np.abs(G), axis=1), 1e-300)
    return dict(n=n, scale=scale, gpu=out, cpu_ns_per_point=cpu_s / n * 1e9,
                lp_max_rel=float(lp_rel.max()), lp_median_rel=float(np.median(lp_rel)),
                g_max_normrel=float(g_rel.max()), finite=bool(np.isfinite(L32).all() and np.isfinite(G32).all()))


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [1_000, 1_000_000]
    for n in ns:
        for scale in (0.5, 1.0, 3.0):
            print(run(n, scale))
