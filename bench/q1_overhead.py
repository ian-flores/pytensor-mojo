"""Q1: per-call overhead and throughput of Python -> Mojo calls vs NumPy/C-ext baselines."""
import sys, timeit, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mojo_src"))
import q1_mod  # noqa: E402


def t(stmt, number, g, repeat=7):
    return min(timeit.repeat(stmt, number=number, repeat=repeat, globals=g)) / number


def main():
    x = np.random.default_rng(0).normal(size=10)
    addr = x.ctypes.data
    def pyfunc(a):
        return a
    g = dict(q1_mod=q1_mod, x=x, addr=addr, np=np, pyfunc=pyfunc, math=__import__("math"))
    N = 200_000
    rows = [
        ("python def f(x): return x", t("pyfunc(x)", N, g)),
        ("C builtin math.sqrt(2.0)", t("math.sqrt(2.0)", N, g)),
        ("mojo noop0()", t("q1_mod.noop0()", N, g)),
        ("mojo noop1(x)", t("q1_mod.noop1(x)", N, g)),
        ("mojo first_ptr(addr) [raw int addr]", t("q1_mod.first_ptr(addr)", N, g)),
        ("mojo first_span(arr) [from_numpy_array]", t("q1_mod.first_span(x)", N, g)),
        ("python x.ctypes.data (addr lookup)", t("x.ctypes.data", N, g)),
        ("numpy x.sum() n=10", t("x.sum()", N, g)),
    ]
    print("## Per-call overhead (min of 7 repeats)")
    print("| call | ns/call |\n|---|---|")
    for name, s in rows:
        print(f"| {name} | {s*1e9:.0f} |")

    print("\n## Throughput: sum and axpy (out = a*x + y, in place)")
    print("| n | np.sum us | mojo sum_span us | mojo sum_ptr (SIMD) us | np axpy (out=) us | mojo axpy_span us | mojo axpy_ptr (SIMD) us |\n|---|---|---|---|---|---|---|")
    for n in [10, 1_000, 100_000, 10_000_000]:
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=n), rng.normal(size=n)
        out = np.empty(n)
        assert np.isclose(q1_mod.sum_span(a), a.sum(), rtol=1e-9)
        q1_mod.axpy_span(2.0, a, b, out)
        assert np.allclose(out, 2.0 * a + b)
        aa, ba, oa = a.ctypes.data, b.ctypes.data, out.ctypes.data
        assert np.isclose(q1_mod.sum_ptr(aa, n), a.sum(), rtol=1e-9)
        out[:] = 0; q1_mod.axpy_ptr(2.0, aa, ba, oa, n)
        assert np.allclose(out, 2.0 * a + b)
        def np_axpy():
            np.multiply(a, 2.0, out=out); np.add(out, b, out=out)
        g2 = dict(q1_mod=q1_mod, a=a, b=b, out=out, np_axpy=np_axpy, aa=aa, ba=ba, oa=oa, n=n)
        num = max(3, int(2e6 // max(n, 1)))
        r = [t("a.sum()", num, g2, 5), t("q1_mod.sum_span(a)", num, g2, 5), t("q1_mod.sum_ptr(aa, n)", num, g2, 5),
             t("np_axpy()", num, g2, 5), t("q1_mod.axpy_span(2.0, a, b, out)", num, g2, 5),
             t("q1_mod.axpy_ptr(2.0, aa, ba, oa, n)", num, g2, 5)]
        print(f"| {n:,} | " + " | ".join(f"{v*1e6:.2f}" for v in r) + " |")


if __name__ == "__main__":
    main()
