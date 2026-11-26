"""Q2: time Mojo eight-schools logp+grad vs PyMC/PyTensor compiled (C and Numba)."""
import pathlib, sys, timeit
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mojo_src")]
import eight_schools as es  # noqa: E402
from pytensor_mojo.eight_schools_ref import J, SIGMA, Y, build_model, compile_flat  # noqa: E402


def best(fn, number, repeat=7):
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number


def main():
    model = build_model()
    q = np.random.default_rng(0).normal(size=J + 2)
    g = np.empty_like(q)
    qa, ga, ya, sa = q.ctypes.data, g.ctypes.data, Y.ctypes.data, SIGMA.ctypes.data
    N = 100_000

    f_c = compile_flat(model, mode=None)          # default: C backend (CVM)
    f_nb = compile_flat(model, mode="NUMBA")
    f_c(q); f_nb(q)
    nb_jit = f_nb.vm.jit_fn                       # raw numba-jitted graph function
    nb_jit(q)

    rows = [
        ("PyTensor C backend, f(q) trust_input", best(lambda: f_c(q), N // 5)),
        ("PyTensor Numba backend, f(q) trust_input", best(lambda: f_nb(q), N // 5)),
        ("Numba jitted graph fn called directly (f.vm.jit_fn)", best(lambda: nb_jit(q), N)),
        ("Mojo logp_dlogp(addr...) from Python", best(lambda: es.logp_dlogp(qa, ga, ya, sa, J), N)),
    ]
    # Toy generated backend (pytensor_mojo.linker), on the value_vars signature.
    import pytensor, pytensor.tensor as pt
    from pytensor_mojo.linker import MOJO
    lp = model.logp(jacobian=True); outs = [lp, *pt.grad(lp, model.value_vars)]
    vv = [np.array(q[0]), np.array(q[1]), q[2:].copy()]
    f_gen = pytensor.function(model.value_vars, outs, mode=MOJO); f_gen.trust_input = True
    f_c3 = pytensor.function(model.value_vars, outs); f_c3.trust_input = True
    f_gen(*vv); f_c3(*vv)
    jf = f_gen.vm.jit_fn
    tab_addr = jf._keepalive[1].ctypes.data
    rows += [
        ("PyTensor C backend, f(mu, tau_log__, theta_tilde)", best(lambda: f_c3(*vv), N // 5)),
        ("Generated Mojo (MojoLinker), same signature, via pytensor.function", best(lambda: f_gen(*vv), N // 5)),
        ("Generated Mojo, raw run(tab_addr) (buffers pre-filled)", best(lambda: jf.mojo_module.run(tab_addr), N)),
    ]
    # compute-only: amortize the Python call over many points
    n = 100_000
    Q = np.ascontiguousarray(np.random.default_rng(1).normal(size=(n, J + 2)))
    G = np.empty_like(Q); L = np.empty(n)
    args = (Q.ctypes.data, G.ctypes.data, L.ctypes.data, ya, sa, J, n)
    rows.append(("Mojo batch, per point (compute only)", best(lambda: es.logp_dlogp_batch(*args), 3, 5) / n))
    print("## eight-schools logp + grad (d=10), min of repeats")
    print("| implementation | us/eval | speedup vs C backend |\n|---|---|---|")
    base = rows[0][1]
    for name, s in rows:
        print(f"| {name} | {s*1e6:.3f} | {base/s:.1f}x |")


if __name__ == "__main__":
    main()
