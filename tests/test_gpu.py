"""Apple-GPU (Metal) float32 checks. Skipped when the GPU build/run is unavailable."""
import pathlib, subprocess, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu"))


def _gpu_ok():
    r = subprocess.run(["mojo", "run", str(ROOT / "gpu" / "vec_add.mojo")], capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


@pytest.fixture(scope="module")
def gpu():
    ok, out = _gpu_ok()
    if not ok:
        pytest.skip(f"Mojo GPU unavailable: {out[-300:]}")
    return out


def test_vec_add_on_gpu(gpu):
    assert "mismatches: 0 of 1048576" in gpu


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
def test_float32_gpu_eight_schools_close_to_float64_cpu(gpu, scale):
    import run_gpu_check
    r = run_gpu_check.run(20_000, scale=scale, seed=3)
    print(r)
    assert r["finite"]
    assert r["lp_max_rel"] < 1e-5
    assert r["g_max_normrel"] < 1e-5
