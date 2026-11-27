# Contributing

Thanks for your interest. This is an experimental research prototype; issues and pull requests
are welcome, especially new ops, models that fail to compile, and accuracy reports.

## Development setup

Requirements: macOS on Apple Silicon, [pixi](https://pixi.sh) and a Rust toolchain.

```bash
pixi install
pixi run build            # baseline Mojo kernels
pixi run build-sampler    # Rust nuts-rs driver
pixi run test             # full pytest suite (about 90 s)
pixi run test-rust        # cargo fmt --check, clippy -D warnings, cargo test
pixi run -e docs docs-serve 8000          # documentation site on 127.0.0.1:8000
```

## Rules

- **Correctness first.** Generated logp and gradient must match PyMC's `compile_logp` /
  `compile_dlogp` at **rtol 1e-10** on at least 20 random points. Loosen a tolerance only with
  a written justification next to it in the test.
- **Evidence for claims.** Speed claims need a benchmark from `bench/` with the median of 5
  runs. Accuracy claims need a test against a high-precision or correctly rounded reference.
- **Comments.** No non-ASCII characters in code comments.
- **Rust checks.** `cargo fmt` and `cargo clippy -- -D warnings` must pass.
- **Generated docs.** After changing ops, the census or benchmark results, run
  `pixi run docs-assets`. `tests/test_docs_generated.py` fails on stale tables.

## Adding an op

1. **Find it.** Run `missing_ops` on the graph that fails (see the
   [troubleshooting guide](https://ian-flores.github.io/pytensor-mojo/guide/troubleshooting/)), or `pixi run census-phase4`.
2. **Scalar op.** Add an entry to `_sop_table` in `pytensor_mojo/ndcodegen.py`, mapping the
   PyTensor scalar op class to a Mojo expression template over `SIMD[DType.float64, w]`. Prefer
   libm through `external_call` where it is correctly rounded. Otherwise implement it in the
   generated header with a documented algorithm.
3. **Tensor op.** Add a branch to `Codegen.node`, add the class to `_TENSOR_OK` (used by
   `missing_ops`), and add a note in `TENSOR_NOTES` in `docs/gen/build_assets.py`.
4. **Tests:**
   - a unit test against the C backend through the `MOJO_ND` mode;
   - for special functions, an accuracy row in `bench/phase4_accuracy.py` with a justified
     limit in `tests/test_special.py`, plus a gradient test;
   - a model-level test against PyMC at rtol 1e-10.
5. **Docs.** Run `pixi run docs-assets` and update `CHANGELOG.md`.

## Pull requests

- **Scope.** Keep each PR to one feature.
- **Description.** Include the test output and, for performance changes, the benchmark table.
