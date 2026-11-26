#!/bin/bash
# Build all Mojo extension modules as shared libs (-O3 is the default for mojo build).
set -e
cd "$(dirname "$0")/mojo_src"
for f in *.mojo; do
  m="${f%.mojo}"
  mojo build "$f" --emit shared-lib -o "$m.so" 2>&1 | grep -v "built for newer macOS" || true
  test -f "$m.so"
done
echo "built: $(ls *.so | tr '\n' ' ')"
# C-ABI model libraries for the Rust sampler: mojo_capi/<name>_capi.mojo -> lib<name>.dylib
cd ../mojo_capi
for f in *_capi.mojo; do
  m="${f%_capi.mojo}"
  mojo build "$f" --emit shared-lib -o "lib$m.dylib" 2>&1 | grep -v "built for newer macOS" || true
  test -f "lib$m.dylib"
done
echo "built: $(ls *.dylib | tr '\n' ' ')"
