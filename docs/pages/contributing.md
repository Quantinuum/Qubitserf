# Contributing

## Repository layout

```
include/qminweight/   public C++ headers
                    (bits, gf2, combinatorics, css, backend, bz, cc, mitm, capi)
src/                core (gf2, css, bz, cc, mitm), backend_cpu, capi
src/metal/          Metal GPU backend (runtime-compiled MSL kernel)
src/cuda/           CUDA GPU backend (built only when a CUDA toolkit is present)
src/tests/          standalone C++ test executables (smoke, backend_compare, mitm_smoke)
python/qminweight/    ctypes wrapper (_native), high-level API (api), code generators (codes)
tests/              pytest validation suite
bench/              benchmarks vs the reference implementations
docs/               this documentation
```

The native core is pure C++17 with no third-party dependencies; the GPU backends are
optional and selected at configure time. Python talks to the compiled
`libqminweight.{dylib,so,dll}` through `ctypes` — there is no compiled extension module, so
the Python layer stays trivially portable.

## Building

See [Installation](installation.md) for the full story. The short version, from the repo
root:

```bash
./build.sh
```

This pins Apple's `clang++` on macOS (so the Metal frameworks resolve), builds a Release
library into `python/qminweight/_lib/`, and — unless you pass `-DQMINWEIGHT_TESTS=OFF` — also
builds the C++ test executables. Extra arguments are forwarded to CMake:

```bash
./build.sh -DQMINWEIGHT_METAL=OFF      # CPU-only on macOS
./build.sh -DQMINWEIGHT_CUDA=OFF       # skip the CUDA probe
```

## Running the tests

### Python (pytest)

The pytest suite is the source of truth for correctness. It validates distances against
codes with **known parameters** (Steane `[[7,1,3]]`, Shor `[[9,1,3]]`, toric `[[2L²,2,L]]`,
surface, repetition, Hamming), checks that all backends and all three algorithms agree, and
includes a regression guard for the GPU stale-device-buffer determinism bug. Run it with
the package on `PYTHONPATH`:

```bash
PYTHONPATH=python python -m pytest tests/ -q
```

The GPU tests skip cleanly on machines without an available accelerator, so the suite
passes everywhere.

Key cross-checks the suite enforces:

- BZ on the CPU equals the known distance for every code, and is marked `proven`.
- `min == min(Z, X)` for CSS codes.
- `mitm` agrees with `bz`; `cc` agrees with `bz`.
- `cc` certifies bivariate-bicycle codes (the gross code `[[144,12,12]]` -> `d=12`,
  `proven`) where BZ cannot.
- GPU agrees with the CPU **deterministically** across repeated runs.

### C++ smoke tests

When built (the default), the standalone test executables land in the build directory and
run without Python:

```bash
./build/qminweight_smoke      # BZ pipeline on codes with known distance (classical + CSS)
./build/qminweight_compare    # backend agreement (CPU vs GPU)
```

`qminweight_smoke` checks BZ distances against hard-coded known values (repetition, Hamming,
Steane, ...) and exits non-zero on any mismatch, so it's a fast sanity gate after a core
change.

## Code conventions

- **GF(2) representation.** Codewords are bit-packed into `uint64` words and operated on
  with hardware `popcount`; matrices are the `GF2Mat` type in `include/qminweight/gf2.hpp`.
  Keep this packing — it is what makes the enumeration fast.
- **Sparse vs dense.** `cc` must receive the *original sparse* `Hx`/`Hz`; do not row-reduce
  before handing matrices to connected cluster. `bz`/`mitm` build their own information-set
  / parity-check representations internally.
- **The C boundary.** All language interop goes through `src/capi.cpp` / the C API; the
  Python `_native.py` mirrors the `QMinWeightResult` struct and the function signatures
  exactly. If you change the C ABI, update both sides in the same change.
- **Determinism.** The CPU backend is the deterministic ground truth and every other
  backend/algorithm is validated against it. Any GPU change must keep the
  determinism/agreement tests green.
- **Three exact methods, one answer.** `bz`, `cc`, and `mitm` must all return the same
  distance on any code where each is applicable; new functionality should preserve that
  invariant (the suite and the differential fuzzer check it).

## Benchmarks

The `bench/` scripts compare against the `codeDistance` reference package. See
[Benchmarks](benchmarks.md) for what each script measures and how to run it. If you change
an algorithm, regenerate the relevant `bench/*.md` so the documented numbers stay honest.
