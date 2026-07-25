# Installation

`distfind` is a C++ core compiled into a single shared library
(`libdistfind.dylib` / `.so` / `.dll`), driven from Python through a thin `ctypes`
binding. Installation is two steps: **build the native library**, then **install the
Python package**.

## Requirements

- **CMake ≥ 3.20**
- A **C++17** compiler
- **Python ≥ 3.9** with **NumPy ≥ 1.20**
- **Optional GPU backends:**
    - **Metal** (Apple Silicon / macOS) — compiled in automatically on Apple platforms.
      The Metal kernel is *compiled at runtime* from embedded MSL source, so the offline
      `metal` command-line tool is **not** required.
    - **CUDA** — built automatically if a CUDA toolkit (`nvcc`) is found at configure
      time. If no toolkit is present, the CUDA backend is silently skipped.

The only Python runtime dependency is NumPy.

## Build the native library

From the repository root:

```bash
./build.sh
```

This configures and builds a Release library into
`python/qubitserf/distfind/_lib/libdistfind.dylib` (the location the Python package looks in).

### Apple-toolchain note

On macOS, `build.sh` pins the compiler to Apple's `clang++` via `xcrun -f clang++`. This
is deliberate: a Homebrew clang on your `PATH` typically lacks the macOS SDK and the
`Metal` / `Foundation` frameworks the Metal backend links against, so the build would fail
if it picked that up. The script handles this for you — you do not need to set anything.

### CUDA

The CUDA implementation is governed by the `DISTFIND_CUDA` CMake option (on by default).
At configure time CMake probes for a CUDA compiler; if found, `src/cuda/backend_cuda.cu`
is compiled and exposed through the public `gpu` backend, otherwise you'll see:

```
distfind: CUDA toolkit not found, skipping CUDA backend
```

and only CPU, plus any other available accelerator implementation, is built.

### Build options

`build.sh` forwards any extra arguments straight to CMake, so you can toggle features:

```bash
./build.sh -DDISTFIND_METAL=OFF      # CPU-only on macOS
./build.sh -DDISTFIND_CUDA=OFF       # skip the CUDA probe
./build.sh -DDISTFIND_TESTS=OFF      # don't build the C++ smoke tests
```

You can also point the build at a different directory with the `BUILD_DIR` environment
variable.

## Install the Python package

Once the library is built, install the package (a standard setuptools project):

```bash
pip install .
```

The native library built by `build.sh` is packaged as package data
(`python/qubitserf/distfind/_lib/*.dylib` / `*.so` / `*.dll`). **Build the native library first,
then `pip install`** — the wheel does not compile the C++ for you.

For an editable/dev install, add the package directory to `PYTHONPATH` instead:

```bash
PYTHONPATH=python python -c "import qubitserf.distfind as df; print(df.version())"
```

### Pointing at a library elsewhere

If your `libdistfind` lives outside `python/qubitserf/distfind/_lib`, set the `DISTFIND_LIB`
environment variable to its full path and the loader will use it directly.

## Verify the install

```python
from qubitserf import distfind as df

print(df.version())              # e.g. "0.1.0"
print(df.available_backends())   # e.g. ['cpu', 'gpu']
```

If `available_backends()` returns only `['cpu']` on a machine you expected to have a GPU,
re-check that the corresponding backend was enabled at build time (see the CMake messages
emitted by `build.sh`).
