# Changelog

All notable changes to **qminweight** are documented here.

## [Unreleased]

### Performance
- **GPU Brouwer–Zimmermann is now a median ~13x faster than the multicore CPU** on codes
  whose enumeration is the actual cost (CPU solve > 10 ms), up from ~1.1x. The dominant
  weight level on toric L=7 dropped 43.6 ms → 7.3 ms (6x) on an Apple M4. Root cause: the
  per-thread scratch arrays were sized to the worst case (`cw[16]`, `pos[32]`) and spilled
  into thread-local memory, collapsing GPU occupancy to ~1.5 %. Fix: the kernel is now
  compiled as a small VARIANT per `(stride, d-bucket)` — `stride` is a compile-time literal
  so the codeword loops unroll and `cw[stride]` lives in registers, and `pos[]` is sized to
  the smallest bucket holding `d`. Variants are compiled lazily and cached (Metal: runtime
  source templating; CUDA: `template<int STRIDE,int POSN>` + a dispatch switch). Also: a
  one-time GPU warmup dispatch, a persistent result buffer, and a lower default
  `QMINWEIGHT_GPU_MIN_WORK` (1<<18) now that the GPU pays off sooner. The CPU remains the
  correctness oracle; `tests/test_distance.py` adds a deep variant-path GPU==CPU check and
  `bench/gpu_vs_cpu.py` reports the >10 ms-code median with a 2x goal gate.

### Changed
- **Connected cluster** (`method="cc"`) now honours `verbose=True` (`-v` on the CLI), which
  previously only affected the `bz`/`mitm` paths. It prints, on stderr in the repo's
  `[cc <dZ|dX>] ...` style: a header (n / checks / logicals / threads), a per-weight-level
  line giving the converging lower bound (`no weight-N logical -> d>N`) or the hit
  (`FOUND weight-N logical`) with per-level timing, and a ~5 s in-level heartbeat
  (`seeds k/n`) so a long weight level is never silent.

## [0.1.0] — 2026-06-22

First release. Exact minimum-distance finding for CSS quantum codes and classical
linear codes, with GPU acceleration.

### Algorithms
- **Brouwer–Zimmermann** (`method="bz"`): information-set enumeration with converging
  upper/lower bounds, matroid-partition information sets, parallel random-information-set
  seeding, and even-weight rounding. Reports a rigorous `[lower, upper]` bracket when its
  bound is too weak to certify.
- **Connected cluster** (`method="cc"`): Tanner-graph frontier search for sparse CSS codes
  (LDPC / topological / bivariate-bicycle); certifies the gross code `[[144,12,12]]` in
  ~0.3 s. CPU, multithreaded over seed qubits.
- **Meet-in-the-Middle** (`method="mitm"`): coordinate-split exact cross-check (CPU).

### Backends
- CPU (multithreaded), **Metal** (Apple GPU, runtime-compiled shaders), and **CUDA**
  (built when a CUDA toolkit is present). Hybrid dispatch sends a weight level to the GPU
  only when its work amortizes launch latency (`QMINWEIGHT_GPU_MIN_WORK`).

### Interfaces
- Python API (`qminweight.css_distance`, `classical_distance`, `available_backends`) and
  code generators (`qminweight.codes`).
- `qminweight` command-line tool (Qubitserf-compatible Pauli-string stdin) and
  `python -m qminweight`.

### Packaging
- `pip`-installable via scikit-build-core (CMake + C++17; CUDA auto-detected).
