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
- **Verbose output now imitates the qubitserf interface** across all three methods
  (`cc`/`bz`/`mitm`). Each ruled-out weight level prints `Distance bound: >N` followed by
  `Elapsed:[<level time>]`, and the result prints `Distance: =N` followed by
  `Elapsed:[<total time>]` — milliseconds under one second (`[15ms]`), seconds with three
  decimals above (`[1.087s]`). The previous per-backend `[cc <dZ|dX>] ...` lines and the
  in-level heartbeat were replaced by this shared format (`include/qminweight/progress.hpp`).
- **Connected cluster** (`method="cc"`) now honours `verbose=True` (`-v` on the CLI), which
  previously only affected the `bz`/`mitm` paths.
- **Connected-cluster CSS distance now interleaves the Z- and X-distance searches** weight
  level by weight level, instead of running one to completion before the other. Both lower
  bounds advance together, so a side that stalls on a hard level no longer starves the other
  of any bound; verbose lines are tagged `[Z]`/`[X]` to keep the two streams distinct. Once a
  side is found, the other's remaining search is capped at that weight (a heavier codeword
  can't lower the min), and the result is still the exact `min(d_Z, d_X)`.

### Benchmarks
- **Comprehensive benchmark extended to n ≤ 288** (`bench/comprehensive.py`): toric and
  surface families now sweep L = 4..12, plus a `bb [[288,12,12]]` bivariate-bicycle code and
  an `hgp(ham4)` (n=241) hypergraph-product code. `cc` certifies all of these in single- to
  low-hundreds of milliseconds while every other method (and the reference) gives out.
- **Warm-robust timing**: a measurement that finishes under `BENCH_REPEAT_BELOW` (1 s) is
  re-run `BENCH_REPEAT` (3) times and the minimum kept. This discards the one-off per-`(stride,d)`
  GPU kernel JIT/dispatch warmup that previously made a smaller code look slower than a larger
  one; expensive enumeration-dominated runs stay single shot.
- **BZ is attempted for n ≤ `BZ_MAX_N` (1024)** — the native GPU ceiling (see *Verified*
  below). Large *sparse* QLDPC codes within this window (toric/surface L ≥ 9, `bb288`) are
  listed in `HARD_CSS_NAMES` so they run a bounded `max_weight` cap rather than a free,
  potentially non-terminating BZ enumeration. (The in-process budget runs the native solver
  on a daemon thread it cannot cancel, so an uncapped BZ that exceeds the budget keeps burning
  a CPU core in the background; capping bounds the work so the thread always finishes.) `cc`
  and the reference still run on every code.
- **Reed–Muller families added** (`reed_muller_r1`, `reed_muller_r2`): dense, non-QLDPC
  quantum Reed–Muller CSS codes `Hx = Hz = G_RM(r,m)` (`codes.quantum_reed_muller`). These
  make the BZ-vs-CC crossover visible: on `qrm(2,6) [[64,20,8]]`, CC takes ~98–136 s (its
  connected-cluster search degenerates to `O(C(n,d))` because every check is dense) while BZ
  finishes in ~18 ms; `qrm(2,7) [[128,70,8]]` times CC out entirely while BZ stays < 1 s.
- Family plots reverted to **one chart per family** (time vs n, all distances together); the
  per-distance grid and `*_vs_d` charts were removed.

### Verified
- **Brouwer–Zimmermann certified correct on both backends up to 1024 qubits.** The codeword
  representation is bit-packed into `ceil(n/64)` words with every host routine sized
  dynamically, so the CPU solver handles arbitrary `n`; the GPU kernels are templated/variant-
  compiled over codeword stride 1..16 words, running natively up to **n = 1024** (stride 16)
  and falling back to the CPU above that for the identical result. `tests/backend_compare`
  now asserts CPU == GPU == 4 on `qrm(1,9) [[512,492,4]]` (stride 8) and
  `qrm(1,10) [[1024,1002,4]]` (stride 16); n = 1058 (stride 17) was confirmed to fall back to
  the CPU and agree. No change to the n < 256 hot path (stride is, and always was, a runtime
  parameter).

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
