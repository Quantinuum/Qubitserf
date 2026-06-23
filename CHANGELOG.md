# Changelog

All notable changes to **qminweight** are documented here.

## [Unreleased]

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
