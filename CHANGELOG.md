# Changelog

All notable changes to **qubitserf** are documented here.

## [Unreleased]

### Changed
- **CSS min distance now interleaves the Z- and X-subproblems for *all three* methods**
  (`bz`, `mitm`, and `cc`) — previously only `cc` did. The Z- and X-searches advance one
  weight level at a time so both lower bounds rise together; a side stalling on a hard level
  no longer starves the other of any bound. Once one side determines the min, the other is
  **capped** (e.g. once dZ=2 is proven, X stops as soon as it has ruled out X<2 — no wasted
  proof of X>2). Verbose progress is tagged `Z-`/`X-`. The exact `min(dZ, dX)` is unchanged.
- **`--zx` computes both distances with an interleaved-but-UNCAPPED search**: each side runs
  to its own full proof (so finding dZ never stops the full dX being found), while both lower
  bounds still advance together. Reported as `dZ dX`.
- **`-o` and `--operator` are now identical**: both take the operator Pauli as a command-line
  argument (`-o PAULI` / `--operator PAULI`), with the stabiliser/gauge generators read from
  stdin. (Previously the C++ `-o` read the operator from the last stdin line.)
- **`method="cc"` on a non-CSS code is rejected** (raises `ValueError` / returns an error)
  instead of silently falling back to MITM — see *Added* below.
- Removed the undocumented/inconsistent `--json` flag from the CLI documentation.

### Added
- **General (non-CSS) stabilizer & subsystem codes** — qubitserf was CSS-only (separate
  `Hx`/`Hz`); it now also takes a **symplectic stabilizer matrix** `S` of shape `(m, 2n)` in
  `[z | x]` column order (row `r` is the Pauli with Z-support `S[r,:n]`, X-support `S[r,n:]`;
  a `Y` sets both) and computes the distance as the minimum **symplectic** weight (number of
  qubits touched) of an operator in the normalizer `C(S)` that is not a stabilizer. The
  reduction: `C(S) = nullspace(swap(S))` where `swap` exchanges the `[z]`/`[x]` halves, and
  the meet-in-the-middle logical detector is stored pre-swapped so the ordinary GF(2) product
  realizes the symplectic product (`⟨a,b⟩ = swap(a)·b`). New:
  - `stabilizer_distance(S, ...)` — non-CSS code distance.
  - `subsystem_stabilizer_distance(G, ...)` — **dressed** distance of a non-CSS subsystem
    code from its (possibly non-commuting) gauge generators `G`; the stabilizer center is the
    symplectic center `nullspace(Gram(G))·G` and the detector uses `C(G)` (`C(G)^⟂ =
    rowspace(G)`).
  - `pauli_operator_weight(G, operator, ...)` — min symplectic weight over the coset
    `operator + rowspace(G)` (the non-CSS analogue of `operator_weight`; `0` iff the operator
    is in the group). Returns a `PauliOpResult`.
  - C ABI: `qubitserf_stabilizer_distance`, `qubitserf_subsystem_stabilizer_distance`,
    `qubitserf_stabilizer_operator_weight`.
  - CLI: a Pauli-stabiliser code containing a `Y` or a row mixing X and Z is auto-detected as
    non-CSS and routed to the symplectic solver; `--symplectic FILE` reads a `(m, 2n)` `[z|x]`
    0/1 matrix directly; `--subsystem` + a non-CSS gauge input gives the dressed distance.

  **Which methods generalize.** **BZ** and **MITM** both work for non-CSS codes; **CC**
  does not.
  - **BZ (default)** uses the weight-doubling isometry `φ:(a|b) → (a|b|a⊕b)`, under which
    `wt_H(φ(v)) = 2·wt_s(v)` (each non-identity qubit adds exactly 2 to the Hamming weight).
    `φ` is `F₂`-linear and injective, so the symplectic-distance problem becomes an ordinary
    binary Hamming-distance problem on a length-`3n` code, solved by the existing BZ (and its
    GPU enumeration); the symplectic distance is half the binary distance. Every `φ`-image has
    even weight, so BZ's even-distance speedup applies for free. This matches the published
    state of the art (Sabater–Vera et al., *Fast Algorithms… Minimum Distance of Quantum
    Codes*, arXiv:2408.10743 — `SAVED_ISOMETRY`).
  - **MITM** enumerates by qubit-support with the three nonzero Paulis `Z/X/Y` per chosen
    qubit (matching the original qubitserf's "middle algorithm").
  - **CC** needs a sparse single-type CSS Tanner graph, which a general non-CSS code does not
    provide, so `method="cc"` on a non-CSS code is **rejected** (the library returns an error;
    the Python API raises `ValueError`) rather than silently substituting another method.

  The CSS fast paths are unchanged: a code whose every row is pure-X/pure-Z is split into
  `Hx/Hz` and solved by the existing BZ/CC/MITM CSS solvers. Verified against a pure-numpy
  brute-force oracle (`_reference.stabilizer_distance_bruteforce` /
  `dressed_stabilizer_distance_bruteforce` / `symplectic_coset_min_weight_bruteforce`), with
  BZ and MITM cross-checked against each other, and against the original qubitserf's
  `interface` binary (the `[[5,1,3]]` perfect code = 3, the `[[8,1,3]]` non-CSS test code = 3).
- **Operator weight** — the minimum weight of a Pauli operator *modulo the stabilizer (or
  gauge) group*, i.e. the minimum-weight coset leader. The Z-part is minimized over
  `z + rowspace(Gz)` and the X-part over `x + rowspace(Gx)`, independently. This carries over
  the feature and CLI shape from the **original qubitserf**, but with a **correctness fix**:
  the original matches MITM syndromes against `M = [Hz; X̄]`, which is a valid parity-check of
  the Z-stabilizer group *only when `Hz·Hzᵀ = 0`*. For codes whose stabilizers are not
  self-orthogonal (surface, toric, bivariate-bicycle) it returns wrong answers — e.g.
  feeding a single Z-stabilizer of the planar `surface(3)` `[[13,1,3]]` code returns `3`
  when the correct answer is `0` (a stabilizer is equivalent to identity → weight 0).
  This package instead reduces the coset-leader problem to the existing `DistProblem` (append
  the operator as an extra generator; a single linear detector separates the two cosets) and
  solves it with the existing **BZ** and **MITM** backends, exactly correct and verified
  against brute force. `method="cc"` falls back to `bz` (the coset's parity check is dense
  even when the generators are sparse). New API `operator_weight(...)` returning `OpResult`,
  and the CLI `-o`/`--operator` flag.
- **Subsystem CSS dressed distance** — the distance of a CSS *subsystem* (gauge) code, given
  its gauge generators. This is the **dressed** distance: the minimum weight of an operator
  that commutes with the stabilizer group (the center of the gauge group) but is *not* in
  the gauge group — distinct from the *bare* distance (which forbids dressing by gauge
  operators). The stabilizer center `Sx/Sz` is computed internally, and the problem maps onto
  the existing `build(Sx, Gz)` `DistProblem`, so **all three backends — BZ, CC, and MITM —
  apply**. CC works on the sparse stabilizer center, so it keeps its sparsity advantage on
  topological subsystem codes such as Bacon-Shor. A stabilizer code is the special case
  `gauge = stabilizers`, where the subsystem distance equals `css_distance`. New API
  `subsystem_css_distance(...)` and the CLI `--subsystem` flag (treat the input X/Z matrices
  as gauge generators).
- **`codes.bacon_shor(d)`** — gauge generators `(Gx, Gz)` for the distance-`d` Bacon-Shor
  subsystem code (`n = d²`, dressed distance `d`).
- **API**: `operator_weight`, `subsystem_css_distance`, and the `OpResult` dataclass
  (`z_weight`, `x_weight`, `weight = max(z, x)`, `proven`, `seconds`, `backend`), all
  exported from the top-level package.
- **CLI**: `--operator` (the last stdin Pauli line — may contain `Y` — is the operator,
  preceding lines are the generators; default output is `max(z, x)`, `--zx` prints `z x`) and
  `--subsystem` (treat the X/Z input as gauge generators and compute the dressed distance).
  Both compose with `--z`/`--x`/`--zx`, `--method`, `--threads`, and the other flags.

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
  `QUBITSERF_GPU_MIN_WORK` (1<<18) now that the GPU pays off sooner. The CPU remains the
  correctness oracle; `tests/test_distance.py` adds a deep variant-path GPU==CPU check and
  `bench/gpu_vs_cpu.py` reports the >10 ms-code median with a 2x goal gate.

### Changed
- **Verbose output now matches the original qubitserf's progress format** across all three
  methods (`cc`/`bz`/`mitm`). Each ruled-out weight level prints `Distance bound: >N` followed by
  `Elapsed:[<level time>]`, and the result prints `Distance: =N` followed by
  `Elapsed:[<total time>]` — milliseconds under one second (`[15ms]`), seconds with three
  decimals above (`[1.087s]`). The previous per-backend `[cc <dZ|dX>] ...` lines and the
  in-level heartbeat were replaced by this shared format (`include/qubitserf/progress.hpp`).
- **Connected cluster** (`method="cc"`) now honours `verbose=True` (`-v` on the CLI), which
  previously only affected the `bz`/`mitm` paths.
- **Connected-cluster CSS distance now interleaves the Z- and X-distance searches** weight
  level by weight level, instead of running one to completion before the other. Both lower
  bounds advance together, so a side that stalls on a hard level no longer starves the other
  of any bound; verbose lines are labelled `Z-`/`X-` to keep the two streams distinct. Once a
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
  only when its work amortizes launch latency (`QUBITSERF_GPU_MIN_WORK`).

### Interfaces
- Python API (`qubitserf.css_distance`, `classical_distance`, `available_backends`) and
  code generators (`qubitserf.codes`).
- `qubitserf` command-line tool (Pauli-string stdin compatible with the original qubitserf)
  and `python -m qubitserf`.

### Packaging
- `pip`-installable via scikit-build-core (CMake + C++17; CUDA auto-detected).
