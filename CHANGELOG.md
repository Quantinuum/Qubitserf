# Changelog

All notable changes to **qubitserf** are documented here.

## [Unreleased] — 2026-07-29

### distfind returns plain ints (breaking)
- **`max_weight` is gone** from every distfind entry point, from the `distfind_*` C ABI,
  from `BZOptions`, and from the native CLI's `--max-weight` flag. It capped different
  quantities per method — the BZ information-set enumeration level (bounded by the code
  dimension `K`), but the codeword weight for `cc`/`mitm` — and only `bz` returned a usable
  `[lower, upper]` bracket when the cap bit; `cc` and `mitm` returned `-1`.
- **`Result`, `OpResult` and `PauliOpResult` are deleted.** All seven entry points
  (`css_distance`, `classical_distance`, `subsystem_css_distance`, `stabilizer_distance`,
  `subsystem_stabilizer_distance`, `operator_weight`, `pauli_operator_weight`) now return a
  plain `int`. A distance of **`-1`** means undefined for the input (empty code, or no
  logical qubits) — the case that used to surface as `proven=False`.
- The dropped fields were carrying little: with no cap, every code tried across all three
  methods came back proven, so `proven` reduced to `distance != -1`. `lower_bound` was the
  raw BZ `outer` bound and could *exceed* the distance after an early stop (the `[8,1,8]`
  repetition code reported `distance=8, lower_bound=16`), so it was not the "certified
  lower bound (== distance when proven)" it was documented to be.
- `operator_weight` returns `max(z_weight, x_weight)`; the separate Z/X coset leaders are no
  longer public. `_native.operator_weight_raw` still reports them, and the operator-weight
  tests use it to keep cross-checking both against the brute-force oracle.
- Benchmarks under `bench/distfind/` no longer cap hard codes; they fall back to the
  existing per-method timeout budget. The committed results files predate this and are
  unchanged.

## [0.1.0-dev] — 2026-07-27

### Unified native core — one library, one enumeration kernel
- **Single native shared library.** The two per-engine libraries (`libdistfind` →
  `qubitserf/distfind/_lib`, `libcodeaut` → `qubitserf/codeaut/_lib`) are merged into one
  **`libqubitserf`** in `python/qubitserf/_lib/`, exposing both flat C ABIs unchanged
  (`distfind_*`, `codeaut_bz_*`, `qaut_leon_*`).
- **Shared C++ core** — `include/qsf/` (namespace `qsf`: bits, combinatorics, GF(2) dense
  matrix, `gf2span` basis, `EnumPlan` + a unified `Backend`) and `src/core/` (`gf2.cpp`,
  `backend_cpu.cpp`, `metal/backend_metal.mm`, `cuda/backend_cuda.cu`). **One two-level
  enumeration kernel serves both engines** via a sink policy: min-logical-weight mode
  (distfind `Backend::enumerate`) and low-weight-collect mode (codeaut `Backend::collect`),
  on CPU / CUDA / Metal alike. `include/distfind/` bits/combinatorics/gf2/backend headers
  are now thin forwards to `qsf`; `include/codeaut/` keeps only `capi.h`; `gf2span.hpp`
  moved to `include/qsf/`.
- **codeaut BZ inherits distfind's optimized kernels.** The codeaut CPU and Metal BZ paths
  now run the two-level incremental kernel instead of their previous naive enumeration:
  measured **~7x faster on CPU** on a heavy case (RM(3,8), m=93, p=6), and the Metal
  collect path is upgraded from the naive kernel. Exact-set parity verified CPU vs Metal
  and old vs new.

### Build & environment renames (legacy env vars still honoured)
- CMake options: `DISTFIND_METAL` / `DISTFIND_CUDA` / `CODEAUT_METAL` / `CODEAUT_CUDA` →
  **`QUBITSERF_METAL`** / **`QUBITSERF_CUDA`**; `DISTFIND_TESTS` → **`QUBITSERF_TESTS`**;
  `DISTFIND_CLI` → **`QUBITSERF_CLI`**.
- Library-path override: **`QUBITSERF_LIB`** (the legacy `DISTFIND_LIB` and
  `CODEAUT_LIB_PATH` are still honoured).
- The no-CMake dev fallback build now compiles `src/{core,distfind,codeaut}/*.cpp` into
  `python/qubitserf/_lib/libqubitserf.{dylib,so}` — both engines get the fallback
  (distfind previously had none).
- GPU tuning knobs unchanged: `DISTFIND_GPU_MIN_WORK` (min mode), `CODEAUT_GPU_MIN_WORK` +
  `CODEAUT_GPU_CAPACITY` (collect mode), `DISTFIND_TPT`.

### Python deduplication (no public API changes)
- Shared `qubitserf/_native.py` (library locate / load / self-build), `qubitserf/_interop.py`
  (the single CSS interop shim; the per-engine `_interop.py` are re-export shims), and
  `qubitserf/_constructions.py` (shared raw-matrix code constructions; the per-engine
  `codes.py` delegate to it). Where constructions genuinely differ — surface HGP vs
  rotated, toric HGP vs lattice — both variants are kept under distinct names.
- All public APIs unchanged; the full pytest suite passes (143 passed, 3 skipped).

## [Unreleased]

### Added — classical `method="bz"` engine
- **`classical_automorphisms` now takes the same `method="auto"|"leon"|"bz"` choice as
  `css_automorphisms`.** `"bz"` is a new classical engine (`codeaut.classical_bz`): certified
  Brouwer–Zimmermann low-weight classes of **both** `C` and `C^perp` (`Aut(C) = Aut(C^perp)`;
  the smaller certified incidence is solved — dimension alone is a bad BZ cost proxy, since an
  LDPC-style code keeps its low-weight words on the primal side while the low-dimensional dual
  basis is dense) feed a coloured coordinate↔codeword incidence solved with nauty/Traces, and
  every generator is
  GF(2)-re-verified to preserve `rowspace(genmat)` — exact at any `dim(C)` (Leon needs
  `2**dim`). Returns a new `codeaut.ClassicalAutResult` (`.order` int, `.generators`,
  `.group()`, `.dim`, `.weight_classes`, `.dualized`, `.method`, `.seconds`). `"auto"` runs
  Leon when `dim(C) ≤ max_dim`, else `bz`, falling through to the other engine on failure; all
  routes are exact and raise rather than approximate. New knobs: `budget`, `backend`,
  `max_threads`, `nauty_timeout`, `traces_timeout`.

### Removed — inexact automorphism entry points (breaking)
- **The `"partial"` CSS method is gone.** `css_automorphisms(..., method="partial")` (and its
  `"affine"` / `"structural"` / `"cyclic"` / `"poly"` / `"fast"` aliases) now raises `ValueError`,
  and `codeaut --method partial` is no longer accepted; `METHODS` is `("auto", "leon", "bz")`.
  The cyclic/affine and Tanner-graph structural subgroups it exposed remain as **internal** stages
  of the `"auto"` ladder, where they are only reached after the exact engines fail and anything
  they return is still flagged `complete=False`. Results for `auto` / `leon` / `bz` are unchanged.
- **The Tanner-graph automorphism wrappers are gone**: `codeaut.css_tanner_graph_automorphisms`
  and `codeaut.classical_tanner_automorphisms`, plus the `codeaut.parity_check_automorphism_group`
  / `codeaut.tanner_permutation_group` re-exports. They returned a basis-dependent *subgroup* of
  the true automorphism group, which is too easy to mistake for the group itself. The
  implementations stay in `codeaut.graphaut` for internal use by the `bz` incidence route.
- The easy interface is now three entry points: `classical_automorphisms`, `css_automorphisms`,
  `group_intersection`.
- **README rewritten** around the library's capabilities (distance, operator weight, automorphism
  groups) rather than its two-module layout, and trimmed to the public, exact surface.

### Merged codeaut into qubitserf
- **`qubitserf` is now a two-part package.** The former standalone `codeaut` project (automorphism
  groups of binary linear and CSS quantum codes) is merged in as the `qubitserf.codeaut`
  subpackage, and the original distance-finding code is renamed to `qubitserf.distfind`. The two
  libraries stay compartmentalized — separate Python subpackages, separate native shared libraries
  (`libdistfind`, `libcodeaut`), and lazy submodule imports so importing one never pulls in the
  other's dependencies.
- **Two CLIs.** The distance CLI is now **`distfind`** (was `qubitserf`); the automorphism CLI
  **`codeaut`** is unchanged. Both are console scripts and both work via
  `python -m qubitserf.distfind` / `python -m qubitserf.codeaut`.
- **Full `distfind` rename of the native layer.** The C++ `namespace qubitserf`, the
  `include/distfind/` headers, the `qubitserf_*` extern-C symbols, and the runtime env vars are
  renamed to `distfind` / `DISTFIND_*` (e.g. `QUBITSERF_LIB` -> `DISTFIND_LIB`,
  `QUBITSERF_GPU_MIN_WORK` -> `DISTFIND_GPU_MIN_WORK`).
- **Repository layout.** `src/distfind` + `src/codeaut`, `include/distfind` + `include/codeaut`,
  `python/qubitserf/{distfind,codeaut}`, `tests/{distfind,codeaut}`, `bench/{distfind,codeaut}`.
  A single `CMakeLists.txt` builds both libraries and `build.sh` drops each into its subpackage.
- **Benchmarks moved out of the README** into [`BENCHMARKS.md`](BENCHMARKS.md).

### Performance
- **GPU (Metal) BZ enumeration ~7x faster** — the Gross `[[144,12,12]]` bivariate-bicycle
  code, previously called out in `bench/gross_code.py` as intractable to *certify* by BZ,
  is now **fully proven (`--zx`, dZ=dX=12, no symmetry assumptions) in under 5 minutes on
  an Apple M4 laptop** (4:41; the d=9 level alone dropped 130s -> 17s). Stacked changes,
  none altering the math (the CPU backend remains the oracle; `qubitserf_compare` agrees):
  - **Zero-contribution information sets are skipped per weight level** (all backends):
    a set with rank `r` adds `max(0, (d+1)-(K-r))` to the BZ lower bound, so a
    rank-deficient set adds *nothing* until `d+1 > K-r` and enumerating it there only
    duplicates the upper-bound hunt. Sets are sorted by rank (descending) and each level
    enumerates only the positive-contribution prefix — for bivariate-bicycle codes
    (one full-rank set + one corank-12 set) this halves every level up to d=11.
    `EnumPlan` gains `num_gamma_total` (upload size) vs `num_gamma` (active prefix).
  - **Two-level incremental enumeration in the Metal kernel**: the weight-d combination
    splits into d-1 outer positions (codeword maintained by XOR-in-place on advance) and
    an inner index swept in a tight x4-unrolled loop where the codeword is
    `cw_base ^ row[last]`, fused into the weight test.
  - **Weight-first early exit**: popcount runs *before* the logical-detector check with a
    per-word exit against the current best (a codeword with weight >= best can never
    improve it), so the check almost never executes on deep levels.
  - **Threadgroup staging of the generators when they fit**, stored transposed
    (word-major) so the unrolled word-0 reads of consecutive rows are contiguous
    (bank-conflict-free). Codes whose matrices exceed the threadgroup-memory budget use a
    device-read kernel variant — fixing a potential out-of-bounds staging for large codes
    (e.g. `n=1024`, `K~1000`).
  - Kernel variants are now specialised per `(stride, d, tgcache)` with the weight level
    baked in as a compile-time literal; threads-per-threadgroup default dropped to 64
    (`QUBITSERF_TPT` overrides).
- **CPU BZ enumeration ~20-31x faster** — the Metal kernel's structure ported to
  `src/backend_cpu.cpp` (two-level incremental enumeration, weight-first per-word early
  exit, word-major transposed gamma for a vectorizable inner sweep; the multithreaded
  chunking is unchanged). Gross Z-side per-level: d=6 2410ms -> 84ms, d=7 25.6s -> 0.84s,
  d=9 = 79.7s (within ~4.4x of the GPU). Guarded by a new randomized oracle test
  (`src/tests/cpu_enum_ref.cpp`, target `qubitserf_cpu_enum_ref`): 84+ random small
  EnumPlans compared against a self-contained reference implementation (the old
  unrank + next_comb + eval_combo path), including multithreaded chunk-boundary cases.
- New `bench/level_bench.sh`: self-contained per-weight-level timing on the Gross code
  (`BACKEND=--cpu` for the CPU kernel).
- New `bench/bch_codes.py` + `bench/bz_vs_cc.py`: quantum BCH codes (CSS from
  dual-containing narrow-sense BCH, pure python/numpy) and a BZ-vs-CC benchmark across
  the sparse/dense divide (results in `bench/bz_vs_cc.md`, `.json`; `--replot`
  regenerates the table from the JSON). Headline: CC certifies sparse
  LDPC (toric, gross) in milliseconds where BZ needs minutes, while BZ certifies the
  dense-check `qbch [[127,71,9]]` in 26s (GPU) / 142s (CPU) where CC times out —
  neither method dominates; the winner follows Tanner-graph sparsity.
- Diagnostic env var `QUBITSERF_NO_EARLY_STOP` (BZ driver): disable the `inner <= outer`
  early stop, for benchmarking full levels.

### Changed
- **Comprehensive benchmark re-run at a 30 s per-run budget with the current solvers;
  all family charts regenerated.** Substantive methodology changes:
  - **Reference columns are now reused from a frozen cache** (`bench/ref_cache.json`)
    rather than re-executing the external `codeDistance` package. qubitserf's `cc`/`bz`/
    `mitm` are still measured live, so a `ref/cc` speedup pairs a fresh qubitserf timing
    with the (unchanged) reference timing; codes with no cached reference — the
    Reed-Muller families — show `n/a`.
  - **Toric/surface BZ is no longer pre-capped/skipped** (removed from `HARD_CSS_NAMES`):
    it runs uncapped and is stopped only by the 30 s budget, so it now *certifies*
    toric/surface `L=9` (GPU: 6.4 s / 3.4 s) and only times out at `L=10`, instead of
    being abandoned at `L=9`. A **drain-on-timeout** was added to `comprehensive.py` so a
    timed-out, uncancellable native solver finishes before the next code runs and can't
    inflate its timing.
  - The **toric/surface plots gained a top distance-`d` axis** (`d`, not raw `n`, is the
    hardness scale for these families).
- **`bench/bz_vs_cc` (the 330 s sparse/dense table) dropped its `d*log10(n)` column and
  its scatter chart (`bz_vs_cc.png`) was retired.** The >30 s measurements are preserved
  (the table is regenerated from `bz_vs_cc.json` via `--replot`, not re-run).
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

---

# Prior `codeaut` changelog (pre-merge)

The history below is the standalone `codeaut` project's changelog as of the merge. Paths and
import references in these older entries reflect the pre-merge standalone layout
(`bench/compare_*.py` is now `bench/codeaut/`; `codeaut.permgroup` is now
`qubitserf.codeaut.permgroup`; the `codeaut` CLI is unchanged).

## codeaut — Unreleased (pre-merge)

- **Exact invariant portfolio.** Added `invariant_automorphism_group(G, method=...)` with forced
  LCD-projector, projective-geometry, twin-compressed incidence, binary-matroid component/wreath,
  fingerprint-cover, pair/triple-moment, small-hull, Schur-projector, Ward residue-span, modular
  section, puncture/shorten minor, bounded circuit/cocircuit, and combined routes. Compact
  relations are treated as overgroups and accepted only when every generator preserves `C`;
  otherwise a guarded Schreier orbit computes the exact rowspace stabilizer, with an exact
  fallback when the orbit cap is exceeded. Added a common labelled-relation/hypergraph
  encoder and `bench/codeaut/compare_invariant_portfolio.py`, whose corpus validates exact order
  and mutual generator containment for every route.

- **Exact component and parallel-column engines.** Added the row-representation-independent
  binary-matroid decomposition, guarded projective local stabilizers, component equivalence
  transporters, parallel-fiber symmetric kernels, and equivalent-component wreath products.

- **Boundary correctness guards.** Leon now rejects dimensions `>=63`, which its uint64 Gray
  counter cannot represent; Python validates all int32 ABI bounds; Ward moduli require actual
  integers; the dreadnaut bridge fails closed on nonzero exits, missing summaries, and parsed-order
  mismatches instead of treating empty/unparseable output as the trivial group.

- **Power-of-two modular-weight automorphisms.** Added `ward_form(G, modulus=2**t)` using
  Ward inclusion--exclusion truncated at degree `t`, symbolic residue-span certificates, and
  `ward_automorphism_group`: a guarded exact route that counts fibers with a reduced decision
  diagram, materializes a low-cost complete spanning residue cover, and solves its colored
  incidence with nauty. Added `bench/codeaut/compare_ward.py`.

- **Leon selectors.** Classical calls accept `spanning_set="congruence"|"auto"` with
  `max_modulus` (arbitrary-modulus complete spanning weight-residue class), and
  `spanning_set="minimal"` (Python alias `"cocircuit"`, support-minimal/cocircuit filtering).

- **CLI.** Surfaces why a result is only a lower bound on stderr; CSS codes are given as Pauli
  stabiliser strings from a file or stdin (dropped `--hx`/`--hz`); `--json` carries the `method`
  diagnostic; explicit engine selection via `--method {auto,leon,bz,partial}` (wall-clock
  `timeout` removed).

- **Result objects.** `AutResult.group()` / `AutResult.n`; `CSSAutResult.order` is a decimal
  string; `backend` + `max_threads` on `css_automorphisms` / `css_automorphism_group`; the five
  high-level convenience entry points (`classical_automorphisms`, `css_automorphisms`,
  `css_tanner_graph_automorphisms`, `classical_tanner_automorphisms`, `group_intersection`).

## codeaut — 0.1.0

First release. A standalone (`numpy` + `ctypes`) package for automorphism groups of binary
linear and CSS quantum codes, extracted from the Quirky automorphism-harvest effort.

- **Leon engine** (`code_automorphism_group`): exact `Aut(C)` of a binary linear code via an
  optimized bit-packed C++ partition-backtracking engine (two-pass, low memory).
- **CSS method ladder** (`css_automorphism_group`): `Aut(Hx) ∩ Aut(Hz)` via Leon + dual-code
  trick, joint Brouwer–Zimmermann + nauty/Traces incidence, single-side rescue, cyclic/affine
  subgroup, and a Tanner-graph subgroup floor.
- **Brouwer–Zimmermann low-weight enumeration** (`low_weight_classes`) with a bit-packed C++
  CPU kernel and **CUDA + Metal** GPU backends (CMake-guarded, with CPU fallback).
- **Vendored permutation-group layer** (`codeaut.permgroup`): deterministic Schreier–Sims for
  exact order/membership/enumeration and an exact group intersection.
- **nauty/Traces** colored-graph solves via the system `dreadnaut` (documented dependency).
- Builtin codes, a `codeaut` CLI, a test suite, and benchmarks.
