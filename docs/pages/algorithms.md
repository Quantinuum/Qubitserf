# Algorithms

`qminweight` implements **three** deterministic, exact distance algorithms. They compute the
same thing — the minimum Hamming weight of a non-trivial logical codeword — and serve as
cross-checks on one another, but they have different cost profiles, so the right choice
depends on the code.

| Method (`method=`) | Idea | Hardware | Best for |
|---|---|---|---|
| `bz` | Brouwer–Zimmermann enumeration | CPU + GPU | dense / random codes |
| `cc` | Connected cluster (Tanner-graph frontier search) | CPU (seed-parallel) | sparse / LDPC / topological / bivariate-bicycle |
| `mitm` | Meet-in-the-middle (syndrome matching) | CPU | a memory-for-time cross-check |

## Rule of thumb

> **Sparse / LDPC / topological / bivariate-bicycle code → `cc`.
> Dense / random code, or to push a GPU → `bz`.**

`mitm` is primarily a validator: it agrees with `bz` on every code (checked by a
differential fuzzer) and exists to give an independent confirmation of the answer.

---

## Brouwer–Zimmermann (`bz`)

The minimum distance is the smallest Hamming weight of a non-trivial logical codeword.
Brouwer–Zimmermann reduces finding it to: for increasing weight `d`, enumerate every
weight-`d` combination of the `K` message rows through a set of systematic
*information-set* generators, keep the lightest logical found, and **stop once a provable
lower bound meets the best weight so far**. The converging upper/lower bounds give early
termination, and — crucially — a rigorous answer even if you stop early.

### Why it maps onto the GPU

That enumeration is **embarrassingly parallel**. The *i*-th weight-`d` combination of `K`
items can be produced independently with the **combinatorial number system** (unranking).
So each GPU thread takes a contiguous slice of the index range, unranks its start, walks
the slice, and folds its minimum into a global `atomicMin` — no inter-thread coordination
at all. Everything runs on bit-packed GF(2) words (`uint64`) with hardware `popcount`.

### Optimizations

- bit-packed GF(2) codewords with hardware popcount;
- combinatorial-number-system work splitting (independent work units);
- converging BZ upper/lower bounds with early termination;
- **matroid partitioning** (Edmonds matroid union) for the maximum number of disjoint
  information sets — the tightest BZ lower bound (the refinement Magma uses), with a greedy
  fallback;
- **parallel random-information-set seeding** (the QDistRnd heuristic, run across cores) to
  start BZ from a near-minimal upper bound, so the exact proof terminates many weight
  levels earlier;
- even-weight rounding of the lower bound for even codes;
- **hybrid CPU/GPU dispatch**: each weight level goes to the GPU only when its candidate
  count is large enough to amortize launch latency; small levels run on the multicore CPU,
  so the GPU is never a slowdown (tunable via `QMINWEIGHT_GPU_MIN_WORK`);
- on the GPU: a **threadgroup tree-reduction** of the per-thread minimum (one atomic per
  threadgroup), per-solve-keyed device buffers (uploaded once), and compact per-thread
  state to keep occupancy high;
- a multithreaded CPU backend as reference and fallback.

Because the random-information-set seed is tight, well-structured codes (toric / surface up
to moderate size) reduce to a few small weight levels, and the CPU is the right tool — the
hybrid backend runs them there. The GPU earns its keep on codes whose BZ lower bound is
weak (high-rate / bivariate-bicycle), where the enumeration is unavoidably deep.

### The `[lower, upper]` bracket

BZ's weakness is codes whose information-set lower bound is loose (high-rate / LDPC). There
the enumeration to certify the lower bound is unavoidably deep, so qminweight reports a
rigorous `[lower_bound, distance]` **bracket** rather than blocking forever: it still
*finds* the true distance instantly via the random-information-set seed, but the
`proven=False` flag tells you the lower bound was not closed. This is exactly the situation
for bivariate-bicycle codes — see [the two-regimes example](#two-regimes-the-gross-code)
below — and the cue to switch to `cc`.

Per [arXiv:2408.10743](https://arxiv.org/abs/2408.10743), the improved BZ is the fastest
known exact method for *dense / random* codes, and its inner enumeration maps cleanly onto
the GPU (constant memory, independent work units).

---

## Connected cluster (`cc`)

The minimum-weight logical operator has **connected support on the Tanner graph** — any
disconnected piece would itself be a lighter logical or stabilizer. So connected cluster
only grows *connected* error clusters: starting from a seed qubit, it repeatedly fixes the
lowest-index unsatisfied check and branches over the qubits in that check, until the
syndrome closes (a codeword) with a non-trivial logical. Weights are tried in increasing
order, so the first hit is exact.

Unlike Brouwer–Zimmermann, this **exploits sparsity**, so it certifies LDPC / topological /
bivariate-bicycle codes where BZ's lower bound is too weak. It is the exact method Webster
*et al.* recommend (after Pryadko) for sparse CSS codes. qminweight's implementation is
multithreaded **over independent seed qubits**.

!!! warning "Pass the original sparse matrices"
    `cc` must be given the **original sparse** `Hx`/`Hz` — sparsity is the whole point. Do
    not row-reduce or densify them first. The Python API and CLI handle this for you.

### Why CC is CPU-only (by design)

GPU acceleration of connected cluster is **not** provided, deliberately. CC is an
irregular, data-dependent depth-first search — each branch has a different depth and
frontier — which maps poorly to SIMT/SIMD execution. The survey paper marks it
non-parallelizable, and qminweight's own research confirmed it. Its natural parallelism is
*across independent seed qubits*, which qminweight exploits on the multicore CPU; that
already certifies the hardest codes in under a second, so a GPU port would add complexity
for likely-negative benefit. (BZ stays the GPU-accelerated path precisely because its
enumeration *is* the embarrassingly parallel part.)

---

## Meet-in-the-middle (`mitm`)

A third exact method, on the CPU, used as a memory-for-time cross-check. It computes the
same quantity as BZ and agrees with it on every code, validated by a differential fuzzer.

The reduction is a coordinate split. Partition the `n` coordinates into a left half and a
right half. A codeword of weight `d` splits into a left part of weight `w_L` and a right
part of weight `w_R = d − w_L` with disjoint support, so

```
H_code · cᵀ = synL ⊕ synR,     check · cᵀ = logL ⊕ logR.
```

Thus `c` is a codeword iff `synL == synR`, and a non-trivial logical iff `logL ⊕ logR ≠ 0`.
For each total weight `d`, MITM hashes the left parts keyed by their syndrome, then probes
with right parts, matching syndromes and demanding the logical mismatch. The first `d` with
a hit is the exact distance.

### Why MITM is CPU-only (by design)

Although its half-weight enumeration is parallelizable in principle, MITM is dominated by
BZ/CC by **2–3 orders of magnitude**. Its real cost is the hash / `std::map` collision
phase plus an exponential `O(C(n/2, d/2))` **memory wall** that the GPU does not relieve —
and the enumeration is not the bottleneck. So a GPU port would not make it competitive.
MITM stays a CPU validator.

---

## Operator weight

The **operator weight** of a Pauli operator is its minimum weight *modulo the stabilizer (or
gauge) group* — the minimum-weight representative of its coset. For a CSS group with X-type
generators `Gx` and Z-type generators `Gz`, the Z- and X-parts are independent:

```
Z-part:  min weight over the coset  z_vec + rowspace(Gz)
X-part:  min weight over the coset  x_vec + rowspace(Gx)
```

### Reduction to `DistProblem`

The minimum weight in the coset `z_vec + rowspace(Gz)` is the lightest nonzero codeword of
the extended code `rowspace([Gz; z_vec])` that lies **outside** `rowspace(Gz)`. The cosets of
`rowspace(Gz)` inside that extended code are exactly `{rowspace(Gz), z_vec + rowspace(Gz)}`,
so a **single** linear detector separates them. That is precisely the core distance problem:

- `code_gen` = a basis of `rowspace([Gz; z_vec])` (append `z_vec` as an extra generator);
- `check` = a single row `φ ∈ nullspace(Gz)` with `φ · z_vecᵀ = 1` (it vanishes on
  `rowspace(Gz)` and is 1 on `z_vec`; it exists iff `z_vec ∉ rowspace(Gz)`).

So operator weight is solved by the **same BZ and MITM engines** as code distance — no
bespoke solver. If `z_vec ∈ rowspace(Gz)` the coset is the group itself and the weight is `0`.
`method="cc"` falls back to `bz`: the parity-check of the extended code is `nullspace([Gz;
z_vec])`, which is dense even when `Gz` is sparse, so the connected-cluster sparsity premise
does not hold.

### Correctness (the qubitserf fix)

qminweight imports this feature and its CLI shape from Quantinuum's **qubitserf**, but fixes a
bug. qubitserf matches MITM syndromes against `M = [Hz; X̄]` (Z-stabilizers stacked with
X-logicals), which is a valid parity-check of the Z-stabilizer group `rowspace(Hz)` **only
when `Hz · Hzᵀ = 0`** — the Z-generators are mutually and self-orthogonal. For codes where
they are not (surface, toric, bivariate-bicycle) it returns wrong answers: feeding a single
Z-stabilizer of the planar `surface(3)` `[[13,1,3]]` code as the operator returns `3`, when
the correct answer is `0` (a stabilizer is equivalent to identity).

The correct syndrome matrix is `nullspace(Gz)` (it equals `[Hx; X̄]` for an ordinary
stabilizer code, but is computed directly and is uniform for stabilizer *and* subsystem
codes). The coset-leader reduction above uses it implicitly, so qminweight is correct on every
code and agrees with qubitserf exactly on self-orthogonal codes (Steane, `Hx = Hz`).

## Subsystem dressed distance

A CSS **subsystem** (gauge) code is given by gauge generators `Gx` (X-type) and `Gz`
(Z-type). Its **stabilizer group is the center of the gauge group**:

```
Sx = { v ∈ rowspace(Gx) : Gz · vᵀ = 0 }     # X-gauge elements commuting with all Z-gauge
Sz = { v ∈ rowspace(Gz) : Gx · vᵀ = 0 }     # Z-gauge elements commuting with all X-gauge
```

The **dressed distance** is the minimum weight over operators that commute with the
stabilizers but are not in the gauge group — `C(S) \ G`. For CSS:

```
d_Z = min weight of a Z-type e with  Sx · eᵀ = 0  AND  e ∉ rowspace(Gz)
d_X = min weight of an X-type e with  Sz · eᵀ = 0  AND  e ∉ rowspace(Gx)
distance = min(d_Z, d_X)
```

The subtlety — and the reason this is "the careful one" — is that the **kernel/normalizer**
constraint uses the **stabilizer center `Sx`**, while the **triviality/quotient** test uses
the **gauge group `Gz`** (not `Sz`). Using `Gx` for the kernel would give the larger, *bare*
distance instead: `ker(Sx)` is strictly larger than `ker(Gx)`, and a dressed operator may
anticommute with individual gauge operators while still commuting with every stabilizer.

This maps directly onto the core distance problem with `build(Sx, Gz)` (and `build(Sz, Gx)`
for the X-part) — the search space is `ker(Sx)`, the detector is a basis of `ker(Gz)` modulo
`rowspace(Sx)` — so **all three engines apply**: BZ, MITM, and CC. CC operates on the sparse
stabilizer center, so it retains its sparsity advantage on topological subsystem codes such as
Bacon-Shor. A stabilizer code is the special case `gauge = stabilizers` (`Gx = Hx`, `Gz = Hz`,
`Sx = Hx`, `Sz = Hz`), where the dressed subsystem distance coincides with `css_distance`.

## General (non-CSS) codes — the symplectic metric

For a **general stabilizer code** the X- and Z-parts no longer decouple. A code is a
symplectic binary matrix `S` (`m × 2n`, `[z | x]` order); two Paulis `a=(a_z|a_x)`,
`b=(b_z|b_x)` commute iff the **symplectic product** `⟨a,b⟩ = a_z·b_x + a_x·b_z = 0`, and the
Pauli weight of `(z|x)` is the **symplectic weight** `#{ j : z_j=1 OR x_j=1 }` — *not*
`wt(z) + wt(x)`. The distance is the minimum symplectic weight over `e ∈ C(S) \ rowspace(S)`,
where `C(S) = nullspace(swap(S))` is the centralizer (`swap` exchanges the `[z]`/`[x]`
halves). Writing the symplectic product as an ordinary dot product, `⟨a,b⟩ = swap(a)·b`, lets
the meet-in-the-middle logical detector reuse the GF(2) machinery verbatim by storing its
rows pre-swapped.

Which engines generalize:

- **BZ — yes** (the default non-CSS path), via the **weight-doubling isometry**
  `φ : (a|b) ↦ (a|b|a⊕b)`. Per qubit, `(a_i,b_i)` contributes `0` to `wt_H(φ(v))` when it is
  identity and `2` otherwise (`Z`=`(1,0)`→`1+0+1`, `X`=`(0,1)`→`0+1+1`, `Y`=`(1,1)`→`1+1+0`),
  so `wt_H(φ(v)) = 2·wt_s(v)`. `φ` is `F₂`-linear and injective, so `C(S)` maps isometrically
  onto a binary linear code of length `3n` with the stabilizer subgroup mapping into it, and
  the "not a stabilizer" detector transfers verbatim (it acts on the first `2n` coordinates;
  the third block is dependent). Therefore the symplectic distance is **half** the binary
  Hamming distance of this length-`3n` code — computed by the *existing* binary BZ (and its GPU
  enumeration). Every `φ`-image has even weight, so BZ's even-distance rounding applies for
  free. This is the `SAVED_ISOMETRY` reduction of Sabater–Vera et al.
  ([arXiv:2408.10743](https://arxiv.org/abs/2408.10743)); `qminweight` builds it in
  `isometry_extend` / `symplectic_bz_distance` (`stab.cpp`).
- **MITM — yes**. The coordinate-split / syndrome-match structure is unchanged; only the
  enumeration changes — a "coordinate" is a **qubit** and each chosen qubit takes one of the
  three nonzero Paulis `Z`, `X`, `Y` (`C(n,w)·3^w` partial operators of weight `w`). This is
  exactly Quantinuum qubitserf's "middle algorithm".
- **CC — no** (falls back to MITM). Connected-cluster grows single-type clusters over a sparse
  CSS Tanner graph, which a general non-CSS code does not provide. (The connectedness argument
  itself does generalize — a min-weight logical has connected Tanner support for any stabilizer
  code — but each clustered qubit would branch over `Z/X/Y` and the syndrome bookkeeping
  becomes the symplectic product, with a `3^{cluster}` branching cost; this is left for future
  work, and `cc` falls back to MITM with a one-line stderr note.)

So `method="bz"` (default) and `"mitm"` both solve genuinely non-CSS codes; only `"cc"` falls
back. A code whose rows are all pure-`X`/pure-`Z` is detected up front and routed to the CSS
solvers above, so the CSS fast paths (and their BZ/CC/GPU acceleration) are fully preserved.
The same symplectic reduction gives the **dressed** distance of a non-CSS subsystem code
(search `C(center(G)) \ rowspace(G)`, with the stabilizer center
`center(G) = nullspace(Gram(G))·G`) and the **operator weight** of a general Pauli (minimum
symplectic weight over `op + rowspace(G)`) — both likewise solvable by BZ (isometry) or MITM.

## Two regimes: the gross code

The IBM gross code `[[144, 12, 12]]` is the canonical illustration of why two algorithms
matter. Its BZ search dimension `K = n − rank(Hx) = 78` **exceeds** `n/2 = 72`, so only
**one** disjoint full information set fits, and the BZ lower bound rises only ~1 per weight
level. BZ therefore:

- *finds* the distance `12` instantly via the random-information-set seed, and
- **brackets** it rigorously (`d ∈ [10, 12]` in one run, `[8, 12]` capped),
- but **certifying** it would require enumerating to weight 10 (`≈ C(78, 10) ≈ 10¹³`
  combinations) — on the order of tens of hours.

**Connected cluster certifies the same code in ~0.3 s** (377 ms measured) by exploiting its
sparsity. This is exactly the paper's recommendation: connected-cluster (or an ILP solver)
for bivariate-bicycle codes, **not** BZ. See the measured numbers in
[Benchmarks](benchmarks.md).

---

## References

- M. Webster *et al.*, *Distance-Finding Algorithms for Quantum Codes and Circuits*,
  [arXiv:2603.22532](https://arxiv.org/abs/2603.22532). The survey of BZ, connected
  cluster, and MITM, and the source of the algorithm recommendations
  (`codeDistance` / `codeDistancePYPI`).
- L. Hernando, E. S. Quintana-Ortí & M. Grassl, *Improved Brouwer–Zimmermann*,
  [arXiv:2408.10743](https://arxiv.org/abs/2408.10743). The improved BZ that qminweight's
  `bz` follows, with the matroid-partitioning lower bound.
- The connected-cluster method is due to L. P. Pryadko (as surveyed in the Webster *et al.*
  paper).
