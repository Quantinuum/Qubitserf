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
