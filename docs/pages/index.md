# qubitserf

**Fast exact code distance and automorphism groups for quantum and classical codes,
on CPU and GPU.** A self-contained C++ core with a Python (`ctypes`) front end.

Qubitserf bundles two engines built on one shared native library, `libqubitserf`:

- **`qubitserf.distfind`** computes the *exact* minimum distance of CSS quantum codes,
  general (non-CSS) stabilizer codes, dressed subsystem codes, and classical linear codes
  — plus the minimum weight of a Pauli operator modulo the stabilizer or gauge group —
  using **deterministic** algorithms (Brouwer–Zimmermann, connected cluster,
  meet-in-the-middle), with the exponential enumeration offloaded to the GPU where it
  pays off.
- **`qubitserf.codeaut`** computes the automorphism group `Aut(C)` of a binary linear
  code (Leon's algorithm, or Brouwer–Zimmermann + nauty incidence) and the
  qubit-permutation group `Aut(Hx) ∩ Aut(Hz)` of a CSS quantum code, with a full
  permutation-group toolkit (exact order, membership, enumeration, intersection) in
  `qubitserf.algebra`.

Everything is exact: a distance result says so via `.proven`; the automorphism functions
return the certified full group or raise. No Monte-Carlo estimates, and never a silently
unverified group.

## The headline results

- **Connected cluster certifies the IBM gross code `[[144, 12, 12]]` in well under a
  second** by exploiting its sparsity, where both qubitserf's own Brouwer–Zimmermann and
  the reference `codeDistance` package time out (>30 s), and where Magma-style BZ would
  need on the order of tens of hours.
- **Brouwer–Zimmermann on the GPU is up to ~500× faster** than the default reference BZ
  on the codes where the reference finishes at all.
- The automorphism engines compute the full automorphism group of sparse codes with
  hundreds of bits/qubits exactly, via the same certified low-weight enumeration.

The distance numbers come from the committed benchmark outputs — see
[Benchmarks](benchmarks.md); they are measurements, not estimates.

## Where to next

| Page | What's there |
|---|---|
| [Code distance](distance.md) | `css_distance`, `classical_distance`, `operator_weight`, subsystem and non-CSS codes |
| [Automorphism groups](automorphisms.md) | `classical_automorphisms`, `css_automorphisms`, working with permutation groups |
| [Benchmarks](benchmarks.md) | Measured numbers vs the reference, and how to reproduce |

## License

MIT.
