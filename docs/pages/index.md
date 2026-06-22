# qminweight

**Fast GPU and multicore-CPU code-distance finding for quantum and
classical codes.** A self-contained C++ core with a Python (`ctypes`) front end.

`qminweight` computes the *exact* minimum distance of CSS quantum stabilizer codes and
classical linear codes using **deterministic** algorithms — Brouwer–Zimmermann (BZ),
connected cluster (CC), and meet-in-the-middle (MITM) — with the exponential enumeration
offloaded to the GPU where it pays off.

It is a from-scratch reimplementation inspired by Mark Webster *et al.*,
[*Distance-Finding Algorithms for Quantum Codes and Circuits*](https://arxiv.org/abs/2603.22532)
(the [`codeDistance`](https://github.com/m-webster/codeDistancePYPI) package), the
Quantinuum [Qubitserf](https://github.com/Quantinuum/Qubitserf) C++ tool, and the
Brouwer–Zimmermann improvements of Hernando, Quintana-Ortí & Grassl,
[arXiv:2408.10743](https://arxiv.org/abs/2408.10743).

## The headline result

The two algorithms cover two complementary regimes:

- **Connected cluster certifies the IBM gross code `[[144, 12, 12]]` in ~0.3 s** (377 ms
  measured) by exploiting its sparsity, where both qminweight's own Brouwer–Zimmermann and
  the reference `codeDistance` package time out (>30 s), and where Magma-style BZ would
  need on the order of tens of hours.
- **Brouwer–Zimmermann on the GPU is up to ~500× faster** than the default
  reference BZ on the codes where the reference finishes at all. On larger codes the
  reference times out (>30 s) where qminweight solves them in well under a second.

These numbers come from [`bench/results.md`](benchmarks.md) and
[`bench/cc_results.md`](benchmarks.md); they are not invented for the docs.

## Features

- **Exact, deterministic distance** for CSS quantum codes and classical linear codes — no
  Monte-Carlo estimates.
- **Three algorithms**, each exact, that cross-validate one another:
    - **Brouwer–Zimmermann** (`bz`) — enumeration-based, accelerated on **CPU and GPU**;
      best for dense / random codes. Reports a rigorous `[lower, upper]` bracket
      when its information-set lower bound is too weak to close.
    - **Connected cluster** (`cc`) — a Tanner-graph frontier search; best for
      sparse / LDPC / topological / bivariate-bicycle codes. CPU, parallel over seeds.
    - **Meet-in-the-middle** (`mitm`) — a CPU memory-for-time cross-check, validated
      against BZ by a differential fuzzer.
- **GPU acceleration where it matters**: BZ's inner enumeration is split across GPU threads
  with the combinatorial number system (no inter-thread coordination), running on
  bit-packed GF(2) words with hardware popcount. A hybrid dispatcher keeps small weight
  levels on the CPU so the GPU is never a slowdown.
- **A clean Python API** (`css_distance`, `classical_distance`, `available_backends`,
  `version`) plus a library of code generators in `qminweight.codes` (toric, surface,
  hypergraph product, bivariate bicycle / gross code, Hamming, repetition, random LDPC).
- **A command-line interface** for piping in Pauli strings or parity-check matrices.

## Where to next

| Page | What's there |
|---|---|
| [Installation](installation.md) | Building the native library and `pip install` |
| [Quickstart](quickstart.md) | Minimal Python and CLI examples |
| [Python API](api.md) | `css_distance`, `classical_distance`, `Result`, `qminweight.codes` |
| [Command line](cli.md) | The `qminweight` / `python -m qminweight` CLI |
| [Algorithms](algorithms.md) | BZ, CC, MITM — how they work and when to use each |
| [Benchmarks](benchmarks.md) | Measured numbers vs the reference, and how to reproduce |
| [Contributing](contributing.md) | Repo layout, building, running the tests |

## License

MIT.
