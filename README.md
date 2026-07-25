<div align="center">

# Qubitserf

*Quantum Error Correction Library*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[Features](#features) • [Installation](#installation) • [Quick start](#quick-start) • [Code distance](#code-distance) • [Automorphism groups](#automorphism-groups) • [Built-in codes](#built-in-codes) • [Benchmarks](#benchmarks)

</div>

Qubitserf is a QEC library with a python frontend and a couple of CLI tools with a C++ backend including GPU acceleration of certain algorithms. It's goal is to be fast, very fast.

## Features

- **Minimum distance** — CSS quantum codes, general (non-CSS) stabilizer codes, dressed subsystem
  codes, and classical linear codes, by three complementary exact algorithms.
- **Operator weight** — the minimum weight of a Pauli operator modulo the stabilizer or gauge
  group, i.e. the minimum-weight coset leader.
- **Automorphism groups** — `Aut(C)` of a binary linear code by Leon's algorithm, and the
  qubit-permutation group `Aut(Hx) ∩ Aut(Hz)` of a CSS quantum code, with a full permutation-group
  toolkit (order, membership, enumeration, intersection).
- **Scales to real codes** — connected-cluster certifies distances of qLDPC codes with up to hundreds of qubits and distances up to 20. Similarly, automorphism groups of sparse codes with hundreds of bits/qubits can be computed exactly with the Brouwer-Zimmerman method.

## Installation

Requirements:

- Python 3.9 or newer, and NumPy
- CMake 3.20 or newer, and a C++17 compiler
- Optional CUDA or Metal for the GPU backends, auto-detected by CMake

```bash
cd qubitserf            # the unpacked package source directory
python -m pip install .
```

This installs the `qubitserf` Python package and two console commands, `distfind` and `codeaut`.

> [!IMPORTANT]
> The automorphism engines additionally need system `nauty` 2.7+ (which provides `dreadnaut`) on
> your `PATH` for the colored-graph solves. Install it with `brew install nauty`,
> `apt install nauty`, or `conda install -c conda-forge nauty`. Distance finding needs no system
> dependency.

To build the native libraries in-tree for local development instead:

```bash
./build.sh              # then run with PYTHONPATH=python
```

## Quick start

Certify the distance of IBM's `[[144,12,12]]` gross code and compute its automorphism group:

```python
from qubitserf import distfind, codeaut
from qubitserf.distfind import codes
from qubitserf.codeaut import codes as acodes

Hx, Hz = codes.gross_code()
r = distfind.css_distance(Hx, Hz, method="cc")
print(r.distance, r.proven, f"{r.seconds:.2f}s")     # 12 True 0.19s (timing varies)

a = codeaut.css_automorphisms(acodes.gross())
print(a.order, a.complete)                            # 144 True
```

The same from the command line:

```console
$ distfind --builtin gross --method cc --zx
12 12

$ printf 'IIIXXXX\nIXXIIXX\nXIXIXIX\nIIIZZZZ\nIZZIIZZ\nZIZIZIZ\n' | codeaut
order=168
1 4 6 0 2 3 5
6 4 1 0 5 3 2
```

---

# Code distance

Minimum distance and operator weight live in `qubitserf.distfind` and the `distfind` CLI.

## Command line

`distfind` reads Pauli stabilizer strings (from a file, stdin, or a pipe), CSS parity-check
matrices, a classical parity-check matrix, or a built-in code. It prints a bare integer, so it
composes with other tools.

```console
$ distfind --builtin steane --zx          # print both CSS components as 'dZ dX'
3 3

$ printf 'XXXX\nZZZZ\n' | distfind - --zx
2 2

$ distfind --hx Hx.txt --hz Hz.txt --method cc --threads 8
3

$ distfind --classical H.mtx --method bz
3
```

Operator weight takes the Pauli as an argument and the generators as input:

```console
$ distfind --builtin steane --operator ZZZZZZZ --zx
3 0

$ distfind --builtin steane --operator IIIZZZZ    # a stabilizer is the identity
0
```

Verbose mode reports the running lower bound on stderr as each weight level is ruled out, then
the exact answer; the last line of stdout is always the result:

```console
$ distfind --builtin surface:5 --method cc --zx -v
...
Z-distance: =5
Elapsed:[0ms]
X-distance bound: >4
Elapsed:[0ms]
X-distance: =5
Elapsed:[0ms]
5 5
```

Useful options: `--method bz|cc|mitm`, `--backend auto|cpu|gpu`, `--which min|z|x`, `--subsystem`,
`--threads N`, `--max-weight N` (cap an expensive search and return a certified bracket), `--json`,
`-v`, `--list-backends`. Run `distfind --help` for the full list, or use
`python -m qubitserf.distfind`.

## Python

```python
import numpy as np
from qubitserf import distfind as df

Hx = np.loadtxt("Hx.txt", dtype=np.uint8)
Hz = np.loadtxt("Hz.txt", dtype=np.uint8)

r = df.css_distance(Hx, Hz, method="bz", backend="gpu")
print(r.distance, r.proven, r.backend)

d_z = df.css_distance(Hx, Hz, which="z", method="cc").distance
d_x = df.css_distance(Hx, Hz, which="x", method="cc").distance

H = np.loadtxt("H.txt", dtype=np.uint8)
print(df.classical_distance(H, method="bz").distance)
```

Every call returns a `Result`:

| field | meaning |
|---|---|
| `distance` | best distance found; exact when `proven` is true |
| `lower_bound` | certified lower bound |
| `proven` | whether `distance == lower_bound` |
| `seconds` | wall-clock runtime |
| `backend` | backend actually used |

## Choosing the algorithm

Method choice matters far more than hardware — this is the single most important thing to know
about the library.

| method | best for | why |
|---|---|---|
| `cc` (connected cluster) | **sparse QLDPC** — bivariate-bicycle, hypergraph-product, toric, surface | exploits low-weight stabilizers; typically certifies in milliseconds regardless of `n` |
| `bz` (Brouwer-Zimmermann) | **dense or random** codes, and anything on the GPU | enumerates low-weight codeword classes; the only route when sparsity is absent |
| `mitm` (meet-in-the-middle) | small-code cross-checks | independent third opinion |

Recorded timings under a uniform 330 s timeout ([`BENCHMARKS.md`](BENCHMARKS.md)):

| code | class | `cc` CPU | `bz` CPU | `bz` GPU |
|---|---|---:|---:|---:|
| toric L=10 `[[200,2,10]]` | sparse LDPC | 10 ms | 135.3 s | 40.7 s |
| gross `[[144,12,12]]` | sparse LDPC | 297 ms | >330 s | 275.5 s |
| qbch `[[127,71,9]]` | dense BCH | >330 s | 142.3 s | 26.3 s |

> [!TIP]
> `backend="gpu"` accelerates Brouwer-Zimmermann; it does **not** remove the connected-cluster
> advantage on sparse Tanner graphs. On a QLDPC code, `method="cc"` on the CPU beats
> `method="bz"` on the GPU by orders of magnitude.

## Operator weight

`operator_weight` returns the minimum weight of a Pauli modulo the stabilizer (or gauge) group.
The Z-part is minimized over `z + rowspace(Gz)` and the X-part over `x + rowspace(Gx)`,
independently.

```python
from qubitserf import distfind as df
from qubitserf.distfind import codes

Gx, Gz = codes.steane()
op = df.operator_weight(Gx, Gz, "ZZZZZZZ")            # a logical Z of Steane
print(op.z_weight, op.x_weight, op.weight)            # 3 0 3

print(df.operator_weight(Gx, Gz, "IIIZZZZ").weight)   # 0 — a stabilizer is the identity
```

The operator may be a Pauli string (`I/X/Y/Z`, where `Y` sets both the Z and X bits), a
`(z_vec, x_vec)` pair, or a length-`2n` symplectic `[z|x]` array. The returned `OpResult` carries
`z_weight`, `x_weight`, `weight` (`= max(z, x)`), `proven`, `seconds`, and `backend`. Operator
weight reduces to the core distance problem, so it accepts `method="bz"` and `method="mitm"`
(`"cc"` falls back to `bz`).

## Subsystem codes

`subsystem_css_distance` takes the **gauge generators** of a CSS subsystem code and returns its
**dressed** distance: the minimum weight of an operator that commutes with the stabilizer group
(the center of the gauge group) but is not itself in the gauge group. The center is computed
internally, and all three engines apply — `cc` keeps its sparsity advantage on topological
subsystem codes.

```python
from qubitserf.distfind import codes

Gx, Gz = codes.bacon_shor(3)                            # gauge generators, n = 9
print(df.subsystem_css_distance(Gx, Gz, method="cc").distance)   # 3
```

A stabilizer code is the special case `gauge = stabilizers`, where the dressed distance coincides
with `css_distance`. On the command line, `--subsystem` switches the same interpretation on.

## General (non-CSS) codes

For any commuting set of Paulis — including operators with `Y`, or stabilizers mixing `X` and `Z` —
pass a **symplectic stabilizer matrix** `S` of shape `(m, 2n)` in `[z | x]` column order: row `r`
is the Pauli with Z-support `S[r, :n]` and X-support `S[r, n:]`. The distance is the minimum
**symplectic** weight (the number of qubits touched, *not* the sum of the Z- and X-weights) of a
normalizer element that is not a stabilizer.

```python
import numpy as np
from qubitserf import distfind as df

def paulis(strings):                    # 'XZZXI' ... -> a [z | x] matrix
    n = len(strings[0]); rows = []
    for s in strings:
        z = np.zeros(n, np.uint8); x = np.zeros(n, np.uint8)
        for j, c in enumerate(s):
            if c in "XY": x[j] = 1
            if c in "ZY": z[j] = 1
        rows.append(np.concatenate([z, x]))
    return np.array(rows, np.uint8)

S = paulis(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])        # the [[5,1,3]] perfect code
print(df.stabilizer_distance(S).distance)               # 3

print(df.pauli_operator_weight(S, S[0]).weight)         # 0 — a stabilizer
print(df.pauli_operator_weight(S, "XXXXX").weight)      # 3 — a logical X, reduced
```

`subsystem_stabilizer_distance(G)` is the non-CSS dressed distance from possibly non-commuting
gauge generators. A matrix whose rows are all pure-X/pure-Z is detected and routed to the fast CSS
solvers, and the CLI auto-detects non-CSS input the same way.

> [!NOTE]
> `method="cc"` has no non-CSS form — connected cluster needs a sparse single-type CSS Tanner
> graph — and requesting it raises an error. Brouwer-Zimmermann and meet-in-the-middle both work.

## Backends and limits

There is no fixed qubit-count limit: codewords are bit-packed into `ceil(n/64)` words and every
host routine is sized dynamically, so the CPU backend handles arbitrary `n`.

The GPU runs its native Brouwer-Zimmermann kernel for codes up to **1024 physical qubits**
(codeword stride ≤ 16 words), and for BZ weight levels up to 32 selected rows. Outside those
bounds — and for problems below an internal work threshold where dispatch latency would dominate —
it transparently falls back to the CPU solver for the same exact result. `threads=0` (the default)
uses all logical cores; pass a count to cap parallelism when running solves concurrently.

```python
print(df.available_backends())                          # ['cpu', 'gpu']
r = df.css_distance(Hx, Hz, method="cc", threads=4)
```

---

# Automorphism groups

Code automorphisms live in `qubitserf.codeaut` and the `codeaut` CLI.

## Command line

A CSS code is given as Pauli stabilizer strings (`I`/`X`/`Z`), one per line, all the same length;
pass a file path, or omit it (or use `-`) to read stdin. Every stabilizer must be pure-X or pure-Z.
The output is the group order followed by its generators as 0-indexed image lists.

```console
$ codeaut steane.txt
order=168
1 4 6 0 2 3 5
6 4 1 0 5 3 2

$ codeaut steane.txt --method bz
order=168
0 1 2 4 3 6 5
0 1 2 5 6 3 4
0 2 1 3 4 6 5
0 3 4 1 2 5 6
1 0 2 3 5 4 6

$ codeaut steane.txt --json
{"order": "168", "complete": true, "generators": [[1, 4, 6, 0, 2, 3, 5], [6, 4, 1, 0, 5, 3, 2]], "method": "..."}
```

Use `--gen` for the automorphism group of a classical generator matrix. Non-CSS input is rejected
with an explanation rather than a wrong answer:

```console
$ printf 'XXXXIII\nYZIIZZI\n' | codeaut
codeaut: stabiliser 1 ('YZIIZZI') contains 'Y': only I, X, Z are allowed -- a Y (or any X/Z mix) means the code is not CSS
```

Options: `--gen FILE|-`, `--method auto|leon|bz`, `--max-dim N` (the Leon enumeration cap),
`--spanning-set minweight|minimal|congruence|auto` (classical Leon incidence selector),
`--backend auto|cpu|gpu`, `--max-threads N`, `--json`, `--list-backends`. Also available as
`python -m qubitserf.codeaut`.

## Python

```python
import numpy as np
from qubitserf import codeaut
from qubitserf.codeaut import codes

# classical: the [7,3,4] simplex code -> GL(3,2)
G = np.array([[0, 0, 0, 1, 1, 1, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
print(codeaut.classical_automorphisms(G).order)         # 168

# CSS: the qubit permutations preserving both Hx and Hz
r = codeaut.css_automorphisms(codes.steane())
print(r.order, r.complete)                              # 168 True
```

`css_automorphisms` accepts a `CSSCode`, an `(Hx, Hz)` pair, two matrices, or any object exposing
`.Hx`/`.Hz`.

> [!NOTE]
> For CSS codes the order is a **decimal string**, since these groups can exceed 64 bits.
> `complete` is `True` when the full group was certified; a result with `complete=False` is a
> verified lower bound, and `.method` says why the engine stopped short.

`AutResult` (classical) carries `.order` (an exact `int`), `.generators`, `.n`, `.dim`,
`.num_codewords`, `.num_incidences`, `.weight_classes` and timing diagnostics. `CSSAutResult`
carries `.order` (a string), `.generators`, `.complete`, `.verified` (every generator re-checked
over GF(2)), `.method`, `.seconds` and `.n`.

## Choosing the engine

| method | what it does | when |
|---|---|---|
| `auto` (default) | tries the engines cheapest-first, stops as soon as one certifies the full group | almost always |
| `leon` | Leon's algorithm per side, then a permutation-group intersection | small and dense codes |
| `bz` | joint Brouwer-Zimmermann + nauty/Traces graph incidence | **LDPC codes** |

Leon's cost is `2**eff_dim` in `eff_dim = max_side min(rank, n−rank)`, so `auto` runs Leon first
when `eff_dim` is small and the Brouwer-Zimmermann route — which sidesteps `eff_dim` entirely —
otherwise.

```python
print(codeaut.css_automorphisms(codes.gross(), method="bz").order)      # 144
print(codeaut.css_automorphisms(codes.steane(), method="leon").order)   # 168
```

`backend="auto"|"cpu"|"gpu"` and `max_threads` steer the Brouwer-Zimmermann enumeration; `"gpu"`
uses CUDA or Metal when built and otherwise falls back to the CPU for the same result.

## Working with the group

Both result types expose `.group()`, which builds a real permutation group — a
`qubitserf.codeaut.permgroup.Group`. That object is what you compute with:

```python
grp = codeaut.css_automorphisms(codes.steane()).group()

print(grp.order())                              # 168
print(grp.reduced_generators())                 # a minimal generating set
print(grp.contains([0, 1, 2, 3, 4, 5, 6]))      # True (the identity)
print(sum(1 for _ in grp))                      # 168 — a Group is directly iterable

for perm in grp.elements():                     # enumerate all elements (small groups only)
    ...
```

`group_intersection` intersects two groups, accepting results, `(generators, degree)` pairs, or
`Group` objects:

```python
A = codeaut.classical_automorphisms(G)
S = codeaut.permgroup.symmetric_group(7)
print(codeaut.group_intersection(A, S).order())          # 168
```

The permutation-group toolkit is available directly as `codeaut.permgroup`:

```python
from qubitserf.codeaut import permgroup as pg

A = pg.Group([[1, 2, 0, 3], [1, 0, 2, 3]], 4)            # 0-indexed image lists
print(A.order(), A.contains([2, 0, 1, 3]))               # 6 True
```

---

## Built-in codes

Standard families ship with both modules, so every example above runs without external data.

| module | generators |
|---|---|
| `qubitserf.distfind.codes` | `steane`, `shor`, `toric(L)`, `surface(L)`, `bivariate_bicycle`, `gross_code`, `bacon_shor(d)`, `hypergraph_product`, `quantum_reed_muller(r,m)`, `reed_muller_generator(r,m)`, `hamming_parity(r)`, `repetition_parity(n)` |
| `qubitserf.codeaut.codes` | `steane`, `shor`, `iceberg(m)`, `toric(L)`, `surface(d)`, `bivariate_bicycle`, `gross`, `hamming_parity(r)`, `repetition(n)` |

The `distfind` CLI reaches them with `--builtin NAME[:L]`:

```console
$ distfind --builtin toric:10 --method cc
10
```


## Citation

```bibtex
@software{qubitserf,
  title  = {Qubitserf : Quantum Coding Theory Library},
  author = {Serban Cercelescu},
  year   = {2026},
  url    = {https://github.com/Quantinuum/qubitserf}
}
```
