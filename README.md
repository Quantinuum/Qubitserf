<div align="center">

# Qubitserf

*Quantum Error Correction Library*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[Features](#features) • [Installation](#installation) • [Quick start](#quick-start) • [Code distance](#code-distance) • [Automorphism groups](#automorphism-groups)

</div>

Qubitserf is a Python QEC library with a C++ backend, including GPU acceleration of certain
algorithms. Its goal is to be fast, very fast.

## Features

- **Code distance** — CSS quantum codes, general (non-CSS) stabilizer codes, subsystem
  codes, and classical linear codes by multiple algorithms
- **Operator weight** — the minimum weight of a Pauli operator modulo the stabilizer or gauge
  group, i.e. the minimum-weight coset leader.
- **Automorphism groups** — `Aut(C)` of a classical binary linear code or of a quantum CSS code with a full permutation-group toolkit (order, membership, enumeration, intersection).
- **Scales to real codes** — connected-cluster certifies distances of qLDPC codes with up to hundreds of qubits and distances up to 20. Similarly, automorphism groups of sparse codes with hundreds of bits/qubits can be computed with the Brouwer-Zimmerman+nauty method.

## Installation

Requirements:

- Python 3.9 or newer, and NumPy
- CMake 3.20 or newer, and a C++17 compiler
- Optional CUDA or Metal for the GPU backends, auto-detected by CMake

```bash
cd qubitserf            # the unpacked package source directory
python -m pip install .
```

> [!IMPORTANT]
> The automorphism engines additionally need system `nauty` 2.7+ (which provides `dreadnaut`) on
> your `PATH` for the colored-graph solves. Install it with `brew install nauty`,
> `apt install nauty`, or `conda install -c conda-forge nauty`. Distance finding needs no system
> dependency.

To build the native library in-tree for local development instead:

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
print(distfind.css_distance(Hx, Hz, method="cc"))     # 12

a = codeaut.css_automorphisms(acodes.gross())
print(a.order())                                      # 144
```

Every `distfind` entry point returns a plain `int` — `-1` when the distance is not defined for
the input (an empty code, or a code with no logical qubits). The `codeaut` functions return the
full group or raise. Pass `verbose=True` to any of them to
stream progress to stderr. Standard code families ship with both modules (`qubitserf.distfind.codes` and `qubitserf.codeaut.codes` —
`steane`, `shor`, `toric(L)`, `surface(L)`, `gross`, `bacon_shor(d)`, Reed–Muller, Hamming, …),
so every example below runs without external data.

---

# Code distance

## CSS codes — `css_distance`

```python
import numpy as np
from qubitserf import distfind as df

Hx = np.loadtxt("Hx.txt", dtype=np.uint8)     # any 0/1 check matrices
Hz = np.loadtxt("Hz.txt", dtype=np.uint8)

print(df.css_distance(Hx, Hz))                # method="bz" is the default
```

choose `method` depending on the codes: use `"cc"` (connected cluster) for sparse qLDPC codes — and `"bz"` (Brouwer-Zimmermann, the default, GPU-capable) for dense or random codes. `"mitm"` is an independent cross-check for
small codes.

```python
from qubitserf.distfind import codes

Hx, Hz = codes.toric(10)                                   # [[200, 2, 10]], sparse
print(df.css_distance(Hx, Hz, method="cc"))                # 10, in milliseconds

print(df.css_distance(Hx, Hz, method="bz", backend="gpu")) # same answer the slow way;
                                                           # falls back to cpu without a GPU

d_z = df.css_distance(Hx, Hz, which="z", method="cc")      # one side only
d_x = df.css_distance(Hx, Hz, which="x", method="cc")
print(d_z, d_x)                                            # 10 10
```

Useful keywords (shared by all distance functions): `which="min"|"z"|"x"`,
`backend="auto"|"cpu"|"gpu"`, `threads=N` (0 = all cores), `verbose=True`.

## Classical codes — `classical_distance`

```python
H = codes.hamming_parity(3)                   # parity-check matrix of the [7,4,3] Hamming code
print(df.classical_distance(H))               # 3
```

## Operator weight — `operator_weight`

The minimum weight of a Pauli modulo the stabilizer group — the Z-part is minimized over
`z + rowspace(Gz)` and the X-part over `x + rowspace(Gx)` independently, and the larger of the
two coset weights is returned:

```python
Gx, Gz = codes.steane()
print(df.operator_weight(Gx, Gz, "ZZZZZZZ"))          # 3 — a logical Z of Steane

print(df.operator_weight(Gx, Gz, "IIIZZZZ"))          # 0 — a stabilizer is the identity
```

The operator may be a Pauli string (`I/X/Y/Z`, where `Y` sets both the Z and X bits), a
`(z_vec, x_vec)` pair, or a length-`2n` symplectic `[z|x]` array.

## Subsystem codes — `subsystem_css_distance`

Pass the **gauge generators**; the stabilizer group (the gauge center) is computed internally and
the **dressed** distance is returned:

```python
Gx, Gz = codes.bacon_shor(3)                            # gauge generators, n = 9
print(df.subsystem_css_distance(Gx, Gz, method="cc"))   # 3
```

## General (non-CSS) codes — `stabilizer_distance`

For any commuting set of Paulis, pass a symplectic stabilizer matrix of shape `(m, 2n)` in
`[z | x]` column order. The distance is the minimum **symplectic** weight (qubits touched) of a
normalizer element that is not a stabilizer:

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
print(df.stabilizer_distance(S))                        # 3

print(df.pauli_operator_weight(S, S[0]))                # 0 — a stabilizer
print(df.pauli_operator_weight(S, "XXXXX"))             # 3 — a logical X, reduced
```

`subsystem_stabilizer_distance(G)` is the non-CSS dressed distance from possibly non-commuting
gauge generators. Pure-X/pure-Z input is detected and routed to the fast CSS solvers.
(`method="cc"` needs a CSS Tanner graph and raises on non-CSS input; `bz` and `mitm` both work.)

---

# Automorphism groups

## Classical codes — `classical_automorphisms`

```python
import numpy as np
from qubitserf import codeaut

# the [7,3,4] simplex code -> GL(3,2)
G = np.array([[0, 0, 0, 1, 1, 1, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)

print(codeaut.classical_automorphisms(G).order)                  # 168
print(codeaut.classical_automorphisms(G, method="bz").order)     # 168 (BZ + nauty route)
```

`method="auto"` (the default) runs Leon's algorithm when the effective dimension
`min(dim, n − dim)` is small and the Brouwer-Zimmermann + nauty incidence route (best for
LDPC-like codes, usable at any dimension) otherwise.

## CSS codes — `css_automorphisms`

The qubit permutations preserving both `Hx` and `Hz`, returned directly as a permutation group (`qubitserf.algebra.Group`):

```python
from qubitserf.codeaut import codes

grp = codeaut.css_automorphisms(codes.steane())
print(grp.order())                                      # 168

print(codeaut.css_automorphisms(codes.gross(), method="bz").order())   # 144
```

It accepts a `CSSCode`, an `(Hx, Hz)` pair, two matrices, or any object exposing `.Hx`/`.Hz`.
The same `method="auto"|"leon"|"bz"` choice applies: `leon` for small effective dimension, `bz`
(joint Brouwer-Zimmermann + nauty/Traces incidence) for LDPC codes.

```python
codeaut.css_automorphisms(codes.gross(), method="leon")
# RuntimeError: css automorphism group: no engine certified the exact Aut(Hx) ∩ Aut(Hz)
#   (method='leon', n=144, eff_dim=66); tried: ... try method='bz' ...
```

## Working with the group

The returned `Group` (classical results expose the same via `.group()`) supports order,
membership, and enumeration:

```python
grp = codeaut.css_automorphisms(codes.steane())

print(grp.order())                              # 168 — a Python int, any size
print(grp.reduced_generators())                 # a minimal generating set
print(grp.contains([0, 1, 2, 3, 4, 5, 6]))      # True (the identity)
print(sum(1 for _ in grp))                      # 168 — a Group is directly iterable
```

## Defining your own group — `algebra.Group`

A permutation group can be defined directly from its generators — each generator is a
0-indexed image list (`perm[i]` = image of point `i`), and the second argument is the
degree:

```python
from qubitserf.algebra import permgroup as pg

# <(0 1 2), (0 1)> acting on 4 points -- S3 on the first three
B = pg.Group([[1, 2, 0, 3], [1, 0, 2, 3]], 4)
print(B.order(), B.contains([2, 0, 1, 3]))      # 6 True

print(pg.symmetric_group(5).order())            # 120
```

This is the same class the automorphism results produce, so hand-built groups compose
with computed ones everywhere (membership, intersection, iteration).

## Intersections — `group_intersection`

Intersect two groups, accepting results, `(generators, degree)` pairs, or `Group` objects:

```python
A = codeaut.classical_automorphisms(G)
print(codeaut.group_intersection(A, pg.symmetric_group(7)).order())   # 168
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
