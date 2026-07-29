# Automorphism groups

Everything is importable from `qubitserf.codeaut`; the permutation-group vocabulary lives
in `qubitserf.algebra`:

```python
from qubitserf import codeaut
from qubitserf.codeaut import codes
from qubitserf.algebra import permgroup as pg
```

| Function | Computes |
|---|---|
| [`classical_automorphisms`](#classical_automorphisms) | `Aut(C)` of a binary linear code |
| [`css_automorphisms`](#css_automorphisms) | `Aut(Hx) ∩ Aut(Hz)` of a CSS quantum code |
| [`group_intersection`](#group_intersection) | the intersection of two permutation groups |

Both automorphism functions are **exact-or-raise**: they return the certified full group,
or raise a `RuntimeError` explaining which engines were tried and what to try instead.
They never return a partial or unverified answer.

> [!IMPORTANT]
> The colored-graph solves need system `nauty` 2.7+ (`dreadnaut`) on your `PATH`
> (`brew install nauty`, `apt install nauty`, or `conda install -c conda-forge nauty`).

## `classical_automorphisms`

```python
classical_automorphisms(genmat, *, method="auto", backend="auto",
                        max_threads=None, verbose=False)
```

The permutation automorphism group `Aut(C)` of the binary linear code
`C = rowspace(genmat)`. `genmat` is any `(m, n)` 0/1 generating set (row-reduced
internally).

```python
import numpy as np

# the [7,3,4] simplex code -> GL(3,2)
G = np.array([[0, 0, 0, 1, 1, 1, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)

r = codeaut.classical_automorphisms(G)
print(r.order)          # 168 — an exact int
print(r.generators)     # 0-indexed image lists, a strong generating set
```

`method` selects the engine:

| method | what it does | when |
|---|---|---|
| `"auto"` (default) | Leon when the effective dimension `min(dim, n − dim)` is ≤ 20, the `bz` route otherwise; falls through to the other engine on failure | almost always |
| `"leon"` | Leon's partition-backtracking algorithm on the cheaper of `C` / `C⊥` (the dual-code trick: `Aut(C) = Aut(C⊥)`, enumeration costs `2**eff_dim`) | small effective dimension |
| `"bz"` | certified-complete Brouwer–Zimmermann low-weight classes + nauty/Traces incidence; exact at any dimension | **LDPC-like codes** |

```python
print(codeaut.classical_automorphisms(G, method="bz").order)     # 168, same group
```

`backend="auto"|"cpu"|"gpu"` and `max_threads` steer the BZ enumeration; `verbose=True`
streams engine progress to stderr. The result carries diagnostics alongside `order` and
`generators` (`n`, `dim`, `num_codewords`, `weight_classes`, and for the `bz` route
`dualized`, `method`, `seconds`).

## `css_automorphisms`

```python
css_automorphisms(code, Hz=None, *, method="auto", backend="auto",
                  max_threads=None, verbose=False) -> algebra.Group
```

The qubit-permutation automorphism group `Aut(Hx) ∩ Aut(Hz)` of a CSS code, returned
directly as a permutation group. `code` is a `CSSCode`, an `(Hx, Hz)` pair (or pass `Hz`
as the second argument), or any object exposing `.Hx`/`.Hz`.

```python
grp = codeaut.css_automorphisms(codes.steane())
print(grp.order())                                                      # 168

print(codeaut.css_automorphisms(codes.gross(), method="bz").order())    # 144
```

The same `method` choice applies: `"leon"` runs Leon per side and intersects, `"bz"` runs
the joint Brouwer–Zimmermann + nauty/Traces graph incidence (best for LDPC codes), and
`"auto"` tries the engines cheapest-first, stopping as soon as one certifies the full
group. Every generator is re-verified over GF(2) to preserve both rowspaces before the
group is returned. If no engine can certify the exact group, a `RuntimeError` lists each
stage tried, why it failed, and what to try:

```python
codeaut.css_automorphisms(codes.gross(), method="leon")
# RuntimeError: css automorphism group: no engine certified the exact Aut(Hx) ∩ Aut(Hz)
#   (method='leon', n=144, eff_dim=66); tried:
#   - leon+dual (stages 1&3): skipped -- eff_dim=66 > 24 (Leon enumerates 2**eff_dim)
#   try method='bz' (joint BZ + nauty/Traces incidence; exact at any eff_dim, ...)
```

## Working with the group

`css_automorphisms` returns a `qubitserf.algebra.Group` directly; classical results
expose the same via `.group()`. It has deterministic Schreier–Sims order, membership,
and enumeration:

```python
grp = codeaut.css_automorphisms(codes.steane())

print(grp.order())                              # 168 — an exact int, any size
print(grp.reduced_generators())                 # a minimal generating set
print(grp.contains([0, 1, 2, 3, 4, 5, 6]))      # True (the identity)
print(sum(1 for _ in grp))                      # 168 — a Group is directly iterable
```

Construct groups directly from 0-indexed image lists:

```python
A = pg.Group([[1, 2, 0, 3], [1, 0, 2, 3]], 4)
print(A.order(), A.contains([2, 0, 1, 3]))      # 6 True
```

## `group_intersection`

```python
group_intersection(g1, g2) -> algebra.Group
```

The intersection of two permutation groups, accepting automorphism results,
`(generators, degree)` pairs, or `Group` objects:

```python
A = codeaut.classical_automorphisms(G)
print(codeaut.group_intersection(A, pg.symmetric_group(7)).order())   # 168
```

## Pauli-string input

`css_from_paulis` parses `I`/`X`/`Z` stabiliser strings (one per line, or an iterable)
into an `(Hx, Hz)` pair, rejecting non-CSS input with an explanation:

```python
Hx, Hz = codeaut.css_from_paulis("""
IIIXXXX
IXXIIXX
XIXIXIX
IIIZZZZ
IZZIIZZ
ZIZIZIZ
""")
print(codeaut.css_automorphisms((Hx, Hz)).order())    # 168
```

## Built-in codes

`qubitserf.codeaut.codes` ships standard families: `steane()`, `shor()`, `iceberg(m)`,
`toric(L)`, `surface(d)`, `bivariate_bicycle(l, m, a_terms, b_terms)`, `gross()`,
`hamming_parity(r)`, `repetition(n)`.
