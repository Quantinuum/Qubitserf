# Code distance

Everything is importable from `qubitserf.distfind`:

```python
from qubitserf import distfind as df
from qubitserf.distfind import codes
```

| Function | Computes |
|---|---|
| [`css_distance`](#css_distance) | exact CSS-code distance |
| [`classical_distance`](#classical_distance) | exact classical linear-code distance |
| [`operator_weight`](#operator_weight) | min weight of a Pauli modulo a CSS group |
| [`subsystem_css_distance`](#subsystem_css_distance) | dressed distance of a CSS subsystem (gauge) code |
| [`stabilizer_distance`](#general-non-css-codes) | exact general (non-CSS) stabilizer-code distance |
| [`subsystem_stabilizer_distance`](#general-non-css-codes) | dressed distance of a non-CSS subsystem code |
| [`pauli_operator_weight`](#general-non-css-codes) | min symplectic weight of a Pauli modulo a non-CSS group |
| `available_backends` / `version` | environment introspection |

All of them share the same keywords:

| Keyword | Default | Meaning |
|---|---|---|
| `method` | `"bz"` | `"bz"` (Brouwer–Zimmermann, GPU-capable), `"cc"` (connected cluster — use for sparse/LDPC codes), `"mitm"` (meet-in-the-middle cross-check) |
| `which` | `"min"` | `"min"` = `min(dX, dZ)`, `"z"` = Z-distance, `"x"` = X-distance (distance functions only) |
| `backend` | `"auto"` | `"auto"` / `"cpu"` / `"gpu"`; used by `bz`. `"gpu"` falls back to the CPU for the identical result when no accelerator is present |
| `threads` | `0` | CPU worker threads; `0` = all hardware cores |
| `max_weight` | `0` | cap an expensive search and return a certified `[lower, upper]` bracket; `0` = no cap |
| `verbose` | `False` | stream per-level progress to stderr |

and return a `Result` with:

| Field | Meaning |
|---|---|
| `distance` | best distance found; exact when `proven` is true |
| `lower_bound` | certified lower bound (`== distance` when proven) |
| `proven` | whether the answer is certified exact |
| `seconds` | wall-clock runtime |
| `backend` | backend actually used |

## `css_distance`

```python
css_distance(Hx, Hz, *, method="bz", which="min", backend="auto",
             threads=0, max_weight=0, verbose=False) -> Result
```

`Hx` and `Hz` are any 0/1 arrays (coerced to `uint8` and reduced mod 2).

```python
Hx, Hz = codes.steane()
r = df.css_distance(Hx, Hz)
print(r.distance, r.proven)                                # 3 True

Hx, Hz = codes.toric(10)                                   # [[200, 2, 10]], sparse
print(df.css_distance(Hx, Hz, method="cc").distance)       # 10, in milliseconds

d_z = df.css_distance(Hx, Hz, which="z", method="cc").distance
d_x = df.css_distance(Hx, Hz, which="x", method="cc").distance
print(d_z, d_x)                                            # 10 10
```

Method choice is the single most important thing to know: `"cc"` exploits low-weight
stabilizers and typically certifies sparse qLDPC codes in milliseconds regardless of `n`;
`"bz"` enumerates low-weight codeword classes and is the route for dense or random codes
(and the only GPU-accelerated one). On a qLDPC code, `method="cc"` on the CPU beats
`method="bz"` on the GPU by orders of magnitude — see [Benchmarks](benchmarks.md).

When BZ cannot close the gap in the allotted `max_weight`, the result is a rigorous
bracket:

```python
Hx, Hz = codes.gross_code()                                # [[144, 12, 12]]
r = df.css_distance(Hx, Hz, method="bz", max_weight=8)
print(r.distance, r.lower_bound, r.proven)                 # 12 10 False
print(df.css_distance(Hx, Hz, method="cc").proven)         # True — cc certifies it
```

## `classical_distance`

```python
classical_distance(H, *, method="bz", backend="auto",
                   threads=0, max_weight=0, verbose=False) -> Result
```

Minimum distance of a classical linear code from its parity-check matrix:

```python
H = codes.hamming_parity(3)                     # the [7, 4, 3] Hamming code
print(df.classical_distance(H).distance)        # 3

print(df.classical_distance(codes.repetition_parity(8)).distance)  # 8
```

## `operator_weight`

```python
operator_weight(Gx, Gz, operator, *, method="bz", backend="auto",
                threads=0, max_weight=0, verbose=False) -> OpResult
```

The minimum weight of a Pauli modulo the group `<Gx, Gz>` — the minimum-weight coset
leader. The Z-part is minimized over `z + rowspace(Gz)` and the X-part over
`x + rowspace(Gx)`, independently:

```python
Gx, Gz = codes.steane()
op = df.operator_weight(Gx, Gz, "ZZZZZZZ")            # a logical Z of Steane
print(op.z_weight, op.x_weight, op.weight)            # 3 0 3

print(df.operator_weight(Gx, Gz, "IIIZZZZ").weight)   # 0 — a stabilizer is the identity
```

The operator may be a Pauli string (`I/X/Y/Z`, where `Y` sets both the Z and X bits), a
`(z_vec, x_vec)` pair, or a length-`2n` symplectic `[z|x]` array. The `OpResult` carries
`z_weight`, `x_weight`, `weight` (`= max(z, x)`), `proven`, `seconds`, and `backend`.
Operator weight reduces to the core distance problem, so it accepts `method="bz"` and
`method="mitm"` (`"cc"` falls back to `bz`).

## `subsystem_css_distance`

```python
subsystem_css_distance(Gx, Gz, *, method="bz", which="min", backend="auto",
                       threads=0, max_weight=0, verbose=False) -> Result
```

Takes the **gauge generators** of a CSS subsystem code and returns its **dressed**
distance: the minimum weight of an operator that commutes with the stabilizer group (the
center of the gauge group) but is not itself in the gauge group. The center is computed
internally, and all three engines apply — `cc` keeps its sparsity advantage on
topological subsystem codes.

```python
Gx, Gz = codes.bacon_shor(3)                            # gauge generators, n = 9
print(df.subsystem_css_distance(Gx, Gz, method="cc").distance)   # 3
```

A stabilizer code is the special case `gauge = stabilizers`, where the dressed distance
coincides with `css_distance`.

## General (non-CSS) codes

For any commuting set of Paulis — including operators with `Y`, or stabilizers mixing `X`
and `Z` — pass a **symplectic stabilizer matrix** `S` of shape `(m, 2n)` in `[z | x]`
column order: row `r` is the Pauli with Z-support `S[r, :n]` and X-support `S[r, n:]`.
The distance is the minimum **symplectic** weight (the number of qubits touched, *not*
the sum of the Z- and X-weights) of a normalizer element that is not a stabilizer.

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

`subsystem_stabilizer_distance(G)` is the non-CSS dressed distance from possibly
non-commuting gauge generators. A matrix whose rows are all pure-X/pure-Z is detected and
routed to the fast CSS solvers.

> [!NOTE]
> `method="cc"` has no non-CSS form — connected cluster needs a sparse single-type CSS
> Tanner graph — and requesting it raises an error. Brouwer–Zimmermann and
> meet-in-the-middle both work.

## Built-in codes

`qubitserf.distfind.codes` ships standard families, so every example above runs without
external data: `steane`, `shor`, `toric(L)`, `surface(L)`, `bacon_shor(d)`,
`hypergraph_product(H1, H2)`, `bivariate_bicycle(l, m, a_terms, b_terms)`, `gross_code()`,
`quantum_reed_muller(r, m)`, `reed_muller_generator(r, m)`, `hamming_parity(r)`,
`repetition_parity(n)`, `cyclic_repetition_parity(n)`, `random_ldpc_parity(m, n)`.

## Backends and limits

There is no fixed qubit-count limit: codewords are bit-packed into `ceil(n/64)` words and
every host routine is sized dynamically, so the CPU backend handles arbitrary `n`. The
GPU runs its native Brouwer–Zimmermann kernel for codes up to 1024 physical qubits and
transparently falls back to the CPU outside that (or when dispatch latency would
dominate) for the same exact result.

```python
print(df.available_backends())                          # ['cpu', 'gpu']
```
