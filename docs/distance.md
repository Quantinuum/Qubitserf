# Code distance

Everything is importable from `qubitserf.distfind`:

```python
from qubitserf import distfind as df
from qubitserf.distfind import codes
```

| Function | Computes |
|---|---|
| [`css_distance`](#css_distance) | CSS-code distance |
| [`classical_distance`](#classical_distance) | classical linear-code distance |
| [`operator_weight`](#operator_weight) | min weight of a Pauli modulo a CSS group |
| [`subsystem_css_distance`](#subsystem_css_distance) | dressed distance of a CSS subsystem (gauge) code |
| [`stabilizer_distance`](#general-non-css-codes) | general (non-CSS) stabilizer-code distance |
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
| `verbose` | `False` | stream per-level progress to stderr |

Each of them returns a plain `int`: the distance (or, for the operator-weight functions, the
coset weight), with `-1` meaning the value is not defined for the input — an empty code, or a
code with no logical qubits, which has no non-trivial operator to measure.

## `css_distance`

```python
css_distance(Hx, Hz=None, *, method="bz", which="min", backend="auto",
             threads=0, verbose=False) -> int
```

`Hx` and `Hz` are any 0/1 arrays (coerced to `uint8` and reduced mod 2).

```python
Hx, Hz = codes.steane()
print(df.css_distance(Hx, Hz))                             # 3

Hx, Hz = codes.toric(10)                                   # [[200, 2, 10]], sparse
print(df.css_distance(Hx, Hz, method="cc"))                # 10, in milliseconds

d_z = df.css_distance(Hx, Hz, which="z", method="cc")
d_x = df.css_distance(Hx, Hz, which="x", method="cc")
print(d_z, d_x)                                            # 10 10
```

Method choice is the single most important thing to know: `"cc"` exploits low-weight
stabilizers and typically certifies sparse qLDPC codes in milliseconds regardless of `n`;
`"bz"` enumerates low-weight codeword classes and is the route for dense or random codes
(and the only GPU-accelerated one). On a qLDPC code, `method="cc"` on the CPU beats
`method="bz"` on the GPU by orders of magnitude — see [Benchmarks](benchmarks.md).

## `classical_distance`

```python
classical_distance(H, *, method="bz", backend="auto",
                   threads=0, verbose=False) -> int
```

Minimum distance of a classical linear code from its parity-check matrix:

```python
H = codes.hamming_parity(3)                     # the [7, 4, 3] Hamming code
print(df.classical_distance(H))                 # 3

print(df.classical_distance(codes.repetition_parity(8)))  # 8
```

A code of dimension 0 has no non-zero codeword, so its distance is undefined:

```python
import numpy as np
print(df.classical_distance(np.eye(6, dtype=np.uint8)))   # -1
```

## `operator_weight`

```python
operator_weight(Gx, Gz, operator, *, method="bz", backend="auto",
                threads=0, verbose=False) -> int
```

The minimum weight of a Pauli modulo the group `<Gx, Gz>` — the minimum-weight coset
leader. The Z-part is minimized over `z + rowspace(Gz)` and the X-part over
`x + rowspace(Gx)` independently, and the larger of the two coset weights is returned:

```python
Gx, Gz = codes.steane()
print(df.operator_weight(Gx, Gz, "ZZZZZZZ"))          # 3 — a logical Z of Steane

print(df.operator_weight(Gx, Gz, "IIIZZZZ"))          # 0 — a stabilizer is the identity
```

The operator may be a Pauli string (`I/X/Y/Z`, where `Y` sets both the Z and X bits), a
`(z_vec, x_vec)` pair, or a length-`2n` symplectic `[z|x]` array.
Operator weight reduces to the core distance problem, so it accepts `method="bz"` and
`method="mitm"` (`"cc"` falls back to `bz`).

## `subsystem_css_distance`

```python
subsystem_css_distance(Gx, Gz=None, *, method="bz", which="min", backend="auto",
                       threads=0, verbose=False) -> int
```

Takes the **gauge generators** of a CSS subsystem code and returns its **dressed**
distance: the minimum weight of an operator that commutes with the stabilizer group (the
center of the gauge group) but is not itself in the gauge group. The center is computed
internally, and all three engines apply — `cc` keeps its sparsity advantage on
topological subsystem codes.

```python
Gx, Gz = codes.bacon_shor(3)                            # gauge generators, n = 9
print(df.subsystem_css_distance(Gx, Gz, method="cc"))   # 3
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
print(df.stabilizer_distance(S))                        # 3

print(df.pauli_operator_weight(S, S[0]))                # 0 — a stabilizer
print(df.pauli_operator_weight(S, "XXXXX"))             # 3 — a logical X, reduced
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
dominate) for the same result.

```python
print(df.available_backends())                          # ['cpu', 'gpu']
```
