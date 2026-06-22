# Python API reference

The public surface is small. Everything is importable from the top-level `qminweight`
package:

```python
import qminweight as df
from qminweight import codes

df.css_distance        # exact CSS-code distance
df.classical_distance  # exact classical linear-code distance
df.available_backends  # list usable backends
df.version             # library version string
df.Result              # the result dataclass
codes                  # code generators (toric, gross_code, hamming_parity, ...)
```

---

## `css_distance`

```python
css_distance(Hx, Hz, *, method="bz", which="min", backend="auto",
             threads=0, max_weight=0, verbose=False) -> Result
```

Exact minimum distance of a CSS code given its X- and Z-check matrices.

**Parameters**

| Name | Type | Default | Meaning |
|---|---|---|---|
| `Hx` | 2-D array of 0/1 | — | X-check (stabilizer) matrix. Coerced to `uint8` and reduced mod 2. |
| `Hz` | 2-D array of 0/1 | — | Z-check matrix, same shape conventions as `Hx`. |
| `method` | str | `"bz"` | `"bz"` (Brouwer–Zimmermann), `"cc"` (connected cluster), or `"mitm"` (meet-in-the-middle). |
| `which` | str | `"min"` | `"min"` = `min(dX, dZ)`, `"z"` = Z-distance, `"x"` = X-distance. (`"m"` is also accepted as `"min"`.) |
| `backend` | str | `"auto"` | `"auto"`, `"cpu"`, or `"gpu"`. Used by `bz`; `gpu` auto-detects the available accelerator. `cc` and `mitm` are CPU-only. |
| `threads` | int | `0` | Number of CPU threads; `0` means use hardware concurrency. |
| `max_weight` | int | `0` | Safety cap on the enumeration weight; `0` means no cap (search up to the full search dimension). |
| `verbose` | bool | `False` | Print per-level progress from the native core. |

**Returns** a [`Result`](#result).

`which` is validated in Python and raises `ValueError` if it is not one of `min`/`z`/`x`.
The matrices are made C-contiguous `uint8` and reduced mod 2 before being passed to the
native library, so you can hand in any 0/1 NumPy array (or anything `np.asarray` accepts).

```python
Hx, Hz = codes.steane()
r = df.css_distance(Hx, Hz)
print(r.distance, r.proven)   # 3 True
```

---

## `classical_distance`

```python
classical_distance(H, *, method="bz", backend="auto",
                   threads=0, max_weight=0, verbose=False) -> Result
```

Minimum distance of a classical linear code from its parity-check matrix `H`. Every
nonzero codeword counts (there is no logical/stabilizer distinction), so this returns the
smallest Hamming weight of a nonzero vector in `ker(H)`.

**Parameters** — as for `css_distance`, minus `which`. (`backend` selects the BZ backend;
`cc` is a CSS-only method, so for classical codes use `bz` or `mitm`.)

**Returns** a [`Result`](#result).

```python
print(df.classical_distance(codes.hamming_parity(3)).distance)   # 3
```

---

## `available_backends`

```python
available_backends() -> list[str]
```

Returns the list of public backends usable on this machine. Always contains `"cpu"`;
adds `"gpu"` if an accelerator backend was compiled in and is available at runtime.

```python
print(df.available_backends())   # ['cpu', 'gpu']
```

---

## `version`

```python
version() -> str
```

The library version string reported by the native core, e.g. `"0.1.0"`.

---

## `Result`

```python
@dataclass
class Result:
    distance: int       # best upper bound found (the exact distance, when proven)
    lower_bound: int    # proven lower bound (== distance when proven)
    proven: bool        # True iff the result is certified exact (lower == upper)
    levels: int         # how many weight levels were enumerated
    seconds: float      # wall-clock time of the solve
    backend: str        # which public backend ran ("cpu" or "gpu")
```

The dataclass returned by `css_distance` and `classical_distance`.

| Field | Meaning |
|---|---|
| `distance` | The best (smallest) logical-codeword weight found — i.e. the minimum distance whenever `proven` is `True`. Acts as the **upper** end of the bracket. |
| `lower_bound` | The proven lower bound. When `proven`, this equals `distance`. When not, `[lower_bound, distance]` is a rigorous bracket containing the true distance. |
| `proven` | `True` when the search closed (`lower_bound == distance`); the distance is then certified exact. |
| `levels` | Number of weight levels the search enumerated — useful for understanding cost. |
| `seconds` | Wall-clock time of the solve. |
| `backend` | The backend that ran the search. |

**Reading a bracketed (non-proven) result.** Brouwer–Zimmermann can return
`proven=False` on codes whose information-set lower bound is weak (e.g. bivariate-bicycle);
then `lower_bound` and `distance` may differ and you have a guaranteed bracket. To certify
such a code, switch to `method="cc"`:

```python
Hx, Hz = codes.gross_code()
bz = df.css_distance(Hx, Hz, method="bz")
print(bz.distance, bz.lower_bound, bz.proven)   # e.g. 12 8 False  -> bracket [8, 12]

cc = df.css_distance(Hx, Hz, method="cc")
print(cc.distance, cc.lower_bound, cc.proven)   # 12 12 True  -> certified
```

---

## `qminweight.codes` — code generators

Generators for classical parity-check matrices and CSS check-matrix pairs. Classical
generators return a single `H`; CSS generators return a `(Hx, Hz)` tuple. All matrices are
`numpy.uint8` 0/1 arrays.

### Classical parity-check matrices

| Function | Returns | Notes |
|---|---|---|
| `repetition_parity(n)` | `H` of the `[n, 1, n]` repetition code | `(n-1) x n` |
| `cyclic_repetition_parity(n)` | closed-loop parity checks (rank `n-1`) | the HGP of this gives the toric code |
| `hamming_parity(r)` | `H` of the `[2^r - 1, 2^r - 1 - r, 3]` Hamming code | columns are the nonzero `r`-bit patterns |
| `random_ldpc_parity(m, n, col_weight=3, seed=0)` | random column-regular LDPC `H` | for benchmarking only |

### CSS check-matrix pairs `(Hx, Hz)`

| Function | Returns | Code |
|---|---|---|
| `steane()` | `(Hx, Hz)` with `Hx = Hz =` Hamming`[7,4,3]` | Steane `[[7, 1, 3]]` |
| `shor()` | `(Hx, Hz)` | Shor `[[9, 1, 3]]` |
| `hypergraph_product(H1, H2)` | `(Hx, Hz)` from the hypergraph product of two parity checks | general HGP construction |
| `toric(L)` | `(Hx, Hz)` | toric code `[[2L², 2, L]]` (HGP of a length-`L` cyclic repetition code) |
| `surface(L)` | `(Hx, Hz)` | planar surface code `[[L² + (L-1)², 1, L]]` (HGP of the `[L, 1, L]` repetition code) |
| `bivariate_bicycle(l, m, a_terms, b_terms)` | `(Hx, Hz)` | bivariate-bicycle CSS code (Bravyi *et al.*, Nature 2024); `[[2 l m, k, d]]` |
| `gross_code()` | `(Hx, Hz)` | IBM gross code `[[144, 12, 12]]`, `l=12, m=6` |

#### `hypergraph_product(H1, H2)`

CSS code from the hypergraph product of two parity-check matrices, with

```
Hx = [ H1 ⊗ I_{n2} | I_{m1} ⊗ H2ᵀ ]
Hz = [ I_{n1} ⊗ H2 | H1ᵀ ⊗ I_{m2} ]
```

#### `bivariate_bicycle(l, m, a_terms, b_terms)`

`x = S_l ⊗ I_m` and `y = I_l ⊗ S_m` are commuting cyclic shifts on `l·m` qubits per
block. `A` is the sum of the monomials in `a_terms`, `B` the sum in `b_terms`, where each
term is a `("x" | "y", power)` pair. Then `Hx = [A | B]` and `Hz = [Bᵀ | Aᵀ]`, giving a
`[[2 l m, k, d]]` code.

```python
Hx, Hz = codes.bivariate_bicycle(
    12, 6,
    [("x", 3), ("y", 1), ("y", 2)],   # A = x^3 + y + y^2
    [("y", 3), ("x", 1), ("x", 2)],   # B = y^3 + x + x^2
)
```

#### `gross_code()`

A convenience wrapper: `bivariate_bicycle(12, 6, [("x",3),("y",1),("y",2)],
[("y",3),("x",1),("x",2)])`, i.e. the IBM `[[144, 12, 12]]` gross code (Bravyi *et al.*,
Nature 2024).

```python
Hx, Hz = codes.gross_code()
print(df.css_distance(Hx, Hz, method="cc").distance)   # 12
```
