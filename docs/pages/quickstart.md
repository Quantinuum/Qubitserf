# Quickstart

This page assumes you've [built the native library and installed the package](installation.md).

## Hello, distance

```python
import qubitserf as df
from qubitserf import codes

# Which backends did the build give us?
print(df.available_backends())          # e.g. ['cpu', 'gpu']

# A CSS quantum code: the toric code on an 8x8 torus, [[128, 2, 8]].
Hx, Hz = codes.toric(8)
r = df.css_distance(Hx, Hz, backend="auto")
print(r.distance, r.backend, r.seconds)  # 8 gpu ...
```

`css_distance` returns a [`Result`](api.md#result) dataclass with the distance, a proven
lower bound, whether the result is certified (`proven`), how many weight levels were
enumerated, the wall-clock seconds, and which backend ran.

## Classical linear codes

Pass a parity-check matrix `H` to `classical_distance`:

```python
import qubitserf as df
from qubitserf import codes

# The [7, 4, 3] Hamming code.
H = codes.hamming_parity(3)
print(df.classical_distance(H).distance)        # 3

# The [n, 1, n] repetition code.
print(df.classical_distance(codes.repetition_parity(8)).distance)  # 8
```

## Choosing a method

By default `css_distance` uses Brouwer–Zimmermann (`method="bz"`). For
**sparse / LDPC / bivariate-bicycle** codes, use **connected cluster** instead — it
certifies codes BZ cannot:

```python
from qubitserf import codes
import qubitserf as df

# IBM gross code: a [[144, 12, 12]] bivariate-bicycle code.
Hx, Hz = codes.gross_code()
r = df.css_distance(Hx, Hz, method="cc")
print(r.distance, r.proven, round(r.seconds, 3))   # 12 True ~0.3
```

The same code under `method="bz"` *finds* distance 12 instantly via the random-information-set
seed but can only **bracket** it as `[lower, upper]` (it would need an intractable
enumeration to certify the lower bound). See [Algorithms](algorithms.md) for the full
story and the [rule of thumb](algorithms.md#rule-of-thumb).

## Z-distance, X-distance, or the minimum

```python
dz   = df.css_distance(Hx, Hz, which="z").distance     # Z-distance only
dx   = df.css_distance(Hx, Hz, which="x").distance     # X-distance only
dmin = df.css_distance(Hx, Hz, which="min").distance   # min(dz, dx)  -- the default
```

## Selecting a backend

Brouwer–Zimmermann can run on `cpu` or `gpu` (or `auto`, which picks the best
available backend). The `gpu` option auto-detects the machine-specific accelerator. The
`cc` and `mitm` methods are CPU-only.

```python
df.css_distance(Hx, Hz, backend="cpu")     # force the multicore CPU backend
df.css_distance(Hx, Hz, backend="gpu")     # use the available GPU accelerator
df.css_distance(Hx, Hz, backend="auto")    # let qubitserf choose (default)
```

## Command line

The same searches are available from a terminal via the `qubitserf` command (or
`python -m qubitserf`). For example, with a builtin code:

```bash
qubitserf --builtin gross --method cc --json
```

or by piping in a code description. See the [CLI reference](cli.md) for input formats and
all options.
