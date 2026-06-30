# qminweight

`qminweight` computes exact minimum distances for CSS quantum codes and classical
linear codes. It provides a C++ core, a Python API, and a command-line tool. Beyond code
distance it also computes:

- **Operator weight** — the minimum weight of a Pauli operator modulo the stabilizer (or
  gauge) group (the minimum-weight coset leader).
- **Subsystem CSS dressed distance** — the distance of a CSS subsystem (gauge) code,
  computed correctly as the dressed distance.

Both are solved by the same Brouwer–Zimmermann / connected-cluster / meet-in-the-middle
engines, by reducing them to the core distance problem.

## Install

Requirements:

- Python 3.9 or newer.
- CMake 3.20 or newer.
- A C++17 compiler.

Install from the package directory:

```bash
cd research/distance/qminweight
python -m pip install .
```

For development, install the Python package with test dependencies:

```bash
python -m pip install -e ".[dev]"
```

To build the native library, C++ CLI, and C++ tests in-tree:

```bash
./build.sh
```

The package install provides the `qminweight` command. The development build also
creates `build/qminweight`.

## Command-Line Use

`qminweight` accepts CSS stabiliser strings, CSS parity-check matrices, or a
classical parity-check matrix. CSS stabiliser input can come from stdin, a file,
or a pipe. End interactive stdin with EOF, usually `Ctrl-D`.

Small stdin example:

```console
$ qminweight --zx
XXXX
ZZZZ
<Ctrl-D>
2 2
```

Here `--zx` prints both CSS components as `dZ dX`.

A slightly larger CSS stabiliser example:

```text
XXXXIII
XXIIXXI
XIXIXIX
ZZZZIII
ZZIIZZI
ZIZIZIZ
```

From stdin:

```console
$ printf 'XXXXIII\nXXIIXXI\nXIXIXIX\nZZZZIII\nZZIIZZI\nZIZIZIZ\n' | qminweight - --zx
3 3
```

From a file containing the same rows:

```console
$ qminweight example_code.txt
3

$ qminweight example_code.txt --zx
3 3

$ qminweight example_code.txt --which z
3

$ qminweight example_code.txt --method cc --threads 8
3
```

Verbose mode prints progress diagnostics to stderr, imitating the qubitserf
interface: a running lower bound (`Distance bound: >N`, the weight level just
ruled out) with its elapsed time, then the exact `Distance: =N` with the total
elapsed time. The final distance still goes to stdout.

For the connected-cluster CSS distance (`--method cc`), the Z- and X-distance
searches are **interleaved** weight level by weight level and labelled
`Z-`/`X-`, so both lower bounds advance in step — if one side stalls on a hard
level, the other still reports its bound. For example, on a toric `[[288,2,12]]`
code:

```console
$ qminweight --hx Hx.txt --hz Hz.txt --method cc -v
Z-distance bound: >1
Elapsed:[0ms]
X-distance bound: >1
Elapsed:[0ms]
...
Z-distance bound: >11
Elapsed:[2ms]
X-distance bound: >11
Elapsed:[2ms]
Z-distance: =12
Elapsed:[9ms]
X-distance: =12
Elapsed:[9ms]
12
```

Timing varies by machine and problem size.

Machine-readable output, with timing and backend varying by machine:

```console
$ qminweight example_code.txt --json
{"distance": 3, "lower_bound": 3, "proven": true, "seconds": 0.001, "backend": "cpu", "which": "d"}
```

For CSS parity-check matrices:

```console
$ qminweight --hx Hx.txt --hz Hz.txt
3

$ qminweight --hx Hx.mtx --hz Hz.mtx --method cc
3

$ qminweight --hx Hx.mtx --hz Hz.mtx --method bz --backend gpu --json
{"distance": 3, "lower_bound": 3, "proven": true, "seconds": 0.001, "backend": "gpu", "which": "d"}
```

For a classical linear code:

```console
$ qminweight --classical H.txt
3

$ qminweight --classical H.mtx --method bz
3

$ qminweight --classical H.txt --method mitm
3
```

Operator weight — the minimum weight of a Pauli operator modulo the stabilizer group. Pass
the operator (it may contain `Y`) as `--operator STR`; the stabilizer (or, with
`--subsystem`, gauge) generators come from stdin or a file. The default output is
`max(z_weight, x_weight)`; `--zx` prints `z_weight x_weight`. A Steane logical `Z`
(`ZZZZZZZ`) has Z-weight 3 and X-weight 0:

```console
$ printf 'IIIXXXX\nIXXIIXX\nXIXIXIX\nIIIZZZZ\nIZZIIZZ\nZIZIZIZ\n' | qminweight --operator ZZZZZZZ --zx
3 0
```

The development C++ binary uses qubitserf's convention instead — the **last** stdin Pauli
line is the operator and the preceding lines are the generators:

```console
$ printf 'IIIXXXX\nIXXIIXX\nXIXIXIX\nIIIZZZZ\nIZZIIZZ\nZIZIZIZ\nZZZZZZZ\n' | ./build/qminweight -o --zx
3 0
```

A stabilizer fed as the operator has weight 0 (it is equivalent to the identity) — including
on codes whose stabilizers are not self-orthogonal, where qubitserf would report a nonzero
weight. See [Operator weight correctness](#operator-weight-correctness) below.

Subsystem distance — with `--subsystem`, the X/Z input is treated as **gauge generators**
and the **dressed** subsystem distance is computed. For the distance-3 Bacon-Shor code piped
as gauge generators:

```console
$ qminweight --subsystem bacon_shor_d3_gauge.txt
3
```

Show the installed compute backends. The exact list depends on the machine:

```console
$ qminweight --list-backends
cpu
gpu
```

Useful options:

- `--method bz`, `--method cc`, or `--method mitm`.
- `--backend auto`, `--backend cpu`, or `--backend gpu`.
- `--which min`, `--which z`, or `--which x` for CSS codes.
- `-o` / `--operator` for operator weight (last stdin line is the operator).
- `--subsystem` to treat the X/Z input as gauge generators (dressed subsystem distance).
- `--threads N` to set CPU worker threads.
- `--max-weight N` to cap an expensive search and return a certified bracket.
- `-v` or `--verbose` to print progress diagnostics to stderr.
- `--list-backends` to show usable backends.

Use `cc` for sparse CSS codes such as LDPC, topological, and bivariate-bicycle
codes. Use `bz` for dense or random codes, or when using the GPU backend. Use
`mitm` mainly as a small-code cross-check.

## Python Use

The Python API works with NumPy-compatible binary matrices.

```python
import numpy as np
import qminweight as df

Hx = np.loadtxt("Hx.txt", dtype=np.uint8)
Hz = np.loadtxt("Hz.txt", dtype=np.uint8)

r = df.css_distance(Hx, Hz, method="bz", backend="gpu")
print(r.distance, r.proven, r.seconds, r.backend)
```

For a distance-3 example, output looks like:

```text
3 True 0.001 gpu
```

The exact `seconds` value varies by machine. The `gpu` backend automatically
uses the available accelerator for the current platform.

For one CSS component:

```python
d_z = df.css_distance(Hx, Hz, which="z", method="cc")
d_x = df.css_distance(Hx, Hz, which="x", method="cc")
```

For a classical code:

```python
H = np.loadtxt("H.txt", dtype=np.uint8)
r = df.classical_distance(H, method="bz")
print(r.distance)
```

### Method selection

Use `method="cc"` (connected cluster) for **QLDPC codes** — sparse codes such as
bivariate-bicycle, hypergraph-product, toric, and surface codes. It exploits the
low-weight structure of the stabilizers and typically certifies the distance in
milliseconds regardless of code size, where Brouwer-Zimmermann cannot finish.

Use `method="bz"` (Brouwer-Zimmermann) for **general CSS codes** — dense, random,
or codes without exploitable sparsity. It is also the right choice when a GPU is
available, since the GPU backend only accelerates BZ.

```python
# QLDPC / sparse code — CC wins
r = df.css_distance(Hx, Hz, method="cc")

# Dense or random code, or GPU run — BZ wins
r = df.css_distance(Hx, Hz, method="bz", backend="gpu")
```

### Code size limits

There is no fixed qubit-count limit. Codewords are bit-packed into
`ceil(n/64)` 64-bit words and every host routine is sized dynamically, so the
CPU backend (both `bz` and `cc`) handles arbitrary `n`.

The **GPU** runs its native BZ kernel for codes up to **1024 physical qubits**
(codeword stride ≤ 16 words); above that it automatically falls back to the CPU
solver for the same exact result, just without GPU acceleration. (The GPU kernel
also falls back to the CPU for Brouwer-Zimmermann *weight levels* deeper than 32
selected rows, an axis independent of `n`.) Small problems always run on the CPU
regardless of backend, below an internal work threshold where GPU dispatch
latency would dominate (tunable via `QMINWEIGHT_GPU_MIN_WORK`).

```python
# 512- and 1024-qubit dense codes run the native GPU BZ kernel:
r = df.css_distance(Hx, Hz, method="bz", backend="gpu")   # n up to 1024
# n > 1024 transparently uses the CPU solver (same answer).
```

### Thread count

`threads=0` (the default) uses all logical CPU cores
(`std::thread::hardware_concurrency()`). For very small enumeration problems the
library falls back to a single thread automatically regardless of this setting.
Pass an explicit count to cap parallelism, for example when running multiple
solves concurrently:

```python
# Use 4 CPU threads instead of all available cores.
r = df.css_distance(Hx, Hz, method="cc", threads=4)
```

`Result` fields:

- `distance`: best distance found; exact when `proven` is true.
- `lower_bound`: certified lower bound.
- `proven`: whether `distance == lower_bound`.
- `seconds`: wall-clock runtime.
- `backend`: backend used for the run.

## Operator weight

`operator_weight` returns the minimum weight of a Pauli operator modulo the stabilizer (or
gauge) group — the minimum-weight coset leader. The Z-part is minimized over
`z + rowspace(Gz)` and the X-part over `x + rowspace(Gx)`, independently.

```python
import qminweight as df
from qminweight import codes

Gx, Gz = codes.steane()
op = df.operator_weight(Gx, Gz, "ZZZZZZZ")   # a logical Z of Steane
print(op.z_weight, op.x_weight, op.weight)   # 3 0 3

# A stabilizer is equivalent to the identity, so it has weight 0:
print(df.operator_weight(Gx, Gz, "IIIZZZZ").weight)   # 0
```

The `operator` argument may be a Pauli string (`I/X/Y/Z`, a `Y` sets both the Z and X bits),
a `(z_vec, x_vec)` pair of 0/1 arrays, or a length-`2n` symplectic `[z|x]` array. The
returned `OpResult` carries `z_weight`, `x_weight`, `weight` (`= max(z, x)`), `proven`,
`seconds`, and `backend`. Operator weight reduces to the core distance problem, so it accepts
`method="bz"` and `method="mitm"` (`"cc"` falls back to `bz`).

### Operator weight correctness

This is the operator-weight feature of Quantinuum's **qubitserf**, but with a correctness
fix. qubitserf matches MITM syndromes against `[Hz; X̄]`, which is a valid parity-check of the
Z-stabilizer group only when `Hz·Hzᵀ = 0`. For codes whose stabilizers are *not*
self-orthogonal — surface, toric, bivariate-bicycle — it returns wrong answers: feeding a
single Z-stabilizer of the planar `surface(3)` `[[13,1,3]]` code returns `3`, when the correct
answer is `0`. qminweight instead computes the minimum-weight coset leader directly (the
correct syndrome matrix is `nullspace(Gz)`), so a stabilizer always has weight 0:

```python
import numpy as np
Gx, Gz = codes.surface(3)
stab = (Gz[0], np.zeros_like(Gz[0]))          # a single Z-stabilizer as the operator
print(df.operator_weight(Gx, Gz, stab).weight)   # 0, not 3
```

## Subsystem distance

`subsystem_css_distance` takes the **gauge generators** of a CSS subsystem code and returns
its **dressed** distance: the minimum weight of an operator that commutes with the stabilizer
group (the center of the gauge group) but is not itself in the gauge group. This is distinct
from the *bare* distance, which additionally forbids dressing by gauge operators. The
stabilizer center is computed internally, and the problem reuses the BZ, CC, and MITM
engines (CC keeps its sparsity advantage on topological subsystem codes).

```python
import qminweight as df
from qminweight import codes

Gx, Gz = codes.bacon_shor(3)                 # gauge generators, n = 9
r = df.subsystem_css_distance(Gx, Gz, method="cc")
print(r.distance)                            # 3
```

A stabilizer code is the special case `gauge = stabilizers`, where the dressed subsystem
distance coincides with `css_distance`:

```python
Hx, Hz = codes.steane()
print(df.subsystem_css_distance(Hx, Hz).distance)   # 3 == css_distance(Hx, Hz)
```

## General (non-CSS) codes

The functions above are CSS-only (separate `Hx`/`Hz`). For a **general stabilizer code** —
any commuting set of Paulis, including ones with `Y` or stabilisers that mix `X` and `Z` —
give a **symplectic stabilizer matrix** `S` of shape `(m, 2n)` in `[z | x]` column order:
row `r` is the Pauli with Z-support `S[r, :n]` and X-support `S[r, n:]` (a `Y` sets both).
The distance is the minimum **symplectic** weight — the number of qubits touched, *not* the
sum of the Z- and X-weights — of an operator in the normalizer that is not a stabilizer.

```python
import numpy as np
import qminweight as df

# The [[5,1,3]] perfect code  XZZXI / IXZZX / XIXZZ / ZXIXZ  as a [z | x] matrix.
def paulis(strings):
    n = len(strings[0]); rows = []
    for s in strings:
        z = np.zeros(n, np.uint8); x = np.zeros(n, np.uint8)
        for j, c in enumerate(s):
            if c in "XY": x[j] = 1
            if c in "ZY": z[j] = 1
        rows.append(np.concatenate([z, x]))
    return np.array(rows, np.uint8)

S = paulis(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
print(df.stabilizer_distance(S).distance)            # 3
```

Both **Brouwer–Zimmermann** and **meet-in-the-middle** work for non-CSS codes:

- `method="bz"` (default) uses the weight-doubling isometry `(a|b) → (a|b|a⊕b)`, under which
  the Hamming weight of the length-`3n` image is exactly twice the symplectic weight. The
  symplectic-distance problem thus becomes an ordinary binary Hamming-distance problem and is
  solved by the existing BZ engine (with its GPU enumeration); the symplectic distance is half
  the binary distance. This is the `SAVED_ISOMETRY` reduction of Sabater–Vera et al.
  ([arXiv:2408.10743](https://arxiv.org/abs/2408.10743)).
- `method="mitm"` enumerates by qubit-support with the three nonzero Paulis `Z/X/Y` per chosen
  qubit.
- `method="cc"` has no non-CSS form (CC needs a sparse single-type CSS Tanner graph), so it
  **falls back to MITM** with a one-line stderr note.

A code whose rows are all pure-X/pure-Z is detected and routed to the fast CSS solvers
unchanged, so `stabilizer_distance` on a CSS matrix matches `css_distance`.

**Non-CSS subsystem (dressed) distance** takes the gauge generators `G` as a `(m, 2n)`
`[z | x]` matrix (the generators may be non-commuting); the stabilizer center is computed
internally:

```python
print(df.subsystem_stabilizer_distance(G).distance)  # dressed distance
```

**Operator weight** of a general Pauli modulo a group `<G>` (the non-CSS analogue of
`operator_weight`) is the minimum symplectic weight over the coset `operator + rowspace(G)`,
which is `0` exactly when the operator is itself in the group:

```python
print(df.pauli_operator_weight(S, S[0]).weight)      # 0 (a stabilizer)
print(df.pauli_operator_weight(S, "XXXXX").weight)   # 3 (logical X, reduced)
```

On the command line, a Pauli-stabiliser code containing a `Y` or a row mixing `X` and `Z` is
auto-detected as non-CSS and routed to the symplectic solver; `--symplectic FILE` reads a
`(m, 2n)` `[z | x]` 0/1 matrix directly:

```bash
printf 'XZZXI\nIXZZX\nXIXZZ\nZXIXZ\n\n' | qminweight -        # 3
qminweight --symplectic S.txt                                 # non-CSS distance
qminweight --symplectic G.txt --subsystem                     # dressed distance
```

## Benchmarks

Benchmark data is generated by the scripts in [`bench/`](bench/). The tables
below are from [`bench/comprehensive_results.md`](bench/comprehensive_results.md)
and [`bench/results.md`](bench/results.md). Reference results use
the [`codedistance`](https://pypi.org/project/codedistance/) package from
[`codeDistancePYPI`](https://github.com/m-webster/codeDistancePYPI), specifically
[`BZDistMW`](https://github.com/m-webster/codeDistancePYPI) and
[`connectedClusterMW`](https://github.com/m-webster/codeDistancePYPI), with a
30 second per-code timeout.

Max stab. weight is the largest row weight of `Hx` = `Hz`; these codes are all QLDPC
(low, `n`-independent stabilizer weight), which is the regime `cc` exploits.

| code | n | d | max stab. weight | qminweight result | reference [`BZDistMW`](https://github.com/m-webster/codeDistancePYPI) | reference [`connectedClusterMW`](https://github.com/m-webster/codeDistancePYPI) |
|---|---:|---:|---:|---|---|---|
| toric L=7 | 98 | 7 | 4 | `cc`: 1.1 ms, `bz gpu`: 21.5 ms | 19.58 s | 14.7 ms |
| toric L=8 | 128 | 8 | 4 | `cc`: 1.3 ms, `bz gpu`: 76.2 ms | timeout | 47.7 ms |
| surface L=12 | 265 | 12 | 4 | `cc`: 5.2 ms, `bz gpu`: `[8,12]` 7.6 s (cpu 153.7 s) | — | 3.42 s |
| toric L=12 | 288 | 12 | 4 | `cc`: 6.3 ms, `bz gpu`: `[8,12]` 14.1 s (cpu 283.8 s) | — | 5.19 s |
| gross `[[144,12,12]]` | 144 | 12 | 6 | `cc`: 185 ms, `bz gpu`: `[8,12]` 287 ms | timeout | timeout |

Summary from the comprehensive benchmark:

- All qminweight methods agreed with the known distance or the reference result
  wherever the reference completed correctly.
- Against reference [`BZDistMW`](https://github.com/m-webster/codeDistancePYPI),
  `qminweight cc` had a median speedup of 12.8x and a maximum speedup of 18232x
  over the completed comparison runs.
- Against reference
  [`connectedClusterMW`](https://github.com/m-webster/codeDistancePYPI),
  `qminweight cc` had a median speedup of 11.3x and a maximum speedup of 821x.
- `qminweight cc` scales where everything else stops: it certifies the exact
  distance of every code out to **n = 288** (toric up to L=12) in single- to
  low-hundreds of milliseconds. Brouwer-Zimmermann is only attempted up to n = 144
  (above that it cannot finish in budget) and the reference `connectedClusterMW`
  is 100–800x slower where it finishes at all (e.g. toric L=12: `cc` 6.3 ms vs
  5.19 s, 821x).
- On Brouwer-Zimmermann runs where the enumeration is the actual cost (CPU solve
  > ~10 ms — the regime the GPU backend is meant for), the GPU is many times
  faster than the CPU on the Apple M4: e.g. toric L=7 11.3x, toric L=8 18.2x,
  gross 15.2x (the `bz_cpu / bz_gpu` column of
  [`comprehensive_results.md`](bench/comprehensive_results.md)); the focused
  [`bench/gpu_vs_cpu.py`](bench/gpu_vs_cpu.py) reports a median **~13x** over the
  codes in that regime. Sub-millisecond codes are dominated by the shared
  random-information-set seed and per-dispatch latency, so CPU and GPU tie there
  (the GPU path falls back to the CPU below `QMINWEIGHT_GPU_MIN_WORK`).
- `qminweight cc` was the only benchmarked method that certified the gross
  `[[144,12,12]]` distance within the timeout.

Each family plot shows wall-clock time vs `n` (log scale) for every method.

### Toric Codes

![Toric code benchmark](bench/comprehensive_toric.png)

### Surface Codes

![Surface code benchmark](bench/comprehensive_surface.png)

### Reed-Muller CSS Codes (dense, non-QLDPC)

The `reed_muller_r1` and `reed_muller_r2` families use quantum Reed-Muller CSS codes
`Hx = Hz = G_RM(r, m)` (valid when `2r < m-1`): parameters `[[2^m, 2^m − 2·Σ C(m,i), 2^(r+1)]]`.
These codes are emphatically **not QLDPC**: the stabilizer (`Hx`=`Hz` row) weights are the
monomial-evaluation weights `2^m, 2^(m-1), …, 2^(m-r)`, so the *maximum* stabilizer weight is
`2^m = n` (the all-ones constant monomial) and even the *minimum* is `n/2^r`, both **growing
with `n`** (the opposite of a QLDPC family, whose stabilizer weight is constant). Because every
check is that dense, the connected-cluster algorithm degenerates to `O(C(n,d))` work just like
MITM, while BZ finds the small-weight codewords quickly. The `bz` column below is the GPU backend.

| family | code | n | d | max stab. weight | `cc` | `bz gpu` |
|---|---|---:|---:|---:|---|---|
| `reed_muller_r1` | `[[16,6,4]]`   | 16  | 4 | 16  | < 1 ms  | < 1 ms |
| `reed_muller_r1` | `[[256,238,4]]`| 256 | 4 | 256 | ~6 ms   | ~9 ms |
| `reed_muller_r1` | `[[512,492,4]]`| 512 | 4 | 512 | ~82 ms  | ~0.11 s |
| `reed_muller_r2` | `[[64,20,8]]`  | 64  | 8 | 64  | ~136 s  | ~12 ms |
| `reed_muller_r2` | `[[128,70,8]]` | 128 | 8 | 128 | timeout | ~0.8 s |
| `reed_muller_r2` | `[[256,182,8]]`| 256 | 8 | 256 | timeout | 216 s |

For `d=4` (r=1 family) the search stays shallow, so every method remains fast (sub-second)
out to `n=512`. For `d=8` (r=2 family) the gap is stark: CC is already ~10⁴× slower than BZ at
`n=64` (~136 s vs ~12 ms) and times out entirely at `n=128` and `n=256`, while BZ still
certifies — instantly at `n=128`, and even at `n=256` (where its own bound converges slowly,
216 s) it is the *only* method that finishes at all.

![Reed-Muller r=1 benchmark](bench/comprehensive_reed_muller_r1.png)

![Reed-Muller r=2 benchmark](bench/comprehensive_reed_muller_r2.png)

Full benchmark tables and additional plots:

- [`bench/comprehensive_results.md`](bench/comprehensive_results.md)
- [`bench/results.md`](bench/results.md)
- [`bench/cc_results.md`](bench/cc_results.md)

## Citation

If you use qminweight in academic work, please cite the package and the underlying methods.

```bibtex
@software{qminweight,
  title  = {qminweight: exact minimum distance of CSS quantum and classical linear codes},
  author = {Serban Cercelescu},
  year   = {2026},
  note   = {C++ core with CPU/CUDA/Metal backends; Brouwer--Zimmermann, connected-cluster,
            and meet-in-the-middle solvers},
  url    = {https://github.com/Quantinuum/qminweight}
}
```
