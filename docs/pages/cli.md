# Command-line interface

!!! note "CLI status"
    This page documents the **intended** command-line interface. The CLI is being built
    alongside the library; the underlying engine (the Python API in [api.md](api.md)) is
    stable, and the CLI is a thin front end over it. Flag names below describe the
    interface shape — consult `qubitserf --help` for the exact, current spelling on your
    build.

`qubitserf` ships a command-line front end so you can find distances without writing Python.
Two equivalent entry points:

```bash
qubitserf ...            # the installed console script
python -m qubitserf ...  # the module form (identical behaviour)
```

## Giving it a code

The CLI accepts a code in one of three ways:

1. **Pauli strings on stdin** — a list of stabilizer generators as Pauli strings (the input
   format inherited from the original qubitserf), one generator per line. This is convenient for piping from
   other tools. A code that is all pure-`X`/pure-`Z` rows is CSS; a code containing a `Y`
   or a row that mixes `X` and `Z` is **non-CSS** and is automatically routed to the
   symplectic solver (a single distance number).

    ```bash
    cat my_code.txt | qubitserf --method cc
    ```

2. **Parity-check matrix files** — pass `--hx` and `--hz` pointing at the X- and Z-check
   matrices (for a CSS code), or a single check matrix for a classical code.

    ```bash
    qubitserf --hx hx.txt --hz hz.txt --method bz
    ```

3. **Symplectic stabilizer matrix** — pass `--symplectic` pointing at a 0/1 matrix with
   `2n` columns in `[z | x]` order (row `r` has Z-support in the first `n` columns and
   X-support in the last `n`), for a general non-CSS code.

    ```bash
    qubitserf --symplectic S.txt              # non-CSS distance
    qubitserf --symplectic G.txt --subsystem  # non-CSS dressed (gauge) distance
    ```

## Options

| Flag | Meaning | Mirrors |
|---|---|---|
| `--method {bz,cc,mitm}` | Which algorithm to run. | `method=` in [`css_distance`](api.md#css_distance) |
| `--backend {auto,cpu,gpu}` | Backend for the BZ enumeration; `gpu` auto-detects the available accelerator (`cc`/`mitm` are CPU). | `backend=` |
| `--which {min,z,x}` | Z-distance, X-distance, or the minimum. | `which=` |
| `-o`, `--operator STR` | Compute **operator weight**: the minimum weight of a Pauli operator (`STR`, may contain `Y`) modulo the stabilizer/gauge group. With C++ stdin input, the last Pauli line is the operator and the preceding lines are the generators. | [`operator_weight`](api.md#operator_weight) |
| `--subsystem` | Treat the X/Z (or symplectic) input as **gauge** generators and compute the **dressed** subsystem distance. | [`subsystem_css_distance`](api.md#subsystem_css_distance) / [`subsystem_stabilizer_distance`](api.md#subsystem_stabilizer_distance) |
| `--symplectic FILE` | A symplectic stabiliser matrix (`2n` columns, `[z\|x]` order) for a general non-CSS code. | [`stabilizer_distance`](api.md#stabilizer_distance) |
| `--threads N` | CPU threads (`0` = hardware concurrency). | `threads=` |
| `--max-weight W` | Safety cap on the enumeration weight (`0` = no cap). | `max_weight=` |
| `--json` | Emit the result as JSON instead of human-readable text. | — |

Each option maps directly onto the corresponding [Python API](api.md) argument, so the
[algorithm guidance](algorithms.md) and the [rule of thumb](algorithms.md#rule-of-thumb)
apply unchanged.

## Examples

Print both CSS components from Pauli-string stdin:

```console
$ qubitserf --zx
XXXX
ZZZZ
<Ctrl-D>
2 2
```

Print machine-readable output from a stabilizer file:

```console
$ qubitserf example_code.txt --json
{"distance": 3, "lower_bound": 3, "proven": true, "seconds": 0.001, "backend": "cpu", "which": "d"}
```

Find the Z-distance of a CSS code from parity-check files on the GPU:

```bash
qubitserf --hx hx.txt --hz hz.txt --method bz --backend gpu --which z
```

Pipe Pauli-string stabilizers in and read a plain-text summary:

```bash
cat stabilizers.txt | qubitserf --method cc
```

When Brouwer–Zimmermann can only bracket a hard code, the output reflects that: `proven`
will be `false` and `lower_bound` < `distance`, giving you the rigorous `[lower, upper]`
range. Switch to `--method cc` for sparse codes to certify them — see
[Algorithms](algorithms.md).

## Operator weight

`--operator` computes the minimum weight of a Pauli operator modulo the stabilizer group —
the minimum-weight coset leader (see [Algorithms](algorithms.md#operator-weight)). The Python
CLI takes the operator (it may contain `Y`) as `--operator STR`, with the generators from
stdin or a file; the development C++ CLI uses the `-o` convention instead, where the
**last** stdin Pauli line is the operator and the preceding lines are the generators (and
spells the flag `-o`). The default output is `max(z_weight, x_weight)`; `--zx` prints
`z_weight x_weight`. A Steane logical `Z` has Z-weight 3 and X-weight 0:

```console
$ printf 'IIIXXXX\nIXXIIXX\nXIXIXIX\nIIIZZZZ\nIZZIIZZ\nZIZIZIZ\n' | qubitserf --operator ZZZZZZZ --zx
3 0
```

A stabilizer fed as the operator returns weight 0, even on codes whose stabilizers are not
self-orthogonal (where the original qubitserf returns a nonzero weight — see
[the correctness note](algorithms.md#operator-weight)). Add `--subsystem` to take operator
weight modulo the **gauge** group instead.

## Subsystem distance

`--subsystem` treats the X/Z input as **gauge** generators of a CSS subsystem code and
computes its **dressed** distance (see [Algorithms](algorithms.md#subsystem-dressed-distance)).
It composes with `--which`, `--method`, and the other flags. For the distance-3 Bacon-Shor
code:

```console
$ qubitserf --subsystem bacon_shor_d3_gauge.txt
3
```
