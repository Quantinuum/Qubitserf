# Command-line interface

!!! note "CLI status"
    This page documents the **intended** command-line interface. The CLI is being built
    alongside the library; the underlying engine (the Python API in [api.md](api.md)) is
    stable, and the CLI is a thin front end over it. Flag names below describe the
    interface shape — consult `qminweight --help` for the exact, current spelling on your
    build.

`qminweight` ships a command-line front end so you can find distances without writing Python.
Two equivalent entry points:

```bash
qminweight ...            # the installed console script
python -m qminweight ...  # the module form (identical behaviour)
```

## Giving it a code

The CLI accepts a code in one of three ways:

1. **Pauli strings on stdin** — a list of stabilizer generators as Pauli strings (the
   Qubitserf-style input), one generator per line. This is convenient for piping from
   other tools.

    ```bash
    cat my_code.txt | qminweight --method cc
    ```

2. **Parity-check matrix files** — pass `--hx` and `--hz` pointing at the X- and Z-check
   matrices (for a CSS code), or a single check matrix for a classical code.

    ```bash
    qminweight --hx hx.txt --hz hz.txt --method bz
    ```

## Options

| Flag | Meaning | Mirrors |
|---|---|---|
| `--method {bz,cc,mitm}` | Which algorithm to run. | `method=` in [`css_distance`](api.md#css_distance) |
| `--backend {auto,cpu,gpu}` | Backend for the BZ enumeration; `gpu` auto-detects the available accelerator (`cc`/`mitm` are CPU). | `backend=` |
| `--which {min,z,x}` | Z-distance, X-distance, or the minimum. | `which=` |
| `--threads N` | CPU threads (`0` = hardware concurrency). | `threads=` |
| `--max-weight W` | Safety cap on the enumeration weight (`0` = no cap). | `max_weight=` |
| `--json` | Emit the result as JSON instead of human-readable text. | — |

Each option maps directly onto the corresponding [Python API](api.md) argument, so the
[algorithm guidance](algorithms.md) and the [rule of thumb](algorithms.md#rule-of-thumb)
apply unchanged.

## Examples

Print both CSS components from Qubitserf-style stdin:

```console
$ qminweight --zx
XXXX
ZZZZ
<Ctrl-D>
2 2
```

Print machine-readable output from a stabilizer file:

```console
$ qminweight example_code.txt --json
{"distance": 3, "lower_bound": 3, "proven": true, "seconds": 0.001, "backend": "cpu", "which": "d"}
```

Find the Z-distance of a CSS code from parity-check files on the GPU:

```bash
qminweight --hx hx.txt --hz hz.txt --method bz --backend gpu --which z
```

Pipe Pauli-string stabilizers in and read a plain-text summary:

```bash
cat stabilizers.txt | qminweight --method cc
```

When Brouwer–Zimmermann can only bracket a hard code, the output reflects that: `proven`
will be `false` and `lower_bound` < `distance`, giving you the rigorous `[lower, upper]`
range. Switch to `--method cc` for sparse codes to certify them — see
[Algorithms](algorithms.md).
