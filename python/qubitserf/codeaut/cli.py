"""Command-line interface: ``codeaut`` -- automorphism groups of codes.

Examples::

    codeaut stabs.txt --json               # CSS code from Pauli stabiliser strings (I/X/Z)
    printf 'XXX...\nZZZ...\n' | codeaut     # ... piped on stdin (no file => read stdin)
    codeaut --gen G.txt                    # Aut(C) of a single classical code
    codeaut --list-backends
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _parse_matrix(text):
    """Parse whitespace-separated 0/1 rows (blank lines ignored) into a GF(2) matrix."""
    rows = [[int(x) for x in line.split()] for line in text.splitlines() if line.strip()]
    return np.array(rows, dtype=np.uint8)


def _read_text(path):
    return sys.stdin.read() if path == "-" else open(path).read()


def _load_matrix(path):
    return _parse_matrix(_read_text(path))


def _css_from_pauli(text):
    """Parse Pauli stabiliser strings into a CSS ``(Hx, Hz)`` pair.

    Each non-blank line is one stabiliser: a string over ``I``/``X``/``Z`` (case-insensitive),
    all of the same length ``n``.  For a **CSS** code every stabiliser must be pure-X or pure-Z;
    an ``X``/``Z`` mix, a ``Y``, or any other character makes the code non-CSS and raises
    ``ValueError``.  X-type rows become ``Hx`` (1 at each X), Z-type rows become ``Hz`` (1 at
    each Z); all-identity rows are dropped.
    """
    rows = [ln.strip().upper() for ln in text.splitlines() if ln.strip()]
    if not rows:
        raise ValueError("no stabilisers on input")
    n = len(rows[0])
    hx, hz = [], []
    for i, s in enumerate(rows):
        if len(s) != n:
            raise ValueError(f"stabiliser {i} has length {len(s)}, expected {n} "
                             "(every stabiliser must span the same number of qubits)")
        bad = sorted(set(s) - set("IXZ"))
        if bad:
            raise ValueError(f"stabiliser {i} ({s!r}) contains {bad[0]!r}: only I, X, Z are "
                             "allowed -- a Y (or any X/Z mix) means the code is not CSS")
        xs = [1 if c == "X" else 0 for c in s]
        zs = [1 if c == "Z" else 0 for c in s]
        if any(xs) and any(zs):
            raise ValueError(f"stabiliser {i} ({s!r}) mixes X and Z: not a CSS code")
        if any(xs):
            hx.append(xs)
        elif any(zs):
            hz.append(zs)
    Hx = np.array(hx, dtype=np.uint8) if hx else np.zeros((0, n), np.uint8)
    Hz = np.array(hz, dtype=np.uint8) if hz else np.zeros((0, n), np.uint8)
    return Hx, Hz


def _print_group(order, generators, *, tag=None):
    """Print just the group order and its generators (one per line)."""
    suffix = f"  ({tag})" if tag else ""
    print(f"order={order}{suffix}")
    for g in generators:
        print(" ".join(str(x) for x in g))


def main(argv=None) -> int:
    import qubitserf.codeaut as codeaut

    ap = argparse.ArgumentParser(prog="codeaut",
                                 description="Automorphism groups of binary linear and CSS codes.")
    ap.add_argument("input", nargs="?", default="-",
                    help="CSS code as Pauli stabiliser strings (I/X/Z), one stabiliser per line: "
                         "a file path, or omitted / '-' to read from stdin. Non-CSS input "
                         "(a Y, or a row mixing X and Z) is rejected.")
    ap.add_argument("--gen", help="GF(2) generator matrix of a single classical code "
                                  "(a file path, or '-' for stdin)")
    ap.add_argument("--method", default="auto", choices=["auto", "leon", "bz"],
                    help="CSS engine: 'auto' (full ladder), 'leon' (Leon + dual-code trick), or "
                         "'bz' (joint Brouwer-Zimmermann + nauty/Traces graph incidence, best "
                         "for LDPC codes)")
    ap.add_argument("--max-dim", type=int, default=24, help="Leon enumeration cap (2**eff_dim)")
    ap.add_argument("--spanning-set", default="minweight",
                    choices=["minweight", "congruence", "auto", "minimal"],
                    help="classical --gen only: Leon codeword selector ('minweight' legacy "
                         "prefix, 'congruence' best spanning weight-residue class, or 'auto' "
                         "use the class only when it shrinks the incidence, or 'minimal' "
                         "support-minimal/cocircuit filtering)")
    ap.add_argument("--max-modulus", type=int, default=None,
                    help="classical --gen only: largest modulus searched by congruence/auto "
                         "(default: n+1)")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"],
                    help="Brouwer-Zimmermann enumeration backend ('gpu' falls back to cpu "
                         "if no GPU is detected)")
    ap.add_argument("--max-threads", type=int, default=None,
                    help="cap CPU worker threads for the BZ enumeration (default: all cores)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--list-backends", action="store_true")
    args = ap.parse_args(argv)

    if args.list_backends:
        print(" ".join(codeaut.available_backends()))
        return 0

    if args.gen:
        res = codeaut.code_automorphism_group(
            _load_matrix(args.gen), max_dim=args.max_dim, spanning_set=args.spanning_set,
            max_modulus=args.max_modulus)
        if args.json:
            print(json.dumps({"order": str(res.order), "generators": res.generators,
                              "spanning_set": res.spanning_set,
                              "modulus": res.modulus, "residue": res.residue,
                              "num_codewords": res.num_codewords,
                              "num_incidences": res.num_incidences,
                              "weight_classes": res.weight_classes}))
        else:
            _print_group(res.order, res.generators)
        return 0

    # CSS code as Pauli stabiliser strings, from the positional file or stdin.
    try:
        Hx, Hz = _css_from_pauli(_read_text(args.input))
    except ValueError as e:
        print(f"codeaut: {e}", file=sys.stderr)
        return 2
    code = codeaut.CSSCode(Hx, Hz)

    res = codeaut.css_automorphism_group(code, method=args.method, max_dim=args.max_dim,
                                         backend=args.backend, max_threads=args.max_threads)
    if args.json:
        print(json.dumps({"order": res.order, "complete": res.complete,
                          "generators": res.generators, "method": res.method}))
    else:
        _print_group(res.order, res.generators,
                     tag=None if res.complete else "lower bound")
        # A lower bound is only ever a partial answer -- tell the user WHY the chosen engine
        # stopped short (e.g. Leon's 2**eff_dim enumeration was over --max-dim) and what to do,
        # instead of leaving a bare "order=1  (lower bound)".  res.method carries the reason.
        if not res.complete:
            print(f"codeaut: {res.method}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
