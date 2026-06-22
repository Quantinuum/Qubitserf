"""Console entry point for qminweight.

Usage::

    qminweight [INPUT] [options]

INPUT is a file of Pauli stabiliser strings, ``-`` for stdin, or omitted when the code
is supplied via ``--hx/--hz``, ``--classical`` or ``--builtin``. By default the minimum
distance is printed to stdout; ``--json`` prints a JSON object instead.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import api, io


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qminweight",
        description="Compute the minimum distance of a quantum (CSS) or classical code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "input sources (choose one):\n"
            "  INPUT                Pauli stabiliser strings from a file, or '-' for stdin\n"
            "  --hx FILE --hz FILE  X and Z parity-check matrices (0/1 text or .mtx)\n"
            "  --classical FILE     a single classical parity-check matrix\n"
            "  --builtin NAME[:L]   a generator from qminweight.codes "
            "(steane, shor, toric:L,\n"
            "                       surface:L, gross, repetition:L, hamming:L)\n"
            "\nexamples:\n"
            "  qminweight --builtin steane\n"
            "  qminweight --builtin gross --method cc\n"
            "  cat code.txt | qminweight -\n"
            "  qminweight --hx Hx.txt --hz Hz.txt --zx\n"
        ),
    )
    p.add_argument("input", nargs="?", default=None,
                   help="Pauli stabiliser file, or '-' for stdin")
    p.add_argument("--hx", metavar="FILE", help="X parity-check matrix file")
    p.add_argument("--hz", metavar="FILE", help="Z parity-check matrix file")
    p.add_argument("--classical", metavar="FILE",
                   help="single classical parity-check matrix file")
    p.add_argument("--builtin", metavar="NAME[:L]",
                   help="use a built-in code generator")

    p.add_argument("--method", choices=["bz", "cc", "mitm"], default="bz",
                   help="distance algorithm (default: bz)")
    p.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto",
                   help="compute backend; gpu auto-selects the available accelerator "
                        "(default: auto)")
    p.add_argument("--which", choices=["min", "z", "x"], default="min",
                   help="CSS component to report (default: min)")
    p.add_argument("--threads", type=int, default=0, metavar="N",
                   help="CPU worker threads (0 => hardware concurrency)")
    p.add_argument("--max-weight", type=int, default=0, metavar="N", dest="max_weight",
                   help="safety cap on enumeration weight (0 => none)")

    p.add_argument("--json", action="store_true",
                   help="emit a JSON object instead of a bare integer")
    p.add_argument("--zx", action="store_true",
                   help="CSS: print '<dZ> <dX>' (or both in JSON)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="verbose diagnostics on stderr")
    p.add_argument("--list-backends", action="store_true",
                   help="list usable backends and exit")
    p.add_argument("--version", action="store_true",
                   help="print the qminweight version and exit")
    return p


def _resolve_input(args):
    """Return ('css', Hx, Hz) or ('classical', H, None) from the chosen input source."""
    sources = [bool(args.hx or args.hz), bool(args.classical), bool(args.builtin)]
    if sum(sources) > 1:
        raise ValueError("choose only one of --hx/--hz, --classical, --builtin")

    if args.builtin:
        return io.load_builtin(args.builtin)

    if args.classical:
        return "classical", io.load_matrix(args.classical), None

    if args.hx or args.hz:
        if not (args.hx and args.hz):
            raise ValueError("--hx and --hz must be given together")
        Hx = io.load_matrix(args.hx)
        Hz = io.load_matrix(args.hz)
        if Hx.shape[1] != Hz.shape[1]:
            raise ValueError("Hx and Hz have differing column counts "
                             "(different number of qubits)")
        return "css", Hx, Hz

    # default: Pauli stabiliser strings from INPUT (file or stdin)
    text = io.read_text(args.input)
    Hx, Hz = io.parse_pauli(text)
    return "css", Hx, Hz


def _emit(result, *, which_label, json_out, zx, second=None):
    """Print a Result (or a pair for --zx) to stdout."""
    if zx:
        z, x = result, second
        if json_out:
            print(json.dumps({
                "dZ": z.distance, "dX": x.distance,
                "dZ_lower_bound": z.lower_bound, "dX_lower_bound": x.lower_bound,
                "proven": bool(z.proven and x.proven),
                "seconds": z.seconds + x.seconds,
                "backend": z.backend,
            }))
        else:
            for r, lbl in ((z, "dZ"), (x, "dX")):
                if not r.proven:
                    print("qminweight: %s in [%d, %d] (not proven)"
                          % (lbl, r.lower_bound, r.distance), file=sys.stderr)
            print("%d %d" % (z.distance, x.distance))
        return

    if json_out:
        print(json.dumps({
            "distance": result.distance,
            "lower_bound": result.lower_bound,
            "proven": bool(result.proven),
            "seconds": result.seconds,
            "backend": result.backend,
            "which": which_label,
        }))
    else:
        if not result.proven:
            print("qminweight: %s in [%d, %d] (not proven)"
                  % (which_label, result.lower_bound, result.distance),
                  file=sys.stderr)
        print(result.distance)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        print(api.version())
        return 0

    if args.list_backends:
        for b in api.available_backends():
            print(b)
        return 0

    try:
        kind, A, B = _resolve_input(args)
    except (ValueError, FileNotFoundError, OSError) as e:
        print("qminweight: %s" % e, file=sys.stderr)
        return 1

    common = dict(method=args.method, backend=args.backend,
                  threads=args.threads, max_weight=args.max_weight,
                  verbose=args.verbose)

    try:
        if kind == "classical":
            if args.zx:
                print("qminweight: --zx is only meaningful for CSS codes",
                      file=sys.stderr)
                return 1
            if args.method == "cc":
                # connected-cluster has no classical entry point in the API
                print("qminweight: --method cc is for CSS codes; use bz or mitm for "
                      "classical codes", file=sys.stderr)
                return 1
            r = api.classical_distance(A, **common)
            _emit(r, which_label="d", json_out=args.json, zx=False)
            return 0

        # CSS code
        Hx, Hz = A, B
        if args.zx:
            z = api.css_distance(Hx, Hz, which="z", **common)
            x = api.css_distance(Hx, Hz, which="x", **common)
            _emit(z, which_label="zx", json_out=args.json, zx=True, second=x)
        else:
            r = api.css_distance(Hx, Hz, which=args.which, **common)
            label = {"min": "d", "z": "dZ", "x": "dX"}[args.which]
            _emit(r, which_label=label, json_out=args.json, zx=False)
        return 0
    except (RuntimeError, ValueError) as e:
        print("qminweight: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
