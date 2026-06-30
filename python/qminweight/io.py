"""Code-file parsers shared by the qminweight command-line interface.

Supported inputs:
  * Pauli stabiliser strings (one stabiliser per line; I/./_ = identity, X, Y, Z).
    A blank line terminates, mirroring Qubitserf. Only CSS codes are accepted, so each
    stabiliser must be pure X-type or pure Z-type; X rows become Hx, Z rows become Hz.
  * 0/1 text matrices (whitespace-separated, one matrix row per line) for --hx / --hz
    and --classical. MatrixMarket ``.mtx`` files are also read (via scipy if installed,
    otherwise with a small built-in coordinate parser).
  * Built-in code generators from ``qminweight.codes`` selected by ``NAME[:L]``.
"""
from __future__ import annotations

import sys
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Pauli stabiliser strings
# --------------------------------------------------------------------------- #
def parse_pauli(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse Pauli stabiliser strings into (Hx, Hz) for a CSS code.

    Reads lines until a blank line or end of input. Raises ValueError on non-CSS input
    (a Y anywhere, or a stabiliser mixing X and Z).
    """
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.strip() == "":
            break  # blank line terminates input, like Qubitserf
        lines.append(line)
    if not lines:
        raise ValueError("no stabilisers found (expected Pauli strings)")

    n = len(lines[0])
    for s in lines:
        if len(s) != n:
            raise ValueError("stabilisers have differing lengths "
                             "(all rows must span the same qubits)")

    xrows, zrows = [], []
    for li, s in enumerate(lines, start=1):
        xrow = np.zeros(n, dtype=np.uint8)
        zrow = np.zeros(n, dtype=np.uint8)
        has_x = has_z = False
        for j, c in enumerate(s):
            if c in "I._ ":
                continue
            if c in "Xx":
                xrow[j] = 1
                has_x = True
            elif c in "Zz":
                zrow[j] = 1
                has_z = True
            elif c in "Yy":
                raise ValueError("stabiliser %d contains a Y -- only CSS codes "
                                 "are supported" % li)
            else:
                raise ValueError("stabiliser %d has an unrecognised character "
                                 "%r" % (li, c))
        if has_x and has_z:
            raise ValueError("stabiliser %d mixes X and Z -- only CSS codes are "
                             "supported" % li)
        if has_x:
            xrows.append(xrow)
        elif has_z:
            zrows.append(zrow)
        # an all-identity row contributes nothing; ignore it

    Hx = np.array(xrows, dtype=np.uint8) if xrows else np.zeros((0, n), dtype=np.uint8)
    Hz = np.array(zrows, dtype=np.uint8) if zrows else np.zeros((0, n), dtype=np.uint8)
    return Hx, Hz


def parse_stabilizers(text: str) -> np.ndarray:
    """Parse general Pauli stabiliser strings into a symplectic ``[z | x]`` matrix.

    Unlike :func:`parse_pauli` this accepts ``Y`` and rows that mix X and Z (i.e. genuine
    non-CSS codes). Reads lines until a blank line or EOF (mirroring Qubitserf). Returns an
    ``(m, 2n)`` uint8 matrix: row r is the Pauli with Z-support in columns ``[0, n)`` and
    X-support in columns ``[n, 2n)`` (a ``Y`` sets both). All-identity rows are dropped.
    """
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.strip() == "":
            break
        lines.append(line)
    if not lines:
        raise ValueError("no stabilisers found (expected Pauli strings)")
    n = len(lines[0])
    for s in lines:
        if len(s) != n:
            raise ValueError("stabilisers have differing lengths "
                             "(all rows must span the same qubits)")
    rows = []
    for li, s in enumerate(lines, start=1):
        z = np.zeros(n, dtype=np.uint8)
        x = np.zeros(n, dtype=np.uint8)
        any_set = False
        for j, c in enumerate(s):
            if c in "I._ ":
                continue
            if c in "Xx":
                x[j] = 1; any_set = True
            elif c in "Zz":
                z[j] = 1; any_set = True
            elif c in "Yy":
                x[j] = 1; z[j] = 1; any_set = True
            else:
                raise ValueError("stabiliser %d has an unrecognised character %r"
                                 % (li, c))
        if any_set:
            rows.append(np.concatenate([z, x]))
    if not rows:
        return np.zeros((0, 2 * n), dtype=np.uint8)
    return np.array(rows, dtype=np.uint8)


def is_css_pauli(text: str) -> bool:
    """True iff every Pauli stabiliser line is pure-X or pure-Z (a CSS code)."""
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.strip() == "":
            break
        has_x = has_z = False
        for c in line:
            if c in "Xx":
                has_x = True
            elif c in "Zz":
                has_z = True
            elif c in "Yy":
                return False
        if has_x and has_z:
            return False
    return True


def parse_operator(s: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a single Pauli string into ``(z_vec, x_vec)`` 0/1 supports.

    Unlike :func:`parse_pauli`, a ``Y`` is allowed: it contributes to BOTH the
    Z- and X-support (``Y = i·X·Z``). ``I``/``.``/``_``/space are identity.
    Raises ValueError if the string length differs from ``n`` or a character is
    unrecognised.
    """
    s = s.strip().rstrip("\r")
    if len(s) != n:
        raise ValueError("operator has length %d, expected n = %d" % (len(s), n))
    z = np.zeros(n, dtype=np.uint8)
    x = np.zeros(n, dtype=np.uint8)
    for j, c in enumerate(s):
        if c in "I._ ":
            continue
        if c in "Xx":
            x[j] = 1
        elif c in "Zz":
            z[j] = 1
        elif c in "Yy":
            x[j] = 1
            z[j] = 1
        else:
            raise ValueError("operator has an unrecognised character %r" % c)
    return z, x


# --------------------------------------------------------------------------- #
# 0/1 matrices (dense text or MatrixMarket)
# --------------------------------------------------------------------------- #
def _parse_mtx(text: str) -> np.ndarray:
    """Minimal MatrixMarket coordinate/array reader (binary entries -> uint8)."""
    lines = [ln for ln in text.splitlines()]
    # find the size header (first non-comment, non-blank line)
    idx = 0
    while idx < len(lines) and (lines[idx].startswith("%") or not lines[idx].strip()):
        idx += 1
    if idx >= len(lines):
        raise ValueError("empty MatrixMarket file")
    header = lines[0].lower() if lines and lines[0].startswith("%%") else ""
    is_array = "array" in header
    dims = lines[idx].split()
    if is_array:
        rows, cols = int(dims[0]), int(dims[1])
        H = np.zeros((rows, cols), dtype=np.uint8)
        vals = []
        for ln in lines[idx + 1:]:
            ln = ln.strip()
            if not ln or ln.startswith("%"):
                continue
            vals.append(float(ln))
        # column-major order per MatrixMarket array format
        k = 0
        for c in range(cols):
            for r in range(rows):
                H[r, c] = 1 if (k < len(vals) and vals[k] != 0.0) else 0
                k += 1
        return H
    # coordinate format
    rows, cols = int(dims[0]), int(dims[1])
    H = np.zeros((rows, cols), dtype=np.uint8)
    for ln in lines[idx + 1:]:
        ln = ln.strip()
        if not ln or ln.startswith("%"):
            continue
        parts = ln.split()
        r, c = int(parts[0]) - 1, int(parts[1]) - 1
        v = float(parts[2]) if len(parts) > 2 else 1.0
        H[r, c] = 1 if v != 0.0 else 0
    return H


def parse_matrix(text: str, *, filename: str = "") -> np.ndarray:
    """Parse a 0/1 matrix from dense whitespace text or MatrixMarket (.mtx) content."""
    if text.lstrip().startswith("%%MatrixMarket") or filename.endswith(".mtx"):
        try:
            import io as _io

            from scipy.io import mmread  # type: ignore
            mat = mmread(_io.StringIO(text))
            arr = mat.todense() if hasattr(mat, "todense") else mat
            return (np.asarray(arr) != 0).astype(np.uint8)
        except ImportError:
            return _parse_mtx(text)

    rows = []
    cols = None
    for ln in text.splitlines():
        toks = ln.split()
        if not toks:
            continue
        row = []
        for t in toks:
            if t in ("0", "1"):
                row.append(int(t))
            else:
                # tolerate floats like "1.0"
                try:
                    row.append(1 if float(t) != 0.0 else 0)
                except ValueError:
                    raise ValueError("non-binary token %r in matrix" % t)
        if cols is None:
            cols = len(row)
        elif len(row) != cols:
            raise ValueError("ragged matrix (rows have differing widths)")
        rows.append(row)
    if not rows:
        raise ValueError("empty matrix")
    return np.array(rows, dtype=np.uint8)


def load_matrix(path: str) -> np.ndarray:
    """Read a 0/1 matrix from a file path (use '-' for stdin)."""
    text = read_text(path)
    return parse_matrix(text, filename=path)


# --------------------------------------------------------------------------- #
# Built-in code generators
# --------------------------------------------------------------------------- #
# Names accepted by --builtin, mapped to a (callable, takes_L) pair. Aliases included.
def _builtin_table():
    from . import codes
    return {
        "steane": (codes.steane, False),
        "shor": (codes.shor, False),
        "toric": (codes.toric, True),
        "surface": (codes.surface, True),
        "gross": (codes.gross_code, False),
        "gross_code": (codes.gross_code, False),
        # classical parity-check generators (single matrix)
        "repetition": (codes.repetition_parity, True),
        "hamming": (codes.hamming_parity, True),
    }


def builtin_names():
    return sorted(_builtin_table().keys())


def load_builtin(spec: str):
    """Resolve ``NAME[:L]`` to either (Hx, Hz) for a CSS code or a single H for a
    classical code. Returns ("css", Hx, Hz) or ("classical", H, None)."""
    if ":" in spec:
        name, _, larg = spec.partition(":")
        try:
            L = int(larg)
        except ValueError:
            raise ValueError("builtin parameter must be an integer in %r" % spec)
    else:
        name, L = spec, None

    table = _builtin_table()
    key = name.strip().lower()
    if key not in table:
        raise ValueError("unknown builtin %r (choices: %s)"
                         % (name, ", ".join(builtin_names())))
    fn, takes_L = table[key]
    if takes_L:
        if L is None:
            raise ValueError("builtin %r needs a size, e.g. --builtin %s:6"
                             % (name, key))
        result = fn(L)
    else:
        if L is not None:
            raise ValueError("builtin %r does not take a size parameter" % name)
        result = fn()

    if isinstance(result, tuple):
        Hx, Hz = result
        return "css", np.asarray(Hx, dtype=np.uint8), np.asarray(Hz, dtype=np.uint8)
    return "classical", np.asarray(result, dtype=np.uint8), None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_text(path: Optional[str]) -> str:
    """Read a file's text. ``None`` or ``"-"`` reads stdin."""
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r") as fh:
        return fh.read()
