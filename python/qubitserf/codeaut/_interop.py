"""CSS interop for codeaut -- re-exports the canonical shim at
:mod:`qubitserf._interop` (see its docstring for the accepted forms), plus the
Pauli-stabiliser-string parser :func:`css_from_paulis`."""
from __future__ import annotations

import numpy as np

from .._interop import as_css, _as_u8

__all__ = ["as_css", "css_from_paulis", "_as_u8"]


def css_from_paulis(text_or_lines):
    """Parse Pauli stabiliser strings into a CSS ``(Hx, Hz)`` pair.

    ``text_or_lines`` is either one string (one stabiliser per line) or an iterable of
    strings (one stabiliser each).  Each non-blank line is one stabiliser: a string over
    ``I``/``X``/``Z`` (case-insensitive), all of the same length ``n``.  For a **CSS** code
    every stabiliser must be pure-X or pure-Z; an ``X``/``Z`` mix, a ``Y``, or any other
    character makes the code non-CSS and raises ``ValueError``.  X-type rows become ``Hx``
    (1 at each X), Z-type rows become ``Hz`` (1 at each Z); all-identity rows are dropped.
    """
    if isinstance(text_or_lines, str):
        lines = text_or_lines.splitlines()
    else:
        lines = list(text_or_lines)
    rows = [ln.strip().upper() for ln in lines if ln.strip()]
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
