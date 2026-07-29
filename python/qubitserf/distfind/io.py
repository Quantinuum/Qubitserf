"""Pauli-operator parsing used by the distfind Python API (:mod:`qubitserf.distfind.api`).

Single entry point :func:`parse_operator`: a Pauli string (one operator; I/./_ = identity,
X, Y, Z) is turned into its ``(z_vec, x_vec)`` 0/1 supports.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def parse_operator(s: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a single Pauli string into ``(z_vec, x_vec)`` 0/1 supports.

    A ``Y`` is allowed: it contributes to BOTH the Z- and X-support (``Y = i·X·Z``).
    ``I``/``.``/``_``/space are identity. Raises ValueError if the string length
    differs from ``n`` or a character is unrecognised.
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
