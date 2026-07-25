"""Structural affine automorphisms for cyclic / prime-length CSS codes.

Many residual codes (self-dual CSS at prime length ``n`` -- QR/BCH-like) are *cyclic*: the shift
``i -> i+1 mod n`` and multiplier maps ``i -> a*i mod n`` (``a`` a unit) are qubit-permutation
automorphisms, giving a verified subgroup of order ``>= n`` that the generic codeword-incidence /
Tanner solvers return as trivial.  This finds the largest verified affine subgroup.
"""

from __future__ import annotations

import math

import numpy as np

from . import gf2
from . import permgroup


def affine_automorphism_group(Hx, Hz, n: int):
    """Largest verified subgroup of ``AGL(1,n) = {i -> a*i + b mod n}`` preserving both
    rowspaces.  Returns ``(order:int, generators:list[list[int]])`` -- ``(1, [])`` if only the
    identity verifies."""
    Hx = gf2.as_uint8(Hx)
    Hz = gf2.as_uint8(Hz)

    def verifies(perm):
        return gf2.preserves_rowspace(Hx, perm) and gf2.preserves_rowspace(Hz, perm)

    gens = []
    shift = [(i + 1) % n for i in range(n)]
    if verifies(shift):
        gens.append(shift)
    for a in range(2, n):
        if math.gcd(a, n) != 1:
            continue
        mul = [(a * i) % n for i in range(n)]
        if verifies(mul):
            gens.append(mul)
            if len(gens) >= 4:                          # a few generators of the unit group suffice
                break
    if not gens:
        return 1, []
    G = permgroup.Group(gens, n)
    return G.order(), gens
