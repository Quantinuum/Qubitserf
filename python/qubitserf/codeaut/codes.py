"""Builtin example CSS codes (as :class:`codeaut.css.CSSCode`) for demos, tests and benchmarks.

Each constructor returns a ``CSSCode(Hx, Hz)`` with ``Hx @ Hz.T == 0`` over GF(2).  The
classical-code helpers (``hamming_parity``, ``repetition``) return plain GF(2) matrices for the
single-code Leon engine.

Raw-matrix construction logic is shared with :mod:`qubitserf.distfind.codes` via
:mod:`qubitserf._constructions`; this module preserves the historical codeaut API
(names, signatures, defaults, and exact output matrices).
"""

from __future__ import annotations

import numpy as np

from .. import _constructions as _c
from .css import CSSCode


# ----------------------------------------------------------------- classical generating matrices

def hamming_parity(r: int = 3) -> np.ndarray:
    """Parity-check matrix of the ``[2**r - 1, 2**r - 1 - r, 3]`` Hamming code (columns are the
    nonzero vectors of ``F_2^r``)."""
    return _c.hamming_parity(r)


def repetition(n: int = 3) -> np.ndarray:
    """Parity-check matrix of the ``[n, 1, n]`` repetition code (an ``(n-1) x n`` matrix)."""
    return _c.repetition_parity(n)


# ----------------------------------------------------------------------------------- CSS codes

def steane() -> CSSCode:
    """The ``[[7,1,3]]`` Steane code (``Hx = Hz`` = the self-orthogonal Hamming/simplex check)."""
    H = np.array([[0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    return CSSCode(H, H, k=1)


def shor() -> CSSCode:
    """The ``[[9,1,3]]`` Shor code."""
    Hx, Hz = _c.shor_checks()
    return CSSCode(Hx, Hz, k=1)


def iceberg(m: int = 2) -> CSSCode:
    """The ``[[2m, 2m-2, 2]]`` iceberg / ``C6``-style code: ``Hx = Hz = `` all-ones row."""
    n = 2 * m
    H = np.ones((1, n), dtype=np.uint8)
    return CSSCode(H, H, k=n - 2)


def toric(L: int = 3) -> CSSCode:
    """The ``[[2 L**2, 2, L]]`` toric code on an ``L x L`` periodic square lattice."""
    Hx, Hz = _c.toric_lattice(L)
    return CSSCode(Hx, Hz, k=2)


def surface(d: int = 3) -> CSSCode:
    """The planar (open-boundary) rotated surface code of distance ``d`` (``[[d**2, 1, d]]``).

    Built from the standard star/plaquette stabilisers on a ``d x d`` array of data qubits.
    """
    Hx, Hz = _c.surface_rotated(d)
    return CSSCode(Hx, Hz)


# -------------------------------------------------------- bivariate-bicycle / quasi-cyclic codes

def bivariate_bicycle(ell: int, m: int, a_terms, b_terms) -> CSSCode:
    """A bivariate-bicycle code (Bravyi et al. 2024).  ``a_terms`` / ``b_terms`` are lists of
    ``(i, j)`` exponents for ``x**i y**j`` (``x = S_ell (x) I_m``, ``y = I_ell (x) S_m``); then
    ``A = sum x**i y**j`` over ``a_terms`` and likewise ``B``, with ``Hx = [A | B]``,
    ``Hz = [B.T | A.T]``.  ``n = 2 ell m``.
    """
    Hx, Hz = _c.bivariate_bicycle(ell, m, a_terms, b_terms)
    return CSSCode(Hx, Hz)


def gross() -> CSSCode:
    """The ``[[144, 12, 12]]`` "gross" bivariate-bicycle code (Bravyi et al. 2024):
    ``ell=12, m=6``, ``A = x**3 + y + y**2``, ``B = y**3 + x + x**2``."""
    Hx, Hz = _c.gross_checks()
    return CSSCode(Hx, Hz)


BUILTIN = {
    "steane": steane, "shor": shor, "iceberg": iceberg,
    "toric": toric, "surface": surface, "gross": gross,
}
