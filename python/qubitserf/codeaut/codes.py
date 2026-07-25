"""Builtin example CSS codes (as :class:`codeaut.css.CSSCode`) for demos, tests and benchmarks.

Each constructor returns a ``CSSCode(Hx, Hz)`` with ``Hx @ Hz.T == 0`` over GF(2).  The
classical-code helpers (``hamming_parity``, ``repetition``) return plain GF(2) matrices for the
single-code Leon engine.
"""

from __future__ import annotations

import numpy as np

from .css import CSSCode


# ----------------------------------------------------------------- classical generating matrices

def hamming_parity(r: int = 3) -> np.ndarray:
    """Parity-check matrix of the ``[2**r - 1, 2**r - 1 - r, 3]`` Hamming code (columns are the
    nonzero vectors of ``F_2^r``)."""
    cols = [[(j >> i) & 1 for i in range(r)] for j in range(1, 1 << r)]
    return np.array(cols, dtype=np.uint8).T


def repetition(n: int = 3) -> np.ndarray:
    """Parity-check matrix of the ``[n, 1, n]`` repetition code (an ``(n-1) x n`` matrix)."""
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


# ----------------------------------------------------------------------------------- CSS codes

def steane() -> CSSCode:
    """The ``[[7,1,3]]`` Steane code (``Hx = Hz`` = the self-orthogonal Hamming/simplex check)."""
    H = np.array([[0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    return CSSCode(H, H, k=1)


def shor() -> CSSCode:
    """The ``[[9,1,3]]`` Shor code."""
    Hz = np.zeros((6, 9), dtype=np.uint8)            # bit-flip checks within each triple
    for b in range(3):
        Hz[2 * b, 3 * b] = Hz[2 * b, 3 * b + 1] = 1
        Hz[2 * b + 1, 3 * b + 1] = Hz[2 * b + 1, 3 * b + 2] = 1
    Hx = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0],      # phase-flip checks across triples
                   [0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    return CSSCode(Hx, Hz, k=1)


def iceberg(m: int = 2) -> CSSCode:
    """The ``[[2m, 2m-2, 2]]`` iceberg / ``C6``-style code: ``Hx = Hz = `` all-ones row."""
    n = 2 * m
    H = np.ones((1, n), dtype=np.uint8)
    return CSSCode(H, H, k=n - 2)


def toric(L: int = 3) -> CSSCode:
    """The ``[[2 L**2, 2, L]]`` toric code on an ``L x L`` periodic square lattice."""
    n = 2 * L * L

    def edge(kind, x, y):       # kind 0 = horizontal, 1 = vertical
        return kind * L * L + (x % L) * L + (y % L)

    Hx_rows, Hz_rows = [], []
    for x in range(L):
        for y in range(L):
            r = np.zeros(n, dtype=np.uint8)          # vertex (star) operator -> X-check
            r[edge(0, x, y)] = 1
            r[edge(0, x - 1, y)] = 1
            r[edge(1, x, y)] = 1
            r[edge(1, x, y - 1)] = 1
            Hx_rows.append(r)
            p = np.zeros(n, dtype=np.uint8)          # plaquette operator -> Z-check
            p[edge(0, x, y)] = 1
            p[edge(0, x, y + 1)] = 1
            p[edge(1, x, y)] = 1
            p[edge(1, x + 1, y)] = 1
            Hz_rows.append(p)
    return CSSCode(np.array(Hx_rows), np.array(Hz_rows), k=2)


def surface(d: int = 3) -> CSSCode:
    """The planar (open-boundary) rotated surface code of distance ``d`` (``[[d**2, 1, d]]``).

    Built from the standard star/plaquette stabilisers on a ``d x d`` array of data qubits.
    """
    n = d * d

    def q(r, c):
        return r * d + c

    Hx_rows, Hz_rows = [], []
    # Z-plaquettes and X-stars on the rotated lattice, with the usual boundary truncation.
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                row = np.zeros(n, dtype=np.uint8)
                for (rr, cc) in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)):
                    row[q(rr, cc)] = 1
                Hz_rows.append(row)
            else:
                row = np.zeros(n, dtype=np.uint8)
                for (rr, cc) in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)):
                    row[q(rr, cc)] = 1
                Hx_rows.append(row)
    # boundary weight-2 stabilisers
    for c in range(0, d - 1, 2):
        row = np.zeros(n, dtype=np.uint8); row[q(0, c)] = 1; row[q(0, c + 1)] = 1
        Hx_rows.append(row)
        row = np.zeros(n, dtype=np.uint8); row[q(d - 1, c + 1)] = 1; row[q(d - 1, c + 2 if c + 2 < d else c)] = 1
        if row.sum() == 2:
            Hx_rows.append(row)
    for r in range(0, d - 1, 2):
        row = np.zeros(n, dtype=np.uint8); row[q(r, d - 1)] = 1; row[q(r + 1, d - 1)] = 1
        Hz_rows.append(row)
        row = np.zeros(n, dtype=np.uint8); row[q(r + 1, 0)] = 1; row[q(r + 2 if r + 2 < d else r, 0)] = 1
        if row.sum() == 2:
            Hz_rows.append(row)
    Hx = np.array(Hx_rows, dtype=np.uint8) if Hx_rows else np.zeros((0, n), np.uint8)
    Hz = np.array(Hz_rows, dtype=np.uint8) if Hz_rows else np.zeros((0, n), np.uint8)
    return CSSCode(Hx, Hz)


# -------------------------------------------------------- bivariate-bicycle / quasi-cyclic codes

def _cyclic_shift(power: int, ell: int) -> np.ndarray:
    """The ``ell x ell`` cyclic-shift permutation matrix ``S**power``."""
    S = np.zeros((ell, ell), dtype=np.uint8)
    for i in range(ell):
        S[i, (i + power) % ell] = 1
    return S


def bivariate_bicycle(ell: int, m: int, a_terms, b_terms) -> CSSCode:
    """A bivariate-bicycle code (Bravyi et al. 2024).  ``a_terms`` / ``b_terms`` are lists of
    ``(i, j)`` exponents for ``x**i y**j`` (``x = S_ell (x) I_m``, ``y = I_ell (x) S_m``); then
    ``A = sum x**i y**j`` over ``a_terms`` and likewise ``B``, with ``Hx = [A | B]``,
    ``Hz = [B.T | A.T]``.  ``n = 2 ell m``.
    """
    def mono(i, j):
        return np.kron(_cyclic_shift(i, ell), _cyclic_shift(j, m)) % 2

    A = np.zeros((ell * m, ell * m), dtype=np.uint8)
    for (i, j) in a_terms:
        A = (A + mono(i, j)) % 2
    B = np.zeros((ell * m, ell * m), dtype=np.uint8)
    for (i, j) in b_terms:
        B = (B + mono(i, j)) % 2
    Hx = np.hstack([A, B]).astype(np.uint8)
    Hz = np.hstack([B.T, A.T]).astype(np.uint8)
    return CSSCode(Hx, Hz)


def gross() -> CSSCode:
    """The ``[[144, 12, 12]]`` "gross" bivariate-bicycle code (Bravyi et al. 2024):
    ``ell=12, m=6``, ``A = x**3 + y + y**2``, ``B = y**3 + x + x**2``."""
    return bivariate_bicycle(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])


BUILTIN = {
    "steane": steane, "shor": shor, "iceberg": iceberg,
    "toric": toric, "surface": surface, "gross": gross,
}
