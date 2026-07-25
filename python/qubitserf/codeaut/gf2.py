"""Minimal GF(2) linear algebra (numpy only).

A *binary matrix* is a ``numpy.ndarray`` of dtype ``uint8`` with entries in ``{0, 1}``.
Every routine coerces its input with :func:`as_uint8` (reducing mod 2), so any integer
array may be passed in freely; results are returned as plain ``uint8`` arrays.

This is the self-contained subset of the Quirky ``lib.utility.gf2`` module that the
``codeaut`` engines need -- no external dependency beyond numpy.
"""

from __future__ import annotations

import numpy as np


def as_uint8(matrix) -> np.ndarray:
    """Coerce to a contiguous ``uint8`` array with entries reduced mod 2."""
    out = np.asarray(matrix, dtype=np.uint8) % np.uint8(2)
    return np.ascontiguousarray(out)


def wt(v) -> int:
    """Hamming weight of a binary vector."""
    return int(as_uint8(v).sum())


def rref_gf2(matrix):
    """Reduced row echelon form over GF(2). Returns ``(rref, pivot_columns)``."""
    A = as_uint8(matrix).copy()
    n_rows, n_cols = A.shape
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        pivot = None
        for row in range(pivot_row, n_rows):
            if A[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != pivot_row:
            A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
        for row in range(n_rows):
            if row != pivot_row and A[row, col]:
                A[row] ^= A[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1
    return A, pivot_cols


def rank_gf2(matrix) -> int:
    """Rank over GF(2)."""
    _, pivots = rref_gf2(matrix)
    return len(pivots)


def row_basis_gf2(matrix) -> np.ndarray:
    """A basis (the nonzero rref rows) for the row space."""
    rref, _ = rref_gf2(matrix)
    rows = [row.copy() for row in rref if np.any(row)]
    if not rows:
        return np.zeros((0, np.asarray(matrix).shape[1]), dtype=np.uint8)
    return np.vstack(rows).astype(np.uint8)


def nullspace_basis_gf2(matrix) -> np.ndarray:
    """A basis for the right null space ``{v : matrix @ v = 0}`` over GF(2)."""
    A = as_uint8(matrix)
    rref, pivot_cols = rref_gf2(A)
    n_cols = A.shape[1]
    free_cols = [col for col in range(n_cols) if col not in pivot_cols]
    if not free_cols:
        return np.zeros((0, n_cols), dtype=np.uint8)
    basis: list[np.ndarray] = []
    for free_col in free_cols:
        vec = np.zeros(n_cols, dtype=np.uint8)
        vec[free_col] = 1
        for row, pivot_col in enumerate(pivot_cols):
            vec[pivot_col] = rref[row, free_col]
        basis.append(vec)
    return np.vstack(basis).astype(np.uint8)


def in_span_gf2(basis, v) -> bool:
    """True iff ``v`` lies in the GF(2) row span of ``basis``."""
    basis = as_uint8(basis)
    v = as_uint8(v).reshape(1, -1)
    if basis.shape[0] == 0:
        return not np.any(v)
    return rank_gf2(basis) == rank_gf2(np.vstack([basis, v]))


def preserves_rowspace(H: np.ndarray, perm) -> bool:
    """True iff column-permuting ``H`` by ``perm`` (0-indexed image list) leaves
    ``rowspace(H)`` invariant -- i.e. ``perm`` is an automorphism of the code ``rowspace(H)``.
    """
    H = as_uint8(H)
    perm = list(perm)
    r = rank_gf2(H)
    return rank_gf2(np.vstack([H, H[:, perm]])) == r


def dual_basis(H: np.ndarray):
    """Pick the cheaper of ``rowspace(H)`` and its dual ``rowspace(H)^perp``.

    Returns ``(B, n, rank, eff)`` where ``B`` is a basis of whichever space has the
    smaller dimension (``eff = min(rank, n - rank)``), ``n`` the length.  ``Aut(C) =
    Aut(C^perp)``, so enumerating the smaller side is sound (the dual-code trick).
    """
    H = as_uint8(H)
    n = H.shape[1]
    r = rank_gf2(H)
    eff = min(r, n - r)
    B = nullspace_basis_gf2(H) if (n - r) < r else row_basis_gf2(H)
    return B, int(n), int(r), int(eff)
