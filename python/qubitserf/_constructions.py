"""Shared raw-matrix code constructions for the two qubitserf engines.

Every builder here returns plain numpy ``uint8`` GF(2) matrices (or ``(Hx, Hz)``
pairs of them) -- no ``CSSCode`` or other engine types are imported.  The
engine-facing modules :mod:`qubitserf.distfind.codes` and
:mod:`qubitserf.codeaut.codes` delegate to these builders and wrap/rename as
their public APIs require.

Where the two engines historically used *different* constructions for the same
code family, both variants are kept under distinct names (``toric_hgp`` vs
``toric_lattice``, ``surface_hgp`` vs ``surface_rotated``) -- they produce
different matrices on purpose.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations as _combinations


# ------------------------------------------------------------------ classical parity checks

def repetition_parity(n: int) -> np.ndarray:
    """Parity-check matrix of the ``[n, 1, n]`` repetition code (an ``(n-1) x n`` matrix)."""
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def cyclic_repetition_parity(n: int) -> np.ndarray:
    """Closed-loop parity checks (rank n-1); HGP of this gives the toric code."""
    H = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        H[i, i] = 1
        H[i, (i + 1) % n] = 1
    return H


def hamming_parity(r: int) -> np.ndarray:
    """Parity-check matrix of the ``[2**r - 1, 2**r - 1 - r, 3]`` Hamming code (columns are the
    nonzero vectors of ``F_2^r``)."""
    n = (1 << r) - 1
    cols = [[(j >> b) & 1 for b in range(r)] for j in range(1, n + 1)]
    return np.array(cols, dtype=np.uint8).T


def random_ldpc_parity(m: int, n: int, col_weight: int = 3, seed: int = 0) -> np.ndarray:
    """Random column-regular LDPC parity-check matrix (for benchmarking only)."""
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), dtype=np.uint8)
    for j in range(n):
        rows = rng.choice(m, size=min(col_weight, m), replace=False)
        H[rows, j] = 1
    return H


def reed_muller_generator(r: int, m: int) -> np.ndarray:
    """Generator matrix of the [2^m, sum_{i<=r} C(m,i), 2^(m-r)] Reed-Muller code RM(r,m).

    Rows are evaluations of all monomials x_S = prod_{j in S} x_j, |S| <= r, over all
    2^m binary points of GF(2)^m.  Row count = sum_{i=0}^{r} C(m,i).  The minimum-weight
    row is the evaluation of any degree-r monomial, which has weight 2^(m-r).
    """
    if r < 0 or r > m:
        raise ValueError(f"Need 0 <= r <= m; got r={r}, m={m}")
    n = 1 << m
    # points[i, b] = bit b of i
    points = np.array([[(i >> b) & 1 for b in range(m)] for i in range(n)], dtype=np.uint8)
    rows = []
    for deg in range(r + 1):
        for S in _combinations(range(m), deg):
            row = np.ones(n, dtype=np.uint8)
            for b in S:
                row = row * points[:, b]
            rows.append(row)
    return np.array(rows, dtype=np.uint8)


# ------------------------------------------------------------------------- small CSS codes

def shor_checks() -> tuple[np.ndarray, np.ndarray]:
    """Check matrices ``(Hx, Hz)`` of the ``[[9,1,3]]`` Shor code."""
    Hz = np.zeros((6, 9), dtype=np.uint8)            # bit-flip checks within each triple
    for blk in range(3):
        Hz[2 * blk, 3 * blk] = Hz[2 * blk, 3 * blk + 1] = 1
        Hz[2 * blk + 1, 3 * blk + 1] = Hz[2 * blk + 1, 3 * blk + 2] = 1
    Hx = np.zeros((2, 9), dtype=np.uint8)            # phase-flip checks across triples
    Hx[0, 0:6] = 1
    Hx[1, 3:9] = 1
    return Hx, Hz


def bacon_shor(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Bacon-Shor subsystem code gauge generators on a ``d x d`` grid.

    Qubit ``(r, c)`` has index ``r*d + c`` (so ``n = d²``).
    X-gauge = ``XX`` on horizontally-adjacent qubits (same row, cols c, c+1);
    Z-gauge = ``ZZ`` on vertically-adjacent qubits (same col, rows r, r+1).
    Returns the gauge generators ``(Gx, Gz)``; the dressed distance is ``d``.
    """
    n = d * d

    def idx(r, c):
        return r * d + c

    gx = []
    for r in range(d):
        for c in range(d - 1):
            v = np.zeros(n, dtype=np.uint8)
            v[idx(r, c)] = 1
            v[idx(r, c + 1)] = 1
            gx.append(v)
    gz = []
    for c in range(d):
        for r in range(d - 1):
            v = np.zeros(n, dtype=np.uint8)
            v[idx(r, c)] = 1
            v[idx(r + 1, c)] = 1
            gz.append(v)
    return np.array(gx, dtype=np.uint8), np.array(gz, dtype=np.uint8)


# ----------------------------------------------------------------- hypergraph-product codes

def hypergraph_product(H1: np.ndarray, H2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSS code from the hypergraph product of two parity-check matrices.

    Hx = [ H1 (x) I_{n2} | I_{m1} (x) H2^T ]
    Hz = [ I_{n1} (x) H2 | H1^T (x) I_{m2} ]
    """
    H1 = np.asarray(H1, dtype=np.uint8)
    H2 = np.asarray(H2, dtype=np.uint8)
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    Hx = np.hstack([np.kron(H1, np.eye(n2, dtype=np.uint8)),
                    np.kron(np.eye(m1, dtype=np.uint8), H2.T)]).astype(np.uint8) % 2
    Hz = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(m2, dtype=np.uint8))]).astype(np.uint8) % 2
    return Hx, Hz


def toric_hgp(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Toric code on an L x L torus as the HGP of a length-L cyclic repetition code
    (the distfind variant).  Yields an [[2L^2, 2, L]] code."""
    Hc = cyclic_repetition_parity(L)
    return hypergraph_product(Hc, Hc)


def surface_hgp(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Planar surface code [[L^2+(L-1)^2, 1, L]] as the HGP of the [L,1,L] repetition code
    (the distfind variant)."""
    Hc = repetition_parity(L)  # (L-1) x L
    return hypergraph_product(Hc, Hc)


# -------------------------------------------------------------- explicit-lattice CSS codes

def toric_lattice(L: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``[[2 L**2, 2, L]]`` toric code on an ``L x L`` periodic square lattice, built from
    explicit star/plaquette operators (the codeaut variant -- NOT the same matrices as
    :func:`toric_hgp`)."""
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
    return np.array(Hx_rows), np.array(Hz_rows)


def surface_rotated(d: int) -> tuple[np.ndarray, np.ndarray]:
    """The planar (open-boundary) rotated surface code of distance ``d`` (``[[d**2, 1, d]]``),
    built from the standard star/plaquette stabilisers on a ``d x d`` array of data qubits
    (the codeaut variant -- NOT the same code as :func:`surface_hgp`)."""
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
    return Hx, Hz


# -------------------------------------------------------- bivariate-bicycle / quasi-cyclic codes

def _cyclic_shift(power: int, ell: int) -> np.ndarray:
    """The ``ell x ell`` cyclic-shift permutation matrix ``S**power``."""
    S = np.zeros((ell, ell), dtype=np.uint8)
    for i in range(ell):
        S[i, (i + power) % ell] = 1
    return S


def bivariate_bicycle(ell: int, m: int, a_terms, b_terms) -> tuple[np.ndarray, np.ndarray]:
    """A bivariate-bicycle code (Bravyi et al., Nature 2024).  ``a_terms`` / ``b_terms`` are
    lists of ``(i, j)`` exponents for ``x**i y**j`` (``x = S_ell (x) I_m``,
    ``y = I_ell (x) S_m``); then ``A = sum x**i y**j`` over ``a_terms`` and likewise ``B``,
    with ``Hx = [A | B]``, ``Hz = [B.T | A.T]``.  ``n = 2 ell m``.
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
    return Hx, Hz


def xy_terms_to_exponents(terms) -> list[tuple[int, int]]:
    """Translate distfind-style ``("x"|"y", power)`` monomial terms to the canonical
    ``(i, j)`` exponent pairs of :func:`bivariate_bicycle` (``"x"`` -> ``(p, 0)``,
    anything else -> ``(0, p)``, matching the historical distfind semantics)."""
    return [(p, 0) if var == "x" else (0, p) for var, p in terms]


def gross_checks() -> tuple[np.ndarray, np.ndarray]:
    """Check matrices of the ``[[144, 12, 12]]`` "gross" bivariate-bicycle code (Bravyi et
    al. 2024): ``ell=12, m=6``, ``A = x**3 + y + y**2``, ``B = y**3 + x + x**2``."""
    return bivariate_bicycle(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
