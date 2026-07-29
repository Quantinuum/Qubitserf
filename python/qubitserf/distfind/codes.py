"""Generators for benchmark/test codes (classical and CSS quantum).

Raw-matrix construction logic is shared with :mod:`qubitserf.codeaut.codes` via
:mod:`qubitserf._constructions`; this module preserves the historical distfind
API (names, signatures, and exact output matrices).
"""
from __future__ import annotations
import numpy as np

from .. import _constructions as _c


def repetition_parity(n: int) -> np.ndarray:
    """Parity-check of the [n,1,n] repetition code."""
    return _c.repetition_parity(n)


def cyclic_repetition_parity(n: int) -> np.ndarray:
    """Closed-loop parity checks (rank n-1); HGP of this gives the toric code."""
    return _c.cyclic_repetition_parity(n)


def hamming_parity(r: int) -> np.ndarray:
    """Parity-check of the [2^r-1, 2^r-1-r, 3] Hamming code."""
    return _c.hamming_parity(r)


def steane() -> tuple[np.ndarray, np.ndarray]:
    """Steane [[7,1,3]] code (Hx = Hz = Hamming[7,4,3])."""
    H = _c.hamming_parity(3)
    return H.copy(), H.copy()


def shor() -> tuple[np.ndarray, np.ndarray]:
    """Shor [[9,1,3]] code."""
    return _c.shor_checks()


def bacon_shor(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Bacon-Shor subsystem code gauge generators on a ``d x d`` grid.

    Qubit ``(r, c)`` has index ``r*d + c`` (so ``n = d²``).
    X-gauge = ``XX`` on horizontally-adjacent qubits (same row, cols c, c+1);
    Z-gauge = ``ZZ`` on vertically-adjacent qubits (same col, rows r, r+1).
    Returns the gauge generators ``(Gx, Gz)``; the dressed distance is ``d``.
    """
    return _c.bacon_shor(d)


def hypergraph_product(H1: np.ndarray, H2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSS code from the hypergraph product of two parity-check matrices.

    Hx = [ H1 (x) I_{n2} | I_{m1} (x) H2^T ]
    Hz = [ I_{n1} (x) H2 | H1^T (x) I_{m2} ]
    """
    return _c.hypergraph_product(H1, H2)


def toric(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Toric code on an L x L torus as the HGP of a length-L cyclic repetition code.

    Yields an [[2L^2, 2, L]] code.
    """
    return _c.toric_hgp(L)


def surface(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Planar surface code [[L^2+(L-1)^2, 1, L]] as the HGP of the [L,1,L] repetition code."""
    return _c.surface_hgp(L)


def bivariate_bicycle(l: int, m: int, a_terms, b_terms) -> tuple[np.ndarray, np.ndarray]:
    """Bivariate bicycle CSS code (Bravyi et al., Nature 2024).

    x = S_l (x) I_m, y = I_l (x) S_m are commuting cyclic shifts on l*m qubits/block.
    A = sum of monomials in `a_terms`, B = sum in `b_terms`, each a list of
    ("x"|"y", power).  Hx = [A | B], Hz = [B^T | A^T], giving an [[2 l m, k, d]] code.
    """
    return _c.bivariate_bicycle(l, m,
                                _c.xy_terms_to_exponents(a_terms),
                                _c.xy_terms_to_exponents(b_terms))


def gross_code() -> tuple[np.ndarray, np.ndarray]:
    """The IBM 'gross code': [[144, 12, 12]] bivariate bicycle code,
    l=12, m=6, A = x^3 + y + y^2, B = y^3 + x + x^2 (Bravyi et al., Nature 2024)."""
    return _c.gross_checks()


def random_ldpc_parity(m: int, n: int, col_weight: int = 3, seed: int = 0) -> np.ndarray:
    """Random column-regular LDPC parity-check matrix (for benchmarking only)."""
    return _c.random_ldpc_parity(m, n, col_weight=col_weight, seed=seed)


def reed_muller_generator(r: int, m: int) -> np.ndarray:
    """Generator matrix of the [2^m, sum_{i<=r} C(m,i), 2^(m-r)] Reed-Muller code RM(r,m).

    Rows are evaluations of all monomials x_S = prod_{j in S} x_j, |S| <= r, over all
    2^m binary points of GF(2)^m.  Row count = sum_{i=0}^{r} C(m,i).  The minimum-weight
    row is the evaluation of any degree-r monomial, which has weight 2^(m-r).
    """
    return _c.reed_muller_generator(r, m)


def quantum_reed_muller(r: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """CSS code with Hx = Hz = G_RM(r, m), valid when 2r < m-1 (self-orthogonality).

    Parameters: n = 2^m, k = 2^m - 2*sum_{i<=r} C(m,i), d = 2^(r+1).
    Rows have weights n, n/2, ..., n/2^r — dense, NOT QLDPC.
    """
    if 2 * r >= m - 1:
        raise ValueError(
            f"RM(r={r}, m={m}) is not self-orthogonal (need 2r < m-1 = {m-1}); "
            f"smallest valid m for this r is {2*r + 2}."
        )
    G = _c.reed_muller_generator(r, m)
    return G.copy(), G.copy()
