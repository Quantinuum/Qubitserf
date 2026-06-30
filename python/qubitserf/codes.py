"""Generators for benchmark/test codes (classical and CSS quantum)."""
from __future__ import annotations
import numpy as np
from itertools import combinations as _combinations


def repetition_parity(n: int) -> np.ndarray:
    """Parity-check of the [n,1,n] repetition code."""
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
    """Parity-check of the [2^r-1, 2^r-1-r, 3] Hamming code."""
    n = (1 << r) - 1
    cols = [[(j >> b) & 1 for b in range(r)] for j in range(1, n + 1)]
    return np.array(cols, dtype=np.uint8).T


def steane() -> tuple[np.ndarray, np.ndarray]:
    """Steane [[7,1,3]] code (Hx = Hz = Hamming[7,4,3])."""
    H = hamming_parity(3)
    return H.copy(), H.copy()


def shor() -> tuple[np.ndarray, np.ndarray]:
    """Shor [[9,1,3]] code."""
    Hz = np.zeros((6, 9), dtype=np.uint8)
    for blk in range(3):
        Hz[2 * blk, 3 * blk] = Hz[2 * blk, 3 * blk + 1] = 1
        Hz[2 * blk + 1, 3 * blk + 1] = Hz[2 * blk + 1, 3 * blk + 2] = 1
    Hx = np.zeros((2, 9), dtype=np.uint8)
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


def toric(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Toric code on an L x L torus as the HGP of a length-L cyclic repetition code.

    Yields an [[2L^2, 2, L]] code.
    """
    Hc = cyclic_repetition_parity(L)
    return hypergraph_product(Hc, Hc)


def surface(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Planar surface code [[L^2+(L-1)^2, 1, L]] as the HGP of the [L,1,L] repetition code."""
    Hc = repetition_parity(L)  # (L-1) x L
    return hypergraph_product(Hc, Hc)


def _cyclic_shift(n: int) -> np.ndarray:
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


def bivariate_bicycle(l: int, m: int, a_terms, b_terms) -> tuple[np.ndarray, np.ndarray]:
    """Bivariate bicycle CSS code (Bravyi et al., Nature 2024).

    x = S_l (x) I_m, y = I_l (x) S_m are commuting cyclic shifts on l*m qubits/block.
    A = sum of monomials in `a_terms`, B = sum in `b_terms`, each a list of
    ("x"|"y", power).  Hx = [A | B], Hz = [B^T | A^T], giving an [[2 l m, k, d]] code.
    """
    Il, Im = np.eye(l, dtype=np.uint8), np.eye(m, dtype=np.uint8)
    x = np.kron(_cyclic_shift(l), Im)
    y = np.kron(Il, _cyclic_shift(m))
    N = l * m

    def monomial(var, p):
        base = x if var == "x" else y
        M = np.eye(N, dtype=np.uint8)
        for _ in range(p):
            M = (M @ base) % 2
        return M

    def poly(terms):
        P = np.zeros((N, N), dtype=np.uint8)
        for var, p in terms:
            P = (P + monomial(var, p)) % 2
        return P

    A, B = poly(a_terms), poly(b_terms)
    Hx = np.hstack([A, B]).astype(np.uint8)
    Hz = np.hstack([B.T, A.T]).astype(np.uint8)
    return Hx, Hz


def gross_code() -> tuple[np.ndarray, np.ndarray]:
    """The IBM 'gross code': [[144, 12, 12]] bivariate bicycle code,
    l=12, m=6, A = x^3 + y + y^2, B = y^3 + x + x^2 (Bravyi et al., Nature 2024)."""
    return bivariate_bicycle(12, 6,
                             [("x", 3), ("y", 1), ("y", 2)],
                             [("y", 3), ("x", 1), ("x", 2)])


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
    G = reed_muller_generator(r, m)
    return G.copy(), G.copy()
