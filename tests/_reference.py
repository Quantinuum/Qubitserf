"""Pure-Python / numpy correctness ORACLE for qminweight's two imported features.

This module has **no dependency on the compiled qminweight library** -- it uses
only numpy. It is the ground-truth ("oracle") against which the package's
C++/Python implementations are cross-checked in the pytest suite.

It provides:

GF(2) linear algebra
    rref, rank, nullspace, rowspace_contains, rowspace_basis

Operator weight (min-weight coset leader) -- the qubitserf-CORRECT definition
    coset_min_weight_rowspace      (a) enumerate rowspace(G), min wt(vec XOR g)
    coset_min_weight_syndrome      (b) increasing-weight syndrome search
    coset_min_weight_bruteforce    dispatches + cross-checks (a) and (b)

Subsystem CSS dressed distance
    css_center                     stabilizer center (Sx, Sz) of a gauge group
    dressed_distance_bruteforce    increasing-weight dressed-distance search
    bacon_shor                     independent Bacon-Shor gauge generators

Plus small helpers used by the self-test (hypergraph_product, surface_3,
steane, hamming_parity, repetition_parity) -- independent copies so the oracle
never imports the package it is meant to check.

See SPEC.md sections 0 (operator weight = min-weight coset leader; the qubitserf
bug), 1 (subsystem dressed distance + stabilizer center) and 5 (oracle contents).
"""
from __future__ import annotations

import itertools

import numpy as np


# --------------------------------------------------------------------------- #
# GF(2) linear algebra
# --------------------------------------------------------------------------- #
def _as_gf2(M) -> np.ndarray:
    """Coerce to a 2-D uint8 GF(2) matrix (entries reduced mod 2)."""
    A = np.asarray(M, dtype=np.uint8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    return (A & 1).astype(np.uint8)


def rref(M):
    """Reduced row-echelon form of ``M`` over GF(2).

    Returns ``(R, pivots)`` where ``R`` is the reduced matrix (same shape as
    ``M``) and ``pivots`` is the list of pivot column indices, in order.
    """
    R = _as_gf2(M).copy()
    rows, cols = R.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        # Find a row at or below `r` with a 1 in column `c`.
        pivot_row = None
        for i in range(r, rows):
            if R[i, c]:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        if pivot_row != r:
            R[[r, pivot_row]] = R[[pivot_row, r]]
        # Eliminate the pivot column from every other row.
        for i in range(rows):
            if i != r and R[i, c]:
                R[i] ^= R[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return R, pivots


def rank(M) -> int:
    """GF(2) rank of ``M`` (number of pivots)."""
    A = _as_gf2(M)
    if A.shape[0] == 0 or A.shape[1] == 0:
        return 0
    _, pivots = rref(A)
    return len(pivots)


def rowspace_basis(M) -> np.ndarray:
    """A basis of rowspace(``M``): the nonzero rows of its RREF.

    Shape ``(rank, n)``; ``(0, n)`` when ``M`` spans only the zero vector.
    """
    A = _as_gf2(M)
    n = A.shape[1]
    if A.shape[0] == 0:
        return np.zeros((0, n), dtype=np.uint8)
    R, pivots = rref(A)
    return R[: len(pivots)].copy() if pivots else np.zeros((0, n), dtype=np.uint8)


def nullspace(M) -> np.ndarray:
    """Basis of the right null space ``{y : M yᵀ = 0}``, returned as ROWS.

    Shape ``(n - rank(M), n)``; ``(0, n)`` when ``M`` has full column rank.
    A matrix with zero rows has the full space as its null space (the identity).
    """
    A = _as_gf2(M)
    n = A.shape[1]
    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if A.shape[0] == 0:
        return np.eye(n, dtype=np.uint8)
    R, pivots = rref(A)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    basis = []
    for f in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        # For each pivot row, R[ri, f] is the coefficient on that pivot var.
        for ri, pc in enumerate(pivots):
            if R[ri, f]:
                v[pc] = 1
        basis.append(v)
    if not basis:
        return np.zeros((0, n), dtype=np.uint8)
    return np.array(basis, dtype=np.uint8)


def rowspace_contains(M, v) -> bool:
    """Is row vector ``v`` an element of rowspace(``M``) over GF(2)?

    True iff ``rank([M; v]) == rank(M)``. An empty ``M`` (zero rows) contains
    only the zero vector.
    """
    A = _as_gf2(M)
    w = _as_gf2(v).reshape(1, -1)
    if A.shape[0] == 0:
        return not bool(np.any(w))
    return rank(np.vstack([A, w])) == rank(A)


def _membership_reducer(M):
    """Return a fast ``contains(v) -> bool`` closure for rowspace(``M``).

    Pre-computes the RREF basis once and reduces candidate vectors against it,
    so repeated membership queries (the inner loop of the brute-force searches)
    avoid recomputing a rank from scratch each time.
    """
    R = rowspace_basis(M)
    pivots = []
    for row in R:
        nz = np.nonzero(row)[0]
        pivots.append(int(nz[0]) if nz.size else -1)

    def contains(v) -> bool:
        w = _as_gf2(v).reshape(-1).copy()
        for row, p in zip(R, pivots):
            if p >= 0 and w[p]:
                w ^= row
        return not bool(np.any(w))

    return contains


def _gf2_matmul(A, B) -> np.ndarray:
    """GF(2) product ``A·B`` (row i = XOR of B rows where A[i,·]=1)."""
    A = _as_gf2(A)
    B = _as_gf2(B)
    if A.shape[0] == 0:
        return np.zeros((0, B.shape[1]), dtype=np.uint8)
    return ((A.astype(np.int64) @ B.astype(np.int64)) & 1).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Operator weight -- minimum-weight coset leader (the qubitserf-CORRECT def.)
#
# The min weight of an operator equivalent to `vec` modulo the multipliable
# group is the minimum Hamming weight over the coset  vec + rowspace(G).
# This is 0 exactly when vec in rowspace(G) -- the case qubitserf gets wrong for
# non-self-orthogonal codes (it returns d instead of 0 for a stabilizer).
# --------------------------------------------------------------------------- #
def coset_min_weight_rowspace(G, vec) -> int:
    """Min weight over ``vec + rowspace(G)`` by ROWSPACE enumeration (method a).

    Enumerates all ``2^rank(G)`` elements ``g`` of rowspace(``G``) and returns
    ``min_g wt(vec XOR g)``. Cheap when ``rank(G)`` is small.
    """
    vec = _as_gf2(vec).reshape(-1)
    B = rowspace_basis(G)
    k = B.shape[0]
    best = int(vec.sum())
    if k == 0:
        return best
    # Gray-code-free straightforward enumeration of the 2^k combinations.
    for bits in range(1 << k):
        g = np.zeros(vec.shape[0], dtype=np.uint8)
        for i in range(k):
            if (bits >> i) & 1:
                g ^= B[i]
        w = int(np.count_nonzero(vec ^ g))
        if w < best:
            best = w
            if best == 0:
                break
    return best


def coset_min_weight_syndrome(G, vec) -> int:
    """Min weight over ``vec + rowspace(G)`` by increasing-weight SYNDROME
    search (method b).

    Let ``P = nullspace(G)`` (rows ``{y : G yᵀ = 0}``); a vector ``u`` lies in
    the coset iff ``P·uᵀ = P·vecᵀ``. For ``w = 0, 1, 2, …`` we enumerate every
    weight-``w`` ``u`` (combinations of ``w`` columns) and return the first
    ``w`` whose syndrome matches the target. Cheap when ``n`` is small.
    """
    vec = _as_gf2(vec).reshape(-1)
    n = vec.shape[0]
    P = nullspace(G)
    target = _gf2_matmul(P, vec.reshape(-1, 1)).reshape(-1)
    for w in range(0, n + 1):
        for cols in itertools.combinations(range(n), w):
            u = np.zeros(n, dtype=np.uint8)
            if cols:
                u[list(cols)] = 1
            syn = _gf2_matmul(P, u.reshape(-1, 1)).reshape(-1)
            if np.array_equal(syn, target):
                return w
    return n  # unreachable: w = n (all-ones) always matches some coset member


def coset_min_weight_bruteforce(G, vec) -> int:
    """Minimum-weight coset leader of ``vec + rowspace(G)`` over GF(2).

    The oracle for the operator-weight feature. Runs whichever of the two
    independent methods are feasible (rowspace enumeration when ``rank(G)`` is
    small; syndrome search when ``n`` is small) and **asserts they agree** when
    both run, so a bug in either is caught. Returns 0 exactly when
    ``vec in rowspace(G)``.
    """
    G = _as_gf2(G)
    vec = _as_gf2(vec).reshape(-1)
    n = vec.shape[0]
    k = rank(G)

    results = []
    if k <= 22:
        results.append(coset_min_weight_rowspace(G, vec))
    if n <= 22:
        results.append(coset_min_weight_syndrome(G, vec))
    if not results:
        # Both too large for brute force: fall back to syndrome search anyway
        # (the tests only invoke this on small instances).
        results.append(coset_min_weight_syndrome(G, vec))
    if len(results) == 2:
        assert results[0] == results[1], (
            f"coset_min_weight methods disagree: rowspace={results[0]} "
            f"syndrome={results[1]}"
        )
    return int(results[0])


# --------------------------------------------------------------------------- #
# Subsystem CSS code: stabilizer center + dressed distance
# --------------------------------------------------------------------------- #
def css_center(Gx, Gz):
    """Stabilizer center ``(Sx, Sz)`` of the CSS gauge group ``<Gx, Gz>``.

    The stabilizer group is the center of the gauge group (SPEC §1):

        Sx = { v in rowspace(Gx) : Gz·vᵀ = 0 }
        Sz = { v in rowspace(Gz) : Gx·vᵀ = 0 }

    Computed via the gauge commutation matrix ``E = Gx·Gzᵀ``: a combination
    ``c·Gx`` lies in ``Sx`` iff ``c`` is in the left null space of ``E``
    (``Eᵀ·cᵀ = 0``); symmetrically ``c·Gz`` lies in ``Sz`` iff ``E·cᵀ = 0``.
    Both returned RREF'd with zero rows dropped.
    """
    Gx = _as_gf2(Gx)
    Gz = _as_gf2(Gz)
    n = Gx.shape[1]
    E = _gf2_matmul(Gx, Gz.T)  # ax x az

    # Sx: c in leftnull(E) = nullspace(Eᵀ); Sx = c·Gx.
    cX = nullspace(E.T)
    Sx = _gf2_matmul(cX, Gx) if cX.shape[0] else np.zeros((0, n), dtype=np.uint8)
    Sx = rowspace_basis(Sx)

    # Sz: c in nullspace(E); Sz = c·Gz.
    cZ = nullspace(E)
    Sz = _gf2_matmul(cZ, Gz) if cZ.shape[0] else np.zeros((0, n), dtype=np.uint8)
    Sz = rowspace_basis(Sz)

    return Sx, Sz


def _min_dressed_weight(S, gauge, n) -> int:
    """Min weight ``e`` with ``S·eᵀ = 0`` (e in ker S) AND ``e ∉ rowspace(gauge)``.

    Increasing-weight enumeration over columns (small ``n`` only). The kernel
    constraint uses the stabilizer center ``S``; the triviality test uses the
    gauge group -- this asymmetry is the whole point of the *dressed* distance
    (SPEC §1): ker(S) is strictly larger than ker(gauge_self), so a dressed
    operator may anticommute with individual gauge ops while commuting with all
    stabilizers.
    """
    S = _as_gf2(S)
    in_gauge = _membership_reducer(gauge)
    for w in range(1, n + 1):
        for cols in itertools.combinations(range(n), w):
            e = np.zeros(n, dtype=np.uint8)
            e[list(cols)] = 1
            if S.shape[0] and np.any(_gf2_matmul(S, e.reshape(-1, 1))):
                continue  # e not in ker(S): not a normalizer element
            if not in_gauge(e):
                return w  # commutes with all stabilizers, nontrivial -> dressed
    return n


def dressed_distance_bruteforce(Gx, Gz, which="min") -> int:
    """Brute-force DRESSED distance of the CSS subsystem code ``<Gx, Gz>``.

    With ``(Sx, Sz) = css_center(Gx, Gz)``:

        dZ = min wt Z-type e with  Sx·eᵀ = 0  AND  e ∉ rowspace(Gz)
        dX = min wt X-type e with  Sz·eᵀ = 0  AND  e ∉ rowspace(Gx)

    ``which`` selects ``'Z'`` -> dZ, ``'X'`` -> dX, ``'min'`` -> min(dZ, dX).
    Note the kernel uses the center (Sx/Sz) but triviality uses the gauge
    group (Gz/Gx), per SPEC §1.
    """
    Gx = _as_gf2(Gx)
    Gz = _as_gf2(Gz)
    n = Gx.shape[1]
    assert Gz.shape[1] == n, "Gx and Gz must have the same number of columns (n)"
    Sx, Sz = css_center(Gx, Gz)

    w = which.lower()
    if w == "z":
        return _min_dressed_weight(Sx, Gz, n)
    if w == "x":
        return _min_dressed_weight(Sz, Gx, n)
    dZ = _min_dressed_weight(Sx, Gz, n)
    dX = _min_dressed_weight(Sz, Gx, n)
    return min(dZ, dX)


# --------------------------------------------------------------------------- #
# General (non-CSS) stabilizer codes -- symplectic [z | x] representation
#
# A stabilizer matrix S is (m, 2n) in [z | x] column order: row r has Z-support
# S[r, :n] and X-support S[r, n:]. The symplectic product of two rows a, b is
#     <a, b> = a_z . b_x + a_x . b_z  (mod 2)
# and the Pauli (symplectic) weight of a row is #{ qubits j : z_j=1 OR x_j=1 }.
# These oracles are independent of the package and of the CSS oracles above; they
# are the ground truth for the non-CSS distance / dressed / operator-weight features.
# --------------------------------------------------------------------------- #
def symplectic_weight(vec, n) -> int:
    """Symplectic weight of a length-2n [z|x] vector over ``n`` qubits."""
    v = _as_gf2(vec).reshape(-1)
    z, x = v[:n], v[n:]
    return int(np.count_nonzero(z | x))


def symplectic_product(a, b, n) -> int:
    """Symplectic product ``<a, b> = a_z.b_x + a_x.b_z`` (mod 2) of two [z|x] vectors."""
    a = _as_gf2(a).reshape(-1)
    b = _as_gf2(b).reshape(-1)
    return int((np.dot(a[:n], b[n:]) + np.dot(a[n:], b[:n])) & 1)


def _swap_zx(M):
    """Exchange the [z] and [x] halves of every row of an (m, 2n) matrix."""
    M = _as_gf2(M)
    n = M.shape[1] // 2
    return np.hstack([M[:, n:], M[:, :n]]).astype(np.uint8)


def centralizer(S) -> np.ndarray:
    """Basis (rows) of the centralizer ``C(S) = {e : <e, s>=0 for all s in S}``.

    ``C(S) = nullspace(swap(S))`` because ``<e, s> = swap(e).s = swap(s).e``.
    """
    return nullspace(_swap_zx(S))


def _iter_paulis_of_weight(n, w):
    """Yield every length-2n [z|x] vector of symplectic weight exactly ``w``.

    Each of the ``w`` chosen qubits is assigned one of Z=(z=1,x=0), X=(0,1), Y=(1,1).
    """
    for cols in itertools.combinations(range(n), w):
        for paulis in itertools.product((1, 2, 3), repeat=w):  # 1=Z, 2=X, 3=Y
            v = np.zeros(2 * n, dtype=np.uint8)
            for q, p in zip(cols, paulis):
                if p in (1, 3):
                    v[q] = 1            # z-bit
                if p in (2, 3):
                    v[n + q] = 1        # x-bit
            yield v


def stabilizer_distance_bruteforce(S) -> int:
    """Brute-force distance of a general stabilizer code ``S`` (m, 2n).

    Minimum symplectic weight over an operator that commutes (symplectically) with
    every stabilizer but is not itself in ``rowspace(S)``. Increasing-weight enumeration
    over Paulis (small n only). Mirrors Qubitserf's ``bruteforce_distance0``.
    """
    S = _as_gf2(S)
    n = S.shape[1] // 2
    if rank(centralizer(S)) <= rank(S):
        return -1                        # no logicals: C(S) == rowspace(S) (k = 0)
    in_S = _membership_reducer(S)
    Sswap = _swap_zx(S)                  # commute test: <s, c> = swap(s).c = Sswap . c
    for w in range(1, n + 1):
        for c in _iter_paulis_of_weight(n, w):
            if S.shape[0] and np.any(_gf2_matmul(Sswap, c.reshape(-1, 1))):
                continue                 # anticommutes with some stabilizer
            if not in_S(c):
                return w                 # in C(S), not a stabilizer -> a logical
    return n


def symplectic_center(G) -> np.ndarray:
    """Stabilizer center of a gauge group ``G`` (m, 2n): the elements of rowspace(G)
    that commute symplectically with every generator. ``v = c.G`` is central iff
    ``c`` lies in the null space of the symplectic Gram matrix ``Gram[i][j]=<g_i,g_j>``.
    Returned RREF'd with zero rows dropped."""
    G = _as_gf2(G)
    n = G.shape[1] // 2
    Gram = _gf2_matmul(G, _swap_zx(G).T)          # m x m, <g_i, g_j>
    c = nullspace(Gram)
    center = _gf2_matmul(c, G) if c.shape[0] else np.zeros((0, 2 * n), dtype=np.uint8)
    return rowspace_basis(center)


def dressed_stabilizer_distance_bruteforce(G) -> int:
    """Brute-force DRESSED distance of a general (non-CSS) subsystem code ``G`` (m, 2n).

    With ``S = symplectic_center(G)``: minimum symplectic weight over an operator that
    commutes with every stabilizer (``e in C(S)``) but is not in ``rowspace(G)``.
    """
    G = _as_gf2(G)
    n = G.shape[1] // 2
    S = symplectic_center(G)
    if rank(centralizer(S)) <= rank(G):
        return -1                        # no dressed logicals: C(S) == rowspace(G)
    in_G = _membership_reducer(G)
    Sswap = _swap_zx(S)
    for w in range(1, n + 1):
        for c in _iter_paulis_of_weight(n, w):
            if S.shape[0] and np.any(_gf2_matmul(Sswap, c.reshape(-1, 1))):
                continue                 # not in C(S)
            if not in_G(c):
                return w                 # commutes with all stabilizers, not in gauge
    return n


def symplectic_coset_min_weight_rowspace(G, op) -> int:
    """Min symplectic weight over ``op + rowspace(G)`` by ROWSPACE enumeration."""
    G = _as_gf2(G)
    op = _as_gf2(op).reshape(-1)
    n = G.shape[1] // 2
    B = rowspace_basis(G)
    k = B.shape[0]
    best = symplectic_weight(op, n)
    for bits in range(1 << k):
        g = np.zeros(op.shape[0], dtype=np.uint8)
        for i in range(k):
            if (bits >> i) & 1:
                g ^= B[i]
        w = symplectic_weight(op ^ g, n)
        if w < best:
            best = w
            if best == 0:
                break
    return best


def symplectic_coset_min_weight_syndrome(G, op) -> int:
    """Min symplectic weight over ``op + rowspace(G)`` by increasing-weight search.

    A vector ``u`` lies in the coset iff ``P.uᵀ = P.opᵀ`` with ``P = nullspace(G)``
    (ordinary rowspace membership). Scans Paulis by increasing symplectic weight.
    """
    G = _as_gf2(G)
    op = _as_gf2(op).reshape(-1)
    n = G.shape[1] // 2
    P = nullspace(G)
    target = _gf2_matmul(P, op.reshape(-1, 1)).reshape(-1)
    # w = 0 matches iff op in rowspace(G) (target all-zero); then onward by weight.
    for w in range(0, n + 1):
        words = _iter_paulis_of_weight(n, w) if w > 0 else [np.zeros(2 * n, np.uint8)]
        for u in words:
            syn = _gf2_matmul(P, u.reshape(-1, 1)).reshape(-1)
            if np.array_equal(syn, target):
                return w
    return n


def symplectic_coset_min_weight_bruteforce(G, op) -> int:
    """Minimum-weight coset leader of ``op + rowspace(G)`` in symplectic weight.

    Runs whichever methods are feasible and asserts they agree. 0 iff op in rowspace(G).
    """
    G = _as_gf2(G)
    op = _as_gf2(op).reshape(-1)
    n = G.shape[1] // 2
    k = rank(G)
    results = []
    if k <= 20:
        results.append(symplectic_coset_min_weight_rowspace(G, op))
    if n <= 12:
        results.append(symplectic_coset_min_weight_syndrome(G, op))
    if not results:
        results.append(symplectic_coset_min_weight_rowspace(G, op))
    if len(results) == 2:
        assert results[0] == results[1], (
            "symplectic coset methods disagree: rowspace=%d syndrome=%d"
            % (results[0], results[1]))
    return int(results[0])


def paulis_to_symplectic(strings):
    """Convert Pauli strings (I/X/Y/Z) to an (m, 2n) [z|x] matrix (a Y sets both)."""
    rows = []
    n = len(strings[0])
    for s in strings:
        z = np.zeros(n, dtype=np.uint8)
        x = np.zeros(n, dtype=np.uint8)
        for j, c in enumerate(s):
            if c in "XY":
                x[j] = 1
            if c in "ZY":
                z[j] = 1
        rows.append(np.concatenate([z, x]))
    return np.array(rows, dtype=np.uint8)


def five_qubit_code():
    """The [[5,1,3]] perfect code as a symplectic [z|x] stabilizer matrix."""
    return paulis_to_symplectic(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])


# --------------------------------------------------------------------------- #
# Independent code constructors (so the oracle never imports the package)
# --------------------------------------------------------------------------- #
def bacon_shor(d):
    """Bacon-Shor gauge generators on a ``d x d`` grid.

    Qubit ``(r, c)`` has index ``r*d + c`` (``n = d²``).
    X-gauge = ``XX`` on horizontally-adjacent pairs (same row, cols c, c+1);
    Z-gauge = ``ZZ`` on vertically-adjacent pairs (same col, rows r, r+1).
    Returns ``(Gx, Gz)``. The dressed distance of this code is ``d``.
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


def repetition_parity(n):
    """Parity check of the [n, 1, n] repetition code (shape (n-1, n))."""
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def hamming_parity(r):
    """Parity check of the [2^r-1, 2^r-1-r, 3] Hamming code (shape (r, 2^r-1))."""
    n = (1 << r) - 1
    cols = [[(j >> b) & 1 for b in range(r)] for j in range(1, n + 1)]
    return np.array(cols, dtype=np.uint8).T


def steane():
    """Steane [[7,1,3]] code (Hx = Hz = Hamming[7,4,3]); returns (Hx, Hz)."""
    H = hamming_parity(3)
    return H.copy(), H.copy()


def hypergraph_product(H1, H2=None):
    """CSS code (Hx, Hz) from the hypergraph product of parity checks.

        Hx = [ H1 ⊗ I_{n2} | I_{m1} ⊗ H2ᵀ ]
        Hz = [ I_{n1} ⊗ H2 | H1ᵀ ⊗ I_{m2} ]
    """
    H1 = _as_gf2(H1)
    H2 = H1 if H2 is None else _as_gf2(H2)
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    Hx = np.hstack([np.kron(H1, np.eye(n2, dtype=np.uint8)),
                    np.kron(np.eye(m1, dtype=np.uint8), H2.T)]).astype(np.uint8) & 1
    Hz = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(m2, dtype=np.uint8))]).astype(np.uint8) & 1
    return Hx, Hz


def surface_3():
    """Planar surface(3) = [[13,1,3]] code (HGP of repetition_parity(3))."""
    return hypergraph_product(repetition_parity(3))


# --------------------------------------------------------------------------- #
# Self-verification (run as a script)
# --------------------------------------------------------------------------- #
def _logical_z_rep(Hx, Hz):
    """A minimum-weight Z-type logical of the CSS code: min wt e in ker(Hx)
    (i.e. Hx·eᵀ=0) that is NOT in rowspace(Hz). Brute force, small n only."""
    n = Hx.shape[1]
    in_Hz = _membership_reducer(Hz)
    for w in range(1, n + 1):
        for cols in itertools.combinations(range(n), w):
            e = np.zeros(n, dtype=np.uint8)
            e[list(cols)] = 1
            if np.any(_gf2_matmul(Hx, e.reshape(-1, 1))):
                continue
            if not in_Hz(e):
                return e
    raise RuntimeError("no logical found")


def _random_css(rng, n, m):
    """A random small CSS code (Hx, Hz) with Hx·Hzᵀ = 0, via a random Gz and
    Hx drawn from a basis of ker(Gz)."""
    while True:
        Gz = (rng.integers(0, 2, size=(m, n))).astype(np.uint8)
        if rank(Gz) == 0:
            continue
        Gz = rowspace_basis(Gz)
        K = nullspace(Gz)  # rows orthogonal to Gz: candidate Hx (Hx·Gzᵀ=0)
        if K.shape[0] < 1:
            continue
        # take a random nonempty subset of K rows as Hx
        nsel = int(rng.integers(1, K.shape[0] + 1))
        idx = rng.choice(K.shape[0], size=nsel, replace=False)
        Hx = rowspace_basis(K[idx])
        if Hx.shape[0] == 0:
            continue
        assert not np.any(_gf2_matmul(Hx, Gz.T)), "CSS condition violated"
        return Hx, Gz


def _selftest():
    ok = 0

    def check(cond, msg):
        nonlocal ok
        assert cond, "FAIL: " + msg
        ok += 1
        print("  PASS:", msg)

    print("== GF(2) primitives ==")
    A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    check(rank(A) == 2, "rank([[110],[011],[101]]) == 2")
    N = nullspace(A)
    check(N.shape[0] == 1 and not np.any(_gf2_matmul(A, N.T)),
          "nullspace basis annihilates A")
    check(rowspace_contains(A, np.array([1, 0, 1])), "rowspace_contains true case")
    check(not rowspace_contains(A, np.array([1, 0, 0])), "rowspace_contains false case")

    print("== Bacon-Shor dressed distance ==")
    Gx3, Gz3 = bacon_shor(3)
    check(Gx3.shape[1] == 9, "bacon_shor(3) has n = 9")
    d3 = dressed_distance_bruteforce(Gx3, Gz3)
    check(d3 == 3, f"bacon_shor(3) dressed distance == 3 (got {d3})")
    dZ3 = dressed_distance_bruteforce(Gx3, Gz3, which="Z")
    dX3 = dressed_distance_bruteforce(Gx3, Gz3, which="X")
    check(dZ3 == 3 and dX3 == 3, f"bacon_shor(3) dZ==dX==3 (got {dZ3},{dX3})")

    print("== Bacon-Shor center invariants ==")
    Sx, Sz = css_center(Gx3, Gz3)
    in_Gx = _membership_reducer(Gx3)
    in_Gz = _membership_reducer(Gz3)
    check(all(in_Gx(row) for row in Sx), "every Sx row in rowspace(Gx)")
    check(not np.any(_gf2_matmul(Sx, Gz3.T)), "Sx·Gzᵀ = 0")
    check(all(in_Gz(row) for row in Sz), "every Sz row in rowspace(Gz)")
    check(not np.any(_gf2_matmul(Sz, Gx3.T)), "Sz·Gxᵀ = 0")

    print("== Steane as a subsystem code (gauge = stabilizers) ==")
    Hx, Hz = steane()
    check(not np.any(_gf2_matmul(Hx, Hz.T)), "Steane Hx·Hzᵀ = 0 (self-orthogonal)")
    dst = dressed_distance_bruteforce(Hx, Hz)
    check(dst == 3, f"Steane subsystem dressed distance == 3 (got {dst})")

    print("== Operator weight: Steane ==")
    # logical Z as the all-ones vector, modulo Gz = Hamming -> 3.
    w_logZ = coset_min_weight_bruteforce(Hz, np.ones(7, dtype=np.uint8))
    check(w_logZ == 3, f"Steane all-ones coset min weight == 3 (got {w_logZ})")
    # a single Hz (stabilizer) row -> 0.
    w_stab = coset_min_weight_bruteforce(Hz, Hz[0])
    check(w_stab == 0, f"Steane single stabilizer row -> 0 (got {w_stab})")

    print("== Operator weight: surface(3) [[13,1,3]] (the qubitserf-bug case) ==")
    sHx, sHz = surface_3()
    check(sHx.shape[1] == 13 and not np.any(_gf2_matmul(sHx, sHz.T)),
          "surface(3): n=13 and Hx·Hzᵀ=0")
    # A single Z-stabilizer row, modulo Gz = Hz -> 0 (qubitserf returns 3: BUG).
    w_surf_stab = coset_min_weight_bruteforce(sHz, sHz[0])
    check(w_surf_stab == 0,
          f"surface(3) stabilizer row coset min weight == 0 (got {w_surf_stab})")
    # An inflated logical Z (logical XOR a few stabilizers) -> 3.
    logz = _logical_z_rep(sHx, sHz)
    inflated = logz.copy()
    inflated ^= sHz[0]
    inflated ^= sHz[2]
    check(coset_min_weight_bruteforce(sHz, logz) == 3,
          "surface(3) bare logical Z coset min weight == 3")
    w_infl = coset_min_weight_bruteforce(sHz, inflated)
    check(w_infl == 3,
          f"surface(3) inflated logical Z coset min weight == 3 (got {w_infl})")

    print("== coset_min_weight methods (a) and (b) agree on random CSS ==")
    rng = np.random.default_rng(0xC05E7)
    agree = 0
    for _ in range(12):
        n = int(rng.integers(4, 10))
        m = int(rng.integers(1, max(2, n - 1)))
        Hx, Gz = _random_css(rng, n, m)
        vec = rng.integers(0, 2, size=n).astype(np.uint8)
        a = coset_min_weight_rowspace(Gz, vec)
        b = coset_min_weight_syndrome(Gz, vec)
        assert a == b, f"methods disagree on n={n}: rowspace={a} syndrome={b}"
        # coset_min_weight_bruteforce internally asserts agreement too.
        c = coset_min_weight_bruteforce(Gz, vec)
        assert c == a
        agree += 1
    check(agree == 12, "12/12 random CSS coset weights: rowspace == syndrome")

    print("== Bacon-Shor d=5 (skipped if slow) ==")
    import time
    Gx5, Gz5 = bacon_shor(5)
    t0 = time.time()
    d5 = dressed_distance_bruteforce(Gx5, Gz5)
    dt = time.time() - t0
    check(d5 == 5, f"bacon_shor(5) dressed distance == 5 (got {d5}, {dt:.1f}s)")

    print(f"\nAll {ok} checks PASSED.")


if __name__ == "__main__":
    _selftest()
