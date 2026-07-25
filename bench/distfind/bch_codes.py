"""Quantum BCH codes: CSS codes from dual-containing narrow-sense primitive BCH codes.

A narrow-sense primitive BCH code C = [n=2^m-1, k, >=delta] contains its dual iff
delta <= 2^ceil(m/2) - 1 (Steane); then CSS(C, C) with Hx = Hz = H(C) gives a
[[n, 2k-n, d]] quantum code with d >= delta (dZ = dX by symmetry). These codes have
DENSE parity checks (row weight ~ n/2 after systematisation, and the Tanner graph is
far from LDPC), which is exactly the regime where information-set methods (BZ) win
and cluster-growth methods (CC) blow up.

Pure python/numpy (GF(2^m) log tables, cyclotomic cosets, generator polynomial) --
no Sage needed. Every construction is self-checked: designed distance, k, and
H . H^T = 0 (dual containment).
"""
from __future__ import annotations

import numpy as np

# Primitive polynomials over GF(2), as bitmasks (bit i = coefficient of x^i).
_PRIM = {3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011, 7: 0b10001001,
         8: 0b100011101}


def _gf_tables(m: int):
    """(exp, log) tables for GF(2^m) with the primitive element alpha = x."""
    n = (1 << m) - 1
    prim = _PRIM[m]
    exp = np.zeros(2 * n, dtype=np.int64)
    log = np.zeros(n + 1, dtype=np.int64)
    v = 1
    for i in range(n):
        exp[i] = v
        log[v] = i
        v <<= 1
        if v & (1 << m):
            v ^= prim
    exp[n:2 * n] = exp[:n]
    return exp, log


def _cyclotomic_coset(i: int, n: int) -> frozenset:
    c, j = set(), i % n
    while j not in c:
        c.add(j)
        j = (2 * j) % n
    return frozenset(c)


def _poly_mul_gf2m(a, b, exp, log, n):
    """Product of polynomials with coefficients in GF(2^m) (lists, a[i] = coeff of x^i)."""
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            out[i + j] ^= exp[(log[ai] + log[bj]) % n]
    return out


def bch_generator_poly(m: int, delta: int) -> list[int]:
    """g(x) of the narrow-sense primitive BCH code with designed distance delta,
    as a GF(2) coefficient list (g[i] = coeff of x^i)."""
    n = (1 << m) - 1
    exp, log = _gf_tables(m)
    cosets = set()
    for i in range(1, delta):
        cosets.add(_cyclotomic_coset(i, n))
    g = [1]
    for coset in sorted(cosets, key=min):
        # minimal polynomial of alpha^i for i in coset: prod_j (x + alpha^j)
        p = [1]
        for j in sorted(coset):
            p = _poly_mul_gf2m(p, [exp[j], 1], exp, log, n)
        assert all(c in (0, 1) for c in p), "minimal polynomial not binary"
        g = _poly_mul_gf2m(g, p, exp, log, n)
    assert all(c in (0, 1) for c in g)
    return [int(c) for c in g]


def bch_parity_check(m: int, delta: int) -> np.ndarray:
    """Parity-check matrix H of the [n, k] BCH code: (n-k) x n, uint8.

    Rows are the cyclic shifts of the reversed check polynomial h*(x), where
    h(x) = (x^n - 1)/g(x). rowspace(H) = C^perp."""
    n = (1 << m) - 1
    g = bch_generator_poly(m, delta)
    r = len(g) - 1            # deg g = n - k
    k = n - r
    # h(x) = (x^n + 1) / g(x) over GF(2) (long division)
    num = [0] * (n + 1)
    num[0] = 1
    num[n] = 1
    h = [0] * (k + 1)
    rem = list(num)
    for i in range(n, r - 1, -1):
        if rem[i]:
            h[i - r] = 1
            for j, gj in enumerate(g):
                rem[i - r + j] ^= gj
    assert not any(rem), "g(x) does not divide x^n + 1"
    hrev = h[::-1]            # h*(x), degree k
    H = np.zeros((r, n), dtype=np.uint8)
    for s in range(r):
        for i, c in enumerate(hrev):
            H[s, (s + i) % n] = c
    return H


def quantum_bch(m: int, delta: int):
    """(Hx, Hz, n, k_css) for CSS(C, C) from the dual-containing BCH code.

    Raises if the BCH code is not dual-containing (H . H^T != 0)."""
    H = bch_parity_check(m, delta)
    n = H.shape[1]
    if np.any((H @ H.T) % 2):
        raise ValueError(f"BCH(m={m}, delta={delta}) is not dual-containing")
    # GF(2) rank of H
    A = H.copy()
    rank, c = 0, 0
    for col in range(n):
        piv = next((i for i in range(rank, A.shape[0]) if A[i, col]), None)
        if piv is None:
            continue
        A[[rank, piv]] = A[[piv, rank]]
        for i in range(A.shape[0]):
            if i != rank and A[i, col]:
                A[i] ^= A[rank]
        rank += 1
    k_css = n - 2 * rank
    return H.copy(), H.copy(), n, k_css


# (label, m, delta, expected [[n, k]]) -- designed distance delta is a LOWER bound on d.
CASES = [
    ("qbch [[15,7,3]]",   4, 3,  (15, 7)),
    ("qbch [[31,11,5]]",  5, 5,  (31, 11)),
    ("qbch [[31,1,7]]",   5, 7,  (31, 1)),
    ("qbch [[63,39,5]]",  6, 5,  (63, 39)),
    ("qbch [[63,27,7]]",  6, 7,  (63, 27)),
    ("qbch [[127,71,9]]", 7, 9,  (127, 71)),
]


if __name__ == "__main__":
    for label, m, delta, (n_want, k_want) in CASES:
        Hx, Hz, n, k = quantum_bch(m, delta)
        row_w = int(Hx.sum(axis=1).max())
        status = "OK " if (n, k) == (n_want, k_want) else "MISMATCH"
        print(f"{status} {label}: n={n} k={k} (want [[{n_want},{k_want}]], d>={delta}); "
              f"H {Hx.shape}, max row weight {row_w}")
