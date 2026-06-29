"""Extra code-family generators for the comprehensive benchmark.

These augment ``qminweight.codes`` with a few named CSS and classical codes that
the comprehensive benchmark sweeps over.  Nothing here mutates the library; we
only *call* ``qminweight.codes`` generators and re-package their outputs with a
benchmark-friendly signature:

    CSS       generator -> (name, Hx, Hz, known_d_or_None)
    classical generator -> (name, H,  known_d_or_None)

The "hard?" / size policy (which methods to cap) lives in ``comprehensive.py``;
this module is purely about *what* codes exist.
"""
from __future__ import annotations
from math import comb

import numpy as np

from qminweight import codes


# --------------------------------------------------------------------------- #
# Bivariate-bicycle helpers
# --------------------------------------------------------------------------- #
# The canonical A = x^3 + y + y^2, B = y^3 + x + x^2 family (Bravyi et al.).
_BB_A = [("x", 3), ("y", 1), ("y", 2)]
_BB_B = [("y", 3), ("x", 1), ("x", 2)]


def bb(l: int, m: int):
    """A bivariate-bicycle code in the canonical A/B family on 2*l*m qubits."""
    return codes.bivariate_bicycle(l, m, _BB_A, _BB_B)


def bb_72_12_6():
    """The [[72,12,6]] bivariate-bicycle code (l=m=6)."""
    Hx, Hz = bb(6, 6)
    return ("bb [[72,12,6]]", Hx, Hz, 6)


def bb_gross_144_12_12():
    """The IBM 'gross' [[144,12,12]] bivariate-bicycle code (l=12, m=6)."""
    Hx, Hz = codes.gross_code()
    return ("gross [[144,12,12]]", Hx, Hz, 12)


def bb_288():
    """A larger [[288,12,12]] bivariate-bicycle code (l=m=12, same A/B family).

    Distance is left unknown (qminweight is ground truth) so a capped BZ bracket
    never trips a spurious mismatch.
    """
    Hx, Hz = bb(12, 12)
    return ("bb [[288,12,12]]", Hx, Hz, None)


# --------------------------------------------------------------------------- #
# Hypergraph-product codes from classical bases
# --------------------------------------------------------------------------- #
def hgp_hamming(r: int):
    """HGP of the [2^r-1, .., 3] Hamming parity check with itself."""
    h = codes.hamming_parity(r)
    Hx, Hz = codes.hypergraph_product(h, h)
    # HGP(H,H) distance equals the classical distance of the base code (=3 for
    # Hamming) when the base has full row rank; we leave it unknown (None) so the
    # benchmark treats qminweight's answer as ground truth and just cross-checks.
    return (f"hgp(ham{r}) ", Hx, Hz, 3)


def hgp_random_ldpc(m: int, n: int, col_weight: int = 3, seed: int = 0):
    """HGP of a random column-regular LDPC parity check with itself."""
    h = codes.random_ldpc_parity(m, n, col_weight, seed=seed)
    Hx, Hz = codes.hypergraph_product(h, h)
    nn = Hx.shape[1]
    return (f"hgp(ldpc{m}x{n}) ", Hx, Hz, None)


def hgp_repetition(nrep: int):
    """HGP of the [nrep,1,nrep] repetition parity check (a surface-like code)."""
    h = codes.repetition_parity(nrep)
    Hx, Hz = codes.hypergraph_product(h, h)
    return (f"hgp(rep{nrep}) ", Hx, Hz, nrep)


# --------------------------------------------------------------------------- #
# Quantum Reed-Muller CSS codes
# --------------------------------------------------------------------------- #
def qrm(r: int, m: int):
    """Quantum Reed-Muller CSS code Hx=Hz=G_RM(r,m), valid when 2r < m-1.

    n=2^m, k=2^m - 2*sum_{i<=r} C(m,i), d=2^(r+1).
    Rows have weight n ... n/2^r — DENSE, NOT QLDPC.  BZ handles these efficiently;
    CC (connected cluster) degenerates to an O(C(n,d)) search because every check
    touches O(n/2^r) qubits.
    """
    Hx, Hz = codes.quantum_reed_muller(r, m)
    n = 1 << m
    k = n - 2 * sum(comb(m, i) for i in range(r + 1))
    d = 1 << (r + 1)
    return (f"qrm(r={r},m={m}) [[{n},{k},{d}]]", Hx, Hz, d)


# --------------------------------------------------------------------------- #
# Classical codes
# --------------------------------------------------------------------------- #
def classical_hamming(r: int):
    """[2^r-1, 2^r-1-r, 3] Hamming code (distance 3)."""
    return (f"hamming r={r}", codes.hamming_parity(r), 3)


def classical_random_ldpc(m: int, n: int, col_weight: int = 3, seed: int = 0):
    """Random column-regular LDPC code (distance unknown -> qminweight decides)."""
    H = codes.random_ldpc_parity(m, n, col_weight, seed=seed)
    return (f"rand_ldpc({m},{n},{col_weight})", H, None)


# --------------------------------------------------------------------------- #
# Family assembly (consumed by comprehensive.py)
# --------------------------------------------------------------------------- #
def css_families():
    """Ordered dict {family_name: [(name, Hx, Hz, known_d)]}.

    Sizes are kept so qminweight finishes fast; the comprehensive driver applies
    per-method time budgets and caps BZ on the hard (sparse, weak-BZ-bound)
    codes so nothing hangs.
    """
    fams: dict[str, list] = {}

    # Small textbook CSS codes.
    sx, sz = codes.steane()
    sh_x, sh_z = codes.shor()
    fams["small"] = [
        ("steane [[7,1,3]]", sx, sz, 3),
        ("shor [[9,1,3]]", sh_x, sh_z, 3),
    ]

    # Toric codes L = 4..12  ([[2L^2, 2, L]], n = 32..288).
    toric = []
    for L in (4, 5, 6, 7, 8, 9, 10, 11, 12):
        Hx, Hz = codes.toric(L)
        toric.append((f"toric L={L}", Hx, Hz, L))
    fams["toric"] = toric

    # Surface codes L = 4..12  ([[L^2+(L-1)^2, 1, L]], n = 25..265).
    surf = []
    for L in (4, 5, 6, 7, 8, 9, 10, 11, 12):
        Hx, Hz = codes.surface(L)
        surf.append((f"surface L={L}", Hx, Hz, L))
    fams["surface"] = surf

    # Bivariate-bicycle codes (sparse; weak BZ lower bound -> "hard" for BZ).
    fams["bivariate_bicycle"] = [
        bb_72_12_6(),
        bb_gross_144_12_12(),
        bb_288(),                                # n=288 (hard)
    ]

    # Hypergraph-product codes from classical bases.
    fams["hypergraph_product"] = [
        hgp_hamming(3),                          # n=58
        hgp_random_ldpc(6, 10, 3, seed=2),       # n=136 (hard-ish)
        hgp_hamming(4),                          # n=241
    ]

    # Quantum Reed-Muller CSS codes — DENSE (NOT QLDPC).
    # Hx = Hz = G_RM(r, m): rows have weight n, n/2, ..., n/2^r.
    # CC degenerates to O(C(n,d)) on these (same as MITM), while BZ handles
    # small-d dense codes efficiently — the crossover between BZ and CC is
    # most visible here.
    #
    # r=1 sub-family: d=4 fixed, n doubles with m. CC and MITM scale as n^4,
    # BZ finds d=4 quickly at any n.
    fams["reed_muller_r1"] = [
        qrm(1, 4),   # [[16, 6, 4]]
        qrm(1, 5),   # [[32, 20, 4]]
        qrm(1, 6),   # [[64, 50, 4]]
        qrm(1, 7),   # [[128, 112, 4]]
        qrm(1, 8),   # [[256, 238, 4]]
        qrm(1, 9),   # [[512, 492, 4]]  (n>256: native GPU BZ; d=4 stays fast)
    ]
    # r=2 sub-family: d=8 fixed. At n=64 CC already nearly times out (C(64,8)≈4.4B
    # DFS nodes); at n=128 CC times out while BZ stays < 1s; at n=256 CC times out
    # while BZ still certifies (slower — the BZ lower bound converges slowly for this
    # large high-rate code, but it stays within the 300s budget).
    fams["reed_muller_r2"] = [
        qrm(2, 6),   # [[64, 20, 8]]   CC ≈98-136s (barely), BZ ≈18ms
        qrm(2, 7),   # [[128, 70, 8]]  CC timeout, BZ ≈0.8s
        qrm(2, 8),   # [[256, 182, 8]] CC timeout, BZ ~minutes (still wins)
    ]

    return fams


def classical_families():
    """List of (name, H, known_d) classical codes."""
    return [
        classical_hamming(3),
        classical_hamming(4),
        classical_random_ldpc(12, 18, 3, seed=1),
    ]


# Which families/codes are "hard" for BZ (sparse QLDPC, weak lower bound): BZ should
# be capped with a max_weight bracket rather than run to certification.  This set
# grows with BZ_MAX_N: whenever a sparse code with d >> BZ_CAP enters the BZ window,
# add it here so uncapped BZ cannot hang and leave an orphan thread.
HARD_CSS_NAMES = {
    "bb [[72,12,6]]",
    "gross [[144,12,12]]",
    "bb [[288,12,12]]",
    "hgp(ldpc6x10) ",
    # Toric L>=9 (n>=162, d=L) — d grows with sqrt(n); uncapped BZ hangs.
    "toric L=9", "toric L=10", "toric L=11", "toric L=12",
    # Surface L>=9 (n>=145, d=L) — same reason.
    "surface L=9", "surface L=10", "surface L=11", "surface L=12",
}
