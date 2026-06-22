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

    # Toric codes L = 4..8  ([[2L^2, 2, L]]).
    toric = []
    for L in (4, 5, 6, 7, 8):
        Hx, Hz = codes.toric(L)
        toric.append((f"toric L={L}", Hx, Hz, L))
    fams["toric"] = toric

    # Surface codes L = 4..7  ([[L^2+(L-1)^2, 1, L]]).
    surf = []
    for L in (4, 5, 6, 7):
        Hx, Hz = codes.surface(L)
        surf.append((f"surface L={L}", Hx, Hz, L))
    fams["surface"] = surf

    # Bivariate-bicycle codes (sparse; weak BZ lower bound -> "hard" for BZ).
    fams["bivariate_bicycle"] = [
        bb_72_12_6(),
        bb_gross_144_12_12(),
    ]

    # Hypergraph-product codes from classical bases.
    fams["hypergraph_product"] = [
        hgp_hamming(3),                          # n=58
        hgp_random_ldpc(6, 10, 3, seed=2),       # n=136 (hard-ish)
    ]

    return fams


def classical_families():
    """List of (name, H, known_d) classical codes."""
    return [
        classical_hamming(3),
        classical_hamming(4),
        classical_random_ldpc(12, 18, 3, seed=1),
    ]


# Which families/codes are "hard" for BZ (sparse, weak lower bound): BZ should
# be capped with a max_weight bracket rather than run to certification.
HARD_CSS_NAMES = {
    "bb [[72,12,6]]",
    "gross [[144,12,12]]",
    "hgp(ldpc6x10) ",
    "toric L=8",
}
