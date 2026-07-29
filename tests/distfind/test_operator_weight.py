"""Cross-check the native operator-weight feature against the pure-Python oracle.

Operator weight is the minimum-weight coset leader of a Pauli operator modulo the
group ``<Gx, Gz>`` (stabilizers for a stabilizer code, gauge generators for a
subsystem code). The Z-part is reduced modulo rowspace(Gz), the X-part modulo
rowspace(Gx); the two are independent (SPEC §0).

Ground truth is ``_reference.coset_min_weight_bruteforce`` -- if native ever
disagrees, the ORACLE wins and the test fails loudly.

Run from the repo root:
    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest

import qubitserf.distfind as df
from qubitserf.distfind import codes, io, _native

import _reference as ref


def _oracle(Gx, Gz, z_vec, x_vec):
    """(z_weight, x_weight) coset leaders per the oracle."""
    return (ref.coset_min_weight_bruteforce(Gz, z_vec),
            ref.coset_min_weight_bruteforce(Gx, x_vec))


def _split(Gx, Gz, z_vec, x_vec, *, method="bz"):
    """The native ``(z_weight, x_weight)`` pair.

    ``df.operator_weight`` returns only ``max(z, x)``; the raw layer still reports
    the two coset leaders separately, and the oracle cross-checks below need both
    (``max`` alone cannot tell (3, 0) from (0, 3)).
    """
    r = _native.operator_weight_raw(Gx, Gz, z_vec, x_vec, method, "cpu", 0, False)
    return int(r.z_weight), int(r.x_weight)


def _check(Gx, Gz, z_vec, x_vec, *, method="bz"):
    """Assert native operator weight matches the oracle; return ``(z, x)``."""
    z_ref, x_ref = _oracle(Gx, Gz, z_vec, x_vec)
    z_got, x_got = _split(Gx, Gz, z_vec, x_vec, method=method)
    assert z_got == z_ref, f"z_weight native={z_got} oracle={z_ref} (method={method})"
    assert x_got == x_ref, f"x_weight native={x_got} oracle={x_ref} (method={method})"
    w = df.operator_weight(Gx, Gz, (z_vec, x_vec), method=method, backend="cpu")
    assert w == max(z_ref, x_ref), f"weight native={w} oracle={max(z_ref, x_ref)}"
    return z_got, x_got


# --------------------------------------------------------------------------- #
# Steane (self-orthogonal): logical, stabilizer, all-ones, Y operators.
# --------------------------------------------------------------------------- #
def test_steane_logical_and_stabilizer():
    Hx, Hz = codes.steane()
    n = 7
    # logical Z (all-ones Z) -> z_weight 3, x_weight 0.
    assert _check(Hx, Hz, np.ones(n, np.uint8), np.zeros(n, np.uint8)) == (3, 0)
    assert df.operator_weight(Hx, Hz, "ZZZZZZZ", backend="cpu") == 3
    # a single stabilizer row -> weight 0.
    assert _check(Hx, Hz, Hz[0], np.zeros(n, np.uint8)) == (0, 0)
    assert df.operator_weight(Hx, Hz, (Hz[0], np.zeros(n, np.uint8)), backend="cpu") == 0


def test_steane_y_operator():
    Hx, Hz = codes.steane()
    n = 7
    z, x = io.parse_operator("YYYYYYY", n)
    assert np.all(z == 1) and np.all(x == 1)
    assert _check(Hx, Hz, z, x) == (3, 3)      # logical in both parts
    # mixed Pauli string with a couple of Ys
    z2, x2 = io.parse_operator("YXZIYXZ", n)
    _check(Hx, Hz, z2, x2)


# --------------------------------------------------------------------------- #
# surface(3) [[13,1,3]] -- regression guard for the original qubitserf's operator-weight bug.
# A single Z-stabilizer row is equivalent to identity -> weight 0 (the original qubitserf
# wrongly returns 3 for non-self-orthogonal codes).
# --------------------------------------------------------------------------- #
def test_surface3_stabilizer_is_weight_zero():
    Hx, Hz = ref.surface_3()
    n = 13
    for row in Hz:
        z_got, _ = _split(Hx, Hz, row, np.zeros(n, np.uint8))
        assert z_got == 0, f"stabilizer row should reduce to 0, got {z_got}"
        assert ref.coset_min_weight_bruteforce(Hz, row) == 0
    for row in Hx:
        _, x_got = _split(Hx, Hz, np.zeros(n, np.uint8), row)
        assert x_got == 0


def test_surface3_inflated_logical_is_distance():
    Hx, Hz = ref.surface_3()
    logz = ref._logical_z_rep(Hx, Hz)
    n = 13
    # bare logical and a logical inflated by a few stabilizers both reduce to 3.
    _check(Hx, Hz, logz, np.zeros(n, np.uint8))
    inflated = logz.copy()
    inflated ^= Hz[0]
    inflated ^= Hz[2]
    assert _check(Hx, Hz, inflated, np.zeros(n, np.uint8)) == (3, 0)


# --------------------------------------------------------------------------- #
# Random small CSS codes vs the brute-force coset leader.
# --------------------------------------------------------------------------- #
def test_random_css_operator_weight():
    rng = np.random.default_rng(0x09E12)
    for _ in range(25):
        n = int(rng.integers(4, 11))
        m = int(rng.integers(1, max(2, n - 1)))
        Hx, Gz = ref._random_css(rng, n, m)  # Gx=Hx, Gz mutually orthogonal
        z = rng.integers(0, 2, size=n).astype(np.uint8)
        x = rng.integers(0, 2, size=n).astype(np.uint8)
        _check(Hx, Gz, z, x)
        # stabilizer rows reduce to 0
        if Gz.shape[0]:
            _check(Hx, Gz, Gz[0], np.zeros(n, np.uint8))
        if Hx.shape[0]:
            _check(Hx, Gz, np.zeros(n, np.uint8), Hx[0])


# --------------------------------------------------------------------------- #
# bz and mitm must agree.
# --------------------------------------------------------------------------- #
def test_bz_and_mitm_agree():
    rng = np.random.default_rng(0xB2117)
    cases = [codes.steane(), ref.surface_3()]
    for _ in range(10):
        n = int(rng.integers(4, 10))
        m = int(rng.integers(1, max(2, n - 1)))
        cases.append(ref._random_css(rng, n, m))
    for Gx, Gz in cases:
        n = Gx.shape[1]
        z = rng.integers(0, 2, size=n).astype(np.uint8)
        x = rng.integers(0, 2, size=n).astype(np.uint8)
        rbz = _split(Gx, Gz, z, x, method="bz")
        rmitm = _split(Gx, Gz, z, x, method="mitm")
        assert rbz == rmitm
        # and both equal the oracle
        assert rbz == _oracle(Gx, Gz, z, x)
        assert df.operator_weight(Gx, Gz, (z, x), backend="cpu") == max(rbz)


# --------------------------------------------------------------------------- #
# Equivalent operator encodings (Pauli string / tuple / symplectic) agree.
# --------------------------------------------------------------------------- #
def test_operator_encodings_equivalent():
    Hx, Hz = codes.steane()
    n = 7
    z, x = io.parse_operator("YZXIYZX", n)
    from_str = df.operator_weight(Hx, Hz, "YZXIYZX", backend="cpu")
    from_tuple = df.operator_weight(Hx, Hz, (z, x), backend="cpu")
    sympl = np.concatenate([z, x])
    from_sympl = df.operator_weight(Hx, Hz, sympl, backend="cpu")
    assert from_tuple == from_str and from_sympl == from_str
