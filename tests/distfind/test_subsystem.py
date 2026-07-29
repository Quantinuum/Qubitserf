"""Cross-check the native subsystem (dressed) CSS distance against the oracle.

A CSS subsystem code is given by GAUGE generators ``Gx`` (X-type) and ``Gz``
(Z-type). The stabilizer group is the center of the gauge group; the *dressed*
distance is the minimum weight of an operator that commutes with every
stabilizer but lies outside the gauge group (SPEC §1). The kernel constraint
uses the center, the triviality test uses the gauge group -- this asymmetry
distinguishes the dressed distance from the (larger) bare distance.

Ground truth: ``_reference.dressed_distance_bruteforce``.

Run from the repo root:
    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest

import qubitserf.distfind as df
from qubitserf.distfind import codes

import _reference as ref


# --------------------------------------------------------------------------- #
# codes.bacon_shor must be identical to the oracle's independent copy.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_codes_bacon_shor_matches_reference(d):
    Gx, Gz = codes.bacon_shor(d)
    rx, rz = ref.bacon_shor(d)
    assert np.array_equal(Gx, rx)
    assert np.array_equal(Gz, rz)
    assert Gx.shape[1] == d * d


# --------------------------------------------------------------------------- #
# Bacon-Shor dressed distance == d.
# --------------------------------------------------------------------------- #
def test_bacon_shor_3():
    Gx, Gz = codes.bacon_shor(3)
    assert df.subsystem_css_distance(Gx, Gz, method="bz", backend="cpu") == 3
    assert ref.dressed_distance_bruteforce(Gx, Gz) == 3
    # Z and X components are each 3.
    assert df.subsystem_css_distance(Gx, Gz, which="z", backend="cpu") == 3
    assert df.subsystem_css_distance(Gx, Gz, which="x", backend="cpu") == 3


def test_bacon_shor_5():
    Gx, Gz = codes.bacon_shor(5)
    assert df.subsystem_css_distance(Gx, Gz, method="bz", backend="cpu") == 5


# --------------------------------------------------------------------------- #
# A stabilizer code passed as a subsystem code (gauge = stabilizers) must give
# the ordinary CSS distance -- the special case Gx=Hx, Gz=Hz, Sx=Hx, Sz=Hz.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,HxHz,expected", [
    ("steane", codes.steane(), 3),
    ("shor", codes.shor(), 3),
    ("surface3", ref.surface_3(), 3),
])
def test_stabilizer_code_as_subsystem_equals_css(name, HxHz, expected):
    Hx, Hz = HxHz
    sub = df.subsystem_css_distance(Hx, Hz, method="bz", backend="cpu")
    css = df.css_distance(Hx, Hz, method="bz", backend="cpu")
    assert sub == css == expected, (
        f"{name}: subsystem={sub} css={css} expected {expected}")
    assert ref.dressed_distance_bruteforce(Hx, Hz) == expected


# --------------------------------------------------------------------------- #
# Random small CSS subsystem codes vs the brute-force dressed distance.
# Only assert on codes with a genuine dressed operator of weight < n in BOTH
# components (otherwise the weight-n boundary is ambiguous / degenerate).
# --------------------------------------------------------------------------- #
def test_random_subsystem_codes():
    rng = np.random.default_rng(0x5A8E3)
    tested = 0
    for _ in range(120):
        n = int(rng.integers(4, 9))
        a = int(rng.integers(1, 5))
        b = int(rng.integers(1, 5))
        Gx = rng.integers(0, 2, size=(a, n)).astype(np.uint8)
        Gz = rng.integers(0, 2, size=(b, n)).astype(np.uint8)
        dZ = ref.dressed_distance_bruteforce(Gx, Gz, which="Z")
        dX = ref.dressed_distance_bruteforce(Gx, Gz, which="X")
        if dZ >= n or dX >= n:
            continue  # no unambiguous interior dressed operator
        rz = df.subsystem_css_distance(Gx, Gz, which="z", method="bz", backend="cpu")
        rx = df.subsystem_css_distance(Gx, Gz, which="x", method="bz", backend="cpu")
        rm = df.subsystem_css_distance(Gx, Gz, which="min", method="bz", backend="cpu")
        assert rz == dZ, f"dZ native={rz} oracle={dZ}\nGx={Gx}\nGz={Gz}"
        assert rx == dX, f"dX native={rx} oracle={dX}\nGx={Gx}\nGz={Gz}"
        assert rm == min(dZ, dX)
        tested += 1
    assert tested >= 20, f"only {tested} random subsystem cases exercised"


# --------------------------------------------------------------------------- #
# bz, cc and mitm must agree on subsystem distance.
# --------------------------------------------------------------------------- #
def test_methods_agree():
    rng = np.random.default_rng(0xC0FFEE)
    cases = [codes.bacon_shor(3), codes.steane()]
    found = 0
    for _ in range(60):
        if found >= 6:
            break
        n = int(rng.integers(4, 8))
        a = int(rng.integers(1, 4))
        b = int(rng.integers(1, 4))
        Gx = rng.integers(0, 2, size=(a, n)).astype(np.uint8)
        Gz = rng.integers(0, 2, size=(b, n)).astype(np.uint8)
        dZ = ref.dressed_distance_bruteforce(Gx, Gz, which="Z")
        dX = ref.dressed_distance_bruteforce(Gx, Gz, which="X")
        if dZ >= n or dX >= n:
            continue
        cases.append((Gx, Gz))
        found += 1
    for Gx, Gz in cases:
        bz = df.subsystem_css_distance(Gx, Gz, method="bz", backend="cpu")
        cc = df.subsystem_css_distance(Gx, Gz, method="cc", backend="cpu")
        mitm = df.subsystem_css_distance(Gx, Gz, method="mitm", backend="cpu")
        assert bz == cc == mitm, f"bz={bz} cc={cc} mitm={mitm}"
