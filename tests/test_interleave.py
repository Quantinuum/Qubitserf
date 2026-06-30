"""The CSS min distance is computed by INTERLEAVING the Z- and X-subproblems weight level
by weight level (for cc, bz, and mitm), so both lower bounds advance together and a side
stalling on a hard level no longer starves the other. Once one side determines the min, the
other is capped. This must not change the answer: interleaved min(dZ, dX) == the true
min(dZ, dX), proven, for every method -- including asymmetric codes (dZ != dX), which
exercise the cap.
"""
import numpy as np
import pytest

import qubitserf as df
from qubitserf import codes


def _components(Hx, Hz):
    dz = df.css_distance(Hx, Hz, which="z", method="bz", backend="cpu").distance
    dx = df.css_distance(Hx, Hz, which="x", method="bz", backend="cpu").distance
    return dz, dx


SYMMETRIC = [
    ("steane", codes.steane()),
    ("shor", codes.shor()),
    ("surface3", codes.surface(3)),
    ("toric4", codes.toric(4)),
]

# Hypergraph products of two *different* repetition codes have dZ != dX -- the cap case.
ASYMMETRIC = [
    ("hgp_2x5", codes.hypergraph_product(codes.repetition_parity(2), codes.repetition_parity(5))),
    ("hgp_3x4", codes.hypergraph_product(codes.repetition_parity(3), codes.repetition_parity(4))),
]


@pytest.mark.parametrize("name,HxHz", SYMMETRIC + ASYMMETRIC)
@pytest.mark.parametrize("method", ["bz", "mitm", "cc"])
def test_interleaved_min_matches_components(name, HxHz, method):
    Hx, Hz = HxHz
    dz, dx = _components(Hx, Hz)
    want = min(dz, dx)
    r = df.css_distance(Hx, Hz, which="min", method=method, backend="cpu")
    assert r.distance == want, f"{name} {method}: got {r.distance}, want min({dz},{dx})={want}"
    assert r.proven, f"{name} {method}: min not proven"


@pytest.mark.parametrize("name,HxHz", ASYMMETRIC)
def test_asymmetric_has_distinct_components(name, HxHz):
    # Guard: these really are asymmetric, so the interleave cap is genuinely exercised.
    dz, dx = _components(*HxHz)
    assert dz != dx, f"{name} expected dZ != dX, got dZ={dz} dX={dx}"
