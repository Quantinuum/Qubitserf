"""Validation tests for qminweight.css_distance / qminweight.classical_distance.

Run from the repo root:

    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python -m pytest tests/ -q

What is checked
---------------
* CSS distances of Steane, Shor, toric(L) and surface(L) on the CPU backend
  against their known values.
* Classical distances of repetition_parity(n) and hamming_parity(3) on the CPU.
* The GPU backend agreeing with the CPU backend, deterministically, on every
  code (a regression guard for the stale-device-buffer bug that was fixed by
  keying the GPU buffer cache on a unique per-solve token rather than on host
  pointer identity).

Expected distances: toric(L) -> L, surface(L) -> L (planar [[L^2+(L-1)^2,1,L]]).
"""
from __future__ import annotations

import numpy as np
import pytest

import qminweight as df
from qminweight import codes

from conftest import HAS_GPU, requires_gpu


def _css_cases():
    cases = [
        ("steane", codes.steane(), 3),
        ("shor", codes.shor(), 3),
    ]
    for L in range(3, 7):           # toric L = 3..6  -> distance L
        cases.append((f"toric{L}", codes.toric(L), L))
    for L in range(3, 6):           # surface L = 3..5 -> distance L
        cases.append((f"surface{L}", codes.surface(L), L))
    return cases


CSS_CASES = _css_cases()


# --------------------------------------------------------------------------- #
# CPU: CSS distances must equal the known values.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,HxHz,expected", CSS_CASES, ids=[c[0] for c in CSS_CASES])
def test_css_distance_cpu(name, HxHz, expected):
    Hx, Hz = HxHz
    r = df.css_distance(Hx, Hz, method="bz", which="min", backend="cpu")
    assert r.distance == expected, f"{name}: cpu d={r.distance}, expected {expected}"
    assert r.proven, f"{name}: cpu result not marked proven"
    assert r.backend == "cpu"


# --------------------------------------------------------------------------- #
# CPU: min == min(Z, X).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,HxHz,expected", CSS_CASES, ids=[c[0] for c in CSS_CASES])
def test_css_distance_z_and_x_consistent(name, HxHz, expected):
    Hx, Hz = HxHz
    dz = df.css_distance(Hx, Hz, which="z", backend="cpu").distance
    dx = df.css_distance(Hx, Hz, which="x", backend="cpu").distance
    dmin = df.css_distance(Hx, Hz, which="min", backend="cpu").distance
    assert dmin == min(dz, dx), f"{name}: min({dz},{dx}) != reported {dmin}"
    assert dmin == expected


# --------------------------------------------------------------------------- #
# CPU determinism.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,HxHz,expected", CSS_CASES, ids=[c[0] for c in CSS_CASES])
def test_css_distance_cpu_deterministic(name, HxHz, expected):
    Hx, Hz = HxHz
    vals = {df.css_distance(Hx, Hz, backend="cpu").distance for _ in range(4)}
    assert vals == {expected}, f"{name}: cpu non-deterministic, saw {vals}"


# --------------------------------------------------------------------------- #
# The meet-in-the-middle algorithm must agree with Brouwer-Zimmermann.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,HxHz,expected", CSS_CASES, ids=[c[0] for c in CSS_CASES])
def test_mitm_matches_bz_cpu(name, HxHz, expected):
    Hx, Hz = HxHz
    bz = df.css_distance(Hx, Hz, method="bz", backend="cpu").distance
    mitm = df.css_distance(Hx, Hz, method="mitm", backend="cpu").distance
    assert mitm == bz == expected, f"{name}: mitm={mitm}, bz={bz}, expected {expected}"


@pytest.mark.parametrize("name,HxHz,expected", CSS_CASES, ids=[c[0] for c in CSS_CASES])
def test_cc_matches_bz(name, HxHz, expected):
    Hx, Hz = HxHz
    bz = df.css_distance(Hx, Hz, method="bz", backend="cpu").distance
    cc = df.css_distance(Hx, Hz, method="cc").distance
    assert cc == bz == expected, f"{name}: cc={cc}, bz={bz}, expected {expected}"


def test_cc_certifies_bivariate_bicycle():
    """Connected cluster certifies sparse BB codes where BZ's bound is too weak."""
    Hx, Hz = codes.gross_code()
    r = df.css_distance(Hx, Hz, method="cc")
    assert r.distance == 12 and r.proven, f"gross code: cc d={r.distance}, expected 12"
    # a smaller BB code [[72,12,6]]
    Hx, Hz = codes.bivariate_bicycle(6, 6, [("x", 3), ("y", 1), ("y", 2)],
                                     [("y", 3), ("x", 1), ("x", 2)])
    r = df.css_distance(Hx, Hz, method="cc")
    assert r.distance == 6 and r.proven, f"bb[[72,12,6]]: cc d={r.distance}, expected 6"


# --------------------------------------------------------------------------- #
# Classical distances on the CPU.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", range(3, 9))
def test_classical_repetition_cpu(n):
    r = df.classical_distance(codes.repetition_parity(n), method="bz", backend="cpu")
    assert r.distance == n, f"repetition({n}): d={r.distance}, expected {n}"
    assert r.proven


def test_classical_hamming_cpu():
    r = df.classical_distance(codes.hamming_parity(3), method="bz", backend="cpu")
    assert r.distance == 3, f"hamming(3): d={r.distance}, expected 3"
    assert r.proven


# --------------------------------------------------------------------------- #
# GPU vs CPU — strict equality, deterministically, on every code.
# --------------------------------------------------------------------------- #
@requires_gpu
@pytest.mark.parametrize("name,HxHz,expected", CSS_CASES, ids=[c[0] for c in CSS_CASES])
def test_gpu_matches_cpu(name, HxHz, expected):
    Hx, Hz = HxHz
    cpu = df.css_distance(Hx, Hz, backend="cpu").distance
    gpu_vals = {df.css_distance(Hx, Hz, backend="gpu").distance for _ in range(6)}
    assert gpu_vals == {cpu} == {expected}, (
        f"{name}: cpu={cpu}, gpu saw {gpu_vals}, expected {expected}"
    )


@requires_gpu
def test_gpu_deterministic_under_zx_alternation():
    """Regression guard for the stale-device-buffer bug: alternating Z- and X-distance
    solves (which freed and reallocated the host buffers) must never return a wrong
    (too-small) distance."""
    for L in (5, 6):
        Hx, Hz = codes.toric(L)
        for _ in range(12):
            z = df.css_distance(Hx, Hz, which="z", backend="gpu").distance
            x = df.css_distance(Hx, Hz, which="x", backend="gpu").distance
            assert z == L and x == L, f"toric{L}: gpu z={z} x={x}, expected {L}"


@requires_gpu
def test_gpu_random_sweep_agrees_with_cpu():
    failures = []
    for fam in ("toric", "surface"):
        for L in (3, 4, 5):
            Hx, Hz = getattr(codes, fam)(L)
            cpu = df.css_distance(Hx, Hz, backend="cpu").distance
            gpu = {df.css_distance(Hx, Hz, backend="gpu").distance for _ in range(4)}
            if gpu != {cpu}:
                failures.append((f"{fam}{L}", cpu, gpu))
    assert not failures, f"gpu/cpu disagreements: {failures}"


# --------------------------------------------------------------------------- #
# available_backends sanity.
# --------------------------------------------------------------------------- #
def test_available_backends_contains_cpu():
    b = df.available_backends()
    assert "cpu" in b
    assert isinstance(b, list)


def test_skip_message_when_no_gpu():
    if not HAS_GPU:
        pytest.skip("gpu backend not available on this machine")
    assert "gpu" in df.available_backends()


@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_public_backend_selection_is_gpu_not_architecture_specific(backend):
    Hx, Hz = codes.steane()
    with pytest.raises(ValueError, match="auto/cpu/gpu"):
        df.css_distance(Hx, Hz, backend=backend)
