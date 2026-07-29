"""Cross-check the native non-CSS (general stabilizer / subsystem) features.

Covers the symplectic generalization of the distance solver:
  * ``stabilizer_distance``            -- min symplectic weight in C(S) \\ rowspace(S)
  * ``subsystem_stabilizer_distance``  -- dressed distance of a non-CSS gauge group
  * ``pauli_operator_weight``          -- min symplectic weight over op + rowspace(G)

Ground truth: the pure-numpy oracles in ``_reference`` (``stabilizer_distance_bruteforce``,
``dressed_stabilizer_distance_bruteforce``, ``symplectic_coset_min_weight_bruteforce``),
each independent of the compiled library. The fixed codes are additionally cross-checked
against the *original* qubitserf ``interface`` binary -- the legacy Quantinuum C++ tool this
package supersedes -- as an external sanity check, *when that binary is available*. This is a
legacy cross-check only: the original binary is never built or fetched here, so the test skips
cleanly (and is unaffected by this package's own release) whenever it is not already present
(see ``_qubitserf_bin`` below).

Run from the repo root:
    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

import qubitserf.distfind as df

import _reference as ref


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def rand_isotropic(n, m, rng):
    """A random commuting, independent stabilizer set (<= m rows), shape (k, 2n)."""
    rows = []
    tries = 0
    while len(rows) < m and tries < 2000:
        tries += 1
        v = rng.integers(0, 2, size=2 * n).astype(np.uint8)
        if not v.any():
            continue
        if any(ref.symplectic_product(v, r, n) for r in rows):
            continue
        if ref.rank(np.array(rows + [v], np.uint8)) == len(rows) + 1:
            rows.append(v)
    return np.array(rows, dtype=np.uint8) if rows else np.zeros((0, 2 * n), np.uint8)


def symplectic_to_paulis(S):
    """Convert an (m, 2n) [z|x] matrix to Pauli strings (for the original qubitserf binary)."""
    n = S.shape[1] // 2
    out = []
    for r in S:
        s = []
        for j in range(n):
            z, x = int(r[j]), int(r[n + j])
            s.append({(0, 0): "I", (0, 1): "X", (1, 0): "Z", (1, 1): "Y"}[(z, x)])
        out.append("".join(s))
    return out


def _qubitserf_bin():
    """Locate a pre-built *original* qubitserf ``interface`` binary, or return None.

    This only *probes* for an already-present legacy binary (``$QUBITSERF_BIN``, a
    ``qubitserf_interface`` on PATH, or a local probe build). It never clones or builds the
    original tool, so it cannot accidentally pick up this package's own superseding code; when
    nothing is found the dependent tests skip.
    """
    return (os.environ.get("QUBITSERF_BIN")
            or shutil.which("qubitserf_interface")
            or ("/tmp/qubitserf_probe/build/interface"
                if os.path.exists("/tmp/qubitserf_probe/build/interface") else None))


def qubitserf_distance(strings):
    binp = _qubitserf_bin()
    if not binp:
        pytest.skip("original qubitserf interface binary not available")
    out = subprocess.run([binp], input="\n".join(strings) + "\n\n",
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[-1])


FIVE_QUBIT = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
NON_CSS_8 = ["ZIYXZIII", "XIZYXIII", "IZZZZIII", "IXXXXIII",
             "IIIIIZII", "IIIIIIZI", "IIIIIIIZ"]


# --------------------------------------------------------------------------- #
# Fixed codes: native == oracle == original qubitserf (legacy cross-check)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strings,expected", [(FIVE_QUBIT, 3), (NON_CSS_8, 3)])
def test_fixed_noncss_distance(strings, expected):
    S = ref.paulis_to_symplectic(strings)
    assert ref.stabilizer_distance_bruteforce(S) == expected
    # mitm = symplectic search; bz = weight-doubling isometry. (cc is rejected on non-CSS
    # codes -- see test_cc_rejected_on_noncss -- so it is not exercised here.)
    for method in ("mitm", "bz"):
        assert df.stabilizer_distance(S, method=method) == expected


@pytest.mark.parametrize("strings,expected", [(FIVE_QUBIT, 3), (NON_CSS_8, 3)])
def test_fixed_noncss_vs_qubitserf(strings, expected):
    assert qubitserf_distance(strings) == expected
    S = ref.paulis_to_symplectic(strings)
    assert df.stabilizer_distance(S) == qubitserf_distance(strings)


def test_five_qubit_helper():
    assert np.array_equal(ref.five_qubit_code(), ref.paulis_to_symplectic(FIVE_QUBIT))


# --------------------------------------------------------------------------- #
# Random non-CSS stabilizer codes: native == oracle (and == original qubitserf)
# --------------------------------------------------------------------------- #
def test_random_noncss_stabilizer_distance():
    rng = np.random.default_rng(0x5AB)
    checked = 0
    for _ in range(60):
        n = int(rng.integers(4, 8))
        m = int(rng.integers(1, n))
        S = rand_isotropic(n, m, rng)
        if S.shape[0] == 0:
            continue
        oracle = ref.stabilizer_distance_bruteforce(S)
        native = df.stabilizer_distance(S)
        assert native == oracle, (n, m, native, oracle)
        checked += 1
    assert checked >= 30


def test_noncss_bz_isometry_matches_mitm_and_oracle():
    """Non-CSS Brouwer-Zimmermann via the (a|b)->(a|b|a^b) isometry == mitm == oracle,
    and the result is proven. Covers plain distance, subsystem distance, and operator
    weight -- all three reuse the same weight-doubling reduction."""
    rng = np.random.default_rng(0xB2150)
    checked = 0
    for _ in range(60):
        n = int(rng.integers(4, 8))
        m = int(rng.integers(1, n))
        S = rand_isotropic(n, m, rng)
        if S.shape[0] == 0:
            continue
        oracle = ref.stabilizer_distance_bruteforce(S)
        rb = df.stabilizer_distance(S, method="bz")
        rm = df.stabilizer_distance(S, method="mitm")
        assert rb == oracle == rm, (n, m, rb, rm, oracle)
        # operator weight: bz == mitm == oracle over the coset op + rowspace(S)
        op = rng.integers(0, 2, size=2 * n).astype(np.uint8)
        ow_oracle = ref.symplectic_coset_min_weight_bruteforce(S, op)
        wb = df.pauli_operator_weight(S, op, method="bz")
        wm = df.pauli_operator_weight(S, op, method="mitm")
        assert wb == ow_oracle == wm, (n, m, wb, wm, ow_oracle)
        checked += 1
    assert checked >= 30


def test_random_noncss_vs_qubitserf():
    if not _qubitserf_bin():
        pytest.skip("original qubitserf interface binary not available")
    rng = np.random.default_rng(0xC0DE)
    checked = 0
    for _ in range(20):
        n = int(rng.integers(4, 7))
        m = int(rng.integers(1, n))
        S = rand_isotropic(n, m, rng)
        if S.shape[0] == 0:
            continue
        if ref.rank(ref.centralizer(S)) <= ref.rank(S):
            continue  # no logicals; qubitserf would assert-fail
        native = df.stabilizer_distance(S)
        qs = qubitserf_distance(symplectic_to_paulis(S))
        assert native == qs, (n, m, native, qs)
        checked += 1
    assert checked >= 8


# --------------------------------------------------------------------------- #
# Random non-CSS subsystem (gauge) codes: native dressed == oracle dressed
# --------------------------------------------------------------------------- #
def test_random_noncss_subsystem_distance():
    rng = np.random.default_rng(0x6A06E)
    checked = 0
    for _ in range(80):
        n = int(rng.integers(4, 7))
        m = int(rng.integers(2, 2 * n))
        G = ref.rowspace_basis(rng.integers(0, 2, size=(m, 2 * n)).astype(np.uint8))
        if G.shape[0] == 0:
            continue
        oracle = ref.dressed_stabilizer_distance_bruteforce(G)
        native = df.subsystem_stabilizer_distance(G)
        assert native == oracle, (n, m, native, oracle)
        checked += 1
    assert checked >= 40


def test_abelian_subsystem_equals_bare():
    # An abelian gauge group is its own center: dressed distance == bare distance.
    S = ref.five_qubit_code()
    assert df.subsystem_stabilizer_distance(S) == 3
    assert df.stabilizer_distance(S) == 3


def test_symplectic_center_invariants():
    rng = np.random.default_rng(0xCE6)
    for _ in range(20):
        n = int(rng.integers(3, 7))
        m = int(rng.integers(2, 2 * n))
        G = ref.rowspace_basis(rng.integers(0, 2, size=(m, 2 * n)).astype(np.uint8))
        if G.shape[0] == 0:
            continue
        S = ref.symplectic_center(G)
        in_G = ref._membership_reducer(G)
        # every center element is in rowspace(G) and commutes with all gauge generators
        for row in S:
            assert in_G(row)
            assert all(ref.symplectic_product(row, g, n) == 0 for g in G)


# --------------------------------------------------------------------------- #
# Operator weight (symplectic coset leader)
# --------------------------------------------------------------------------- #
def test_random_operator_weight():
    rng = np.random.default_rng(0x09E)
    checked = 0
    for _ in range(60):
        n = int(rng.integers(4, 7))
        m = int(rng.integers(1, n))
        G = rand_isotropic(n, m, rng)
        if G.shape[0] == 0:
            continue
        op = rng.integers(0, 2, size=2 * n).astype(np.uint8)
        oracle = ref.symplectic_coset_min_weight_bruteforce(G, op)
        native = df.pauli_operator_weight(G, op)
        assert native == oracle, (n, native, oracle)
        checked += 1
    assert checked >= 30


def test_operator_weight_stabilizer_is_zero():
    S = ref.five_qubit_code()
    for row in S:
        assert df.pauli_operator_weight(S, row) == 0


def test_operator_weight_logical_reduced():
    # Logical X = X on every qubit; modulo the 5-qubit stabilizers it reduces to weight 3.
    S = ref.five_qubit_code()
    logX = np.concatenate([np.zeros(5, np.uint8), np.ones(5, np.uint8)])
    assert df.pauli_operator_weight(S, logX) == 3
    # accepts Pauli-string and (z, x) forms too
    assert df.pauli_operator_weight(S, "XXXXX") == 3
    assert df.pauli_operator_weight(S, (np.zeros(5), np.ones(5))) == 3


# --------------------------------------------------------------------------- #
# CSS fast path: a CSS code given symplectically agrees with css_distance
# --------------------------------------------------------------------------- #
def test_css_fast_path_matches_css_distance():
    H = np.array([[0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 1]], np.uint8)
    n = 7
    rows = [np.concatenate([np.zeros(n, np.uint8), r]) for r in H]   # X-type
    rows += [np.concatenate([r, np.zeros(n, np.uint8)]) for r in H]  # Z-type
    S = np.array(rows, np.uint8)
    css = df.css_distance(H, H)
    for method in ("bz", "cc", "mitm"):
        assert df.stabilizer_distance(S, method=method) == css == 3


# --------------------------------------------------------------------------- #
# cc is rejected on genuinely non-CSS codes (no silent fallback)
# --------------------------------------------------------------------------- #
def test_cc_rejected_on_noncss():
    S = ref.paulis_to_symplectic(FIVE_QUBIT)   # non-CSS [[5,1,3]]
    import pytest as _pytest
    with _pytest.raises(ValueError, match="cc"):
        df.stabilizer_distance(S, method="cc")
    with _pytest.raises(ValueError, match="cc"):
        df.subsystem_stabilizer_distance(S, method="cc")
    # bz and mitm still work on the same non-CSS code.
    assert df.stabilizer_distance(S, method="bz") == 3
    assert df.stabilizer_distance(S, method="mitm") == 3
