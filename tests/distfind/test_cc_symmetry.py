"""Exhaustive small-code oracles for row-space translations and CSS duality."""
import re

import numpy as np
import pytest

from qubitserf import distfind as df
from qubitserf.codeaut.gf2 import rank_gf2, row_basis_gf2
from qubitserf.distfind import _native
from test_cc_pruning import brute_distance


def torus_checks(height, width):
    size = height * width
    identity = np.eye(size, dtype=np.uint8)
    shift_x = np.zeros_like(identity)
    shift_y = np.zeros_like(identity)
    for row in range(height):
        for column in range(width):
            source = row * width + column
            shift_x[source, ((row + 1) % height) * width + column] = 1
            shift_y[source, row * width + (column + 1) % width] = 1
    first = identity ^ shift_x
    second = identity ^ shift_y
    return np.hstack([first, second]), np.hstack([second.T, first.T])


def assert_oracle(hx, hz, threads):
    assert not np.any((hx @ hz.T) & 1)
    z_distance = brute_distance(hx, hz)
    x_distance = brute_distance(hz, hx)
    assert z_distance > 0 and x_distance > 0
    expected = {"z": z_distance, "x": x_distance,
                "min": min(z_distance, x_distance)}
    for side, distance in expected.items():
        assert df.css_distance(hx, hz, method="cc", which=side, backend="cpu",
                               threads=threads) == distance
        for threshold in {1, distance, distance + 1, min(hx.shape[1], distance + 3)}:
            assert df.css_distance_at_least(
                hx, hz, lower_bound=threshold, which=side, threads=threads,
            ) == (distance >= threshold)
    result = _native.css_distance_raw(hx, hz, "cc", "M", "cpu", threads, False)
    assert result.proven
    assert result.distance == expected["min"]
    assert result.lower_bound == expected["min"]
    return expected


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (2, 4)])
@pytest.mark.parametrize("representation", ["full", "omit_dependent", "row_basis", "reordered"])
@pytest.mark.parametrize("threads", [1, 2])
def test_two_dimensional_translations_against_exhaustive_oracle(shape, representation, threads):
    hx, hz = torus_checks(*shape)
    if representation == "omit_dependent":
        assert rank_gf2(hx[:-1]) == rank_gf2(hx)
        assert rank_gf2(hz[:-1]) == rank_gf2(hz)
        hx, hz = hx[:-1], hz[:-1]
    elif representation == "row_basis":
        hx, hz = row_basis_gf2(hx), row_basis_gf2(hz)
    elif representation == "reordered":
        hx = np.vstack([hx[::-1], hx[:1], np.zeros((1, hx.shape[1]), dtype=np.uint8)])
        hz = np.vstack([np.roll(hz, 1, axis=0), hz[-2:]])
    expected = assert_oracle(hx, hz, threads)
    assert expected["z"] == expected["x"] == min(shape)


@pytest.mark.parametrize("threads", [1, 2])
def test_omitted_generators_still_reduce_translation_roots(threads, capfd):
    hx, hz = torus_checks(2, 3)
    hx, hz = hx[:-1], hz[:-1]
    for matrix in (hx, hz):
        permutation = np.arange(12).reshape(4, 3)
        permutation = np.roll(permutation, 1, axis=1).reshape(-1)
        assert {row.tobytes() for row in matrix} != {row.tobytes() for row in matrix[:, permutation]}
        assert rank_gf2(np.vstack([matrix, matrix[:, permutation]])) == rank_gf2(matrix)
    expected = assert_oracle(hx, hz, threads)
    capfd.readouterr()
    assert df.css_distance(hx, hz, method="cc", which="z", backend="cpu",
                           threads=threads, verbose=True) == expected["z"]
    diagnostic = capfd.readouterr().err
    seed_count = re.search(r"Z-CC: (\d+) seeds", diagnostic)
    assert seed_count is not None
    assert int(seed_count.group(1)) < hx.shape[1]


@pytest.mark.parametrize("threads", [1, 2])
def test_qubit_permutation_preserves_exact_and_bound_results(threads, capfd):
    hx, hz = torus_checks(2, 3)
    permutation = np.array([9, 2, 7, 0, 11, 4, 8, 1, 6, 10, 3, 5])
    hx, hz = hx[::-1, permutation], np.roll(hz, 1, axis=0)[:, permutation]
    expected = assert_oracle(hx, hz, threads)
    capfd.readouterr()
    assert df.css_distance(hx, hz, method="cc", which="min", backend="cpu",
                           threads=threads, verbose=True) == expected["min"]
    assert "verified equivalent X/Z sectors" not in capfd.readouterr().err


@pytest.mark.parametrize("threads", [1, 2])
def test_equivalent_sectors_emit_progress_for_both_sides(threads, capfd):
    hx, hz = torus_checks(2, 3)
    hx, hz = hx[:-1], hz[:-1]
    expected = assert_oracle(hx, hz, threads)
    capfd.readouterr()
    assert df.css_distance(hx, hz, method="cc", which="min", backend="cpu",
                           threads=threads, verbose=True) == expected["min"]
    diagnostic = capfd.readouterr().err
    assert "verified equivalent X/Z sectors" in diagnostic
    assert f"Z-distance: ={expected['z']}" in diagnostic
    assert f"X-distance: ={expected['x']}" in diagnostic
    assert df.css_distance_at_least(hx, hz, lower_bound=expected["min"],
                                    threads=threads, verbose=True)
    diagnostic = capfd.readouterr().err
    assert f"Z-distance bound: >{expected['z'] - 1}" in diagnostic
    assert f"X-distance bound: >{expected['x'] - 1}" in diagnostic


@pytest.mark.parametrize("threads", [1, 2])
def test_equal_rank_asymmetric_sectors_cannot_be_identified(threads, capfd):
    hx = np.ones((1, 8), dtype=np.uint8)
    hz = np.zeros_like(hx)
    hz[0, :2] = 1
    assert rank_gf2(hx) == rank_gf2(hz)
    expected = assert_oracle(hx, hz, threads)
    assert expected == {"z": 2, "x": 1, "min": 1}
    capfd.readouterr()
    assert df.css_distance(hx, hz, method="cc", which="min", backend="cpu",
                           threads=threads, verbose=True) == 1
    assert "verified equivalent X/Z sectors" not in capfd.readouterr().err
    assert not df.css_distance_at_least(hx, hz, lower_bound=2, threads=threads)


@pytest.mark.parametrize("threads", [1, 2])
def test_check_translation_must_also_preserve_trivial_space(threads):
    hx = np.ones((1, 8), dtype=np.uint8)
    hz = np.zeros((3, 8), dtype=np.uint8)
    hz[0, [0, 1]] = 1
    hz[1, [0, 2]] = 1
    hz[2, [0, 3]] = 1
    expected = assert_oracle(hx, hz, threads)
    assert expected == {"z": 2, "x": 1, "min": 1}


@pytest.mark.parametrize("threads", [1, 2])
def test_identity_duality_with_different_sparse_generator_bases(threads, capfd):
    hx = np.array([[1, 1, 1, 1, 0, 0, 0],
                   [1, 1, 0, 0, 1, 1, 0],
                   [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    hz = hx.copy()
    hz[0] ^= hz[1]
    hz[1] ^= hz[2]
    assert int((hx[0] ^ hx[1]).sum()) == max(int(hx[0].sum()), int(hx[1].sum()))
    expected = assert_oracle(hx, hz, threads)
    assert expected == {"z": 3, "x": 3, "min": 3}
    capfd.readouterr()
    assert df.css_distance(hx, hz, method="cc", threads=threads, verbose=True) == 3
    assert "verified equivalent X/Z sectors" in capfd.readouterr().err


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("redundant", [False, True])
def test_shorter_xor_check_augmentation_preserves_quotient(threads, redundant):
    hx = np.array([[1, 1, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)
    hz = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    assert int((hx[0] ^ hx[1]).sum()) < min(int(hx[0].sum()), int(hx[1].sum()))
    if redundant:
        hx = np.vstack([hx, hx[:1], np.zeros_like(hx[:1])])
        hz = np.vstack([hz, hz, np.zeros_like(hz)])
    assert assert_oracle(hx, hz, threads) == {"z": 3, "x": 1, "min": 1}
    assert assert_oracle(hz, hx, threads) == {"z": 1, "x": 3, "min": 1}


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("syndrome_rows", [65, 129, 193, 257, 513])
@pytest.mark.parametrize("empty_trivial", [False, True])
def test_duplicate_syndrome_rows_cover_fixed_and_dynamic_words(threads, syndrome_rows, empty_trivial):
    hx = np.ones((syndrome_rows, 8), dtype=np.uint8)
    hx[::7] = 0
    hz = np.zeros((0 if empty_trivial else syndrome_rows, 8), dtype=np.uint8)
    if not empty_trivial:
        hz[:, :2] = 1
        hz[::11] = 0
    assert assert_oracle(hx, hz, threads) == {"z": 2, "x": 1, "min": 1}


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("with_trivial_component", [False, True])
def test_disconnected_first_check_is_not_lost_during_ordering(threads, with_trivial_component):
    hx = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.uint8)
    hz = np.array([[1, 1, 0, 0]], dtype=np.uint8) if with_trivial_component else np.zeros((0, 4), dtype=np.uint8)
    assert assert_oracle(hx, hz, threads) == {"z": 2, "x": 1, "min": 1}
    assert assert_oracle(hx[::-1], hz, threads) == {"z": 2, "x": 1, "min": 1}
