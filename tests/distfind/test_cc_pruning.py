"""Independent small-code oracles for exact CC pruning and symmetry."""
import numpy as np
import pytest

from qubitserf import distfind as df
from qubitserf.codeaut.gf2 import nullspace_basis_gf2
from qubitserf.distfind import codes


def packed_rows(matrix):
    return [sum(int(bit) << q for q, bit in enumerate(row)) for row in matrix]


def brute_distance(checks, trivial):
    span = {0}
    for row in packed_rows(trivial):
        span |= {word ^ row for word in span}
    rows = packed_rows(checks)
    weights = [bin(word).count("1") for word in range(1, 1 << checks.shape[1])
               if word not in span and all(bin(word & row).count("1") % 2 == 0
                                           for row in rows)]
    return min(weights, default=-1)


def assert_matches_oracle(hx, hz, threads):
    z, x = brute_distance(hx, hz), brute_distance(hz, hx)
    expected = min((d for d in (z, x) if d >= 0), default=-1)
    for which, value in (("z", z), ("x", x), ("min", expected)):
        assert df.css_distance(hx, hz, method="cc", which=which, backend="cpu",
                               threads=threads) == value
        if value < 0:
            with pytest.raises(ValueError, match="undefined"):
                df.css_distance_at_least(hx, hz, lower_bound=1, which=which, threads=2)
        else:
            for bound in {1, value, value + 1, min(hx.shape[1], value + 3)}:
                assert df.css_distance_at_least(hx, hz, lower_bound=bound,
                                               which=which, threads=2) == (value >= bound)


@pytest.mark.parametrize("seed", range(128))
def test_random_commuting_codes(seed):
    rng = np.random.default_rng(seed)
    n = seed % 10 + 1
    hx = rng.integers(0, 2, size=(seed % 5, n), dtype=np.uint8)
    kernel = nullspace_basis_gf2(hx)
    coefficients = rng.integers(0, 2, size=((seed // 3) % 5, len(kernel)), dtype=np.uint8)
    hz = (coefficients @ kernel) & 1
    assert_matches_oracle(hx, hz, 1 if seed % 2 else 4)


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_verified_cyclic_reduction_and_permuted_fallback(m):
    rng = np.random.default_rng(m)
    identity = np.eye(m, dtype=np.uint8)
    a = identity ^ np.roll(identity, 1, axis=1) ^ np.roll(identity, 2, axis=1)
    b = identity ^ np.roll(identity, 1, axis=1) ^ np.roll(identity, 3, axis=1)
    hx, hz = np.hstack([a, b]), np.hstack([b.T, a.T])
    assert_matches_oracle(hx, hz, 4)
    perm = rng.permutation(2 * m)
    assert_matches_oracle(hx[:, perm], hz[:, perm], 1)
    # Duplicate/zero generators preserve both the code and the symmetry.
    hx = np.vstack([hx, hx[:1], np.zeros((1, 2 * m), np.uint8)])
    hz = np.vstack([hz, hz[:2]])
    assert_matches_oracle(hx, hz, 4)


def test_check_symmetry_does_not_imply_quotient_symmetry(capfd):
    # H alone is invariant under the candidate two-block shift, but its
    # stabilizer quotient is not. Reducing seeds using H alone is unsound.
    hx = np.ones((1, 8), np.uint8)
    hz = np.zeros((2, 8), np.uint8)
    hz[0, [0, 1]] = 1
    hz[1, [0, 2]] = 1
    assert_matches_oracle(hx, hz, 4)
    df.css_distance(hx, hz, method="cc", which="z", backend="cpu",
                    threads=1, verbose=True)
    assert "8 seeds" in capfd.readouterr().err


@pytest.mark.parametrize("threads", [1, 4])
def test_sparse_deep_search_and_odd_distances(threads):
    for family, size in (("toric", 7), ("surface", 5)):
        hx, hz = getattr(codes, family)(size)
        assert df.css_distance(hx, hz, method="cc", backend="cpu",
                               threads=threads) == size


def test_multiword_syndrome_and_logical_detector():
    hx = np.zeros((130, 132), dtype=np.uint8)
    hx[np.arange(130), np.arange(130)] = 1
    hz = np.zeros((0, 132), dtype=np.uint8)
    assert df.css_distance(hx, hz, method="cc", which="z", backend="cpu",
                           threads=4) == 1
    # Empty syndrome columns, but >64 independent logical detectors.
    hx = np.zeros((0, 132), dtype=np.uint8)
    assert df.css_distance(hx, hz, method="cc", backend="cpu", threads=4) == 1


def test_duplicate_column_pair_completion():
    hx = np.ones((70, 10), dtype=np.uint8)
    hz = np.zeros((0, 10), dtype=np.uint8)
    assert_matches_oracle(hx, hz, 4)


def test_large_input_without_pair_table():
    hx = np.zeros((0, 2050), dtype=np.uint8)
    hz = np.zeros_like(hx)
    assert df.css_distance(hx, hz, method="cc", which="z", backend="cpu",
                           threads=1) == 1


@pytest.mark.parametrize("n", [64, 65, 130])
def test_local_check_mask_width_and_fallback(n):
    hx = np.ones((1, n), dtype=np.uint8)
    hz = np.zeros((0, n), dtype=np.uint8)
    assert df.css_distance(hx, hz, method="cc", which="z", backend="cpu",
                           threads=2) == 2


@pytest.mark.parametrize("size", [8, 10])
def test_degenerate_larger_distance(size):
    hx, hz = codes.toric(size)
    assert df.css_distance(hx, hz, method="cc", backend="cpu", threads=2) == size
    assert df.css_distance_at_least(hx, hz, lower_bound=size, threads=2)
    assert not df.css_distance_at_least(hx, hz, lower_bound=size + 1, threads=2)


def test_direct_bound_catches_shorter_odd_logical_in_pair_tail():
    hx, hz = codes.steane()
    assert not df.css_distance_at_least(hx, hz, lower_bound=5, which="z", threads=2)


def test_direct_bound_validates_input():
    hx, hz = codes.steane()
    for lower in (0, -1):
        with pytest.raises(ValueError):
            df.css_distance_at_least(hx, hz, lower_bound=lower, threads=2)
    with pytest.raises(TypeError):
        df.css_distance_at_least(hx, hz, lower_bound=2.5, threads=2)
    with pytest.raises(ValueError):
        df.css_distance_at_least(hx, hz, lower_bound=3, threads=-1)
    with pytest.raises(ValueError):
        df.css_distance_at_least(hx, hz, lower_bound=3, threads=2**40)
    with pytest.raises(ValueError, match="undefined"):
        df.css_distance_at_least(np.zeros((2, 0), np.uint8), np.zeros((1, 0), np.uint8),
                                lower_bound=1, threads=2)
    assert not df.css_distance_at_least((hx, hz), lower_bound=10**100, threads=2)
    assert df.css_distance_at_least((hx, hz), lower_bound=np.int64(3), threads=2)
    with pytest.raises(ValueError):
        df.css_distance_at_least(np.eye(2, dtype=np.uint8), np.eye(2, dtype=np.uint8),
                                lower_bound=2, threads=2)


@pytest.mark.parametrize("m", [3, 5, 7])
def test_cyclic_gap_preserves_complete_periodic_left_logical(m):
    hx = np.zeros((m, 2 * m), dtype=np.uint8)
    hz = np.zeros((m, 2 * m), dtype=np.uint8)
    for j in range(m):
        hx[j, [j, (j + 1) % m]] = 1
        hz[j, m + j] = 1
    assert df.css_distance(hx, hz, method="cc", which="z", backend="cpu", threads=2) == m
    assert df.css_distance_at_least(hx, hz, lower_bound=m, which="z", threads=2)
    assert not df.css_distance_at_least(hx, hz, lower_bound=m + 1, which="z", threads=2)


def test_large_packed_syndrome():
    hx = np.ones((1100, 10), dtype=np.uint8)
    hz = np.zeros((0, 10), dtype=np.uint8)
    assert df.css_distance(hx, hz, method="cc", which="z", backend="cpu", threads=2) == 2
    assert df.css_distance_at_least(hx, hz, lower_bound=2, which="z", threads=2)
    assert not df.css_distance_at_least(hx, hz, lower_bound=3, which="z", threads=2)
