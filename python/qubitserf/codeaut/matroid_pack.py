"""Optimal information-set packing over GF(2) via matroid union (Edmonds base packing).

The Brouwer--Zimmermann completeness bound in :mod:`lowweight` strengthens with the number of
*pairwise-disjoint* information sets (column bases) of the generator matrix: ``q`` disjoint full
bases certify weight ``q(p+1)-1`` after enumerating message weight ``<= p``.  Greedy peeling can
find fewer than the maximum ``q`` (the rank-2 binary counterexample in the self-test), paying a
much larger ``p``.  This module computes the **maximum** number of disjoint bases (Edmonds'
matroid base-packing optimum ``floor(min_X |E\\X| / (r(E)-r(X)))``) by matroid-union
augmenting paths, plus an optimal final partial set.

``pack_information_sets(B, max_sets=None) -> (infosets, fresh_counts)``: ``B`` is a ``(k, n)``
uint8 GF(2) matrix of rank ``k``; each returned infoset is a sorted length-``k`` column basis,
the first ``q`` pairwise disjoint, and ``fresh_counts[j]`` is the number of columns of infoset
``j`` not used by an earlier one.  Dependency-free (numpy only), deterministic.

:mod:`lowweight` consumes this only as a packing *hint*: it independently re-verifies each set
is a rank-``k`` basis and recomputes the fresh counts, so a defect here can only loosen the
bound (never invalidate the completeness certificate), with the greedy packing as fallback.

Provenance: implemented by ``codex`` per ``research/matroid_bz_codex.md``; self-test validates
the rank-2 greedy-failure counterexample and 200 random matrices against the brute-force
Edmonds optimum.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Optional, Sequence

import numpy as np


__all__ = ["pack_information_sets"]


def _as_gf2_matrix(B) -> np.ndarray:
    A = np.asarray(B, dtype=np.uint8)
    if A.ndim != 2:
        raise ValueError("B must be a two-dimensional GF(2) matrix")
    return np.ascontiguousarray(A & 1, dtype=np.uint8)


def _columns_as_ints(B: np.ndarray) -> list[int]:
    k, n = B.shape
    cols: list[int] = []
    for j in range(n):
        x = 0
        for i in range(k):
            if int(B[i, j]):
                x |= 1 << i
        cols.append(x)
    return cols


def _rank_ints(vectors: Iterable[int]) -> int:
    basis: dict[int, int] = {}
    rank = 0
    for value in vectors:
        x = int(value)
        while x:
            pivot = x.bit_length() - 1
            reducer = basis.get(pivot)
            if reducer is None:
                basis[pivot] = x
                rank += 1
                break
            x ^= reducer
    return rank


class _Reducer:
    def __init__(self) -> None:
        self.basis: dict[int, int] = {}
        self.rank = 0

    def reduce(self, value: int) -> int:
        x = int(value)
        while x:
            pivot = x.bit_length() - 1
            reducer = self.basis.get(pivot)
            if reducer is None:
                return x
            x ^= reducer
        return 0

    def add(self, value: int) -> bool:
        x = self.reduce(value)
        if x == 0:
            return False
        self.basis[x.bit_length() - 1] = x
        self.rank += 1
        return True


def _extend_to_basis(cols: Sequence[int], k: int, seed: Sequence[int], order: Sequence[int]) -> list[int]:
    reducer = _Reducer()
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for e in seed:
        if e in chosen_set:
            raise AssertionError("duplicate seed column while extending basis")
        if not reducer.add(cols[e]):
            raise AssertionError("seed columns are not independent")
        chosen.append(int(e))
        chosen_set.add(int(e))
    for e in order:
        e = int(e)
        if e in chosen_set:
            continue
        if reducer.add(cols[e]):
            chosen.append(e)
            chosen_set.add(e)
            if len(chosen) == k:
                return sorted(chosen)
    if len(chosen) != k:
        raise AssertionError("could not extend independent set to a full basis")
    return sorted(chosen)


class _UnionPacking:
    """Matroid-union augmenting-path packing for copies of one binary matroid."""

    def __init__(self, cols: Sequence[int], k: int, capacities: Sequence[int]) -> None:
        self.cols = list(map(int, cols))
        self.k = int(k)
        self.n = len(self.cols)
        self.capacities = list(map(int, capacities))
        self.num_sets = len(self.capacities)
        self.sets: list[set[int]] = [set() for _ in self.capacities]
        self.owner = [-1] * self.n

    def _span_representation(self, elems: Sequence[int], e: int) -> tuple[bool, int]:
        """Return (dependent, mask). If dependent, mask is e's support in elems."""
        basis: dict[int, tuple[int, int]] = {}
        for pos, col_idx in enumerate(elems):
            x = self.cols[col_idx]
            mask = 1 << pos
            while x:
                pivot = x.bit_length() - 1
                reducer = basis.get(pivot)
                if reducer is None:
                    break
                x ^= reducer[0]
                mask ^= reducer[1]
            if x == 0:
                raise AssertionError("internal set lost independence")
            basis[x.bit_length() - 1] = (x, mask)

        x = self.cols[e]
        mask = 0
        while x:
            pivot = x.bit_length() - 1
            reducer = basis.get(pivot)
            if reducer is None:
                return False, mask
            x ^= reducer[0]
            mask ^= reducer[1]
        return True, mask

    @staticmethod
    def _mask_elements(elems: Sequence[int], mask: int) -> list[int]:
        out: list[int] = []
        while mask:
            bit = mask & -mask
            pos = bit.bit_length() - 1
            out.append(int(elems[pos]))
            mask ^= bit
        return out

    def _can_add(self, color: int, e: int) -> bool:
        if len(self.sets[color]) >= self.capacities[color] or e in self.sets[color]:
            return False
        dependent, _ = self._span_representation(sorted(self.sets[color]), e)
        return not dependent

    def greedy_start(self) -> None:
        for color, cap in enumerate(self.capacities):
            if cap <= 0:
                continue
            for e in range(self.n):
                if len(self.sets[color]) >= cap:
                    break
                if self.owner[e] < 0 and self._can_add(color, e):
                    self.sets[color].add(e)
                    self.owner[e] = color

    def augment_once(self) -> bool:
        parent = [-3] * self.n
        edge_color = [-1] * self.n
        queue: deque[int] = deque()
        for e in range(self.n):
            if self.owner[e] < 0:
                parent[e] = -2
                queue.append(e)

        terminal = -1
        terminal_color = -1
        while queue and terminal < 0:
            x = queue.popleft()
            for color in range(self.num_sets):
                if x in self.sets[color]:
                    continue
                elems = sorted(self.sets[color])
                dependent, mask = self._span_representation(elems, x)
                if not dependent:
                    if len(elems) < self.capacities[color]:
                        terminal = x
                        terminal_color = color
                        break
                    exchange_out = elems
                else:
                    exchange_out = self._mask_elements(elems, mask)

                for y in exchange_out:
                    if y != x and parent[y] == -3:
                        parent[y] = x
                        edge_color[y] = color
                        queue.append(y)

        if terminal < 0:
            return False

        self.sets[terminal_color].add(terminal)
        self.owner[terminal] = terminal_color
        cur = terminal
        while parent[cur] != -2:
            prev = parent[cur]
            color = edge_color[cur]
            self.sets[color].remove(cur)
            self.sets[color].add(prev)
            self.owner[prev] = color
            cur = prev
        return True

    def maximize(self) -> int:
        self.greedy_start()
        while self.augment_once():
            pass
        return self.size

    @property
    def size(self) -> int:
        return sum(len(s) for s in self.sets)

    def sorted_sets(self) -> list[list[int]]:
        return [sorted(s) for s in self.sets]


def _full_basis_packing(cols: Sequence[int], k: int, q: int) -> _UnionPacking:
    packing = _UnionPacking(cols, k, [k] * q)
    packing.maximize()
    return packing


def pack_information_sets(B, max_sets: Optional[int] = None) -> tuple[list[list[int]], list[int]]:
    """Pack GF(2) information sets optimally by matroid union.

    Returns (infosets, fresh_counts). The first returned sets are a maximum
    number of pairwise-disjoint full bases of the column matroid, subject to
    max_sets if supplied. If another returned set is possible, the last one is
    a full basis whose number of fresh columns is maximum over all optimal full
    base packings.
    """
    A = _as_gf2_matrix(B)
    k, n = A.shape
    if max_sets is not None:
        max_sets = int(max_sets)
        if max_sets < 0:
            raise ValueError("max_sets must be nonnegative or None")
        if max_sets == 0:
            return [], []
    if k == 0:
        return [], []

    cols = _columns_as_ints(A)
    if _rank_ints(cols) != k:
        raise ValueError("B must have full row rank over GF(2)")

    nonloops = sum(1 for c in cols if c != 0)
    q_limit = nonloops // k
    if max_sets is not None:
        q_limit = min(q_limit, max_sets)

    best_q = 0
    best_packing: Optional[_UnionPacking] = None
    for q in range(1, q_limit + 1):
        packing = _full_basis_packing(cols, k, q)
        if packing.size == q * k:
            best_q = q
            best_packing = packing
        else:
            break

    if best_q == 0:
        basis = _extend_to_basis(cols, k, [], list(range(n)))
        return [basis], [k]

    can_return_partial = max_sets is None or best_q < max_sets
    chosen_packing = best_packing
    partial: list[int] = []
    if can_return_partial:
        max_f = min(k - 1, nonloops - best_q * k)
        lo, hi = 1, max_f
        while lo <= hi:
            f = (lo + hi) // 2
            candidate = _UnionPacking(cols, k, [k] * best_q + [f])
            if candidate.maximize() == best_q * k + f:
                chosen_packing = candidate
                partial = sorted(candidate.sets[-1])
                lo = f + 1
            else:
                hi = f - 1

    if chosen_packing is None:
        raise AssertionError("matroid union failed to keep a feasible full packing")

    full_sets = [sorted(chosen_packing.sets[i]) for i in range(best_q)]
    infosets = [list(s) for s in full_sets]
    if partial:
        old_columns = sorted(set().union(*(set(s) for s in full_sets)))
        infosets.append(_extend_to_basis(cols, k, partial, old_columns))

    fresh_counts: list[int] = []
    used: set[int] = set()
    for info in infosets:
        fresh_counts.append(sum(1 for e in info if e not in used))
        used.update(info)
    return infosets, fresh_counts


def _greedy_peeling_count(cols: Sequence[int], k: int) -> int:
    unused = set(range(len(cols)))
    count = 0
    while True:
        reducer = _Reducer()
        basis: list[int] = []
        for e in range(len(cols)):
            if e in unused and reducer.add(cols[e]):
                basis.append(e)
                if len(basis) == k:
                    break
        if len(basis) != k:
            return count
        count += 1
        unused.difference_update(basis)


def _edmonds_optimum_bruteforce(cols: Sequence[int], k: int) -> int:
    n = len(cols)
    best = n // k
    for mask in range(1 << n):
        subset_rank = _rank_ints(cols[i] for i in range(n) if (mask >> i) & 1)
        if subset_rank < k:
            outside = n - mask.bit_count()
            best = min(best, outside // (k - subset_rank))
    return best


def _assert_basis(cols: Sequence[int], k: int, info: Sequence[int]) -> None:
    assert len(info) == k
    assert len(set(info)) == k
    assert _rank_ints(cols[e] for e in info) == k


def _random_full_rank_matrix(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    while True:
        B = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        if _rank_ints(_columns_as_ints(B)) == k:
            return B


def _self_test() -> None:
    counter = np.array(
        [
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    cols = _columns_as_ints(counter)
    assert _greedy_peeling_count(cols, 2) == 1
    infosets, fresh = pack_information_sets(counter)
    assert fresh[:2] == [2, 2]
    assert len([f for f in fresh if f == 2]) == 2

    rng = np.random.default_rng(12345)
    for _ in range(200):
        k = int(rng.integers(1, 6))
        n = int(rng.integers(k, 11))
        B = _random_full_rank_matrix(rng, k, n)
        cols = _columns_as_ints(B)
        expected = _edmonds_optimum_bruteforce(cols, k)
        infosets, fresh = pack_information_sets(B)
        got = sum(1 for f in fresh if f == k)
        assert got == expected, (k, n, expected, got, infosets, fresh)

        used: set[int] = set()
        for info, f in zip(infosets, fresh):
            _assert_basis(cols, k, info)
            assert f == sum(1 for e in info if e not in used)
            if f == k:
                assert set(info).isdisjoint(used)
            used.update(info)


if __name__ == "__main__":
    _self_test()

