"""Exact automorphisms from binary-matroid connected components.

For a binary code ``C = rowspace(G)``, coordinate permutations preserving ``C`` are exactly
the automorphisms of the column matroid of ``G``.  Matroid connected components give the
canonical direct-sum decomposition of ``C``; automorphisms act internally on each component
and may permute equivalent components.  This module computes that wreath-product action
without enumerating the global code.

Local components are handled by a guarded projective-column search.  If ``B`` has rank ``r``,
its columns define a multiplicity function ``mu: GF(2)^r -> N``.  The outer local group is

    {A in GL(r,2) : mu(A v) = mu(v) for every v}.

The search fixes a basis of present column types and enumerates its possible images.  Partial
maps are rejected by comparing the multiplicities on each newly exposed affine coset.  The
comparison iterates only present column types, not all ``2**r`` vectors.  Permutations inside
equal-column fibres are added separately.  Resource-limit paths may use Leon on the local code
(or a caller-supplied exact solver); no inexact subgroup is ever returned.

The public result follows the other ``codeaut`` engines: it contains generators and an exact
order and has a :meth:`ComponentAutResult.group` convenience method.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Optional

import numpy as np

from . import gf2
from . import permgroup


class ComponentGuardExceeded(RuntimeError):
    """A guarded component/projective search could not finish exactly."""


@dataclass
class ComponentAutResult:
    """Exact result returned by :func:`component_automorphism_group`."""

    generators: list[list[int]]
    order: int
    n: int
    dim: int
    num_components: int
    component_sizes: tuple[int, ...]
    component_dims: tuple[int, ...]
    type_counts: tuple[int, ...]
    component_classes: tuple[tuple[int, ...], ...]
    method: str
    construction_seconds: float
    search_seconds: float
    total_seconds: float
    diagnostics: dict[str, Any]

    def group(self) -> permgroup.Group:
        """Return the result as a permutation group on the code coordinates."""
        return permgroup.Group(self.generators, self.n)


@dataclass
class _TypeConfig:
    rank: int
    length: int
    types: tuple[int, ...]
    fibres: dict[int, tuple[int, ...]]
    multiplicity: dict[int, int]
    fingerprints: Optional[dict[int, tuple]]
    signature: tuple


@dataclass
class _Component:
    index: int
    coordinates: tuple[int, ...]
    basis: np.ndarray
    config: _TypeConfig


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.parent[y] = x
        self.size[x] += self.size[y]


def _columns_as_ints(B: np.ndarray) -> list[int]:
    r, n = B.shape
    out = [0] * n
    for j in range(n):
        value = 0
        for i in range(r):
            if int(B[i, j]):
                value |= 1 << i
        out[j] = value
    return out


def _rank_ints(vectors) -> int:
    pivots: dict[int, int] = {}
    rank = 0
    for value in vectors:
        x = int(value)
        while x:
            pivot = x.bit_length() - 1
            reducer = pivots.get(pivot)
            if reducer is None:
                pivots[pivot] = x
                rank += 1
                break
            x ^= reducer
    return rank


def _coefficient_reducer(basis: list[int]) -> dict[int, tuple[int, int]]:
    """Pivot table mapping a vector in ``span(basis)`` to its basis coefficients."""
    pivots: dict[int, tuple[int, int]] = {}
    for i, value in enumerate(basis):
        x = int(value)
        coeff = 1 << i
        while x:
            pivot = x.bit_length() - 1
            reducer = pivots.get(pivot)
            if reducer is None:
                pivots[pivot] = (x, coeff)
                break
            x ^= reducer[0]
            coeff ^= reducer[1]
        if x == 0:
            raise AssertionError("internal projective basis became dependent")
    return pivots


def _coefficients(value: int, pivots: dict[int, tuple[int, int]]) -> Optional[int]:
    x = int(value)
    coeff = 0
    while x:
        pivot = x.bit_length() - 1
        reducer = pivots.get(pivot)
        if reducer is None:
            return None
        x ^= reducer[0]
        coeff ^= reducer[1]
    return coeff


def _combination(basis: list[int], mask: int) -> int:
    out = 0
    while mask:
        bit = mask & -mask
        out ^= basis[bit.bit_length() - 1]
        mask ^= bit
    return out


def _type_fingerprint(multiplicity: dict[int, int], value: int) -> tuple:
    """An equivariant additive-neighbourhood fingerprint of one present type."""
    counts = Counter((multiplicity[u], multiplicity.get(value ^ u, 0))
                     for u in multiplicity)
    return tuple(sorted((key, count) for key, count in counts.items()))


def _type_config(B: np.ndarray, *, fingerprint_limit: int) -> _TypeConfig:
    B = gf2.row_basis_gf2(B)
    r, n = B.shape
    fibres_mut: dict[int, list[int]] = {}
    for j, value in enumerate(_columns_as_ints(B)):
        fibres_mut.setdefault(value, []).append(j)
    fibres = {value: tuple(indices) for value, indices in fibres_mut.items()}
    multiplicity = {value: len(indices) for value, indices in fibres.items()}
    types = tuple(sorted(fibres))

    fingerprints = None
    if len(types) <= fingerprint_limit:
        fingerprints = {value: _type_fingerprint(multiplicity, value) for value in types}
        marked = tuple(sorted((multiplicity[value], fingerprints[value]) for value in types))
    else:
        marked = tuple(sorted(multiplicity.values()))
    signature = (r, n, multiplicity.get(0, 0), marked)
    return _TypeConfig(r, n, types, fibres, multiplicity, fingerprints, signature)


def _matroid_components(B: np.ndarray) -> list[tuple[int, ...]]:
    """Connected components of the column matroid, via fundamental circuits."""
    B = gf2.row_basis_gf2(B)
    _r, n = B.shape
    if n == 0:
        return []
    systematic, pivots = gf2.rref_gf2(B)
    pivot_set = set(pivots)
    uf = _UnionFind(n)
    for j in range(n):
        if j in pivot_set:
            continue
        support_rows = [i for i in range(len(pivots)) if int(systematic[i, j])]
        if not support_rows:
            # A zero column is a matroid loop and hence a singleton component.
            continue
        circuit = [j] + [int(pivots[i]) for i in support_rows]
        anchor = circuit[0]
        for other in circuit[1:]:
            uf.union(anchor, other)
    groups: dict[int, list[int]] = {}
    for j in range(n):
        groups.setdefault(uf.find(j), []).append(j)
    return [tuple(group) for group in sorted(groups.values(), key=lambda x: (min(x), len(x)))]


def _select_source_basis(config: _TypeConfig) -> list[int]:
    if config.rank == 0:
        return []
    if config.fingerprints is None:
        raise ComponentGuardExceeded("projective type fingerprints exceed max_types")
    key_frequency = Counter((config.multiplicity[v], config.fingerprints[v])
                            for v in config.types if v)
    candidates = sorted((v for v in config.types if v),
                        key=lambda v: (key_frequency[(config.multiplicity[v],
                                                     config.fingerprints[v])],
                                       config.multiplicity[v], v))
    chosen: list[int] = []
    rank = 0
    for value in candidates:
        new_rank = _rank_ints(chosen + [value])
        if new_rank > rank:
            chosen.append(value)
            rank = new_rank
            if rank == config.rank:
                return chosen
    raise AssertionError("present column types do not span the local row rank")


def _coset_matches(src: _TypeConfig, dst: _TypeConfig,
                   src_assigned: list[int], dst_assigned: list[int],
                   src_new: int, dst_new: int) -> bool:
    """Compare multiplicities on one new affine coset under a partial linear map."""
    src_reducer = _coefficient_reducer(src_assigned)
    dst_reducer = _coefficient_reducer(dst_assigned)
    expected: dict[int, int] = {}
    for value, count in src.multiplicity.items():
        coeff = _coefficients(value ^ src_new, src_reducer)
        if coeff is not None:
            image = dst_new ^ _combination(dst_assigned, coeff)
            expected[image] = count
    for image, count in expected.items():
        if dst.multiplicity.get(image, 0) != count:
            return False
    for value, count in dst.multiplicity.items():
        if _coefficients(value ^ dst_new, dst_reducer) is not None:
            if expected.get(value, 0) != count:
                return False
    return True


def _linear_type_map(src: _TypeConfig, dst: _TypeConfig,
                     src_basis: list[int], dst_basis: list[int]) -> dict[int, int]:
    reducer = _coefficient_reducer(src_basis)
    mapping: dict[int, int] = {}
    for value in src.types:
        coeff = _coefficients(value, reducer)
        if coeff is None:
            raise AssertionError("source type lies outside a full projective basis")
        image = _combination(dst_basis, coeff)
        if src.multiplicity[value] != dst.multiplicity.get(image, 0):
            raise AssertionError("completed projective map does not preserve multiplicities")
        mapping[value] = image
    return mapping


def _enumerate_type_isomorphisms(src: _TypeConfig, dst: _TypeConfig, *,
                                 max_types: int, max_candidates: int,
                                 max_solutions: int, max_seconds: Optional[float],
                                 first_only: bool):
    """Enumerate full basis images defining type-multiset isomorphisms."""
    if src.signature != dst.signature:
        return [], {"candidates": 0, "nodes": 0, "solutions": 0}
    if max(len(src.types), len(dst.types)) > max_types:
        raise ComponentGuardExceeded(
            f"projective search has {max(len(src.types), len(dst.types))} column types, "
            f"above max_types={max_types}")
    if src.rank != dst.rank or src.length != dst.length:
        return [], {"candidates": 0, "nodes": 0, "solutions": 0}
    if src.multiplicity.get(0, 0) != dst.multiplicity.get(0, 0):
        return [], {"candidates": 0, "nodes": 0, "solutions": 0}

    src_basis = _select_source_basis(src)
    if not src_basis:
        return [[]], {"candidates": 0, "nodes": 1, "solutions": 1,
                      "source_basis": []}
    if src.fingerprints is None or dst.fingerprints is None:
        raise ComponentGuardExceeded("projective fingerprints unavailable above max_types")

    pools: list[list[int]] = []
    for value in src_basis:
        key = (src.multiplicity[value], src.fingerprints[value])
        pool = [candidate for candidate in dst.types if candidate and
                (dst.multiplicity[candidate], dst.fingerprints[candidate]) == key]
        pools.append(sorted(pool))

    stats = {"candidates": 0, "nodes": 0, "solutions": 0,
             "source_basis": list(src_basis)}
    solutions: list[list[int]] = []
    chosen: list[int] = []
    deadline = None if max_seconds is None else time.perf_counter() + max_seconds

    def visit(depth: int) -> bool:
        stats["nodes"] += 1
        if depth == src.rank:
            # The flag of checked cosets covers all of GF(2)^r.  Retain a defensive full check
            # over the present support before accepting the map.
            _linear_type_map(src, dst, src_basis, chosen)
            stats["solutions"] += 1
            if stats["solutions"] > max_solutions:
                raise ComponentGuardExceeded(
                    f"projective search exceeded max_solutions={max_solutions}")
            solutions.append(list(chosen))
            return first_only

        dst_reducer = _coefficient_reducer(chosen)
        src_new = src_basis[depth]
        for candidate in pools[depth]:
            stats["candidates"] += 1
            if stats["candidates"] > max_candidates:
                raise ComponentGuardExceeded(
                    f"projective search exceeded max_candidates={max_candidates}")
            if (deadline is not None and (stats["candidates"] & 255) == 0 and
                    time.perf_counter() > deadline):
                raise ComponentGuardExceeded(
                    f"projective search exceeded max_projective_seconds={max_seconds}")
            if _coefficients(candidate, dst_reducer) is not None:
                continue
            if not _coset_matches(src, dst, src_basis[:depth], chosen,
                                  src_new, candidate):
                continue
            chosen.append(candidate)
            stop = visit(depth + 1)
            chosen.pop()
            if stop:
                return True
        return False

    visit(0)
    return solutions, stats


def _coordinate_transporter(src: _TypeConfig, dst: _TypeConfig,
                            src_basis: list[int], dst_basis: list[int]) -> list[int]:
    mapping = _linear_type_map(src, dst, src_basis, dst_basis)
    out = [-1] * src.length
    for value, image in mapping.items():
        left = src.fibres[value]
        right = dst.fibres[image]
        if len(left) != len(right):
            raise AssertionError("type transporter encountered unequal fibres")
        for i, j in zip(left, right):
            out[i] = j
    if sorted(out) != list(range(src.length)):
        raise AssertionError("type transporter is not a coordinate permutation")
    return out


def _transports_code(src_basis: np.ndarray, dst_basis: np.ndarray, perm) -> bool:
    perm = [int(x) for x in perm]
    if src_basis.shape[1] != dst_basis.shape[1] or len(perm) != src_basis.shape[1]:
        return False
    inverse = permgroup.inv(perm)
    transported = src_basis[:, inverse]
    rank = gf2.rank_gf2(dst_basis)
    return (gf2.rank_gf2(src_basis) == rank and
            gf2.rank_gf2(np.vstack([dst_basis, transported])) == rank)


def _parse_local_group(value, degree: int) -> tuple[list[list[int]], int]:
    if isinstance(value, permgroup.Group):
        return value.gens(), value.order()
    if hasattr(value, "generators") and hasattr(value, "order"):
        generators = value.generators
        generators = generators() if callable(generators) else generators
        order = value.order() if callable(value.order) else value.order
        return [[int(x) for x in g] for g in generators], int(order)
    if isinstance(value, tuple) and len(value) >= 2:
        generators, order = value[:2]
        return [[int(x) for x in g] for g in generators], int(order)
    raise TypeError("local_solver must return a Group, an AutResult-like object, or (gens, order)")


def _default_local_group(B: np.ndarray, max_component_dim: int):
    from . import leon

    economical, _n, _rank, effective_dim = gf2.dual_basis(B)
    if effective_dim > max_component_dim:
        raise ComponentGuardExceeded(
            f"local effective dimension {effective_dim} exceeds "
            f"max_component_dim={max_component_dim}")
    result = leon.automorphism_group(economical, max_dim=max_component_dim)
    return result.generators, result.order, {
        "solver": "Leon local fallback",
        "effective_dim": effective_dim,
        "num_codewords": result.num_codewords,
    }


def _projective_local_group(config: _TypeConfig, *, max_types: int,
                            max_candidates: int, max_solutions: int,
                            max_seconds: Optional[float]):
    solutions, stats = _enumerate_type_isomorphisms(
        config, config, max_types=max_types, max_candidates=max_candidates,
        max_solutions=max_solutions, max_seconds=max_seconds, first_only=False)
    src_basis = stats.get("source_basis", [])
    type_index = {value: i for i, value in enumerate(config.types)}
    outer_perms: list[list[int]] = []
    for dst_basis in solutions:
        mapping = _linear_type_map(config, config, src_basis, dst_basis)
        outer_perms.append([type_index[mapping[value]] for value in config.types])
    outer_group = permgroup.Group(outer_perms, len(config.types))
    if outer_group.order() != len(solutions):
        raise AssertionError("projective maps did not form the expected faithful outer group")

    generators: list[list[int]] = []
    for outer in outer_group.reduced_generators():
        mapping = {value: config.types[outer[i]] for i, value in enumerate(config.types)}
        coordinate = [-1] * config.length
        for value, image in mapping.items():
            for left, right in zip(config.fibres[value], config.fibres[image]):
                coordinate[left] = right
        generators.append(coordinate)

    fibre_kernel_order = 1
    for fibre in config.fibres.values():
        size = len(fibre)
        fibre_kernel_order *= math.factorial(size)
        if size <= 1:
            continue
        transposition = list(range(config.length))
        transposition[fibre[0]], transposition[fibre[1]] = fibre[1], fibre[0]
        generators.append(transposition)
        if size > 2:
            cycle = list(range(config.length))
            for i, point in enumerate(fibre):
                cycle[point] = fibre[(i + 1) % size]
            generators.append(cycle)

    group = permgroup.Group(generators, config.length)
    expected_order = len(solutions) * fibre_kernel_order
    if group.order() != expected_order:
        raise AssertionError("lifted projective/fibre group has an unexpected order")
    return group.reduced_generators(), expected_order, {
        "solver": "projective column-type stabilizer",
        "outer_order": len(solutions),
        "fibre_kernel_order": fibre_kernel_order,
        **stats,
    }


def _local_group(component: _Component, *, max_types: int, max_candidates: int,
                 max_solutions: int, max_seconds: Optional[float],
                 max_component_dim: int,
                 local_solver: Optional[Callable]):
    # Native exhaustive Leon is generally cheaper than Python projective backtracking in this
    # small effective-dimension window, especially for high-symmetry configurations.  Keep very
    # small ranks on the dependency-free projective path, where they finish almost immediately.
    effective_dim = min(component.basis.shape[0],
                        component.basis.shape[1] - component.basis.shape[0])
    if local_solver is None and 5 <= effective_dim <= min(12, max_component_dim):
        try:
            generators, order, detail = _default_local_group(
                component.basis, max_component_dim)
            detail["selection"] = "cost-aware small-effective-dimension fast path"
            return generators, order, detail
        except Exception:
            # The native library is optional at import time.  If it is unavailable, retain the
            # self-contained guarded projective route below.
            pass
    try:
        return _projective_local_group(
            component.config, max_types=max_types, max_candidates=max_candidates,
            max_solutions=max_solutions, max_seconds=max_seconds)
    except ComponentGuardExceeded as projective_error:
        if local_solver is not None:
            generators, order = _parse_local_group(local_solver(component.basis),
                                                    component.basis.shape[1])
            detail = {"solver": "caller local fallback",
                      "projective_guard": str(projective_error)}
        else:
            try:
                generators, order, detail = _default_local_group(
                    component.basis, max_component_dim)
                detail["projective_guard"] = str(projective_error)
            except Exception as fallback_error:
                if isinstance(fallback_error, ComponentGuardExceeded):
                    raise ComponentGuardExceeded(
                        f"component {component.index}: {projective_error}; "
                        f"local fallback unavailable: {fallback_error}") from fallback_error
                raise ComponentGuardExceeded(
                    f"component {component.index}: {projective_error}; "
                    f"local fallback failed: {fallback_error}") from fallback_error

        if not all(gf2.preserves_rowspace(component.basis, g) for g in generators):
            raise AssertionError("local fallback returned a generator outside Aut(component)")
        check = permgroup.Group(generators, component.basis.shape[1])
        if check.order() != order:
            raise AssertionError("local fallback's generators and reported order disagree")
        return check.reduced_generators(), int(order), detail


def _default_equivalence(src: _Component, dst: _Component, max_component_dim: int):
    """Use Aut(src direct-sum dst) to decide/extract a component swap."""
    from . import leon

    r = src.basis.shape[0]
    n = src.basis.shape[1]
    zeros = np.zeros_like(src.basis)
    direct = np.vstack([np.hstack([src.basis, zeros]),
                        np.hstack([zeros, dst.basis])]).astype(np.uint8)
    economical, _length, _rank, effective_dim = gf2.dual_basis(direct)
    if effective_dim > max_component_dim:
        raise ComponentGuardExceeded(
            f"pair equivalence effective dimension {effective_dim} exceeds "
            f"max_component_dim={max_component_dim}")
    result = leon.automorphism_group(economical, max_dim=max_component_dim)
    first = set(range(n))
    second = set(range(n, 2 * n))
    for generator in result.generators:
        if {generator[i] for i in first} == second:
            transporter = [int(generator[i] - n) for i in range(n)]
            if _transports_code(src.basis, dst.basis, transporter):
                return transporter, {"solver": "Leon direct-sum equivalence fallback",
                                     "effective_dim": effective_dim}
    return None, {"solver": "Leon direct-sum equivalence fallback",
                  "effective_dim": effective_dim}


def _component_isomorphism(src: _Component, dst: _Component, *, max_types: int,
                           max_candidates: int, max_solutions: int,
                           max_seconds: Optional[float],
                           max_component_dim: int,
                           equivalence_solver: Optional[Callable]):
    if src.config.signature != dst.config.signature:
        return None, {"solver": "invariant rejection"}
    if np.array_equal(src.basis, dst.basis):
        return list(range(src.basis.shape[1])), {"solver": "identical local basis"}
    try:
        solutions, stats = _enumerate_type_isomorphisms(
            src.config, dst.config, max_types=max_types, max_candidates=max_candidates,
            max_solutions=max_solutions, max_seconds=max_seconds, first_only=True)
        if not solutions:
            return None, {"solver": "projective equivalence", **stats}
        source_basis = stats.get("source_basis", [])
        transporter = _coordinate_transporter(src.config, dst.config,
                                              source_basis, solutions[0])
        if not _transports_code(src.basis, dst.basis, transporter):
            raise AssertionError("projective component transporter failed rowspace verification")
        return transporter, {"solver": "projective equivalence", **stats}
    except ComponentGuardExceeded as projective_error:
        if equivalence_solver is not None:
            transporter = equivalence_solver(src.basis, dst.basis)
            detail = {"solver": "caller equivalence fallback",
                      "projective_guard": str(projective_error)}
            if transporter is None:
                return None, detail
            transporter = [int(x) for x in transporter]
            if not _transports_code(src.basis, dst.basis, transporter):
                raise AssertionError("equivalence_solver returned an invalid transporter")
            return transporter, detail
        try:
            transporter, detail = _default_equivalence(src, dst, max_component_dim)
            detail["projective_guard"] = str(projective_error)
            return transporter, detail
        except Exception as fallback_error:
            if isinstance(fallback_error, ComponentGuardExceeded):
                raise ComponentGuardExceeded(
                    f"components {src.index}/{dst.index}: {projective_error}; "
                    f"equivalence fallback unavailable: {fallback_error}") from fallback_error
            raise ComponentGuardExceeded(
                f"components {src.index}/{dst.index}: {projective_error}; "
                f"equivalence fallback failed: {fallback_error}") from fallback_error


def _lift_local(local_perm, coordinates: tuple[int, ...], degree: int) -> list[int]:
    out = list(range(degree))
    for i, image in enumerate(local_perm):
        out[coordinates[i]] = coordinates[int(image)]
    return out


def _block_permutation(members: list[_Component], transporters: list[list[int]],
                       block_perm: list[int], degree: int) -> list[int]:
    """Lift a component permutation using coherent representative identifications."""
    out = list(range(degree))
    inverses = [permgroup.inv(q) for q in transporters]
    for block, component in enumerate(members):
        target_block = block_perm[block]
        target = members[target_block]
        source_to_rep = inverses[block]
        rep_to_target = transporters[target_block]
        for local, coordinate in enumerate(component.coordinates):
            representative_local = source_to_rep[local]
            target_local = rep_to_target[representative_local]
            out[coordinate] = target.coordinates[target_local]
    return out


def component_automorphism_group(
        generator_matrix, *, max_types: int = 64, max_candidates: int = 2_000_000,
        max_solutions: int = 50_000, max_projective_seconds: Optional[float] = 5.0,
        max_component_dim: int = 20,
        local_solver: Optional[Callable] = None,
        equivalence_solver: Optional[Callable] = None,
        verify_order: bool = False) -> ComponentAutResult:
    """Compute exact ``Aut(rowspace(generator_matrix))`` by matroid components.

    ``max_types``, ``max_candidates``, ``max_solutions``, and ``max_projective_seconds`` guard
    each sparse projective-column search.  Set the time guard to ``None`` to disable it.  On a
    guard hit, a connected component is passed to ``local_solver(B)`` if supplied; otherwise
    native Leon is used when the component's effective dimension is at most ``max_component_dim``.
    A supplied solver must be exact and return either a
    :class:`~codeaut.permgroup.Group`, an AutResult-like object, or ``(generators, order)``.

    Component equivalence normally uses the same projective search.  On its guard path,
    ``equivalence_solver(B1, B2)`` may return an image-list transporter from ``B1`` to ``B2``
    (or ``None`` if inequivalent); otherwise Leon solves the direct sum when its effective
    dimension fits ``max_component_dim``.  Exhausting all exact options raises
    :class:`ComponentGuardExceeded` rather than returning a subgroup.

    Every lifted generator is always verified against the original rowspace.  ``verify_order``
    additionally asks the pure-Python Schreier--Sims engine to recompute the full wreath-product
    order from those generators.  It defaults to false because that redundant audit can dominate
    otherwise-polynomial decomposition on many large isomorphic components; the reported order
    is already exact by the proved wreath formula and the independently exact local orders.
    """
    started = time.perf_counter()
    G = np.asarray(generator_matrix)
    if G.ndim != 2:
        raise ValueError("generator_matrix must be two-dimensional")
    if min(max_types, max_candidates, max_solutions, max_component_dim) < 1:
        raise ValueError("all component/projective guards must be positive")
    if max_projective_seconds is not None and max_projective_seconds <= 0:
        raise ValueError("max_projective_seconds must be positive or None")
    C = gf2.row_basis_gf2(G)
    k, n = C.shape

    before = time.perf_counter()
    coordinate_components = _matroid_components(C)
    components: list[_Component] = []
    for index, coordinates in enumerate(coordinate_components):
        local_basis = gf2.row_basis_gf2(C[:, list(coordinates)])
        config = _type_config(local_basis, fingerprint_limit=max_types)
        components.append(_Component(index, coordinates, local_basis, config))
    if sum(component.basis.shape[0] for component in components) != k:
        raise AssertionError("matroid components did not reproduce the global row rank")
    construction_seconds = time.perf_counter() - before

    search_started = time.perf_counter()
    classes: list[dict[str, Any]] = []
    equivalence_diagnostics: list[dict[str, Any]] = []
    for component in components:
        placed = False
        for class_index, cls in enumerate(classes):
            representative: _Component = cls["members"][0]
            transporter, detail = _component_isomorphism(
                representative, component, max_types=max_types,
                max_candidates=max_candidates, max_solutions=max_solutions,
                max_seconds=max_projective_seconds,
                max_component_dim=max_component_dim,
                equivalence_solver=equivalence_solver)
            equivalence_diagnostics.append({
                "representative": representative.index,
                "candidate": component.index,
                "equivalent": transporter is not None,
                **detail,
            })
            if transporter is not None:
                cls["members"].append(component)
                cls["transporters"].append(transporter)
                placed = True
                break
        if not placed:
            classes.append({"members": [component],
                            "transporters": [list(range(component.basis.shape[1]))]})

    global_generators: list[list[int]] = []
    total_order = 1
    local_diagnostics: list[dict[str, Any]] = []
    for class_index, cls in enumerate(classes):
        members: list[_Component] = cls["members"]
        transporters: list[list[int]] = cls["transporters"]
        representative = members[0]
        local_generators, local_order, detail = _local_group(
            representative, max_types=max_types, max_candidates=max_candidates,
            max_solutions=max_solutions, max_seconds=max_projective_seconds,
            max_component_dim=max_component_dim,
            local_solver=local_solver)
        for generator in local_generators:
            global_generators.append(_lift_local(generator, representative.coordinates, n))

        copies = len(members)
        total_order *= (int(local_order) ** copies) * math.factorial(copies)
        local_diagnostics.append({
            "class": class_index,
            "representative": representative.index,
            "members": [member.index for member in members],
            "local_order": int(local_order),
            **detail,
        })

        if copies >= 2:
            transposition = list(range(copies))
            transposition[0], transposition[1] = 1, 0
            global_generators.append(
                _block_permutation(members, transporters, transposition, n))
        if copies >= 3:
            cycle = [(i + 1) % copies for i in range(copies)]
            global_generators.append(_block_permutation(members, transporters, cycle, n))

    if not all(gf2.preserves_rowspace(C, generator) for generator in global_generators):
        raise AssertionError("a lifted component/wreath generator does not preserve the code")
    global_group = permgroup.Group(global_generators, n)
    if verify_order and global_group.order() != total_order:
        raise AssertionError(
            f"lifted component group order {global_group.order()} != wreath formula {total_order}")
    generators = global_group.reduced_generators() if verify_order else global_group.gens()
    search_seconds = time.perf_counter() - search_started

    diagnostics = {
        "components": [list(component.coordinates) for component in components],
        "component_signatures": [component.config.signature for component in components],
        "local_solvers": local_diagnostics,
        "equivalence_checks": equivalence_diagnostics,
        "wreath_formula_order": total_order,
        "guards": {
            "max_types": max_types,
            "max_candidates": max_candidates,
            "max_solutions": max_solutions,
            "max_projective_seconds": max_projective_seconds,
            "max_component_dim": max_component_dim,
        },
    }
    return ComponentAutResult(
        generators=generators,
        order=int(total_order),
        n=n,
        dim=k,
        num_components=len(components),
        component_sizes=tuple(len(component.coordinates) for component in components),
        component_dims=tuple(component.basis.shape[0] for component in components),
        type_counts=tuple(len(component.config.types) for component in components),
        component_classes=tuple(tuple(member.index for member in cls["members"])
                                for cls in classes),
        method="binary-matroid components + projective/guarded local exact solvers",
        construction_seconds=construction_seconds,
        search_seconds=search_seconds,
        total_seconds=time.perf_counter() - started,
        diagnostics=diagnostics,
    )


def _self_test(seed: int = 7, random_trials: int = 30) -> None:
    """Small deterministic validation; callable without a test-framework dependency."""
    cases = [
        (np.zeros((0, 4), dtype=np.uint8), math.factorial(4)),
        (np.eye(4, dtype=np.uint8), math.factorial(4)),
        (np.ones((1, 4), dtype=np.uint8), math.factorial(4)),
        (np.array([[1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1, 1]], dtype=np.uint8), 72),
    ]
    for matrix, expected in cases:
        result = component_automorphism_group(matrix)
        assert result.order == expected
        assert all(gf2.preserves_rowspace(matrix, g) for g in result.generators)

    from . import leon
    rng = np.random.default_rng(seed)
    for _ in range(random_trials):
        k = int(rng.integers(1, 6))
        n = int(rng.integers(k, 10))
        matrix = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        result = component_automorphism_group(matrix)
        basis, _n, _rank, _effective = gf2.dual_basis(matrix)
        reference = leon.automorphism_group(basis, max_dim=10).group()
        actual = result.group()
        assert actual.order() == reference.order()
        assert all(reference.contains(g) for g in actual.gens())
        assert all(actual.contains(g) for g in reference.gens())


__all__ = ["ComponentGuardExceeded", "ComponentAutResult",
           "component_automorphism_group"]
