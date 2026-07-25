"""Experimental exact automorphism portfolio built from characteristic code invariants.

The production Leon engine encodes complete low-weight codeword classes until they span the
code.  This module explores smaller exact certificates and compact invariant *overgroups*:

* LCD orthogonal projectors;
* projective-column geometry, including parallel-coordinate symmetric kernels;
* twin-compressed and fingerprint-selected codeword incidences;
* binary-matroid component labels and BZ/fixed-support bounded circuits/cocircuits;
* pair/triple moments, SSA-style minor hull enumerators, Schur/conductor projectors;
* Ward residue-span layers and modular singleton/pair sections;
* puncture/shorten minor signatures.

Every public call still returns an exact group.  Direct certificates prove equality.  A compact
relation that is only known to contain ``Aut(C)`` is accepted iff every generator returned by
the graph solver preserves ``C``; otherwise a guarded Schreier traversal computes the exact
rowspace stabilizer, or the method falls back to ordinary exact minimum-weight incidence when
that orbit/dimension guard permits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import itertools
import math
import subprocess
import time
from typing import Optional

import numpy as np

from . import gf2
from . import graphaut
from . import lowweight
from . import permgroup
from . import ward


METHODS = (
    "lcd", "geometry", "compressed", "components", "fingerprint", "moments",
    "hull", "schur", "residue", "modular", "minors", "bounded", "combined", "auto",
)
METHOD_REGISTRY = {name: name for name in METHODS}


class InvariantLimitExceeded(RuntimeError):
    """A configured enumeration, geometry, relation, or derived-code guard was exceeded."""


@dataclass
class InvariantAutResult:
    generators: list
    order: int
    n: int
    dim: int
    method: str
    requested_method: str
    exact: bool = True
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    preprocessing_seconds: float = 0.0
    search_seconds: float = 0.0
    stabilizer_seconds: float = 0.0
    seconds: float = 0.0
    num_vertices: Optional[int] = None
    num_codewords: Optional[int] = None
    num_incidences: Optional[int] = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.exact

    @property
    def num_edges(self) -> Optional[int]:
        return self.num_incidences

    def group(self):
        return permgroup.Group(self.generators, self.n)


@dataclass(frozen=True)
class _Enumeration:
    messages: np.ndarray
    words: np.ndarray
    weights: np.ndarray


def available_methods():
    """Names accepted by :func:`automorphism_group`."""
    return METHODS


def _code_basis(generator_matrix) -> np.ndarray:
    array = np.asarray(generator_matrix)
    if array.ndim != 2:
        raise ValueError("generator_matrix must be 2-D")
    return gf2.row_basis_gf2(array)


def _inverse_gf2(matrix) -> np.ndarray:
    A = gf2.as_uint8(matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("GF(2) inverse requires a square matrix")
    size = A.shape[0]
    augmented = np.hstack([A.copy(), np.eye(size, dtype=np.uint8)])
    for column in range(size):
        candidates = np.flatnonzero(augmented[column:, column])
        if not len(candidates):
            raise np.linalg.LinAlgError("singular matrix over GF(2)")
        pivot = column + int(candidates[0])
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        for row in range(size):
            if row != column and augmented[row, column]:
                augmented[row] ^= augmented[column]
    return np.ascontiguousarray(augmented[:, size:])


def orthogonal_projector(generator_matrix) -> Optional[np.ndarray]:
    """Binary orthogonal projector onto an LCD code, or ``None`` when the hull is nontrivial.

    For a row basis ``B``, ``Q=B.T (B B.T)^-1 B``.  A coordinate permutation preserves the
    code exactly iff it preserves ``Q``.
    """
    B = _code_basis(generator_matrix)
    n = B.shape[1]
    if not len(B):
        return np.zeros((n, n), dtype=np.uint8)
    gram = (B @ B.T) % 2
    if gf2.rank_gf2(gram) != len(B):
        return None
    inverse = _inverse_gf2(gram)
    return np.ascontiguousarray((B.T @ inverse @ B) % 2, dtype=np.uint8)


def _hull_basis(C: np.ndarray) -> np.ndarray:
    if not len(C):
        return np.zeros((0, C.shape[1]), dtype=np.uint8)
    gram = (C @ C.T) % 2
    messages = gf2.nullspace_basis_gf2(gram)
    return gf2.row_basis_gf2((messages @ C) % 2)


def _hull_dimension(C: np.ndarray) -> int:
    return int(len(C) - gf2.rank_gf2((C @ C.T) % 2))


def _schur_power(C: np.ndarray, degree: int = 2, *,
                 max_products: Optional[int] = None) -> np.ndarray:
    """Span of coordinatewise products of up to ``degree`` basis rows (repetitions allowed)."""
    C = _code_basis(C)
    if degree < 1:
        raise ValueError("Schur degree must be positive")
    if not len(C):
        return C.copy()
    product_count = math.comb(len(C) + degree - 1, degree)
    if max_products is not None and product_count > max_products:
        raise InvariantLimitExceeded(
            f"Schur power needs {product_count} products > "
            f"max_schur_products={max_products}")
    products = []
    for indices in itertools.combinations_with_replacement(range(len(C)), degree):
        row = np.ones(C.shape[1], dtype=np.uint8)
        for index in indices:
            row &= C[index]
        products.append(row)
    return gf2.row_basis_gf2(np.vstack(products)) if products else C.copy()


def _schur_product(C: np.ndarray, D: np.ndarray, *,
                   max_products: Optional[int] = None) -> np.ndarray:
    """Span of all coordinatewise products ``c*d`` for ``c in C`` and ``d in D``."""
    C = _code_basis(C)
    D = _code_basis(D)
    if C.shape[1] != D.shape[1]:
        raise ValueError("Schur-product codes must have the same length")
    product_count = len(C) * len(D)
    if max_products is not None and product_count > max_products:
        raise InvariantLimitExceeded(
            f"Schur product needs {product_count} products > "
            f"max_schur_products={max_products}")
    if not product_count:
        return np.zeros((0, C.shape[1]), dtype=np.uint8)
    products = np.asarray([left & right for left in C for right in D], dtype=np.uint8)
    return gf2.row_basis_gf2(products)


def _conductor(C: np.ndarray, D: np.ndarray, *,
               max_products: Optional[int] = None) -> np.ndarray:
    """Return ``Cond(C,D) = {x : x*C subseteq D}`` as a binary linear code.

    The identity ``Cond(C,D) = (C * D^perp)^perp`` turns the universal containment
    condition into two GF(2) nullspaces and one guarded Schur product.
    """
    C = _code_basis(C)
    D = _code_basis(D)
    if C.shape[1] != D.shape[1]:
        raise ValueError("conductor codes must have the same length")
    dual = gf2.nullspace_basis_gf2(D)
    constraints = _schur_product(C, dual, max_products=max_products)
    return gf2.row_basis_gf2(gf2.nullspace_basis_gf2(constraints))


def _intersection_with_dual(C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Basis of ``C intersect D^perp``; both constructions are coordinate-characteristic."""
    C = _code_basis(C)
    D = _code_basis(D)
    if not len(C) or not len(D):
        return C.copy()
    messages = gf2.nullspace_basis_gf2(((C @ D.T) % 2).T)
    return gf2.row_basis_gf2((messages @ C) % 2)


def _intersection_codes(C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Basis of the rowspace intersection ``C intersect D``."""
    C = _code_basis(C)
    D = _code_basis(D)
    if C.shape[1] != D.shape[1]:
        raise ValueError("intersection codes must have the same length")
    return _intersection_with_dual(C, gf2.nullspace_basis_gf2(D))


def _enumerate_code(C: np.ndarray, *, max_dim: int, max_enumerated: int) -> _Enumeration:
    C = _code_basis(C)
    k, n = C.shape
    if k >= 63 or k > max_dim:
        raise InvariantLimitExceeded(f"dimension {k} exceeds enumerator limit {min(max_dim, 62)}")
    total = 1 << k
    if total > max_enumerated:
        raise InvariantLimitExceeded(
            f"enumeration needs {total} messages > max_enumerated={max_enumerated}")
    if k == 0:
        return _Enumeration(np.zeros((0, 0), np.uint8), np.zeros((0, n), np.uint8),
                            np.zeros(0, np.int64))
    indices = np.arange(1, total, dtype=np.uint64)
    shifts = np.arange(k, dtype=np.uint64)
    messages = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    words = np.ascontiguousarray((messages @ C) % 2, dtype=np.uint8)
    weights = words.sum(axis=1, dtype=np.int64)
    return _Enumeration(messages, words, weights)


def _minimum_weight_prefix(C: np.ndarray, enumeration: _Enumeration):
    """Complete ascending exact-weight cells through the first spanning weight."""
    cells = []
    stacked = np.zeros((0, C.shape[1]), dtype=np.uint8)
    for weight in sorted(set(int(x) for x in enumeration.weights)):
        indices = np.flatnonzero(enumeration.weights == weight)
        rows = np.ascontiguousarray(enumeration.words[indices])
        messages = np.ascontiguousarray(enumeration.messages[indices])
        cells.append({"label": ("weight", weight), "weight": weight,
                      "rows": rows, "messages": messages})
        stacked = gf2.row_basis_gf2(np.vstack([stacked, rows]))
        if len(stacked) == len(C):
            return cells
    if len(C):
        raise AssertionError("nonzero codeword enumeration did not span its code")
    return []


def _cells_cost(cells) -> tuple[int, int]:
    words = sum(len(cell["rows"]) for cell in cells)
    edges = sum(sum(min(int(row.sum()), row.size - int(row.sum())) for row in cell["rows"])
                for cell in cells)
    return words, edges


def _cells_to_class_groups(cells):
    return [[(cell["label"], cell["rows"]) for cell in cells]]


def _choose_cell_cover(cells, dimension: int, *, max_subsets: int = 1_000_000):
    """Choose complete invariant cells spanning the code; exact subset search then greedy."""
    candidates = []
    for cell in cells:
        basis = gf2.row_basis_gf2(cell["messages"])
        if len(basis):
            item = dict(cell)
            item["basis"] = basis
            candidates.append(item)
    candidates.sort(key=lambda cell: (len(cell["rows"]), int(cell["rows"].sum()),
                                       repr(cell["label"])))
    count = len(candidates)
    best = None
    if count < 63 and (1 << count) <= max_subsets:
        for size in range(1, count + 1):
            for indices in itertools.combinations(range(count), size):
                basis = np.vstack([candidates[index]["basis"] for index in indices])
                if gf2.rank_gf2(basis) != dimension:
                    continue
                chosen = [candidates[index] for index in indices]
                score = (*_cells_cost(chosen), size, indices)
                if best is None or score < best[0]:
                    best = (score, chosen)
        return None if best is None else best[1]
    chosen = []
    basis = np.zeros((0, dimension), dtype=np.uint8)
    remaining = list(candidates)
    while len(basis) < dimension:
        current_rank = len(basis)
        options = []
        for cell in remaining:
            merged = gf2.row_basis_gf2(np.vstack([basis, cell["basis"]]))
            gain = len(merged) - current_rank
            if gain:
                options.append((-gain / len(cell["rows"]), len(cell["rows"]),
                                int(cell["rows"].sum()), repr(cell["label"]), cell, merged))
        if not options:
            return None
        *_score, cell, basis = min(options, key=lambda item: item[:4])
        chosen.append(cell)
        remaining.remove(cell)
    for cell in list(reversed(chosen)):
        trial = [other for other in chosen if other is not cell]
        if trial and gf2.rank_gf2(np.vstack([other["basis"] for other in trial])) == dimension:
            chosen = trial
    return chosen


def _column_masks(C: np.ndarray):
    masks = []
    for column in range(C.shape[1]):
        value = 0
        for row in range(C.shape[0]):
            value |= int(C[row, column]) << row
        masks.append(value)
    return masks


def _mask_vector(mask: int, dimension: int) -> np.ndarray:
    return np.fromiter(((mask >> bit) & 1 for bit in range(dimension)),
                       dtype=np.uint8, count=dimension)


def _mask_rank(masks, dimension: int) -> int:
    values = list(masks)
    if not values:
        return 0
    matrix = np.column_stack([_mask_vector(mask, dimension) for mask in values])
    return gf2.rank_gf2(matrix)


def _matroid_components(C: np.ndarray):
    """Circuit components of the binary column matroid, from fundamental circuits."""
    C = _code_basis(C)
    n = C.shape[1]
    reduced, pivots = gf2.rref_gf2(C)
    pivot_set = set(pivots)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for column in range(n):
        if column in pivot_set:
            continue
        circuit = [column] + [pivots[row] for row in np.flatnonzero(reduced[:, column])]
        for point in circuit[1:]:
            union(circuit[0], point)
    groups = defaultdict(list)
    for point in range(n):
        groups[find(point)].append(point)
    return sorted(groups.values(), key=lambda cell: cell[0])


def _component_labels(C: np.ndarray):
    masks = _column_masks(C)
    labels = [None] * C.shape[1]
    components = _matroid_components(C)
    for component in components:
        multiplicities = Counter(masks[index] for index in component)
        local = C[:, component]
        fingerprint = (
            len(component), gf2.rank_gf2(local), len(multiplicities),
            tuple(sorted(multiplicities.values())),
            sum(mask == 0 for mask in (masks[index] for index in component)),
        )
        for point in component:
            labels[point] = ("component", fingerprint, multiplicities[masks[point]])
    return labels, components


def _support_fingerprint(C: np.ndarray, row: np.ndarray, hull: np.ndarray,
                         schur: np.ndarray):
    support = np.flatnonzero(row)
    complement = np.flatnonzero(1 - row)
    shortening_dimension = len(C) - gf2.rank_gf2(C[:, complement])
    puncture_rank = gf2.rank_gf2(C[:, support])
    hull_shortening = len(hull) - gf2.rank_gf2(hull[:, complement]) if len(hull) else 0
    hull_puncture = gf2.rank_gf2(hull[:, support]) if len(hull) else 0
    schur_puncture = gf2.rank_gf2(schur[:, support]) if len(schur) else 0
    in_hull = int(not np.any((row @ C.T) % 2))
    return (shortening_dimension, puncture_rank, in_hull, hull_shortening,
            hull_puncture, schur_puncture)


def _fingerprint_cells(C: np.ndarray, prefix, *, max_fingerprint_words: int):
    if len(C) == 0:
        return []
    total = sum(len(cell["rows"]) for cell in prefix)
    if total > max_fingerprint_words:
        raise InvariantLimitExceeded(
            f"fingerprinting needs {total} words > max_fingerprint_words={max_fingerprint_words}")
    hull = _hull_basis(C)
    schur = _schur_power(C, 2)
    buckets = defaultdict(lambda: {"rows": [], "messages": []})
    for cell in prefix:
        for row, message in zip(cell["rows"], cell["messages"]):
            feature = (cell["weight"],) + _support_fingerprint(C, row, hull, schur)
            buckets[feature]["rows"].append(row)
            buckets[feature]["messages"].append(message)
    cells = []
    for feature in sorted(buckets):
        item = buckets[feature]
        cells.append({
            "label": ("fingerprint",) + feature,
            "weight": feature[0],
            "rows": np.ascontiguousarray(np.vstack(item["rows"]), dtype=np.uint8),
            "messages": np.ascontiguousarray(np.vstack(item["messages"]), dtype=np.uint8),
        })
    chosen = _choose_cell_cover(cells, len(C))
    if chosen is None:
        raise AssertionError("complete invariant fingerprint cells failed to span")
    return chosen


def _make_result(group, C: np.ndarray, requested: str, method: str, started: float, *,
                 preprocessing_seconds=0.0, search_seconds=0.0, num_vertices=None,
                 num_codewords=None, num_incidences=None, used_fallback=False,
                 fallback_reason=None, diagnostics=None, stabilizer_seconds=0.0):
    generators = group.reduced_generators()
    return InvariantAutResult(
        generators=generators,
        order=group.order(),
        n=C.shape[1],
        dim=len(C),
        method=method,
        requested_method=requested,
        exact=True,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        preprocessing_seconds=float(preprocessing_seconds),
        search_seconds=float(search_seconds),
        stabilizer_seconds=float(stabilizer_seconds),
        seconds=time.perf_counter() - started,
        num_vertices=num_vertices,
        num_codewords=num_codewords,
        num_incidences=num_incidences,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def _group_preserves_code(group, C: np.ndarray) -> bool:
    return all(gf2.preserves_rowspace(C, generator) for generator in group.gens())


def rowspace_stabilizer(candidate_group, generator_matrix, *, max_orbit: int = 20_000):
    """Exact stabilizer of a code inside a permutation overgroup, by a Schreier orbit.

    The candidate group must contain ``Aut(C)``.  Orbit states are canonical RREF rowspaces.
    If more than ``max_orbit`` distinct images are encountered, the guarded routine raises
    :class:`InvariantLimitExceeded`; it never returns a partial stabilizer.

    Returns ``(stabilizer_group, orbit_size)``.  The order identity
    ``|Stab(C)| * |Orb(C)| = |candidate_group|`` is checked independently.
    """
    C = _code_basis(generator_matrix)
    n = C.shape[1]
    if isinstance(max_orbit, bool) or int(max_orbit) < 1:
        raise ValueError("max_orbit must be a positive integer")
    if not isinstance(candidate_group, permgroup.Group):
        candidate_group = permgroup.Group(candidate_group, n)
    if candidate_group.degree != n:
        raise ValueError("candidate group and code act on different numbers of coordinates")

    def key(basis):
        reduced = gf2.row_basis_gf2(basis)
        return reduced.tobytes(), reduced

    identity = list(range(n))
    root_key, root = key(C)
    states = [root]
    state_keys = [root_key]
    index = {root_key: 0}
    transporters = [identity]
    generators = candidate_group.gens()
    symmetric_generators = generators + [permgroup.inv(g) for g in generators]
    stabilizer_generators = []
    cursor = 0
    while cursor < len(states):
        state = states[cursor]
        transporter = transporters[cursor]
        for generator in symmetric_generators:
            image_key, image_basis = key(state[:, generator])
            path = permgroup.compose(transporter, generator)
            target = index.get(image_key)
            if target is None:
                if len(states) >= max_orbit:
                    raise InvariantLimitExceeded(
                        f"rowspace orbit exceeded max_stabilizer_orbit={max_orbit}")
                target = len(states)
                index[image_key] = target
                state_keys.append(image_key)
                states.append(image_basis)
                transporters.append(path)
            else:
                schreier = permgroup.compose(path, permgroup.inv(transporters[target]))
                if schreier != identity:
                    stabilizer_generators.append(schreier)
        cursor += 1
    stabilizer = permgroup.Group(stabilizer_generators, n)
    candidate_order = candidate_group.order()
    if stabilizer.order() * len(states) != candidate_order:
        raise AssertionError("rowspace Schreier stabilizer failed orbit-stabilizer order check")
    if not _group_preserves_code(stabilizer, C):
        raise AssertionError("rowspace Schreier generator failed code preservation")
    return stabilizer, len(states)


def _native_fallback(C: np.ndarray, requested: str, reason: str, started: float, *,
                     max_dim: int, preprocessing_seconds: float, fallback: bool,
                     diagnostics=None, stabilizer_seconds: float = 0.0):
    if not fallback:
        raise InvariantLimitExceeded(reason)
    from . import leon
    effective, _n, _rank, effective_dim = gf2.dual_basis(C)
    before = time.perf_counter()
    base = leon.automorphism_group(effective, max_dim=max_dim, spanning_set="minweight")
    wall = time.perf_counter() - before
    group = base.group()
    info = {} if diagnostics is None else dict(diagnostics)
    info.update({"fallback_effective_dim": effective_dim,
                 "fallback_enumeration_seconds": base.enumeration_seconds,
                 "fallback_search_seconds": base.search_seconds})
    return _make_result(
        group, C, requested, "leon-minweight", started,
        preprocessing_seconds=preprocessing_seconds + (base.enumeration_seconds or 0.0),
        search_seconds=base.search_seconds if base.search_seconds is not None else wall,
        num_vertices=C.shape[1] + base.num_codewords,
        num_codewords=base.num_codewords,
        num_incidences=base.num_incidences,
        used_fallback=True,
        fallback_reason=reason,
        diagnostics=info,
        stabilizer_seconds=stabilizer_seconds,
    )


def _relation_tables(matrices, vertex_extra=None, edge_extra=None):
    matrices = [gf2.as_uint8(matrix) for matrix in matrices]
    if matrices:
        n = matrices[0].shape[0]
        if any(matrix.shape != (n, n) for matrix in matrices):
            raise ValueError("all relation matrices must be n x n")
    elif vertex_extra is not None:
        n = len(vertex_extra)
    elif edge_extra is not None:
        n = len(edge_extra)
    else:
        raise ValueError("at least one relation layer is required")
    vertex_extra = [()] * n if vertex_extra is None else list(vertex_extra)
    if len(vertex_extra) != n:
        raise ValueError("vertex relation labels have the wrong length")
    if edge_extra is None:
        edge_extra = [[()] * n for _ in range(n)]
    vertex_labels = []
    edges = [[None] * n for _ in range(n)]
    for i in range(n):
        extra = vertex_extra[i] if isinstance(vertex_extra[i], tuple) else (vertex_extra[i],)
        vertex_labels.append(tuple(int(matrix[i, i]) for matrix in matrices) + extra)
        for j in range(n):
            value = edge_extra[i][j]
            extra_edge = value if isinstance(value, tuple) else (value,)
            edges[i][j] = tuple(int(matrix[i, j]) for matrix in matrices) + extra_edge
    return vertex_labels, edges


def _solve_relation(C: np.ndarray, requested: str, started: float, vertex_labels, edge_labels,
                    *, preprocessing_seconds: float, timeout, nauty_timeout, traces_timeout,
                    max_dim: int, fallback: bool, method: str, diagnostics=None,
                    max_stabilizer_orbit: int = 0):
    before = time.perf_counter()
    try:
        group, vertices = graphaut.relation_group(
            vertex_labels, edge_labels, timeout=timeout, nauty_timeout=nauty_timeout,
            traces_timeout=traces_timeout)
        order = group.order()
    except (RuntimeError, OSError, FileNotFoundError, subprocess.SubprocessError) as exc:
        return _native_fallback(
            C, requested, f"{method} graph solve failed: {exc}", started, max_dim=max_dim,
            preprocessing_seconds=preprocessing_seconds, fallback=fallback,
            diagnostics=diagnostics)
    search_seconds = time.perf_counter() - before
    if not _group_preserves_code(group, C):
        info = {} if diagnostics is None else dict(diagnostics)
        info.update({"overgroup_order": order, "relation_vertices": vertices,
                     "relation_exact_hit": False})
        stabilizer_seconds = 0.0
        if max_stabilizer_orbit:
            stabilizer_started = time.perf_counter()
            try:
                stabilizer, orbit_size = rowspace_stabilizer(
                    group, C, max_orbit=max_stabilizer_orbit)
                stabilizer_seconds = time.perf_counter() - stabilizer_started
                info.update({"rowspace_orbit": orbit_size,
                             "relation_exactified": True})
                return _make_result(
                    stabilizer, C, requested, method + "+rowspace-stabilizer", started,
                    preprocessing_seconds=preprocessing_seconds,
                    search_seconds=search_seconds,
                    stabilizer_seconds=stabilizer_seconds,
                    num_vertices=vertices,
                    diagnostics=info)
            except InvariantLimitExceeded as exc:
                stabilizer_seconds = time.perf_counter() - stabilizer_started
                info["stabilizer_guard"] = str(exc)
        reason = f"{method} relation is a strict overgroup"
        if info.get("stabilizer_guard"):
            reason += f"; {info['stabilizer_guard']}"
        return _native_fallback(
            C, requested, reason, started,
            max_dim=max_dim, preprocessing_seconds=preprocessing_seconds + search_seconds,
            fallback=fallback, diagnostics=info,
            stabilizer_seconds=stabilizer_seconds)
    info = {} if diagnostics is None else dict(diagnostics)
    info["relation_exact_hit"] = True
    pair_values = [edge_labels[i][j] for i in range(len(edge_labels))
                   for j in range(i + 1, len(edge_labels))]
    default_pair = Counter(pair_values).most_common(1)[0][0] if pair_values else None
    relation_gadgets = sum(value != default_pair for value in pair_values)
    return _make_result(
        group, C, requested, method, started,
        preprocessing_seconds=preprocessing_seconds,
        search_seconds=search_seconds,
        num_vertices=vertices,
        num_incidences=relation_gadgets,
        diagnostics=info,
    )


def _geometry_group(C: np.ndarray, *, max_rank: int, max_candidates: int):
    """Exact stabilizer of the represented projective multiset, including parallel kernels."""
    effective, _n, _rank, _effective_dim = gf2.dual_basis(C)
    B = _code_basis(effective)
    k, n = B.shape
    if k > max_rank:
        raise InvariantLimitExceeded(
            f"effective projective rank {k} exceeds max_geometry_rank={max_rank}")
    if k == 0:
        group = permgroup.symmetric_group(n)
        return group, {"effective_rank": 0, "column_types": 1,
                       "linear_candidates": 1, "linear_stabilizers": 1}
    masks = _column_masks(B)
    positions = defaultdict(list)
    for coordinate, mask in enumerate(masks):
        positions[mask].append(coordinate)
    present = sorted(mask for mask in positions if mask)
    if _mask_rank(present, k) != k:
        raise AssertionError("full-row-rank generator columns failed to span message space")
    by_multiplicity = defaultdict(list)
    for mask in present:
        by_multiplicity[len(positions[mask])].append(mask)
    ordered = sorted(present, key=lambda mask: (len(by_multiplicity[len(positions[mask])]),
                                                -len(positions[mask]), mask))
    anchor = []
    for mask in ordered:
        if _mask_rank(anchor + [mask], k) > len(anchor):
            anchor.append(mask)
            if len(anchor) == k:
                break
    V = np.column_stack([_mask_vector(mask, k) for mask in anchor])
    V_inverse = _inverse_gf2(V)
    generators = []
    candidate_count = 0
    stabilizer_count = 0

    def mapped_mask(A, mask):
        vector = (A @ _mask_vector(mask, k)) % 2
        return sum(int(vector[bit]) << bit for bit in range(k))

    def inspect(images):
        nonlocal candidate_count, stabilizer_count
        candidate_count += 1
        if candidate_count > max_candidates:
            raise InvariantLimitExceeded(
                f"projective search exceeded max_geometry_candidates={max_candidates}")
        U = np.column_stack([_mask_vector(mask, k) for mask in images])
        A = (U @ V_inverse) % 2
        image_by_mask = {mask: mapped_mask(A, mask) for mask in positions}
        if any(len(positions[mask]) != len(positions.get(image, ()))
               for mask, image in image_by_mask.items()):
            return
        stabilizer_count += 1
        permutation = list(range(n))
        for source_mask, source_points in positions.items():
            target_points = positions[image_by_mask[source_mask]]
            for source, target in zip(source_points, target_points):
                permutation[source] = target
        generators.append(permutation)

    def recurse(images):
        depth = len(images)
        if depth == k:
            inspect(images)
            return
        multiplicity = len(positions[anchor[depth]])
        for image in by_multiplicity[multiplicity]:
            if image in images:
                continue
            if _mask_rank(images + [image], k) != depth + 1:
                continue
            recurse(images + [image])

    recurse([])
    # Every bijection inside one equal-column class is an automorphism; adjacent swaps generate
    # the full symmetric kernel without factorial enumeration.
    for points in positions.values():
        for index in range(len(points) - 1):
            transposition = list(range(n))
            a, b = points[index:index + 2]
            transposition[a], transposition[b] = b, a
            generators.append(transposition)
    group = permgroup.Group(generators, n)
    if not _group_preserves_code(group, C):
        raise AssertionError("projective-column stabilizer failed rowspace verification")
    return group, {"effective_rank": k, "column_types": len(positions),
                   "linear_candidates": candidate_count,
                   "linear_stabilizers": stabilizer_count}


def _incidence_result(C: np.ndarray, requested: str, started: float, cells, *,
                      coordinate_labels=None, preprocessing_seconds=0.0, timeout=None,
                      nauty_timeout=5.0, traces_timeout=None, method="compressed-incidence",
                      used_fallback=False, fallback_reason=None, diagnostics=None,
                      stabilizer_seconds: float = 0.0):
    words, edges = _cells_cost(cells)
    before = time.perf_counter()
    group, vertices, twin_types, quotient_edges = graphaut.compressed_incidence_group(
        C.shape[1], _cells_to_class_groups(cells), coordinate_labels=coordinate_labels,
        timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout)
    order = group.order()
    search_seconds = time.perf_counter() - before
    if not _group_preserves_code(group, C):
        raise AssertionError("spanning invariant incidence did not preserve its code")
    info = {} if diagnostics is None else dict(diagnostics)
    info.update({"twin_types": twin_types, "uncompressed_vertices": C.shape[1] + words,
                 "uncompressed_incidences": edges, "incidence_order": order})
    return _make_result(
        group, C, requested, method, started,
        preprocessing_seconds=preprocessing_seconds,
        search_seconds=search_seconds,
        num_vertices=vertices,
        num_codewords=words,
        num_incidences=quotient_edges,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        diagnostics=info,
        stabilizer_seconds=stabilizer_seconds,
    )


def _pair_moment_tables(cells, n: int):
    vertex = [[] for _ in range(n)]
    edge = [[[] for _ in range(n)] for _ in range(n)]
    for cell in cells:
        rows = cell["rows"].astype(np.int64, copy=False)
        singles = rows.sum(axis=0, dtype=np.int64)
        pairs = rows.T @ rows
        for i in range(n):
            vertex[i].append(int(singles[i]))
            for j in range(n):
                edge[i][j].append(int(pairs[i, j]))
    return [tuple(values) for values in vertex], \
        [[tuple(values) for values in row] for row in edge]


def _triple_moment_group(cells, n: int, vertex, edges, *, max_triples: int, timeout,
                         nauty_timeout, traces_timeout):
    triples = math.comb(n, 3) if n >= 3 else 0
    if triples > max_triples:
        raise InvariantLimitExceeded(
            f"triple moments need {triples} triples > max_moment_triples={max_triples}")
    pair_by_label = defaultdict(list)
    pair_counts = Counter(edges[i][j] for i in range(n) for j in range(i + 1, n))
    pair_default = pair_counts.most_common(1)[0][0] if pair_counts else None
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] != pair_default:
                pair_by_label[edges[i][j]].append((i, j))
    triple_values = []
    for support in itertools.combinations(range(n), 3):
        label = tuple(int(np.all(cell["rows"][:, support], axis=1).sum()) for cell in cells)
        triple_values.append((support, label))
    triple_counts = Counter(label for _support, label in triple_values)
    triple_default = triple_counts.most_common(1)[0][0] if triple_counts else None
    triple_by_label = defaultdict(list)
    for support, label in triple_values:
        if label != triple_default:
            triple_by_label[label].append(support)
    relations = [(('pair', label), supports) for label, supports in pair_by_label.items()]
    relations += [(('triple', label), supports) for label, supports in triple_by_label.items()]
    group, vertices = graphaut.hypergraph_group(
        n, vertex, relations, timeout=timeout, nauty_timeout=nauty_timeout,
        traces_timeout=traces_timeout)
    return group, vertices, {"moment_triples": triples,
                             "pair_gadgets": sum(map(len, pair_by_label.values())),
                             "triple_gadgets": sum(map(len, triple_by_label.values()))}


def _minimal_cells(C: np.ndarray, prefix, *, require_spanning: bool = True,
                   label_prefix: str = "cocircuit"):
    selected = []
    for cell in prefix:
        rows = []
        messages = []
        for row, message in zip(cell["rows"], cell["messages"]):
            complement = np.flatnonzero(1 - row)
            if gf2.rank_gf2(C[:, complement]) == len(C) - 1:
                rows.append(row)
                messages.append(message)
        if rows:
            selected.append({
                "label": (label_prefix, cell["weight"]),
                "weight": cell["weight"],
                "rows": np.ascontiguousarray(np.vstack(rows), dtype=np.uint8),
                "messages": np.ascontiguousarray(np.vstack(messages), dtype=np.uint8),
            })
    if (require_spanning and len(C) and
            (not selected or
             gf2.rank_gf2(np.vstack([cell["rows"] for cell in selected])) != len(C))):
        raise AssertionError("support-minimal prefix failed to span")
    return selected


def _enumerated_cells(C: np.ndarray, strategy: str, *, max_dim: int, max_enumerated: int,
                      max_fingerprint_words: int):
    enumeration = _enumerate_code(C, max_dim=max_dim, max_enumerated=max_enumerated)
    prefix = _minimum_weight_prefix(C, enumeration)
    if strategy == "prefix":
        return prefix
    if strategy == "minimal":
        return _minimal_cells(C, prefix)
    if strategy == "fingerprint":
        return _fingerprint_cells(C, prefix,
                                  max_fingerprint_words=max_fingerprint_words)
    raise ValueError(f"unknown incidence strategy {strategy!r}")


def _lcd_method(C: np.ndarray, requested: str, started: float, *, max_dim: int,
                timeout, nauty_timeout, traces_timeout, fallback: bool,
                max_stabilizer_orbit: int, **_kwargs):
    before = time.perf_counter()
    projector = orthogonal_projector(C)
    preprocessing = time.perf_counter() - before
    if projector is None:
        return _native_fallback(
            C, requested, f"C has nontrivial hull dimension {_hull_dimension(C)}", started,
            max_dim=max_dim, preprocessing_seconds=preprocessing, fallback=fallback,
            diagnostics={"hull_dimension": _hull_dimension(C)})
    labels, edges = _relation_tables([projector])
    return _solve_relation(
        C, requested, started, labels, edges, preprocessing_seconds=preprocessing,
        timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
        max_dim=max_dim, fallback=fallback, method="lcd-projector",
        diagnostics={"hull_dimension": 0, "projector_rank": len(C)},
        max_stabilizer_orbit=max_stabilizer_orbit)


def _geometry_method(C: np.ndarray, requested: str, started: float, *, max_dim: int,
                     max_geometry_rank: int, max_geometry_candidates: int, fallback: bool,
                     **_kwargs):
    before = time.perf_counter()
    try:
        group, diagnostics = _geometry_group(
            C, max_rank=max_geometry_rank, max_candidates=max_geometry_candidates)
        group.order()
    except InvariantLimitExceeded as exc:
        return _native_fallback(
            C, requested, str(exc), started, max_dim=max_dim,
            preprocessing_seconds=time.perf_counter() - before, fallback=fallback)
    elapsed = time.perf_counter() - before
    return _make_result(
        group, C, requested, "projective-geometry", started,
        preprocessing_seconds=elapsed,
        search_seconds=0.0,
        num_vertices=diagnostics["column_types"],
        diagnostics=diagnostics)


def _incidence_method(C: np.ndarray, requested: str, started: float, strategy: str, *,
                      max_dim: int, max_enumerated: int, max_fingerprint_words: int,
                      timeout, nauty_timeout, traces_timeout, fallback: bool,
                      component_labels: bool = False, method=None, **_kwargs):
    before = time.perf_counter()
    try:
        cells = _enumerated_cells(
            C, strategy, max_dim=max_dim, max_enumerated=max_enumerated,
            max_fingerprint_words=max_fingerprint_words)
    except InvariantLimitExceeded as exc:
        return _native_fallback(
            C, requested, str(exc), started, max_dim=max_dim,
            preprocessing_seconds=time.perf_counter() - before, fallback=fallback)
    labels = None
    components = None
    if component_labels:
        labels, components = _component_labels(C)
    preprocessing = time.perf_counter() - before
    diagnostics = {"selector": strategy}
    if components is not None:
        diagnostics.update({"matroid_components": len(components),
                            "component_sizes": [len(cell) for cell in components]})
    try:
        return _incidence_result(
            C, requested, started, cells, coordinate_labels=labels,
            preprocessing_seconds=preprocessing, timeout=timeout,
            nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
            method=method or f"{strategy}-compressed-incidence", diagnostics=diagnostics)
    except (RuntimeError, OSError, FileNotFoundError, subprocess.SubprocessError) as exc:
        return _native_fallback(
            C, requested, f"compressed incidence graph solve failed: {exc}", started,
            max_dim=max_dim, preprocessing_seconds=preprocessing, fallback=fallback,
            diagnostics=diagnostics)


def _moments_method(C: np.ndarray, requested: str, started: float, *, max_dim: int,
                    max_enumerated: int, max_fingerprint_words: int, timeout,
                    nauty_timeout, traces_timeout, max_moment_triples: int,
                    fallback: bool, max_stabilizer_orbit: int, **_kwargs):
    before = time.perf_counter()
    try:
        cells = _enumerated_cells(
            C, "fingerprint", max_dim=max_dim, max_enumerated=max_enumerated,
            max_fingerprint_words=max_fingerprint_words)
    except InvariantLimitExceeded as exc:
        return _native_fallback(
            C, requested, str(exc), started, max_dim=max_dim,
            preprocessing_seconds=time.perf_counter() - before, fallback=fallback)
    vertex, edges = _pair_moment_tables(cells, C.shape[1])
    preprocessing = time.perf_counter() - before
    try:
        graph_started = time.perf_counter()
        group, vertices = graphaut.relation_group(
            vertex, edges, timeout=timeout, nauty_timeout=nauty_timeout,
            traces_timeout=traces_timeout)
        overgroup_order = group.order()
        relation_search = time.perf_counter() - graph_started
    except (RuntimeError, OSError, FileNotFoundError, subprocess.SubprocessError) as exc:
        if not fallback:
            raise InvariantLimitExceeded(f"pair-moment graph solve failed: {exc}") from exc
        return _incidence_result(
            C, requested, started, cells, preprocessing_seconds=preprocessing,
            timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
            method="fingerprint-incidence", used_fallback=True,
            fallback_reason=f"pair-moment graph solve failed: {exc}",
            diagnostics={"relation_exact_hit": False})
    if _group_preserves_code(group, C):
        return _make_result(
            group, C, requested, "pair-moments", started,
            preprocessing_seconds=preprocessing, search_seconds=relation_search,
            num_vertices=vertices,
            diagnostics={"relation_exact_hit": True, "moment_cells": len(cells)})
    pair_overgroup_order = overgroup_order
    exactifier_group = group
    exactifier_vertices = vertices
    exactifier_method = "pair-moments"
    exactifier_diagnostics = {}
    # Pair moments collapse on designs.  A guarded third moment is the next invariant in the
    # same hierarchy; it remains an overgroup until generator verification succeeds.
    try:
        triple_started = time.perf_counter()
        triple_group, triple_vertices, triple_diagnostics = _triple_moment_group(
            cells, C.shape[1], vertex, edges, max_triples=max_moment_triples,
            timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout)
        triple_order = triple_group.order()
        triple_search = time.perf_counter() - triple_started
        if _group_preserves_code(triple_group, C):
            return _make_result(
                triple_group, C, requested, "pair-triple-moments", started,
                preprocessing_seconds=preprocessing,
                search_seconds=relation_search + triple_search,
                num_vertices=triple_vertices,
                diagnostics={"relation_exact_hit": True, "moment_cells": len(cells),
                             "pair_overgroup_order": pair_overgroup_order,
                             **triple_diagnostics})
        triple_reason = f"triple moments form strict overgroup of order {triple_order}"
        exactifier_group = triple_group
        exactifier_vertices = triple_vertices
        exactifier_method = "pair-triple-moments"
        exactifier_diagnostics = triple_diagnostics
        relation_search += triple_search
    except (InvariantLimitExceeded, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        triple_reason = str(exc)
    stabilizer_seconds = 0.0
    if max_stabilizer_orbit:
        stabilizer_started = time.perf_counter()
        try:
            stabilizer, orbit_size = rowspace_stabilizer(
                exactifier_group, C, max_orbit=max_stabilizer_orbit)
            stabilizer_seconds = time.perf_counter() - stabilizer_started
            return _make_result(
                stabilizer, C, requested,
                exactifier_method + "+rowspace-stabilizer", started,
                preprocessing_seconds=preprocessing,
                search_seconds=relation_search,
                stabilizer_seconds=stabilizer_seconds,
                num_vertices=exactifier_vertices,
                diagnostics={"relation_exact_hit": False,
                             "relation_exactified": True,
                             "rowspace_orbit": orbit_size,
                             "pair_overgroup_order": pair_overgroup_order,
                             "moment_cells": len(cells),
                             **exactifier_diagnostics})
        except InvariantLimitExceeded as exc:
            stabilizer_seconds = time.perf_counter() - stabilizer_started
            triple_reason += f"; stabilizer guard: {exc}"
    if not fallback:
        raise InvariantLimitExceeded(
            f"pair moments form a strict overgroup; triple attempt: {triple_reason}")
    return _incidence_result(
        C, requested, started, cells,
        preprocessing_seconds=preprocessing + relation_search,
        timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
        method="fingerprint-incidence", used_fallback=True,
        fallback_reason=f"pair moments form a strict overgroup; triple attempt: {triple_reason}",
        diagnostics={"relation_exact_hit": False, "overgroup_order": pair_overgroup_order,
                     "relation_vertices": vertices, "moment_cells": len(cells),
                     "triple_attempt": triple_reason},
        stabilizer_seconds=stabilizer_seconds)


def _hull_moment_layer(C: np.ndarray, *, max_dim: int, max_enumerated: int):
    hull = _hull_basis(C)
    enumeration = _enumerate_code(
        hull, max_dim=max_dim, max_enumerated=max_enumerated)
    cells = []
    for weight in sorted(set(int(x) for x in enumeration.weights)):
        indices = np.flatnonzero(enumeration.weights == weight)
        cells.append({"label": ("hull-weight", weight), "weight": weight,
                      "rows": np.ascontiguousarray(enumeration.words[indices]),
                      "messages": np.ascontiguousarray(enumeration.messages[indices])})
    vertex, edges = _pair_moment_tables(cells, C.shape[1])
    return vertex, edges, {"hull_dimension": len(hull), "hull_words": len(enumeration.words),
                           "hull_weight_cells": len(cells)}


def _hull_method(C: np.ndarray, requested: str, started: float, *, max_dim: int,
                 max_enumerated: int, timeout, nauty_timeout, traces_timeout,
                 fallback: bool, max_stabilizer_orbit: int, max_minor_pairs: int,
                 max_hull_dimension: int, max_hull_words: int,
                 max_schur_products: int, max_minor_hull_work: int, **_kwargs):
    before = time.perf_counter()
    component, _components = _component_labels(C)
    vertex_layers = [component]
    edge_layers = []
    diagnostics = {"support_splitting": [], "hull_dimension": _hull_dimension(C)}
    try:
        hull_vertex, hull_edges, hull_diagnostics = _hull_moment_layer(
            C, max_dim=max_dim, max_enumerated=max_enumerated)
        vertex_layers.append(hull_vertex)
        edge_layers.append(hull_edges)
        diagnostics.update(hull_diagnostics)
        diagnostics["support_splitting"].append("hull-weight-moments")
    except InvariantLimitExceeded as exc:
        diagnostics["hull_moments_skipped"] = str(exc)
    try:
        minor_vertex, minor_edges, minor_diagnostics = _minor_layer(
            C, max_minor_pairs=max_minor_pairs,
            max_hull_dimension=max_hull_dimension,
            max_hull_words=max_hull_words,
            max_schur_products=max_schur_products,
            max_minor_hull_work=max_minor_hull_work)
        vertex_layers.append(minor_vertex)
        edge_layers.append(minor_edges)
        diagnostics["minor_hull_signatures"] = minor_diagnostics
        diagnostics["support_splitting"].append("puncture/shorten-hull-enumerators")
    except InvariantLimitExceeded as exc:
        diagnostics["minor_hull_signatures_skipped"] = str(exc)
    if not edge_layers:
        edge_layers.append([[0] * C.shape[1] for _ in range(C.shape[1])])
    vertex, edges = _merge_relation_layers(C.shape[1], vertex_layers, edge_layers)
    return _solve_relation(
        C, requested, started, vertex, edges,
        preprocessing_seconds=time.perf_counter() - before,
        timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
        max_dim=max_dim, fallback=fallback, method="hull-support-splitting",
        diagnostics=diagnostics, max_stabilizer_orbit=max_stabilizer_orbit)


def _characteristic_codes(C: np.ndarray, *, max_schur_products: int):
    """Named derived codes fixed setwise by every automorphism of ``C``.

    Besides the code, hull, and square, include the square of the hull and several conductors.
    Every operation commutes with coordinate permutations, so each successful layer is
    characteristic.  Oversized conductor products are skipped individually and diagnosed.
    """
    C = _code_basis(C)
    hull = _hull_basis(C)
    dual = gf2.nullspace_basis_gf2(C)
    skipped = []
    candidates = [("code", C), ("hull", hull)]

    def add_power(name, source, degree=2):
        try:
            power = _schur_power(
                source, degree, max_products=max_schur_products)
            candidates.extend([(name, power), (name + "-hull", _hull_basis(power))])
            return power
        except InvariantLimitExceeded as exc:
            skipped.append((name, str(exc)))
            return None

    schur2 = add_power("schur2", C)
    hull_schur2 = add_power("hull-schur2", hull)
    add_power("hull-schur3", hull, 3)
    dual_schur2 = add_power("dual-schur2", dual)
    if schur2 is not None:
        candidates.append(("schur-annihilator", _intersection_with_dual(C, schur2)))
    if hull_schur2 is not None:
        candidates.append(("code-intersect-hull-schur2",
                           _intersection_codes(C, hull_schur2)))
    conductor_pairs = [
        ("conductor-code-code", C, C),
        ("conductor-dual-dual", dual, dual),
    ]
    if schur2 is not None:
        conductor_pairs.extend([
            ("conductor-code-schur2", C, schur2),
            ("conductor-schur2-code", schur2, C),
            ("conductor-schur2-schur2", schur2, schur2),
        ])
    if dual_schur2 is not None:
        conductor_pairs.append(
            ("conductor-dual-dual-schur2", dual, dual_schur2))
    if len(hull) and hull_schur2 is not None:
        conductor_pairs.extend([
            ("conductor-hull-code", hull, C),
            ("conductor-hull-hull-square", hull, hull_schur2),
            ("conductor-hull-square-hull", hull_schur2, hull),
            ("conductor-hull-square-code", hull_schur2, C),
            ("conductor-code-hull-square", C, hull_schur2),
        ])
        if schur2 is not None:
            conductor_pairs.extend([
                ("conductor-hull-schur2", hull, schur2),
                ("conductor-hull-square-schur2", hull_schur2, schur2),
            ])
    for name, source, target in conductor_pairs:
        try:
            conductor = _conductor(
                source, target, max_products=max_schur_products)
            candidates.extend([(name, conductor),
                               ("code-intersect-" + name,
                                _intersection_codes(C, conductor))])
        except InvariantLimitExceeded as exc:
            skipped.append((name, str(exc)))
    seen = set()
    answer = []
    for name, basis in candidates:
        basis = _code_basis(basis)
        key = (basis.shape, basis.tobytes())
        if key not in seen:
            seen.add(key)
            answer.append((name, basis))
    return answer, skipped


def _projector_layers(C: np.ndarray, *, max_schur_products: int):
    layers = []
    diagnostics = []
    characteristic, skipped = _characteristic_codes(
        C, max_schur_products=max_schur_products)
    for name, code in characteristic:
        projector = orthogonal_projector(code)
        saturated = len(code) in (0, C.shape[1])
        diagnostics.append((name, len(code), _hull_dimension(code),
                            projector is not None, saturated))
        if projector is not None and not saturated:
            layers.append((name, code, projector))
    return layers, diagnostics, skipped


def _schur_method(C: np.ndarray, requested: str, started: float, *, max_dim: int,
                  timeout, nauty_timeout, traces_timeout, fallback: bool,
                  max_stabilizer_orbit: int, max_schur_products: int, **_kwargs):
    before = time.perf_counter()
    if len(C) in (0, C.shape[1]):
        group = permgroup.symmetric_group(C.shape[1])
        return _make_result(
            group, C, requested, "schur-trivial-code", started,
            preprocessing_seconds=time.perf_counter() - before,
            num_vertices=C.shape[1], diagnostics={"derived_codes": []})
    try:
        layers, layer_diagnostics, skipped = _projector_layers(
            C, max_schur_products=max_schur_products)
    except InvariantLimitExceeded as exc:
        return _native_fallback(
            C, requested, str(exc), started, max_dim=max_dim,
            preprocessing_seconds=time.perf_counter() - before, fallback=fallback)
    preprocessing = time.perf_counter() - before
    if not layers:
        return _native_fallback(
            C, requested, "no nonzero characteristic LCD projector layer", started,
            max_dim=max_dim, preprocessing_seconds=preprocessing, fallback=fallback,
            diagnostics={"derived_codes": layer_diagnostics,
                         "skipped_derived_codes": skipped})
    labels, edges = _relation_tables([projector for _name, _code, projector in layers])
    return _solve_relation(
        C, requested, started, labels, edges, preprocessing_seconds=preprocessing,
        timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
        max_dim=max_dim, fallback=fallback, method="schur-projector-layers",
        diagnostics={"derived_codes": layer_diagnostics,
                     "skipped_derived_codes": skipped,
                     "projector_layers": [name for name, _code, _Q in layers]},
        max_stabilizer_orbit=max_stabilizer_orbit)


def _basis_cover(candidates, dimension: int, cost):
    """Greedy characteristic-subcode cover; each selected object remains invariant itself."""
    chosen = []
    basis = np.zeros((0, candidates[0][1].shape[1] if candidates else 0), dtype=np.uint8)
    remaining = list(candidates)
    while len(basis) < dimension:
        options = []
        for item in remaining:
            merged = gf2.row_basis_gf2(np.vstack([basis, item[1]]))
            gain = len(merged) - len(basis)
            if gain:
                options.append((-gain / max(cost(item), 1), cost(item), repr(item[0]),
                                item, merged))
        if not options:
            return None
        *_score, item, basis = min(options, key=lambda value: value[:3])
        chosen.append(item)
        remaining.remove(item)
    for item in list(reversed(chosen)):
        trial = [other for other in chosen if other is not item]
        if trial and gf2.rank_gf2(np.vstack([other[1] for other in trial])) == dimension:
            chosen = trial
    return chosen


def _residue_spans(C: np.ndarray, modulus: int, *, max_form_operations: int,
                   max_residue_indicator_terms: int):
    form = ward.ward_form(C, modulus, max_operations=max_form_operations)
    spans = []
    seen = set()
    for residue in range(form.modulus):
        certificate = ward.residue_span(
            form, residue, max_indicator_terms=max_residue_indicator_terms)
        basis = certificate.code_basis
        key = (basis.shape, basis.tobytes())
        if key in seen:
            continue
        seen.add(key)
        spans.append((residue, basis, certificate.indicator_terms))
    return form, spans


def _residue_method(C: np.ndarray, requested: str, started: float, *, modulus: int,
                    max_dim: int, max_enumerated: int, max_form_operations: int,
                    max_residue_indicator_terms: int,
                    timeout, nauty_timeout, traces_timeout, fallback: bool,
                    max_stabilizer_orbit: int, **_kwargs):
    before = time.perf_counter()
    if len(C) == 0:
        group = permgroup.symmetric_group(C.shape[1])
        return _make_result(
            group, C, requested, "residue-span-zero-code", started,
            preprocessing_seconds=time.perf_counter() - before,
            num_vertices=C.shape[1], diagnostics={"selected_residues": []})
    try:
        form, spans = _residue_spans(
            C, modulus, max_form_operations=max_form_operations,
            max_residue_indicator_terms=max_residue_indicator_terms)
    except (ward.WardLimitExceeded, MemoryError, RecursionError) as exc:
        return _native_fallback(
            C, requested, f"residue-span construction failed: {exc}", started,
            max_dim=max_dim, preprocessing_seconds=time.perf_counter() - before,
            fallback=fallback)
    # First try the genuinely compact case: LCD residue spans whose sum is C.  Each projector
    # encodes preservation of its characteristic span exactly.
    lcd = []
    for residue, basis, terms in spans:
        projector = orthogonal_projector(basis)
        if projector is not None and len(basis):
            lcd.append(((residue, terms), basis, projector))
    cover = _basis_cover(lcd, len(C), lambda item: int(item[2].sum()) + C.shape[1]) if lcd else None
    if cover:
        labels, edges = _relation_tables([item[2] for item in cover])
        preprocessing = time.perf_counter() - before
        return _solve_relation(
            C, requested, started, labels, edges, preprocessing_seconds=preprocessing,
            timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
            max_dim=max_dim, fallback=fallback, method="residue-span-projectors",
            diagnostics={"modulus": form.modulus, "form_terms": form.num_terms,
                         "span_dimensions": [(r, len(B), terms) for r, B, terms in spans],
                         "selected_residues": [item[0][0] for item in cover]},
            max_stabilizer_orbit=max_stabilizer_orbit)

    # Non-LCD spans are still useful: enumerate low-weight classes *inside the characteristic
    # subcodes*, choose spans whose sum is C, and build one exact layered incidence.
    subcodes = [(residue, basis, terms) for residue, basis, terms in spans if len(basis)]
    chosen = _basis_cover(subcodes, len(C), lambda item: 1 << min(len(item[1]), 62))
    if not chosen:
        return _native_fallback(
            C, requested, "residue spans failed to cover C", started, max_dim=max_dim,
            preprocessing_seconds=time.perf_counter() - before, fallback=fallback)
    cells = []
    try:
        for residue, basis, _terms in chosen:
            enumeration = _enumerate_code(
                basis, max_dim=max_dim, max_enumerated=max_enumerated)
            for cell in _minimum_weight_prefix(basis, enumeration):
                item = dict(cell)
                item["label"] = ("residue-span", residue, cell["weight"])
                cells.append(item)
    except InvariantLimitExceeded as exc:
        return _native_fallback(
            C, requested, f"derived residue incidence: {exc}", started, max_dim=max_dim,
            preprocessing_seconds=time.perf_counter() - before, fallback=fallback)
    preprocessing = time.perf_counter() - before
    try:
        return _incidence_result(
            C, requested, started, cells, preprocessing_seconds=preprocessing,
            timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
            method="residue-span-incidence",
            diagnostics={"modulus": form.modulus, "form_terms": form.num_terms,
                         "span_dimensions": [(r, len(B), terms) for r, B, terms in spans],
                         "selected_residues": [item[0] for item in chosen]})
    except (RuntimeError, OSError, FileNotFoundError, subprocess.SubprocessError) as exc:
        return _native_fallback(
            C, requested, f"residue-span graph solve failed: {exc}", started,
            max_dim=max_dim, preprocessing_seconds=preprocessing, fallback=fallback)


def _modular_section_layer(C: np.ndarray, modulus: int, *, max_form_operations: int,
                           max_bdd_nodes: int, max_bdd_states: Optional[int],
                           max_bdd_key_terms: int, max_bdd_count_cells: int,
                           max_modular_pairs: int):
    form = ward.ward_form(C, modulus, max_operations=max_form_operations)
    diagram = ward.WardDecisionDiagram(
        form, max_nodes=max_bdd_nodes, max_states=max_bdd_states,
        max_key_terms=max_bdd_key_terms, max_count_cells=max_bdd_count_cells)
    n = form.n
    if n * (n - 1) // 2 > max_modular_pairs:
        raise InvariantLimitExceeded(
            f"modular sections need {n * (n - 1) // 2} pairs > "
            f"max_modular_pairs={max_modular_pairs}")
    masks = _column_masks(form.generator)
    singleton_cache = {}
    vertex = []
    for mask in masks:
        table = singleton_cache.get(mask)
        if table is None:
            table = diagram.conditioned_residue_counts(
                [mask], max_cells=max_bdd_count_cells)
            singleton_cache[mask] = table
        vertex.append(tuple(int(table[residue][1]) for residue in range(form.modulus)))
    pair_cache = {}
    edges = [[None] * n for _ in range(n)]
    for i in range(n):
        edges[i][i] = ()
        for j in range(i + 1, n):
            key = tuple(sorted((masks[i], masks[j])))
            table = pair_cache.get(key)
            if table is None:
                table = diagram.conditioned_residue_counts(
                    list(key), max_cells=max_bdd_count_cells)
                pair_cache[key] = table
            label = tuple(int(table[residue][3]) for residue in range(form.modulus))
            edges[i][j] = edges[j][i] = label
    return vertex, edges, {
        "modulus": form.modulus,
        "form_terms": form.num_terms,
        "bdd_nodes": diagram.num_nodes,
        "bdd_states": diagram.num_states,
        "column_types": len(set(masks)),
        "conditioned_singletons": len(singleton_cache),
        "conditioned_pairs": len(pair_cache),
    }


def _modular_method(C: np.ndarray, requested: str, started: float, *, modulus: int,
                    max_dim: int, max_form_operations: int, max_bdd_nodes: int,
                    max_bdd_states: Optional[int], max_bdd_key_terms: int,
                    max_bdd_count_cells: int, max_modular_pairs: int, timeout,
                    nauty_timeout, traces_timeout, fallback: bool,
                    max_stabilizer_orbit: int, **_kwargs):
    before = time.perf_counter()
    try:
        vertex, edges, diagnostics = _modular_section_layer(
            C, modulus, max_form_operations=max_form_operations,
            max_bdd_nodes=max_bdd_nodes, max_bdd_states=max_bdd_states,
            max_bdd_key_terms=max_bdd_key_terms,
            max_bdd_count_cells=max_bdd_count_cells,
            max_modular_pairs=max_modular_pairs)
    except (ward.WardLimitExceeded, InvariantLimitExceeded, MemoryError, RecursionError) as exc:
        return _native_fallback(
            C, requested, f"modular section construction failed: {exc}", started,
            max_dim=max_dim, preprocessing_seconds=time.perf_counter() - before,
            fallback=fallback)
    components, _cells = _component_labels(C)
    vertex = [tuple(vertex[i]) + (components[i],) for i in range(C.shape[1])]
    return _solve_relation(
        C, requested, started, vertex, edges,
        preprocessing_seconds=time.perf_counter() - before,
        timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
        max_dim=max_dim, fallback=fallback, method="modular-section-signatures",
        diagnostics=diagnostics, max_stabilizer_orbit=max_stabilizer_orbit)


def _minor_code(C: np.ndarray, coordinates, shorten: bool) -> np.ndarray:
    coordinates = sorted(set(int(x) for x in coordinates))
    keep = [i for i in range(C.shape[1]) if i not in coordinates]
    if shorten and coordinates:
        constraints = C[:, coordinates].T
        messages = gf2.nullspace_basis_gf2(constraints)
        basis = gf2.row_basis_gf2((messages @ C) % 2)
    else:
        basis = C
    return gf2.row_basis_gf2(basis[:, keep])


def _minor_code_signature(C: np.ndarray, *, max_hull_dimension: int,
                          max_hull_words: int, max_schur_products: int,
                          expand_hull_dimensions=None):
    """Fixed-parameter support-splitting signature of one punctured/shortened minor."""
    C = _code_basis(C)
    hull = _hull_basis(C)
    total = 1 << len(hull)
    expand = (len(hull) <= max_hull_dimension and total <= max_hull_words and
              (expand_hull_dimensions is None or len(hull) in expand_hull_dimensions))
    if expand and len(hull):
        enumeration = _enumerate_code(
            hull, max_dim=max_hull_dimension, max_enumerated=max_hull_words)
        distribution = (
            "weights",
            tuple(sorted(Counter(int(weight) for weight in enumeration.weights).items())))
    elif len(hull):
        # Expansion is selected solely from the invariant hull dimension, so this coarse marker
        # remains permutation-equivariant and does not force the entire minor layer to abort.
        distribution = ("unexpanded", len(hull), total)
    else:
        distribution = ("weights", ())
    try:
        if not expand:
            raise InvariantLimitExceeded("minor hull stratum was not expanded")
        hull_square = _schur_power(hull, 2, max_products=max_schur_products)
        square_signature = ("expanded", len(hull_square), _hull_dimension(hull_square))
    except InvariantLimitExceeded:
        square_signature = ("unexpanded", math.comb(len(hull) + 1, 2))
    return (len(C), len(hull), distribution,
            square_signature)


def _minor_invariant(C: np.ndarray, coordinates, *, max_hull_dimension: int,
                     max_hull_words: int, max_schur_products: int):
    punctured = _minor_code(C, coordinates, False)
    shortened = _minor_code(C, coordinates, True)
    return (_minor_code_signature(
                punctured, max_hull_dimension=max_hull_dimension,
                max_hull_words=max_hull_words,
                max_schur_products=max_schur_products),
            _minor_code_signature(
                shortened, max_hull_dimension=max_hull_dimension,
                max_hull_words=max_hull_words,
                max_schur_products=max_schur_products))


def _minor_layer(C: np.ndarray, *, max_minor_pairs: int, max_hull_dimension: int,
                 max_hull_words: int, max_schur_products: int,
                 max_minor_hull_work: int):
    n = C.shape[1]
    if n * (n - 1) // 2 > max_minor_pairs:
        raise InvariantLimitExceeded(
            f"minor signatures need {n * (n - 1) // 2} pairs > "
            f"max_minor_pairs={max_minor_pairs}")
    coordinate_sets = [(i,) for i in range(n)]
    coordinate_sets.extend((i, j) for i in range(n) for j in range(i + 1, n))
    minor_codes = {}
    hull_dimension_counts = Counter()
    for coordinates in coordinate_sets:
        punctured = _minor_code(C, coordinates, False)
        shortened = _minor_code(C, coordinates, True)
        minor_codes[coordinates] = (punctured, shortened)
        hull_dimension_counts[_hull_dimension(punctured)] += 1
        hull_dimension_counts[_hull_dimension(shortened)] += 1
    expanded_dimensions = set()
    expanded_work = 0
    for dimension in sorted(hull_dimension_counts):
        per_code = 1 << dimension
        stratum_work = hull_dimension_counts[dimension] * per_code
        if dimension > max_hull_dimension or per_code > max_hull_words:
            continue
        if expanded_work + stratum_work > max_minor_hull_work:
            continue
        expanded_dimensions.add(dimension)
        expanded_work += stratum_work
    signature_kwargs = dict(
        max_hull_dimension=max_hull_dimension,
        max_hull_words=max_hull_words,
        max_schur_products=max_schur_products,
        expand_hull_dimensions=expanded_dimensions)

    def signature(coordinates):
        return tuple(_minor_code_signature(code, **signature_kwargs)
                     for code in minor_codes[tuple(coordinates)])

    vertex = [signature((i,)) for i in range(n)]
    edges = [[None] * n for _ in range(n)]
    for i in range(n):
        edges[i][i] = ()
        for j in range(i + 1, n):
            edges[i][j] = edges[j][i] = signature((i, j))
    return vertex, edges, {"minor_pairs": n * (n - 1) // 2,
                           "max_hull_dimension": max_hull_dimension,
                           "max_hull_words": max_hull_words,
                           "max_minor_hull_work": max_minor_hull_work,
                           "hull_dimension_counts": sorted(hull_dimension_counts.items()),
                           "expanded_hull_dimensions": sorted(expanded_dimensions),
                           "expanded_hull_work": expanded_work,
                           "minor_vertex_colours": len(set(vertex)),
                           "minor_pair_colours": len(set(edges[i][j] for i in range(n)
                                                         for j in range(i + 1, n)))}


def _minors_method(C: np.ndarray, requested: str, started: float, *, max_dim: int,
                   max_minor_pairs: int, max_hull_dimension: int, max_hull_words: int,
                   max_schur_products: int, max_minor_hull_work: int,
                   timeout, nauty_timeout, traces_timeout,
                   fallback: bool, max_stabilizer_orbit: int, **_kwargs):
    before = time.perf_counter()
    try:
        vertex, edges, diagnostics = _minor_layer(
            C, max_minor_pairs=max_minor_pairs,
            max_hull_dimension=max_hull_dimension,
            max_hull_words=max_hull_words,
            max_schur_products=max_schur_products,
            max_minor_hull_work=max_minor_hull_work)
    except InvariantLimitExceeded as exc:
        return _native_fallback(
            C, requested, str(exc), started, max_dim=max_dim,
            preprocessing_seconds=time.perf_counter() - before, fallback=fallback)
    components, _cells = _component_labels(C)
    vertex = [tuple(vertex[i]) + (components[i],) for i in range(C.shape[1])]
    return _solve_relation(
        C, requested, started, vertex, edges,
        preprocessing_seconds=time.perf_counter() - before,
        timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
        max_dim=max_dim, fallback=fallback, method="puncture-shorten-minors",
        diagnostics=diagnostics, max_stabilizer_orbit=max_stabilizer_orbit)


def _bounded_invariant_cells(C: np.ndarray, *, max_support_weight: int,
                             max_bounded_subsets: int, max_bounded_bz_budget: int,
                             max_bounded_class_size: int):
    """Complete support-minimal code/dual cells through ``max_support_weight``.

    A binary word is determined by its support.  Enumerating coordinate subsets and testing
    their parity checks therefore costs ``sum(comb(n,w), w<=b)`` rather than ``2**dim``.  This
    is polynomial in ``n`` for fixed support bound ``b`` and is guarded before allocation.  If
    the subset guard truncates that scan, a guarded BZ certificate may add higher complete
    classes without a ``2**dim`` traversal.
    """
    C = _code_basis(C)
    n = C.shape[1]
    requested_bound = min(max_support_weight, n)
    dual = gf2.nullspace_basis_gf2(C)
    sides = (("cocircuits", C, dual), ("circuits", dual, C))
    buckets = {name: defaultdict(list) for name, _code, _check in sides}
    bz_diagnostics = {}
    has_spanning_side = False
    for name, code, _parity_check in sides:
        if has_spanning_side:
            bz_diagnostics[name] = {"skipped": "earlier complete classes already span"}
            continue
        try:
            classes, info = lowweight.low_weight_classes(
                code, want_span=True, max_weight=requested_bound,
                budget=max_bounded_bz_budget, full_enum_max_dim=0,
                max_class_size=max_bounded_class_size)
            bz_diagnostics[name] = {
                key: info.get(key) for key in
                ("method", "spans", "budget_hit", "p", "W_cert", "classes")
            }
            has_spanning_side |= bool(info.get("spans"))
            for weight, rows in classes:
                if weight <= requested_bound:
                    buckets[name][int(weight)].extend(np.asarray(rows, dtype=np.uint8))
        except (RuntimeError, MemoryError, ValueError) as exc:
            bz_diagnostics[name] = {"error": str(exc)}

    # A complete spanning prefix already proves the exact group, so the coordinate-subset
    # backend is only needed when BZ leaves both sides partial.  This keeps high-dimensional
    # repetition/component families output-sensitive instead of paying n-choose-b first.
    subset_count = 0
    bound = 0
    if not has_spanning_side:
        for weight in range(1, requested_bound + 1):
            next_count = subset_count + math.comb(n, weight)
            if next_count > max_bounded_subsets:
                break
            subset_count = next_count
            bound = weight
        for weight in range(1, bound + 1):
            for support in itertools.combinations(range(n), weight):
                for name, _code, parity_check in sides:
                    if len(parity_check) and np.any(
                            np.bitwise_xor.reduce(parity_check[:, support], axis=1)):
                        continue
                    row = np.zeros(n, dtype=np.uint8)
                    row[list(support)] = 1
                    buckets[name][weight].append(row)
    cells = []
    for name, code, _parity_check in sides:
        for weight, rows in sorted(buckets[name].items()):
            minimal = []
            seen = set()
            for row in rows:
                key = np.asarray(row, dtype=np.uint8).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                complement = np.flatnonzero(1 - row)
                if gf2.rank_gf2(code[:, complement]) == len(code) - 1:
                    minimal.append(row)
            if minimal:
                cells.append({
                    "label": (name, weight),
                    "weight": weight,
                    "rows": np.ascontiguousarray(np.vstack(minimal), dtype=np.uint8),
                    "messages": np.zeros((len(minimal), 0), dtype=np.uint8),
                })
    return cells, {
        "subsets_tested": subset_count,
        "subset_support_weight": bound,
        "requested_support_weight": requested_bound,
        "bz": bz_diagnostics,
    }


def _bounded_method(C: np.ndarray, requested: str, started: float, *, max_dim: int,
                    max_enumerated: int, max_fingerprint_words: int, timeout,
                    nauty_timeout, traces_timeout, fallback: bool,
                    max_support_weight: int, max_bounded_subsets: int,
                    max_bounded_bz_budget: int, max_bounded_class_size: int,
                    max_stabilizer_orbit: int, **_kwargs):
    """Use all circuits and cocircuits through a fixed support weight as an overgroup.

    Unlike the complete minimal-prefix selector, the chosen families need not span either side.
    They are nevertheless complete invariant sets, hence their coloured hypergraph contains
    ``Aut(C)``.  Generator verification or the guarded rowspace stabilizer makes the final answer
    exact; Leon remains the last resort.
    """
    before = time.perf_counter()
    try:
        cells, discovery = _bounded_invariant_cells(
            C, max_support_weight=max_support_weight,
            max_bounded_subsets=max_bounded_subsets,
            max_bounded_bz_budget=max_bounded_bz_budget,
            max_bounded_class_size=max_bounded_class_size)
    except InvariantLimitExceeded as exc:
        return _native_fallback(
            C, requested, str(exc), started, max_dim=max_dim,
            preprocessing_seconds=time.perf_counter() - before, fallback=fallback)
    component_labels, components = _component_labels(C)

    def encoded_support(row):
        weight = int(row.sum())
        return np.flatnonzero(row if weight <= row.size - weight else 1 - row)

    relation_cells = [
        (cell["label"], [encoded_support(row) for row in cell["rows"]])
        for cell in cells
    ]
    word_count = sum(len(cell["rows"]) for cell in cells)
    incidence_count = sum(len(encoded_support(row)) for cell in cells for row in cell["rows"])
    side_summary = [(cell["label"], len(cell["rows"])) for cell in cells]
    preprocessing = time.perf_counter() - before
    try:
        search_started = time.perf_counter()
        group, vertices = graphaut.hypergraph_group(
            C.shape[1], component_labels, relation_cells,
            timeout=timeout, nauty_timeout=nauty_timeout,
            traces_timeout=traces_timeout)
        overgroup_order = group.order()
        search_seconds = time.perf_counter() - search_started
    except (RuntimeError, OSError, FileNotFoundError, subprocess.SubprocessError) as exc:
        return _native_fallback(
            C, requested, f"bounded circuit/cocircuit graph solve failed: {exc}", started,
            max_dim=max_dim, preprocessing_seconds=preprocessing, fallback=fallback)
    diagnostics = {
        "max_support_weight": max_support_weight,
        "bounded_discovery": discovery,
        "bounded_cells": side_summary,
        "matroid_components": len(components),
        "relation_exact_hit": _group_preserves_code(group, C),
        "overgroup_order": overgroup_order,
    }
    if diagnostics["relation_exact_hit"]:
        return _make_result(
            group, C, requested, "bounded-circuit-cocircuit-hypergraph", started,
            preprocessing_seconds=preprocessing, search_seconds=search_seconds,
            num_vertices=vertices, num_codewords=word_count,
            num_incidences=incidence_count, diagnostics=diagnostics)
    stabilizer_seconds = 0.0
    stabilizer_guard = None
    if max_stabilizer_orbit:
        stabilizer_started = time.perf_counter()
        try:
            stabilizer, orbit_size = rowspace_stabilizer(
                group, C, max_orbit=max_stabilizer_orbit)
            stabilizer_seconds = time.perf_counter() - stabilizer_started
            diagnostics.update({"relation_exactified": True,
                                "rowspace_orbit": orbit_size})
            return _make_result(
                stabilizer, C, requested,
                "bounded-circuit-cocircuit-hypergraph+rowspace-stabilizer", started,
                preprocessing_seconds=preprocessing, search_seconds=search_seconds,
                stabilizer_seconds=stabilizer_seconds,
                num_vertices=vertices, num_codewords=word_count,
                num_incidences=incidence_count, diagnostics=diagnostics)
        except InvariantLimitExceeded as exc:
            stabilizer_seconds = time.perf_counter() - stabilizer_started
            stabilizer_guard = str(exc)
            diagnostics["stabilizer_guard"] = stabilizer_guard
    reason = "bounded circuit/cocircuit relation is a strict overgroup"
    if stabilizer_guard:
        reason += f"; {stabilizer_guard}"
    return _native_fallback(
        C, requested, reason, started, max_dim=max_dim,
        preprocessing_seconds=preprocessing + search_seconds, fallback=fallback,
        diagnostics=diagnostics, stabilizer_seconds=stabilizer_seconds)


def _merge_relation_layers(n: int, vertex_layers, edge_layers):
    vertex = []
    edges = [[None] * n for _ in range(n)]
    for i in range(n):
        vertex.append(tuple(layer[i] for layer in vertex_layers))
        for j in range(n):
            edges[i][j] = tuple(layer[i][j] for layer in edge_layers)
    return vertex, edges


def _combined_method(C: np.ndarray, requested: str, started: float, *, modulus: int,
                     max_dim: int, max_enumerated: int, max_fingerprint_words: int,
                     max_schur_products: int,
                     max_hull_dimension: int, max_hull_words: int,
                     max_minor_hull_work: int,
                     max_support_weight: int, max_bounded_subsets: int,
                     max_bounded_bz_budget: int, max_bounded_class_size: int,
                     max_form_operations: int, max_residue_indicator_terms: int,
                     max_bdd_nodes: int,
                     max_bdd_states: Optional[int], max_bdd_key_terms: int,
                     max_bdd_count_cells: int, max_modular_pairs: int,
                     max_minor_pairs: int, timeout, nauty_timeout, traces_timeout,
                     fallback: bool, max_stabilizer_orbit: int, **_kwargs):
    before = time.perf_counter()
    n = C.shape[1]
    if n <= 1:
        group = permgroup.symmetric_group(n)
        return _make_result(group, C, requested, "combined-trivial", started,
                            preprocessing_seconds=time.perf_counter() - before,
                            num_vertices=n, diagnostics={"layers": ["trivial"]})
    vertex_layers = []
    edge_layers = []
    layer_names = []
    layer_diagnostics = {}

    try:
        projectors, projector_diagnostics, skipped_projectors = _projector_layers(
            C, max_schur_products=max_schur_products)
        for name, _code, projector in projectors:
            vertex_layers.append([int(projector[i, i]) for i in range(n)])
            edge_layers.append([[int(projector[i, j]) for j in range(n)] for i in range(n)])
            layer_names.append(f"projector:{name}")
        layer_diagnostics["projectors"] = projector_diagnostics
        layer_diagnostics["skipped_projectors"] = skipped_projectors
    except InvariantLimitExceeded as exc:
        layer_diagnostics["projectors_skipped"] = str(exc)

    try:
        residue_form, residue_spans = _residue_spans(
            C, modulus, max_form_operations=max_form_operations,
            max_residue_indicator_terms=max_residue_indicator_terms)
        residue_layers = []
        for residue, basis, terms in residue_spans:
            projector = orthogonal_projector(basis)
            if projector is None or not len(basis):
                continue
            vertex_layers.append([int(projector[i, i]) for i in range(n)])
            edge_layers.append([[int(projector[i, j]) for j in range(n)] for i in range(n)])
            layer_names.append(f"residue-projector:{residue}")
            residue_layers.append((residue, len(basis), terms))
        layer_diagnostics["residue_projectors"] = {
            "modulus": residue_form.modulus,
            "form_terms": residue_form.num_terms,
            "layers": residue_layers,
        }
    except (ward.WardLimitExceeded, MemoryError, RecursionError) as exc:
        layer_diagnostics["residue_projectors_skipped"] = str(exc)

    component_labels, components = _component_labels(C)
    vertex_layers.append(component_labels)
    layer_names.append("matroid-components")
    layer_diagnostics["component_sizes"] = [len(cell) for cell in components]

    try:
        hull_vertex, hull_edges, hull_diagnostics = _hull_moment_layer(
            C, max_dim=max_dim, max_enumerated=max_enumerated)
        vertex_layers.append(hull_vertex)
        edge_layers.append(hull_edges)
        layer_names.append("hull-sections")
        layer_diagnostics["hull"] = hull_diagnostics
    except InvariantLimitExceeded as exc:
        layer_diagnostics["hull_skipped"] = str(exc)

    try:
        minor_vertex, minor_edges, minor_diagnostics = _minor_layer(
            C, max_minor_pairs=max_minor_pairs,
            max_hull_dimension=max_hull_dimension,
            max_hull_words=max_hull_words,
            max_schur_products=max_schur_products,
            max_minor_hull_work=max_minor_hull_work)
        vertex_layers.append(minor_vertex)
        edge_layers.append(minor_edges)
        layer_names.append("puncture-shorten")
        layer_diagnostics["minors"] = minor_diagnostics
    except InvariantLimitExceeded as exc:
        layer_diagnostics["minors_skipped"] = str(exc)

    try:
        modular_vertex, modular_edges, modular_diagnostics = _modular_section_layer(
            C, modulus, max_form_operations=max_form_operations,
            max_bdd_nodes=max_bdd_nodes, max_bdd_states=max_bdd_states,
            max_bdd_key_terms=max_bdd_key_terms,
            max_bdd_count_cells=max_bdd_count_cells,
            max_modular_pairs=max_modular_pairs)
        vertex_layers.append(modular_vertex)
        edge_layers.append(modular_edges)
        layer_names.append("modular-sections")
        layer_diagnostics["modular"] = modular_diagnostics
    except (ward.WardLimitExceeded, InvariantLimitExceeded, MemoryError, RecursionError) as exc:
        layer_diagnostics["modular_skipped"] = str(exc)

    incidence_cells = None
    try:
        incidence_cells = _enumerated_cells(
            C, "fingerprint", max_dim=max_dim, max_enumerated=max_enumerated,
            max_fingerprint_words=max_fingerprint_words)
        moment_vertex, moment_edges = _pair_moment_tables(incidence_cells, n)
        vertex_layers.append(moment_vertex)
        edge_layers.append(moment_edges)
        layer_names.append("fingerprint-moments")
        layer_diagnostics["moment_cells"] = len(incidence_cells)
    except InvariantLimitExceeded as exc:
        layer_diagnostics["moments_skipped"] = str(exc)

    bounded_cells = []
    try:
        bounded_cells, bounded_discovery = _bounded_invariant_cells(
            C, max_support_weight=max_support_weight,
            max_bounded_subsets=max_bounded_subsets,
            max_bounded_bz_budget=max_bounded_bz_budget,
            max_bounded_class_size=max_bounded_class_size)
        if bounded_cells:
            layer_names.append("bounded-circuit-cocircuit-hyperedges")
        layer_diagnostics["bounded"] = {
            "max_support_weight": max_support_weight,
            "discovery": bounded_discovery,
            "cells": [(cell["label"], len(cell["rows"])) for cell in bounded_cells],
        }
    except InvariantLimitExceeded as exc:
        layer_diagnostics["bounded_skipped"] = str(exc)

    # Relations with no edge layer are still meaningful vertex colourings; supply the all-zero
    # pair relation so the common graph encoder can be used.
    if not edge_layers:
        edge_layers.append([[0] * n for _ in range(n)])
    vertex, edges = _merge_relation_layers(n, vertex_layers, edge_layers)
    pair_values = [edges[i][j] for i in range(n) for j in range(i + 1, n)]
    default_pair = Counter(pair_values).most_common(1)[0][0] if pair_values else None
    pair_cells = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] != default_pair:
                pair_cells[edges[i][j]].append((i, j))
    mixed_relations = [(('pair-relation', label), supports)
                       for label, supports in pair_cells.items()]
    mixed_relations.extend(
        (("bounded-support", cell["label"]),
         [np.flatnonzero(row if int(row.sum()) <= row.size - int(row.sum()) else 1 - row)
          for row in cell["rows"]])
        for cell in bounded_cells)
    preprocessing = time.perf_counter() - before
    try:
        graph_started = time.perf_counter()
        group, vertices = graphaut.hypergraph_group(
            n, vertex, mixed_relations, timeout=timeout,
            nauty_timeout=nauty_timeout, traces_timeout=traces_timeout)
        overgroup_order = group.order()
        relation_search = time.perf_counter() - graph_started
    except (RuntimeError, OSError, FileNotFoundError, subprocess.SubprocessError) as exc:
        if incidence_cells is not None and fallback:
            return _incidence_result(
                C, requested, started, incidence_cells,
                coordinate_labels=component_labels,
                preprocessing_seconds=preprocessing,
                timeout=timeout, nauty_timeout=nauty_timeout,
                traces_timeout=traces_timeout, method="combined-fingerprint-incidence",
                used_fallback=True, fallback_reason=f"combined graph solve failed: {exc}",
                diagnostics={"layers": layer_names, **layer_diagnostics})
        return _native_fallback(
            C, requested, f"combined graph solve failed: {exc}", started,
            max_dim=max_dim, preprocessing_seconds=preprocessing,
            fallback=fallback, diagnostics={"layers": layer_names, **layer_diagnostics})
    if _group_preserves_code(group, C):
        return _make_result(
            group, C, requested, "combined-relations", started,
            preprocessing_seconds=preprocessing, search_seconds=relation_search,
            num_vertices=vertices,
            diagnostics={"layers": layer_names, "relation_exact_hit": True,
                         **layer_diagnostics})
    stabilizer_guard = None
    stabilizer_seconds = 0.0
    if max_stabilizer_orbit:
        stabilizer_started = time.perf_counter()
        try:
            stabilizer, orbit_size = rowspace_stabilizer(
                group, C, max_orbit=max_stabilizer_orbit)
            return _make_result(
                stabilizer, C, requested,
                "combined-relations+rowspace-stabilizer", started,
                preprocessing_seconds=preprocessing,
                search_seconds=relation_search,
                stabilizer_seconds=time.perf_counter() - stabilizer_started,
                num_vertices=vertices,
                diagnostics={"layers": layer_names, "relation_exact_hit": False,
                             "relation_exactified": True, "rowspace_orbit": orbit_size,
                             "overgroup_order": overgroup_order, **layer_diagnostics})
        except InvariantLimitExceeded as exc:
            stabilizer_seconds = time.perf_counter() - stabilizer_started
            stabilizer_guard = str(exc)
    strict_reason = "combined relation is a strict overgroup"
    if stabilizer_guard:
        strict_reason += f"; {stabilizer_guard}"
    if incidence_cells is not None and fallback:
        return _incidence_result(
            C, requested, started, incidence_cells,
            coordinate_labels=component_labels,
            preprocessing_seconds=preprocessing + relation_search,
            timeout=timeout, nauty_timeout=nauty_timeout,
            traces_timeout=traces_timeout, method="combined-fingerprint-incidence",
            used_fallback=True, fallback_reason=strict_reason,
            diagnostics={"layers": layer_names, "relation_exact_hit": False,
                         "overgroup_order": overgroup_order,
                         "stabilizer_guard": stabilizer_guard, **layer_diagnostics},
            stabilizer_seconds=stabilizer_seconds)
    return _native_fallback(
        C, requested, strict_reason, started,
        max_dim=max_dim, preprocessing_seconds=preprocessing + relation_search,
        fallback=fallback,
        diagnostics={"layers": layer_names, "relation_exact_hit": False,
                     "overgroup_order": overgroup_order,
                     "stabilizer_guard": stabilizer_guard, **layer_diagnostics},
        stabilizer_seconds=stabilizer_seconds)


def _components_method(C: np.ndarray, requested: str, started: float, *, max_dim: int,
                       max_enumerated: int, max_fingerprint_words: int,
                       max_geometry_candidates: int, timeout, nauty_timeout,
                       traces_timeout, fallback: bool, **kwargs):
    # Prefer the direct component-wreath implementation when its guarded canonicalization fits.
    try:
        from .components import component_automorphism_group, ComponentGuardExceeded
    except ImportError:
        component_automorphism_group = None
        ComponentGuardExceeded = InvariantLimitExceeded
    before = time.perf_counter()
    if component_automorphism_group is not None:
        try:
            result = component_automorphism_group(
                C, max_candidates=max_geometry_candidates,
                max_component_dim=max_dim)
            if not all(gf2.preserves_rowspace(C, generator)
                       for generator in result.generators):
                raise AssertionError("component-wreath generator failed verification")
            return InvariantAutResult(
                generators=[list(generator) for generator in result.generators],
                order=int(result.order), n=C.shape[1], dim=len(C),
                method="matroid-component-wreath", requested_method=requested,
                preprocessing_seconds=float(result.construction_seconds),
                search_seconds=float(result.search_seconds),
                seconds=time.perf_counter() - started,
                num_vertices=sum(result.type_counts),
                diagnostics=dict(result.diagnostics))
        except ComponentGuardExceeded:
            pass
    # The exact universal fallback still uses component fingerprints and the twin quotient;
    # it attempts the structural refinement without relying on guarded component isomorphism.
    return _incidence_method(
        C, requested, started, "prefix", max_dim=max_dim,
        max_enumerated=max_enumerated,
        max_fingerprint_words=max_fingerprint_words,
        timeout=timeout, nauty_timeout=nauty_timeout, traces_timeout=traces_timeout,
        fallback=fallback, component_labels=True,
        method="component-refined-twin-incidence", **kwargs)


def _auto_method(C: np.ndarray, requested: str, started: float, **kwargs):
    attempts = []
    # Cheap exact certificates first, then increasingly rich verified relations.  Each trial is
    # run with fallback disabled so a strict overgroup does not terminate the portfolio early.
    for name, runner in (
        ("lcd", _lcd_method),
        ("geometry", _geometry_method),
        ("components", _components_method),
        ("combined", _combined_method),
        ("fingerprint", lambda *args, **kw: _incidence_method(
            *args, strategy="fingerprint", method="fingerprint-compressed-incidence", **kw)),
    ):
        try:
            trial_kwargs = dict(kwargs)
            trial_kwargs["fallback"] = False
            result = runner(C, requested, started, **trial_kwargs)
            result.requested_method = requested
            result.diagnostics = dict(result.diagnostics)
            result.diagnostics["portfolio_attempts"] = attempts + [(name, "exact")]
            return result
        except (InvariantLimitExceeded, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            attempts.append((name, str(exc)))
    result = _native_fallback(
        C, requested, "automatic invariant portfolio exhausted", started,
        max_dim=kwargs["max_dim"], preprocessing_seconds=time.perf_counter() - started,
        fallback=kwargs.get("fallback", True), diagnostics={"portfolio_attempts": attempts})
    return result


def automorphism_group(generator_matrix, *, method: str = "auto", modulus: int = 8,
                       max_dim: int = 20, max_words: int = 200_000,
                       max_enumerated: int = 1 << 18,
                       max_fingerprint_words: int = 100_000,
                       max_schur_products: int = 200_000,
                       max_hull_dimension: int = 12,
                       max_hull_words: int = 1 << 14,
                       max_minor_hull_work: int = 2_000_000,
                       max_support_weight: int = 8,
                       max_bounded_subsets: int = 2_000_000,
                       max_bounded_bz_budget: int = 10_000_000,
                       max_bounded_class_size: int = 200_000,
                       max_geometry_rank: int = 6,
                       max_geometry_candidates: int = 100_000,
                       max_form_operations: int = 5_000_000,
                       max_residue_indicator_terms: int = 2_000_000,
                       max_bdd_nodes: int = 500_000,
                       max_bdd_states: Optional[int] = None,
                       max_bdd_key_terms: int = 2_000_000,
                       max_bdd_count_cells: int = 5_000_000,
                       max_modular_pairs: int = 20_000,
                       max_minor_pairs: int = 20_000,
                       max_moment_triples: int = 20_000,
                       max_stabilizer_orbit: int = 5_000,
                       timeout: Optional[float] = None,
                       nauty_timeout: Optional[float] = 5.0,
                       traces_timeout: Optional[float] = None,
                       fallback: bool = True) -> InvariantAutResult:
    """Compute exact ``Aut(C)`` through one invariant certificate or the guarded portfolio.

    Methods are listed by :func:`available_methods`.  Relation-only methods (moments, hull,
    Schur, modular sections, bounded dependencies, and minors) may produce a strict overgroup.
    A guarded rowspace stabilizer first tries to exactify it; with the default ``fallback=True``
    an exhausted orbit transparently returns an exact incidence/Leon result and exposes the miss
    through ``used_fallback`` and ``fallback_reason``. Set ``fallback=False`` to benchmark the
    certificate/exactifier regime and receive :class:`InvariantLimitExceeded` otherwise.

    ``max_words`` is accepted for API compatibility with the Ward benchmark harness; explicit
    invariant incidences are instead guarded by ``max_enumerated`` and
    ``max_fingerprint_words``. Bounded dependency, minor-hull, and Schur/conductor work have
    their own ``max_bounded_*``, ``max_hull_*``/``max_minor_hull_work``, and
    ``max_schur_products`` caps.
    """
    started = time.perf_counter()
    method = str(method).lower().replace("_", "-")
    aliases = {"projector": "lcd", "projective": "geometry", "twin": "compressed",
               "component": "components", "pair-moments": "moments",
               "support-splitting": "hull", "ward-spans": "residue",
               "ward-sections": "modular", "circuits": "bounded",
               "portfolio": "combined"}
    method = aliases.get(method, method)
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    positive_limits = {
        "max_dim": max_dim, "max_words": max_words,
        "max_enumerated": max_enumerated,
        "max_fingerprint_words": max_fingerprint_words,
        "max_schur_products": max_schur_products,
        "max_hull_dimension": max_hull_dimension,
        "max_hull_words": max_hull_words,
        "max_minor_hull_work": max_minor_hull_work,
        "max_support_weight": max_support_weight,
        "max_bounded_subsets": max_bounded_subsets,
        "max_bounded_bz_budget": max_bounded_bz_budget,
        "max_bounded_class_size": max_bounded_class_size,
        "max_geometry_rank": max_geometry_rank,
        "max_geometry_candidates": max_geometry_candidates,
        "max_form_operations": max_form_operations, "max_bdd_nodes": max_bdd_nodes,
        "max_residue_indicator_terms": max_residue_indicator_terms,
        "max_bdd_key_terms": max_bdd_key_terms,
        "max_bdd_count_cells": max_bdd_count_cells,
        "max_modular_pairs": max_modular_pairs, "max_minor_pairs": max_minor_pairs,
        "max_moment_triples": max_moment_triples,
    }
    if max_bdd_states is not None:
        positive_limits["max_bdd_states"] = max_bdd_states
    for name, value in positive_limits.items():
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                or int(value) < 1):
            raise ValueError(f"{name} must be a positive integer")
    if (isinstance(max_stabilizer_orbit, bool)
            or not isinstance(max_stabilizer_orbit, (int, np.integer))
            or int(max_stabilizer_orbit) < 0):
        raise ValueError("max_stabilizer_orbit must be a nonnegative integer")
    if method in {"residue", "modular", "combined", "auto"}:
        # Validate even when the code has length zero or one: trivial instances must not make
        # malformed API arguments appear to work.
        modulus = ward._validate_modulus(modulus)[0]
    del max_words
    C = _code_basis(generator_matrix)
    if C.shape[1] <= 1:
        group = permgroup.symmetric_group(C.shape[1])
        return _make_result(group, C, method, "trivial-length", started,
                            num_vertices=C.shape[1], diagnostics={"requested": method})
    common = dict(
        modulus=modulus,
        max_dim=int(max_dim),
        max_enumerated=int(max_enumerated),
        max_fingerprint_words=int(max_fingerprint_words),
        max_schur_products=int(max_schur_products),
        max_hull_dimension=int(max_hull_dimension),
        max_hull_words=int(max_hull_words),
        max_minor_hull_work=int(max_minor_hull_work),
        max_support_weight=int(max_support_weight),
        max_bounded_subsets=int(max_bounded_subsets),
        max_bounded_bz_budget=int(max_bounded_bz_budget),
        max_bounded_class_size=int(max_bounded_class_size),
        max_geometry_rank=int(max_geometry_rank),
        max_geometry_candidates=int(max_geometry_candidates),
        max_form_operations=int(max_form_operations),
        max_residue_indicator_terms=int(max_residue_indicator_terms),
        max_bdd_nodes=int(max_bdd_nodes),
        max_bdd_states=(None if max_bdd_states is None else int(max_bdd_states)),
        max_bdd_key_terms=int(max_bdd_key_terms),
        max_bdd_count_cells=int(max_bdd_count_cells),
        max_modular_pairs=int(max_modular_pairs),
        max_minor_pairs=int(max_minor_pairs),
        max_moment_triples=int(max_moment_triples),
        max_stabilizer_orbit=int(max_stabilizer_orbit),
        timeout=timeout,
        nauty_timeout=nauty_timeout,
        traces_timeout=traces_timeout,
        fallback=bool(fallback),
    )
    if method == "lcd":
        return _lcd_method(C, method, started, **common)
    if method == "geometry":
        return _geometry_method(C, method, started, **common)
    if method == "compressed":
        return _incidence_method(
            C, method, started, "prefix", method="twin-compressed-incidence", **common)
    if method == "components":
        return _components_method(C, method, started, **common)
    if method == "fingerprint":
        return _incidence_method(
            C, method, started, "fingerprint",
            method="fingerprint-compressed-incidence", **common)
    if method == "moments":
        return _moments_method(C, method, started, **common)
    if method == "hull":
        return _hull_method(C, method, started, **common)
    if method == "schur":
        return _schur_method(C, method, started, **common)
    if method == "residue":
        return _residue_method(C, method, started, **common)
    if method == "modular":
        return _modular_method(C, method, started, **common)
    if method == "minors":
        return _minors_method(C, method, started, **common)
    if method == "bounded":
        return _bounded_method(C, method, started, **common)
    if method == "combined":
        return _combined_method(C, method, started, **common)
    return _auto_method(C, method, started, **common)


__all__ = [
    "METHODS", "METHOD_REGISTRY", "InvariantLimitExceeded", "InvariantAutResult",
    "available_methods", "orthogonal_projector", "rowspace_stabilizer",
    "automorphism_group",
]
