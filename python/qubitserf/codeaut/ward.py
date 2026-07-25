"""Power-of-two modular-weight invariants via Ward's inclusion--exclusion form.

For a row basis ``G = (g_0, ..., g_{k-1})`` and message ``x in F_2^k``, truncate the
inclusion--exclusion formula at ``t = log2(m)``::

    wt(xG) = sum_{1 <= |I| <= t} (-2)^(|I|-1)
                 wt(and_{i in I} g_i) prod_{i in I} x_i       (mod m),

where ``m = 2**t``.  Higher terms vanish modulo ``m``.  The resulting multilinear Ward form
has ``O(n k**t)`` construction cost for fixed ``t`` and does not enumerate ``2**k`` codewords.

This module uses the compact form in an exact, output-sensitive automorphism route:

1. Compile the form into a reduced multi-terminal decision diagram and count all residue fibers
   without scanning the ``2**k`` messages.
2. Materialize only small, complete fibers and choose a low-cost collection whose union spans
   ``C`` (minimum-word when the guarded exact subset search fits, greedy otherwise).
3. Give every residue/exact-weight class a separate colour in the coordinate--codeword incidence
   graph and solve its automorphism group with nauty.

Every code automorphism preserves all complete fibers.  Conversely, every automorphism of the
chosen incidence preserves their span, which is ``C``.  Hence the graph group is exactly
``Aut(C)``.  The Ward form has polynomial size for fixed ``t``.  A residue fiber itself may be
exponential and a decision diagram has no polynomial worst-case bound, so both have explicit
resource guards and fall back to the current min-weight Leon implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import operator
import subprocess
import time
from typing import Optional

import numpy as np

from . import gf2
from . import permgroup


class WardLimitExceeded(RuntimeError):
    """A configured Ward form, decision-diagram, or fiber-size guard was exceeded."""


# Monomials in message bits are Python-int masks.  Integer polynomials use dict[mask, coeff];
# Boolean ANFs use set[mask], with symmetric difference as addition.


def _validate_modulus(modulus: int) -> tuple[int, int]:
    if isinstance(modulus, bool):
        raise ValueError("modulus must be a power of two >= 2")
    try:
        modulus = operator.index(modulus)
    except TypeError:
        raise ValueError("modulus must be an integer power of two >= 2") from None
    if modulus < 2 or modulus & (modulus - 1):
        raise ValueError("modulus must be a power of two >= 2")
    return modulus, modulus.bit_length() - 1


def _add_mod(poly: dict[int, int], mono: int, coeff: int, modulus: int) -> None:
    value = (poly.get(mono, 0) + coeff) % modulus
    if value:
        poly[mono] = value
    else:
        poly.pop(mono, None)


def _xor_lift(bit_poly: set[int], modulus: int) -> dict[int, int]:
    """The unique multilinear integer polynomial modulo ``modulus`` taking the XOR of the
    Boolean monomials in ``bit_poly``.  Terms whose coefficient vanishes modulo the power of two
    are dropped, which truncates the degree automatically."""
    out: dict[int, int] = {}
    for mono in sorted(bit_poly, key=lambda x: (x.bit_count(), x)):
        old = list(out.items())
        _add_mod(out, mono, 1, modulus)
        for other, coeff in old:
            _add_mod(out, mono | other, -2 * coeff, modulus)
    return out


def _gf2_mul(left: set[int], right: set[int], *, max_terms: Optional[int] = None) -> set[int]:
    out: set[int] = set()
    for a in left:
        for b in right:
            mono = a | b
            if mono in out:
                out.remove(mono)
            else:
                out.add(mono)
            if max_terms is not None and len(out) > max_terms:
                raise WardLimitExceeded(
                    f"Boolean ANF product exceeded max_indicator_terms={max_terms}")
    return out


@dataclass(frozen=True)
class WardForm:
    """Compact multilinear form for ``wt(xG) mod modulus``.

    ``coefficients`` maps a tuple of message-bit indices to its coefficient in ``Z_modulus``.
    Its degree is at most ``log2(modulus)``.
    """

    generator: np.ndarray
    modulus: int
    degree: int
    coefficients: dict[tuple[int, ...], int]

    @property
    def n(self) -> int:
        return int(self.generator.shape[1])

    @property
    def dim(self) -> int:
        return int(self.generator.shape[0])

    @property
    def num_terms(self) -> int:
        return len(self.coefficients)

    def evaluate(self, message) -> int:
        x = np.asarray(message, dtype=np.uint8).reshape(-1) % 2
        if x.size != self.dim:
            raise ValueError(f"message has length {x.size}; expected {self.dim}")
        value = 0
        for mono, coeff in self.coefficients.items():
            if all(x[i] for i in mono):
                value += coeff
        return int(value % self.modulus)

    def _mask_coefficients(self) -> dict[int, int]:
        return {sum(1 << i for i in mono): int(c) for mono, c in self.coefficients.items()}

    def bit_polynomials(self) -> list[set[int]]:
        """Boolean ANFs of the low ``degree`` bits of the modular weight.

        The 2-adic peeling is coefficient-exact in the Boolean quotient ``x_i**2=x_i``.  At
        each level the XOR lift is reduced modulo the remaining power of two, so its size stays
        polynomial for fixed modulus.
        """
        current = self._mask_coefficients()
        bits: list[set[int]] = []
        remaining = self.modulus
        for _ in range(self.degree):
            bit = {mono for mono, coeff in current.items() if coeff & 1}
            bits.append(bit)
            lift = _xor_lift(bit, remaining)
            keys = set(current) | set(lift)
            next_modulus = remaining // 2
            nxt: dict[int, int] = {}
            for mono in keys:
                diff = (current.get(mono, 0) - lift.get(mono, 0)) % remaining
                if diff & 1:
                    raise AssertionError("non-even Ward carry during 2-adic peeling")
                value = (diff // 2) % next_modulus if next_modulus > 1 else 0
                if value:
                    nxt[mono] = value
            current = nxt
            remaining = next_modulus
        return bits

    def residue_indicator(self, residue: int, *, max_terms: Optional[int] = None) -> set[int]:
        """Boolean ANF equal to one exactly when ``wt(xG) == residue (mod modulus)``."""
        if max_terms is not None and int(max_terms) < 1:
            raise ValueError("max_terms must be positive or None")
        residue = int(residue) % self.modulus
        indicator = {0}
        for bit_index, bit_poly in enumerate(self.bit_polynomials()):
            factor = set(bit_poly)
            if ((residue >> bit_index) & 1) == 0:
                if 0 in factor:
                    factor.remove(0)
                else:
                    factor.add(0)
            indicator = _gf2_mul(indicator, factor, max_terms=max_terms)
            if not indicator:
                break
        return indicator


@dataclass(frozen=True)
class ResidueSpan:
    modulus: int
    residue: int
    message_basis: np.ndarray
    code_basis: np.ndarray
    indicator_terms: int

    @property
    def dimension(self) -> int:
        return int(self.message_basis.shape[0])


def ward_form(generator_matrix, modulus: int = 4, *,
              max_operations: Optional[int] = None) -> WardForm:
    """Construct the compact Ward form of ``rowspace(generator_matrix)`` modulo ``2**t``.

    The input is row-reduced first.  Complexity is
    ``O(n * sum(comb(k,j), j=1..t))`` in the dense worst case, for fixed ``t``.
    """
    modulus, degree = _validate_modulus(modulus)
    G = gf2.row_basis_gf2(generator_matrix)
    k, n = G.shape
    if max_operations is not None:
        max_operations = int(max_operations)
        if max_operations < 1:
            raise ValueError("max_operations must be positive, or None")
    # Equal generator columns define the same linear form.  Accumulate their multiplicity first:
    # a type repeated a multiple of ``modulus`` contributes nothing at all, and common
    # quasi-cyclic/repetition structures avoid redundant subset expansion.
    column_types: dict[tuple[int, ...], int] = {}
    for col in range(n):
        support = tuple(int(i) for i in np.flatnonzero(G[:, col]))
        column_types[support] = column_types.get(support, 0) + 1

    coeff: dict[int, int] = {}
    operations = 0
    for support, multiplicity in column_types.items():
        for size in range(1, min(degree, len(support)) + 1):
            value = (multiplicity * ((-2) ** (size - 1))) % modulus
            if not value:
                continue
            operations += math.comb(len(support), size)
            if max_operations is not None and operations > max_operations:
                raise WardLimitExceeded(
                    f"Ward form needs more than max_form_operations={max_operations} "
                    "column/subset updates")
            for subset in itertools.combinations(support, size):
                mask = sum(1 << i for i in subset)
                _add_mod(coeff, mask, value, modulus)
    public = {tuple(i for i in range(k) if (mask >> i) & 1): value
              for mask, value in coeff.items()}
    return WardForm(G, modulus, degree, public)


def residue_span(form_or_generator, residue: int, modulus: int = 4, *,
                 max_indicator_terms: Optional[int] = None) -> ResidueSpan:
    """Span of all codewords with weight ``residue mod modulus``, without listing the fiber.

    If ``delta(x)`` is the Boolean indicator of the residue fiber, a linear form ``a.x``
    vanishes on the fiber iff the ANF ``delta(x) * (a.x)`` is identically zero.  The latter is
    a linear system in ``a``.  Its kernel is the orthogonal complement of the fiber span.
    """
    form = form_or_generator if isinstance(form_or_generator, WardForm) else \
        ward_form(form_or_generator, modulus)
    residue = int(residue) % form.modulus
    delta = form.residue_indicator(residue, max_terms=max_indicator_terms)
    k = form.dim

    equations: dict[int, int] = {}
    for mono in delta:
        for i in range(k):
            target = mono | (1 << i)
            equations[target] = equations.get(target, 0) ^ (1 << i)
    rows = [bits for bits in equations.values() if bits]
    A = np.zeros((len(rows), k), dtype=np.uint8)
    for row, bits in enumerate(rows):
        for i in range(k):
            A[row, i] = (bits >> i) & 1
    annihilator = gf2.nullspace_basis_gf2(A)
    messages = gf2.nullspace_basis_gf2(annihilator)
    code_basis = ((messages @ form.generator) % 2).astype(np.uint8)
    code_basis = gf2.row_basis_gf2(code_basis)
    return ResidueSpan(form.modulus, residue, messages, code_basis, len(delta))


class WardDecisionDiagram:
    """Reduced ordered decision diagram for a :class:`WardForm`.

    Leaves are the ``modulus`` possible residues.  Internal nodes hold ``(variable, low, high)``;
    identical restrictions and identical low/high children are merged.  For structured forms
    this can be exponentially smaller than the truth table.  ``max_nodes`` is a hard guard:
    difficult forms fall back to the legacy engine instead of consuming unbounded memory.
    """

    def __init__(self, form: WardForm, *, max_nodes: int = 500_000,
                 max_states: Optional[int] = None, max_key_terms: int = 2_000_000,
                 max_count_cells: int = 5_000_000):
        self.form = form
        self.max_nodes = int(max_nodes)
        self.max_states = int(max_states if max_states is not None else 4 * max_nodes)
        self.max_key_terms = int(max_key_terms)
        self.max_count_cells = int(max_count_cells)
        if min(self.max_nodes, self.max_states, self.max_key_terms,
               self.max_count_cells) < 1:
            raise ValueError("all Ward decision-diagram guards must be positive")
        self.nodes: dict[int, tuple[int, int, int]] = {}
        self._unique: dict[tuple[int, int, int], int] = {}
        self._memo: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
        self._states_built = 0
        self._key_terms_built = 0
        self._next = form.modulus
        self.root = self._build(0, form._mask_coefficients())
        # Restriction states are only needed while constructing the reduced DAG.  Some compact
        # diagrams arise from large intermediate polynomial keys, so retaining this cache would
        # needlessly keep their full peak memory alive during fiber enumeration.
        self._memo.clear()
        self._residue_counts_cache: Optional[tuple[int, ...]] = None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes) + self.form.modulus

    @property
    def num_states(self) -> int:
        return self._states_built

    def _canonical(self, poly: dict[int, int]) -> tuple[tuple[int, int], ...]:
        return tuple(sorted((mask, coeff % self.form.modulus)
                            for mask, coeff in poly.items() if coeff % self.form.modulus))

    def _build(self, position: int, poly: dict[int, int]) -> int:
        if len(poly) > self.max_key_terms - self._key_terms_built:
            raise WardLimitExceeded(
                f"Ward decision diagram exceeded max_key_terms={self.max_key_terms}")
        canonical = self._canonical(poly)
        self._key_terms_built += len(canonical)
        if self._key_terms_built > self.max_key_terms:
            raise WardLimitExceeded(
                f"Ward decision diagram exceeded max_key_terms={self.max_key_terms}")
        key = (position, canonical)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        self._states_built += 1
        if self._states_built > self.max_states:
            raise WardLimitExceeded(
                f"Ward decision diagram exceeded max_states={self.max_states}")
        if position == self.form.dim:
            leaf = int(poly.get(0, 0) % self.form.modulus)
            self._memo[key] = leaf
            return leaf

        bit = 1 << position
        low: dict[int, int] = {}
        high: dict[int, int] = {}
        for mask, coeff in poly.items():
            coeff %= self.form.modulus
            if not coeff:
                continue
            if mask & bit:
                _add_mod(high, mask ^ bit, coeff, self.form.modulus)
            else:
                _add_mod(low, mask, coeff, self.form.modulus)
                _add_mod(high, mask, coeff, self.form.modulus)
        lo = self._build(position + 1, low)
        hi = self._build(position + 1, high)
        if lo == hi:
            node = lo
        else:
            signature = (position, lo, hi)
            node = self._unique.get(signature)
            if node is None:
                if len(self.nodes) >= self.max_nodes:
                    raise WardLimitExceeded(
                        f"Ward decision diagram exceeded max_nodes={self.max_nodes}")
                node = self._next
                self._next += 1
                self._unique[signature] = node
                self.nodes[node] = signature
        self._memo[key] = node
        return node

    def residue_counts(self) -> list[int]:
        if self._residue_counts_cache is not None:
            return list(self._residue_counts_cache)
        memo: dict[tuple[int, int], tuple[int, ...]] = {}
        count_cells = 0

        def visit(node: int, start: int) -> tuple[int, ...]:
            nonlocal count_cells
            key = (node, start)
            if key in memo:
                return memo[key]
            count_cells += self.form.modulus
            if count_cells > self.max_count_cells:
                raise WardLimitExceeded(
                    f"Ward residue counts exceeded max_count_cells={self.max_count_cells}")
            if node < self.form.modulus:
                out = [0] * self.form.modulus
                out[node] = 1 << (self.form.dim - start)
                answer = tuple(out)
            else:
                variable, lo, hi = self.nodes[node]
                factor = 1 << (variable - start)
                a = visit(lo, variable + 1)
                b = visit(hi, variable + 1)
                answer = tuple(factor * (x + y) for x, y in zip(a, b))
            memo[key] = answer
            return answer

        self._residue_counts_cache = visit(self.root, 0)
        return list(self._residue_counts_cache)

    def message_masks(self, residue: int, *, limit: Optional[int] = None) -> list[int]:
        residue = int(residue) % self.form.modulus
        expected = self.residue_counts()[residue]
        if limit is not None and expected > limit:
            raise ValueError(f"residue {residue} has {expected} messages > limit={limit}")
        out: list[int] = []
        reachable: dict[int, bool] = {}

        def can_reach(node: int) -> bool:
            cached = reachable.get(node)
            if cached is not None:
                return cached
            if node < self.form.modulus:
                answer = node == residue
            else:
                _variable, lo, hi = self.nodes[node]
                answer = can_reach(lo) or can_reach(hi)
            reachable[node] = answer
            return answer

        def visit(node: int, start: int, prefix: int) -> None:
            if not can_reach(node):
                return
            if node < self.form.modulus:
                if node == residue:
                    for suffix in range(1 << (self.form.dim - start)):
                        out.append(prefix | (suffix << start))
                return
            variable, lo, hi = self.nodes[node]
            for skipped in range(1 << (variable - start)):
                base = prefix | (skipped << start)
                if can_reach(lo):
                    visit(lo, variable + 1, base)
                if can_reach(hi):
                    visit(hi, variable + 1, base | (1 << variable))

        visit(self.root, 0, 0)
        if len(out) != expected:
            raise AssertionError(f"Ward BDD enumerated {len(out)} messages; expected {expected}")
        return out

    def conditioned_residue_counts(self, linear_masks, *, max_cells: Optional[int] = None):
        """Count residues jointly with up to three linear Boolean conditions.

        ``linear_masks`` contains message-bit masks for forms ``ell_j(x)``.  The result is a
        ``modulus x 2**q`` tuple table; entry ``[r][s]`` counts messages of Ward residue ``r``
        whose condition bits form state ``s``.  Skipped decision-diagram variables are summed by
        their tiny parity distribution, so this does not enumerate messages.  It underlies the
        modular singleton/pair/triple section signatures used by :mod:`codeaut.invariants`.
        """
        masks = tuple(int(mask) for mask in linear_masks)
        q = len(masks)
        if q > 3:
            raise ValueError("at most three linear conditions are supported")
        if any(mask < 0 or mask >> self.form.dim for mask in masks):
            raise ValueError("linear condition uses a message bit outside the Ward form")
        states = 1 << q
        limit = self.max_count_cells if max_cells is None else int(max_cells)
        if limit < 1:
            raise ValueError("max_cells must be positive")
        memo = {}
        cells = 0

        def delta_at(variable: int) -> int:
            value = 0
            for condition, mask in enumerate(masks):
                value |= ((mask >> variable) & 1) << condition
            return value

        def parity_distribution(start: int, stop: int):
            distribution = [0] * states
            distribution[0] = 1
            for variable in range(start, stop):
                delta = delta_at(variable)
                updated = [0] * states
                for state, count in enumerate(distribution):
                    updated[state] += count
                    updated[state ^ delta] += count
                distribution = updated
            return distribution

        def convolve(table, distribution):
            answer = [[0] * states for _ in range(self.form.modulus)]
            for residue, row in enumerate(table):
                for state, count in enumerate(row):
                    if not count:
                        continue
                    for skipped, multiplicity in enumerate(distribution):
                        if multiplicity:
                            answer[residue][state ^ skipped] += count * multiplicity
            return answer

        def visit(node: int, start: int):
            nonlocal cells
            key = (node, start)
            cached = memo.get(key)
            if cached is not None:
                return cached
            cells += self.form.modulus * states
            if cells > limit:
                raise WardLimitExceeded(
                    f"Ward conditioned counts exceeded max_cells={limit}")
            if node < self.form.modulus:
                base = [[0] * states for _ in range(self.form.modulus)]
                base[node][0] = 1
                answer = convolve(base, parity_distribution(start, self.form.dim))
            else:
                variable, lo, hi = self.nodes[node]
                low = visit(lo, variable + 1)
                high = visit(hi, variable + 1)
                delta = delta_at(variable)
                branched = [[0] * states for _ in range(self.form.modulus)]
                for residue in range(self.form.modulus):
                    for state in range(states):
                        branched[residue][state] = (low[residue][state] +
                                                     high[residue][state ^ delta])
                answer = convolve(branched, parity_distribution(start, variable))
            frozen = tuple(tuple(row) for row in answer)
            memo[key] = frozen
            return frozen

        return visit(self.root, 0)


@dataclass
class WardAutResult:
    generators: list
    order: int
    n: int
    dim: int
    modulus: int
    residues: tuple[int, ...]
    form_terms: int
    bdd_nodes: int
    bdd_states: int
    residue_counts: tuple[int, ...]
    num_codewords: int
    num_incidences: Optional[int]
    weight_classes: list[int]
    form_seconds: float
    bdd_seconds: float
    enumeration_seconds: float
    search_seconds: float
    seconds: float
    method: str
    components: list

    @property
    def residue(self) -> Optional[int]:
        return self.residues[0] if len(self.residues) == 1 else None

    @property
    def spanning_set(self) -> str:
        return "ward" if self.residues else "minweight"

    def group(self):
        return permgroup.Group(self.generators, self.n)


def _messages_from_masks(masks: list[int], dimension: int) -> np.ndarray:
    messages = np.empty((len(masks), dimension), dtype=np.uint8)
    for bit in range(dimension):
        messages[:, bit] = np.fromiter(((mask >> bit) & 1 for mask in masks),
                                      dtype=np.uint8, count=len(masks))
    return messages


def _fiber_candidate(form: WardForm, diagram: WardDecisionDiagram, residue: int,
                     max_words: int) -> dict:
    masks = diagram.message_masks(residue, limit=max_words + (residue == 0))
    masks = [mask for mask in masks if mask]
    messages = _messages_from_masks(masks, form.dim)
    basis = gf2.row_basis_gf2(messages)
    return {
        "residue": int(residue),
        "messages": messages,
        "basis": basis,
        "rank": int(basis.shape[0]),
        "words": len(masks),
    }


def _materialize_candidate(form: WardForm, candidate: dict) -> None:
    rows = ((candidate["messages"] @ form.generator) % 2).astype(np.uint8)
    weights = rows.sum(axis=1, dtype=np.int64) if len(rows) else np.zeros(0, np.int64)
    classes = [(int(weight), np.ascontiguousarray(rows[weights == weight]))
               for weight in sorted(set(int(x) for x in weights))]
    candidate["classes"] = classes
    candidate["edges"] = int(weights.sum())
    candidate["weights"] = [weight for weight, _rows in classes]


def _choose_cover(candidates: list[dict], dimension: int, max_words: int,
                  max_subsets: int) -> Optional[list[dict]]:
    """Low-cost spanning residue cover.

    The cover is minimum-word when the guarded exact subset search fits and greedy otherwise.
    The choice affects graph size only, never exactness.
    """
    candidates = [item for item in candidates if item["rank"]]
    candidates.sort(key=lambda item: (item["words"], item["residue"]))
    count = len(candidates)
    best = None
    if count < 63 and (1 << count) <= max_subsets:
        for size in range(1, count + 1):
            for indices in itertools.combinations(range(count), size):
                words = sum(candidates[i]["words"] for i in indices)
                if words > max_words:
                    continue
                stacked = np.vstack([candidates[i]["basis"] for i in indices])
                if gf2.rank_gf2(stacked) != dimension:
                    continue
                score = (words, size, indices)
                if best is None or score < best[0]:
                    best = (score, indices)
        return None if best is None else [candidates[i] for i in best[1]]

    # Large powers of two make exact subset enumeration unattractive.  Greedily maximize rank
    # gained per word, then remove redundant fibers.  This affects graph size, never exactness.
    chosen: list[dict] = []
    basis = np.zeros((0, dimension), dtype=np.uint8)
    remaining = list(candidates)
    while gf2.rank_gf2(basis) < dimension and remaining:
        old_rank = gf2.rank_gf2(basis)
        ranked = []
        for item in remaining:
            new_rank = gf2.rank_gf2(np.vstack([basis, item["basis"]]))
            gain = new_rank - old_rank
            if gain:
                ranked.append((-gain / max(item["words"], 1), item["words"],
                               item["residue"], item, new_rank))
        if not ranked:
            return None
        *_score, item, _new_rank = min(ranked, key=lambda value: value[:3])
        chosen.append(item)
        remaining.remove(item)
        basis = gf2.row_basis_gf2(np.vstack([basis, item["basis"]]))
        if sum(x["words"] for x in chosen) > max_words:
            return None
    for item in list(reversed(chosen)):
        trial = [other for other in chosen if other is not item]
        if trial and gf2.rank_gf2(np.vstack([x["basis"] for x in trial])) == dimension:
            chosen = trial
    return chosen if sum(x["words"] for x in chosen) <= max_words else None


def automorphism_group(generator_matrix, *, modulus: int = 8, residue: Optional[int] = None,
                       max_dim: int = 20, max_words: int = 200_000,
                       max_form_operations: int = 5_000_000,
                       max_bdd_nodes: int = 500_000, max_bdd_states: Optional[int] = None,
                       max_bdd_key_terms: int = 2_000_000,
                       max_bdd_count_cells: int = 5_000_000,
                       max_cover_subsets: int = 1_000_000,
                       timeout: Optional[float] = None,
                       nauty_timeout: Optional[float] = 5.0,
                       traces_timeout: Optional[float] = None) -> WardAutResult:
    """Compute exact ``Aut(C)`` from complete mod-``2**t`` weight fibers when economical.

    ``modulus`` must be a power of two.  With ``residue=None``, all fibers small enough to
    materialize are considered and a low-cost spanning collection is selected (minimum-word
    while the exact subset-search guard fits, greedy otherwise).  Passing a residue restricts
    the construction to that one complete fiber.  ``max_form_operations``,
    ``max_bdd_nodes``/``max_bdd_states``/``max_bdd_key_terms``/``max_bdd_count_cells``, and
    ``max_words`` are hard resource guards.  If a
    guard is hit, no small spanning fiber cover exists, or nauty is unavailable, the ordinary
    min-weight Leon solve (on the cheaper of the code and its dual) is used instead.

    The returned group is exact in both paths.  Compact Ward-form construction is polynomial
    for fixed ``log2(modulus)``; decision-diagram size, fiber output, and graph automorphism are
    deliberately not advertised as polynomial-time.
    """
    from . import graphaut
    from . import leon

    started = time.perf_counter()
    C = gf2.row_basis_gf2(generator_matrix)
    k, n = C.shape
    modulus, _degree = _validate_modulus(modulus)
    if max_words < 1 or max_cover_subsets < 1:
        raise ValueError("max_words and max_cover_subsets must be positive")

    form = None
    diagram = None
    counts: tuple[int, ...] = ()
    form_seconds = bdd_seconds = enumeration_seconds = search_seconds = 0.0

    def fallback(reason: str) -> WardAutResult:
        nonlocal enumeration_seconds, search_seconds
        basis, _n, _rank, _eff = gf2.dual_basis(C)
        base = leon.automorphism_group(basis, max_dim=max_dim, spanning_set="minweight")
        enumeration_seconds += base.enumeration_seconds or 0.0
        search_seconds += base.search_seconds or 0.0
        return WardAutResult(
            base.generators, base.order, n, k, modulus, (),
            0 if form is None else form.num_terms,
            0 if diagram is None else diagram.num_nodes,
            0 if diagram is None else diagram.num_states,
            counts, base.num_codewords, base.num_incidences, base.weight_classes,
            form_seconds, bdd_seconds, enumeration_seconds, search_seconds,
            time.perf_counter() - started,
            f"Leon min-weight fallback ({reason})", [])

    try:
        before = time.perf_counter()
        form = ward_form(C, modulus, max_operations=max_form_operations)
        form_seconds = time.perf_counter() - before
        before = time.perf_counter()
        diagram = WardDecisionDiagram(form, max_nodes=max_bdd_nodes,
                                      max_states=max_bdd_states,
                                      max_key_terms=max_bdd_key_terms,
                                      max_count_cells=max_bdd_count_cells)
        counts = tuple(diagram.residue_counts())
        bdd_seconds = time.perf_counter() - before
    except (WardLimitExceeded, RecursionError) as exc:
        if form is None:
            form_seconds = time.perf_counter() - before
        else:
            bdd_seconds = time.perf_counter() - before
        return fallback(str(exc))

    selected_residues = ([int(residue) % modulus] if residue is not None else
                         sorted(range(modulus),
                                key=lambda r: (counts[r] - (1 if r == 0 else 0), r)))
    candidates = []
    best_single_words = max_words + 1
    before = time.perf_counter()
    for current in selected_residues:
        nonzero_count = counts[current] - (1 if current == 0 else 0)
        if not (0 < nonzero_count <= max_words):
            continue
        # Once a spanning single fiber is known, no larger fiber can participate in a cheaper
        # cover.  Counts are sorted in automatic mode, so this avoids materializing the rest of
        # the code on the structured instances for which Ward is intended.
        if residue is None and nonzero_count > best_single_words:
            break
        candidate = _fiber_candidate(form, diagram, current, max_words)
        candidates.append(candidate)
        if candidate["rank"] == k:
            best_single_words = min(best_single_words, candidate["words"])
    enumeration_seconds = time.perf_counter() - before
    picked = _choose_cover(candidates, k, max_words, max_cover_subsets)
    if not picked:
        detail = (f"residue {int(residue) % modulus} does not give a <= {max_words}-word "
                  "spanning fiber" if residue is not None else
                  f"no <= {max_words}-word spanning residue cover")
        return fallback(detail)
    if graphaut.nauty_binary() is None:
        return fallback("nauty/dreadnaut is unavailable")

    for item in picked:
        _materialize_candidate(form, item)

    before = time.perf_counter()
    try:
        group, vertices = graphaut.incidence_group(
            n, [item["classes"] for item in picked], timeout=timeout,
            nauty_timeout=nauty_timeout,
            traces_timeout=traces_timeout if traces_timeout is not None else timeout)
    except (RuntimeError, OSError, FileNotFoundError, subprocess.SubprocessError) as exc:
        search_seconds = time.perf_counter() - before
        return fallback(f"residue-incidence graph solve failed: {exc}")
    if not all(gf2.preserves_rowspace(C, generator) for generator in group.gens()):
        raise AssertionError("Ward residue-incidence generator does not preserve the code")
    generators = group.gens()
    order = group.order()
    search_seconds = time.perf_counter() - before

    components = [{
        "residue": item["residue"],
        "rank": item["rank"],
        "num_codewords": item["words"],
        "num_incidences": item["edges"],
        "weight_classes": item["weights"],
    } for item in picked]
    words = sum(item["words"] for item in picked)
    edges = sum(item["edges"] for item in picked)
    weights = sorted({weight for item in picked for weight in item["weights"]})
    residues = tuple(item["residue"] for item in picked)
    return WardAutResult(
        generators, order, n, k, modulus, residues, form.num_terms,
        diagram.num_nodes, diagram.num_states, counts, words, edges, weights,
        form_seconds, bdd_seconds, enumeration_seconds, search_seconds,
        time.perf_counter() - started,
        f"Ward mod-{modulus} complete residue fiber incidence ({vertices} vertices)",
        components)


__all__ = ["WardLimitExceeded", "WardForm", "WardDecisionDiagram", "ResidueSpan",
           "WardAutResult", "ward_form", "residue_span", "automorphism_group"]
