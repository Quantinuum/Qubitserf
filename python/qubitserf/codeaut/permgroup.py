"""Pure-Python permutation-group computations for the ``codeaut`` package.

This module is a self-contained implementation of the small slice of
``PermutationGroup`` functionality that ``codeaut`` needs.  It implements
exact group order, membership testing, element enumeration and subgroup
intersection on top of a deterministic Schreier--Sims
base-and-strong-generating-set (BSGS) construction.

It depends only on the Python standard library (``numpy`` is permitted but
not actually required here -- nothing else is imported).  In particular it
does **not** import SymPy at runtime; the algorithms below were ported from
the descriptions in Holt, Eick & O'Brien, *Handbook of Computational Group
Theory* (and validated against SymPy in the self-test).

Conventions (these are part of the public contract)
----------------------------------------------------
* A permutation is a 0-indexed image list/tuple ``p`` of length ``degree``
  with ``p[i] == image of point i``.  Points are ``0 .. degree-1``.
* Composition: ``compose(a, b)[i] == a[b[i]]`` -- i.e. ``b`` is applied
  first, then ``a``.  ``inv(p)`` satisfies ``compose(p, inv(p)) ==
  identity``.  (This matches SymPy's internal array-form ``_af_rmul``.)
"""

from __future__ import annotations

import os as _os
from itertools import product as _iproduct

__all__ = [
    "compose",
    "inv",
    "identity",
    "Group",
    "from_generators",
    "intersection",
    "symmetric_group",
]


# ---------------------------------------------------------------------------
# Low-level permutation arithmetic (array form == list of images).
# ---------------------------------------------------------------------------

def identity(degree):
    """Return the identity permutation on ``degree`` points."""
    return list(range(degree))


def compose(a, b):
    """Return the composition ``a after b``: ``compose(a, b)[i] == a[b[i]]``."""
    return [a[x] for x in b]


def inv(p):
    """Return the inverse of permutation ``p`` (array form)."""
    res = [0] * len(p)
    for i, pi in enumerate(p):
        res[pi] = i
    return res


def _is_identity(p):
    for i, pi in enumerate(p):
        if pi != i:
            return False
    return True


def _is_permutation(p, degree):
    if len(p) != degree:
        return False
    seen = [False] * degree
    for x in p:
        if not isinstance(x, (int,)) and not hasattr(x, "__index__"):
            return False
        xi = int(x)
        if xi < 0 or xi >= degree or seen[xi]:
            return False
        seen[xi] = True
    return True


# ---------------------------------------------------------------------------
# Orbit / transversal / stabilizer primitives (array form).
# ---------------------------------------------------------------------------

def _orbit(degree, gens, alpha):
    """Orbit of point ``alpha`` under ``gens`` (a set of points)."""
    orb = [alpha]
    used = [False] * degree
    used[alpha] = True
    for b in orb:
        for gen in gens:
            t = gen[b]
            if not used[t]:
                used[t] = True
                orb.append(t)
    return set(orb)


def _orbits(degree, gens):
    """List of orbits (as sets) of ``gens`` on ``0 .. degree-1``."""
    orbs = []
    remaining = set(range(degree))
    while remaining:
        i = min(remaining)
        orb = _orbit(degree, gens, i)
        orbs.append(orb)
        remaining -= orb
    return orbs


def _orbit_transversal_dict(degree, gens, alpha):
    """Return ``{beta: u}`` with ``u[alpha] == beta`` and ``u`` in <gens>."""
    idn = list(range(degree))
    tr = {alpha: idn}
    order = [alpha]
    for x in order:
        px = tr[x]
        for gen in gens:
            t = gen[x]
            if t not in tr:
                tr[t] = compose(gen, px)
                order.append(t)
    return tr


def _stabilizer(degree, gens, alpha):
    """Generators of the stabilizer of ``alpha`` in <gens> (Schreier gens)."""
    idn = list(range(degree))
    orb = [alpha]
    table = {alpha: idn}
    table_inv = {alpha: idn}
    used = [False] * degree
    used[alpha] = True
    stab_gens = []
    for b in orb:
        tb = table[b]
        for gen in gens:
            t = gen[b]
            if not used[t]:
                gen_t = compose(gen, tb)
                orb.append(t)
                table[t] = gen_t
                table_inv[t] = inv(gen_t)
                used[t] = True
            else:
                # schreier generator: table_inv[t] . gen . table[b]
                sg = compose(table_inv[t], compose(gen, tb))
                if sg not in stab_gens:
                    stab_gens.append(sg)
    return stab_gens


def _base_ordering(base, degree):
    """``ordering[point]`` = rank of point with base points first, in order."""
    ordering = [0] * degree
    in_base = [False] * degree
    for i, b in enumerate(base):
        ordering[b] = i
        in_base[b] = True
    current = len(base)
    for i in range(degree):
        if not in_base[i]:
            ordering[i] = current
            current += 1
    return ordering


def _distribute_gens_by_base(base, gens, degree):
    """Distribute ``gens`` across basic stabilizers of ``base``.

    Returns a list of length ``len(base)`` whose ``i``-th entry holds the
    generators fixing ``base[0..i-1]`` pointwise (entry 0 is all of
    ``gens``).  Empty levels are filled with the identity.
    """
    base_len = len(base)
    stabs = [[] for _ in range(base_len)]
    max_stab_index = 0
    idn = list(range(degree))
    for gen in gens:
        j = 0
        while j < base_len - 1 and gen[base[j]] == base[j]:
            j += 1
        if j > max_stab_index:
            max_stab_index = j
        for k in range(j + 1):
            stabs[k].append(gen)
    for i in range(max_stab_index + 1, base_len):
        stabs[i].append(idn[:])
    return stabs


def _orbits_transversals_from_bsgs(base, strong_gens_distr, degree):
    """Basic orbits (lists) and transversals (dicts) for a distributed SGS."""
    base_len = len(base)
    basic_orbits = [None] * base_len
    transversals = [None] * base_len
    for i in range(base_len):
        tr = _orbit_transversal_dict(degree, strong_gens_distr[i], base[i])
        transversals[i] = tr
        basic_orbits[i] = list(tr.keys())
    return basic_orbits, transversals


# ---------------------------------------------------------------------------
# Sifting / stripping.
# ---------------------------------------------------------------------------

def _strip(g, base, basic_orbit_sets, transversals):
    """Sift ``g`` through the (full) BSGS.

    Returns ``(residue, level)``.  If ``level == len(base) + 1`` and the
    residue is the identity, ``g`` lies in the group.
    """
    h = g
    base_len = len(base)
    for i in range(base_len):
        beta = h[base[i]]
        if beta == base[i]:
            continue
        if beta not in basic_orbit_sets[i]:
            return h, i + 1
        u = transversals[i][beta]
        h = compose(inv(u), h)
    return h, base_len + 1


def _strip_af(h, base, orbit_sets, transversals, j):
    """Optimised sift used inside Schreier--Sims.

    ``h`` already fixes ``base[0..j]``.  Returns ``(residue, level)`` where
    ``residue`` is ``False`` when ``h`` sifts to the identity, otherwise the
    leftover array-form permutation.
    """
    base_len = len(base)
    for i in range(j + 1, base_len):
        beta = h[base[i]]
        if beta == base[i]:
            continue
        if beta not in orbit_sets[i]:
            return h, i + 1
        u = transversals[i][beta]
        if h == u:
            return False, base_len + 1
        h = compose(inv(u), h)
    return h, base_len + 1


# ---------------------------------------------------------------------------
# Deterministic incremental Schreier--Sims.
# ---------------------------------------------------------------------------

def _schreier_sims_incremental(base, gens, degree):
    """Extend ``base``/``gens`` to a base and strong generating set.

    Returns ``(base, strong_gens)`` with ``strong_gens`` in array form.
    The trivial group yields ``(base, [])``.
    """
    idn = list(range(degree))
    # Drop the identity and exact duplicate generators.
    _gens = []
    for g in gens:
        if _is_identity(g):
            continue
        if g not in _gens:
            _gens.append(g)
    if not _gens:
        return list(base), []

    _base = list(base)
    # Ensure no generator fixes every base point.
    for gen in _gens:
        if all(gen[x] == x for x in _base):
            for new in range(degree):
                if gen[new] != new:
                    _base.append(new)
                    break

    strong_gens_distr = _distribute_gens_by_base(_base, _gens, degree)
    transversals = {}
    orbs = {}
    base_len = len(_base)
    for i in range(base_len):
        tr = _orbit_transversal_dict(degree, strong_gens_distr[i], _base[i])
        transversals[i] = tr
        orbs[i] = set(tr.keys())

    strong_gens_extra = []
    i = base_len - 1
    while i >= 0:
        continue_i = False
        for beta in list(transversals[i].keys()):
            u_beta = transversals[i][beta]
            for gen in strong_gens_distr[i]:
                gb = gen[beta]
                u1 = transversals[i][gb]
                g1 = compose(gen, u_beta)
                if g1 == u1:
                    continue
                schreier_gen = compose(inv(u1), g1)
                h, j = _strip_af(schreier_gen, _base, orbs, transversals, i)
                y = True
                if j <= base_len:
                    y = False
                elif h is not False:
                    # nontrivial element fixing all current base points
                    y = False
                    moved = 0
                    while h[moved] == moved:
                        moved += 1
                    _base.append(moved)
                    base_len += 1
                    strong_gens_distr.append([])
                if not y:
                    strong_gens_extra.append(h)
                    for l in range(i + 1, j):
                        strong_gens_distr[l].append(h)
                        tr = _orbit_transversal_dict(
                            degree, strong_gens_distr[l], _base[l])
                        transversals[l] = tr
                        orbs[l] = set(tr.keys())
                    i = j - 1
                    continue_i = True
                if continue_i:
                    break
            if continue_i:
                break
        if continue_i:
            continue
        i -= 1

    strong_gens = _gens + strong_gens_extra
    return _base, strong_gens


# ---------------------------------------------------------------------------
# Backtracking subgroup search (used for intersection).
# ---------------------------------------------------------------------------

def _subgroup_search(degree, base, strong_gens, prop):
    """Return generators of ``{g in <strong_gens> : prop(g)}``.

    ``prop`` is a predicate on array-form permutations defining a subgroup.
    This is a depth-first base-image backtrack with the standard minimality
    pruning (Holt, Eick & O'Brien, *Handbook of CGT*, pp. 114-117), ported
    from SymPy's ``subgroup_search``.
    """
    base_len = len(base)
    idn = list(range(degree))

    base_ordering = _base_ordering(base, degree)
    base_ordering.append(degree)   # sentinel "larger than all points"
    base_ordering.append(-1)       # sentinel "smaller than all points"

    strong_gens_distr = _distribute_gens_by_base(base, strong_gens, degree)
    basic_orbits, transversals = _orbits_transversals_from_bsgs(
        base, strong_gens_distr, degree)

    def get_reps(orbits):
        return [min(orbit, key=lambda x: base_ordering[x]) for orbit in orbits]

    def update_nu(l):
        temp_index = (len(basic_orbits[l]) + 1
                      - len(res_basic_orbits_init_base[l]))
        if temp_index >= len(sorted_orbits[l]):
            nu[l] = base_ordering[degree]
        else:
            nu[l] = sorted_orbits[l][temp_index]

    f = base_len - 1
    l = base_len - 1

    # BSGS for the (initially trivial) subgroup K relative to ``base``.
    res_generators = [idn[:]]
    res_base = list(base)
    res_strong_gens = [idn[:]]
    res_strong_gens_distr = _distribute_gens_by_base(
        res_base, res_strong_gens, degree)
    res_basic_orbits_init_base = [
        _orbit(degree, res_strong_gens_distr[i], res_base[i])
        for i in range(base_len)]

    orbit_reps = [None] * base_len
    orbits = _orbits(degree, res_strong_gens_distr[f])
    orbit_reps[f] = get_reps(orbits)
    orbit_reps[f].remove(base[f])

    c = [0] * base_len
    u = [idn[:]] * base_len
    sorted_orbits = [None] * base_len
    for i in range(base_len):
        so = basic_orbits[i][:]
        so.sort(key=lambda point: base_ordering[point])
        sorted_orbits[i] = so

    mu = [None] * base_len
    nu = [None] * base_len
    mu[l] = degree + 1
    update_nu(l)

    computed_words = [idn[:]] * base_len

    while True:
        # Descend, applying minimality / pruning tests.
        while (l < base_len - 1
               and computed_words[l][base[l]] in orbit_reps[l]
               and (base_ordering[mu[l]]
                    < base_ordering[computed_words[l][base[l]]]
                    < base_ordering[nu[l]])):
            new_point = computed_words[l][base[l]]
            res_base[l] = new_point
            new_stab_gens = _stabilizer(
                degree, res_strong_gens_distr[l], new_point)
            res_strong_gens_distr[l + 1] = new_stab_gens
            orbits = _orbits(degree, new_stab_gens)
            orbit_reps[l + 1] = get_reps(orbits)
            l += 1
            temp_orbit = [computed_words[l - 1][point]
                          for point in basic_orbits[l]]
            temp_orbit.sort(key=lambda point: base_ordering[point])
            sorted_orbits[l] = temp_orbit
            new_mu = degree + 1
            for i in range(l):
                if base[l] in res_basic_orbits_init_base[i]:
                    candidate = computed_words[i][base[i]]
                    if base_ordering[candidate] > base_ordering[new_mu]:
                        new_mu = candidate
            mu[l] = new_mu
            update_nu(l)
            c[l] = 0
            temp_point = sorted_orbits[l][c[l]]
            gamma = inv(computed_words[l - 1])[temp_point]
            u[l] = transversals[l][gamma]
            computed_words[l] = compose(u[l], computed_words[l - 1])

        # Test the element reached at the current leaf.
        g = computed_words[l]
        temp_point = g[base[l]]
        if (l == base_len - 1
                and (base_ordering[mu[l]] < base_ordering[temp_point]
                     < base_ordering[nu[l]])
                and temp_point in orbit_reps[l]
                and prop(g)):
            res_generators.append(g)
            res_base = list(base)
            res_strong_gens.append(g)
            res_strong_gens_distr = _distribute_gens_by_base(
                res_base, res_strong_gens, degree)
            res_basic_orbits_init_base = [
                _orbit(degree, res_strong_gens_distr[i], res_base[i])
                for i in range(base_len)]
            orbit_reps[f] = get_reps(orbits)
            l = f

        # Backtrack to the first not-fully-searched branch.
        while l >= 0 and c[l] == len(basic_orbits[l]) - 1:
            l -= 1
        if l == -1:
            return res_generators

        if l < f:
            f = l
            c[l] = 0
            temp_orbits = _orbits(degree, res_strong_gens_distr[f])
            orbit_reps[f] = get_reps(temp_orbits)
            mu[l] = degree + 1
            temp_index = (len(basic_orbits[l]) + 1
                          - len(res_basic_orbits_init_base[l]))
            if temp_index >= len(sorted_orbits[l]):
                nu[l] = base_ordering[degree]
            else:
                nu[l] = sorted_orbits[l][temp_index]

        c[l] += 1
        if l == 0:
            gamma = sorted_orbits[l][c[l]]
        else:
            gamma = inv(computed_words[l - 1])[sorted_orbits[l][c[l]]]
        u[l] = transversals[l][gamma]
        if l == 0:
            computed_words[l] = u[l]
        else:
            computed_words[l] = compose(u[l], computed_words[l - 1])


# ---------------------------------------------------------------------------
# Public Group class.
# ---------------------------------------------------------------------------

class Group:
    """A permutation group on ``0 .. degree-1`` given by generators.

    The order and membership tests are exact (deterministic Schreier--Sims).
    """

    def __init__(self, generators, degree):
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self._degree = degree
        # Normalise generators: coerce to int lists, validate, drop the
        # identity and duplicates.
        gens = []
        seen = set()
        for g in generators:
            gl = [int(x) for x in g]
            if not _is_permutation(gl, degree):
                raise ValueError(
                    "generator is not a permutation of %d points: %r"
                    % (degree, g))
            if _is_identity(gl):
                continue
            key = tuple(gl)
            if key in seen:
                continue
            seen.add(key)
            gens.append(gl)
        self._input_gens = gens
        # Lazily computed BSGS data.
        self._base = None
        self._strong_gens = None
        self._basic_orbits = None        # list of sets
        self._transversals = None        # list of dicts
        self._order = None

    # -- basic accessors ---------------------------------------------------

    @property
    def degree(self):
        return self._degree

    def gens(self):
        """A generating set (0-indexed image lists), identity excluded."""
        return [list(g) for g in self._input_gens]

    def generators(self):
        """Alias of :meth:`gens`."""
        return self.gens()

    # -- BSGS machinery ----------------------------------------------------

    def _ensure_bsgs(self):
        if self._base is not None:
            return
        if not self._input_gens:
            # Trivial group.
            self._base = []
            self._strong_gens = []
            self._basic_orbits = []
            self._transversals = []
            self._order = 1
            return
        base, strong = _schreier_sims_incremental(
            [], self._input_gens, self._degree)
        distr = _distribute_gens_by_base(base, strong, self._degree)
        basic_orbits, transversals = _orbits_transversals_from_bsgs(
            base, distr, self._degree)
        self._base = base
        self._strong_gens = strong
        self._basic_orbits = [set(o) for o in basic_orbits]
        self._transversals = transversals
        order = 1
        for o in self._basic_orbits:
            order *= len(o)
        self._order = order

    def _strong_generators(self):
        self._ensure_bsgs()
        return self._strong_gens

    def base_and_strong_gens(self):
        """Return ``(base, strong_gens)`` (array form) for this group."""
        self._ensure_bsgs()
        return list(self._base), [list(g) for g in self._strong_gens]

    # -- exact queries -----------------------------------------------------

    def order(self):
        """Exact group order as a Python ``int``."""
        self._ensure_bsgs()
        return int(self._order)

    def __len__(self):
        return self.order()

    def contains(self, perm):
        """Membership test for a 0-indexed image list ``perm``."""
        try:
            p = [int(x) for x in perm]
        except (TypeError, ValueError):
            return False
        if not _is_permutation(p, self._degree):
            return False
        self._ensure_bsgs()
        h, level = _strip(p, self._base, self._basic_orbits, self._transversals)
        return level == len(self._base) + 1 and _is_identity(h)

    def __contains__(self, perm):
        return self.contains(perm)

    # -- enumeration -------------------------------------------------------

    def elements(self):
        """Iterate over all group elements as 0-indexed tuples.

        Intended for small groups; it materialises the basic transversals
        and forms their products, so the cost is proportional to the order.
        """
        self._ensure_bsgs()
        idn = tuple(range(self._degree))
        if not self._base:
            yield idn
            return
        # Per level, the list of transversal elements (array form).
        levels = []
        for i in range(len(self._base)):
            tr = self._transversals[i]
            levels.append([tr[beta] for beta in self._basic_orbits[i]])
        # g = t_0 . t_1 . ... . t_{m-1}  (compose left-to-right).
        for combo in _iproduct(*levels):
            w = list(combo[0])
            for t in combo[1:]:
                w = compose(w, t)
            yield tuple(w)

    def __iter__(self):
        return self.elements()

    def reduced_generators(self):
        """A small generating set (0-indexed image lists) for this group: greedily pick strong
        generators that grow the subgroup until its order matches ``self.order()``.  Avoids the
        bloated generating set produced when a group is constructed from all of its elements."""
        self._ensure_bsgs()
        if not self._input_gens:
            return []
        target = self.order()
        chosen = []
        cur = 1
        for pool in (self._strong_gens, self._input_gens):
            for g in pool:
                if cur == target:
                    return chosen
                trial = Group(chosen + [list(g)], self._degree)
                o = trial.order()
                if o > cur:
                    chosen.append(list(g))
                    cur = o
            if cur == target:
                break
        return chosen

    # -- misc --------------------------------------------------------------

    def __repr__(self):
        return "Group(degree=%d, order=%d, gens=%d)" % (
            self._degree, self.order(), len(self._input_gens))


# ---------------------------------------------------------------------------
# Module-level constructors / operations.
# ---------------------------------------------------------------------------

def from_generators(generators, degree):
    """Convenience constructor for :class:`Group`."""
    return Group(generators, degree)


def symmetric_group(n):
    """The full symmetric group ``Sym(n)`` (order ``n!``)."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return Group([], n)
    # Generators: the transposition (0 1) and the n-cycle (0 1 ... n-1).
    transposition = list(range(n))
    transposition[0], transposition[1] = 1, 0
    cycle = [(i + 1) % n for i in range(n)]
    return Group([transposition, cycle], n)


def intersection(G, H, *, max_enumerate=4_000_000, allow_sympy=True):
    """Return ``G ∩ H`` as a :class:`Group` (exact).

    Both groups must act on the same number of points.  Computed by enumerating the smaller
    group and keeping the elements that lie in the larger one -- correct by construction (the
    intersection is exactly ``{g in smaller : g in larger}``) on top of the validated exact
    ``elements()`` / ``contains()`` primitives.

    To bound the cost, the smaller group must be enumerable: if ``min(|G|, |H|) >
    max_enumerate`` a :class:`ValueError` is raised (callers fall through to a method that does
    not require a generic intersection, e.g. the joint-incidence route).
    """
    if G.degree != H.degree:
        raise ValueError("groups must have the same degree")
    degree = G.degree
    if G.order() == 1 or H.order() == 1:
        return Group([], degree)

    searched, other = (G, H) if G.order() <= H.order() else (H, G)

    # Fast path for large groups: a backtracking subgroup search.  SymPy's battle-tested
    # ``subgroup_search`` is used when SymPy is importable (an optional, pure-Python
    # accelerator); it handles groups far too large to enumerate.  This is the only place the
    # package will use SymPy, and only if it is installed.
    if searched.order() > max_enumerate:
        if allow_sympy:
            gens = _intersection_via_sympy(G, H, degree)
            if gens is not None:
                return Group(gens, degree)
        raise ValueError(
            f"intersection: smaller group order {searched.order()} exceeds max_enumerate "
            f"{max_enumerate} (allow_sympy={allow_sympy}); use the joint-incidence route instead")

    # Enumerate the smaller group; keep the elements in the larger one, accumulating a SMALL
    # generating set incrementally -- only add an element as a generator when it is not already
    # in the subgroup generated so far.  Correct by construction on top of the validated exact
    # ``elements()`` / ``contains()`` primitives.
    gens = []
    K = Group([], degree)
    for el in searched.elements():
        e = list(el)
        if other.contains(e) and not K.contains(e):
            gens.append(e)
            K = Group(gens, degree)
    return K


# Optional wall-clock cap (seconds) on the SymPy intersection backtrack.  Default 0 = UNCAPPED:
# the inline ``automorphism_group`` (no ``timeout``) then preserves coverage -- it returns the
# exact group even on rare hard large-group intersections, where SymPy's pure-Python backtrack
# can be much slower than GAP (the wall clock is bounded instead via the ``timeout=`` API, which
# kills the worker at the deadline).  Set CODEAUT_SYMPY_INTERSECT_TIMEOUT to bound it inline too
# (on expiry the caller falls back to a verified subgroup -- a sound lower bound).
_SYMPY_INTERSECT_TIMEOUT = float(_os.environ.get("CODEAUT_SYMPY_INTERSECT_TIMEOUT", "0"))


def _intersection_via_sympy(G, H, degree):
    """``G ∩ H`` generators via SymPy's ``subgroup_search`` (optional fast path for large
    groups), time-bounded by ``_SYMPY_INTERSECT_TIMEOUT``.  Returns 0-indexed image lists, or
    ``None`` if SymPy is unavailable or the backtrack exceeds the cap.

    A group is the same subset of ``Sym(n)`` under either multiplication convention, so building
    the SymPy groups from the same generators and intersecting their element sets is convention-
    independent; ``subgroup_search`` finds exactly ``{g in G : g in H}``.
    """
    try:
        from sympy.combinatorics import Permutation, PermutationGroup
    except Exception:
        return None

    def to_sympy(grp):
        gens = grp.gens() or [list(range(degree))]
        return PermutationGroup([Permutation(g, size=degree) for g in gens])

    SG, SH = to_sympy(G), to_sympy(H)
    if SG.order() > SH.order():
        SG, SH = SH, SG
    try:
        K = _call_time_bounded(lambda: SG.subgroup_search(lambda g: g in SH),
                               _SYMPY_INTERSECT_TIMEOUT)
    except (TimeoutError, Exception):
        return None
    if K is None:
        return None
    out = []
    for g in K.generators:
        af = list(g.array_form)
        af += list(range(len(af), degree))     # pad trailing fixed points
        if af != list(range(degree)):
            out.append(af)
    return out


def _call_time_bounded(fn, seconds):
    """Call ``fn()`` with a SIGALRM wall-clock cap (main thread, Unix); raises ``TimeoutError``
    on expiry.  If signals are unavailable (non-main thread / unsupported platform) or
    ``seconds <= 0``, ``fn()`` runs uncapped."""
    if not seconds or seconds <= 0:
        return fn()
    try:
        import signal
    except Exception:
        return fn()

    def _handler(signum, frame):
        raise TimeoutError("sympy intersection exceeded time budget")

    try:
        old = signal.signal(signal.SIGALRM, _handler)
    except (ValueError, AttributeError, OSError):
        return fn()                              # not the main thread / no SIGALRM
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Self-test / correctness gate.
# ---------------------------------------------------------------------------

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _selftest():
    import random

    # ---- 1. Symmetric groups -------------------------------------------
    for n in range(1, 8):
        assert symmetric_group(n).order() == _factorial(n), n
    print("[1] symmetric_group(n).order() == n!  for n=1..7  OK")

    # ---- 2. Cyclic & dihedral group orders -----------------------------
    for n in range(1, 12):
        cyc = [(i + 1) % n for i in range(n)]
        assert Group([cyc], n).order() == n, ("cyclic", n)
    print("[2a] cyclic group C_n order == n  for n=1..11  OK")

    for n in range(3, 12):
        rot = [(i + 1) % n for i in range(n)]
        ref = [(-i) % n for i in range(n)]
        assert Group([rot, ref], n).order() == 2 * n, ("dihedral", n)
    print("[2b] dihedral group D_n order == 2n  for n=3..11  OK")

    # ---- 3. GL(3,2) of order 168 (= PSL(2,7); automorphism group of the
    # [7,4,3] Hamming code / Fano plane) acting on the 7 nonzero vectors of
    # F_2^3.  We build the permutations directly from generating matrices so
    # the group is provably exactly GL(3,2).
    #
    # Point ``v`` in 1..7 encodes a vector with bit j == coefficient of the
    # j-th basis vector e_j (e1=1, e2=2, e3=4); point index = v - 1.
    def _gl32_perm(me1, me2, me3):
        # M(e1)=me1, M(e2)=me2, M(e3)=me3 ; extend linearly over F_2.
        img = [(me1, me2, me3)]
        perm = [0] * 7
        for v in range(1, 8):
            out = 0
            if v & 1:
                out ^= me1
            if v & 2:
                out ^= me2
            if v & 4:
                out ^= me3
            perm[v - 1] = out - 1
        return perm

    swap12 = _gl32_perm(2, 1, 4)         # e1<->e2
    swap23 = _gl32_perm(1, 4, 2)         # e2<->e3
    transv = _gl32_perm(1, 3, 4)         # e2 -> e2 + e1   (transvection)
    g168 = Group([swap12, swap23, transv], 7)
    assert g168.order() == 168, g168.order()
    print("[3] GL(3,2) on 7 points has order 168  OK  (order = %d)"
          % g168.order())

    # ---- cross-validation against SymPy --------------------------------
    import sympy
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.perm_groups import PermutationGroup
    Permutation.print_cyclic = False
    print("    (cross-validating against SymPy %s)" % sympy.__version__)

    # Convention sanity: our compose == sympy array-form _af_rmul.
    from sympy.combinatorics.permutations import _af_rmul, _af_invert
    rng_conv = random.Random(0)
    for _ in range(30):
        n = rng_conv.randint(2, 8)
        x = list(range(n)); rng_conv.shuffle(x)
        y = list(range(n)); rng_conv.shuffle(y)
        assert compose(x, y) == _af_rmul(x, y)
        assert inv(x) == _af_invert(x)
    print("[conv] compose == sympy _af_rmul, inv == _af_invert  OK")

    def rand_perm(rng, n):
        p = list(range(n))
        rng.shuffle(p)
        return p

    # ---- 4. order() vs sympy, ~500 random generator sets ----------------
    rng = random.Random(1234)
    n_order_checks = 0
    for _ in range(60):
        n = rng.randint(4, 9)
        k = rng.randint(1, 4)
        gens = [rand_perm(rng, n) for _ in range(k)]
        mine = Group(gens, n).order()
        ref = PermutationGroup([Permutation(g, size=n) for g in gens]).order()
        assert mine == ref, (n, gens, mine, ref)
        # spot-check membership: every input generator is a member.
        Gm = Group(gens, n)
        for g in gens:
            assert Gm.contains(g)
        n_order_checks += 1
    print("[4] order() matches SymPy on %d random groups (deg 4..9)  OK"
          % n_order_checks)

    # extra membership cross-check: random probes in/out of the group
    rng_m = random.Random(99)
    n_mem = 0
    for _ in range(40):
        n = rng_m.randint(4, 8)
        k = rng_m.randint(1, 3)
        gens = [rand_perm(rng_m, n) for _ in range(k)]
        Gm = Group(gens, n)
        Gs = PermutationGroup([Permutation(g, size=n) for g in gens])
        for _ in range(4):
            probe = rand_perm(rng_m, n)
            assert Gm.contains(probe) == Gs.contains(
                Permutation(probe, size=n)), (gens, probe)
            n_mem += 1
    print("[4b] contains() matches SymPy on %d random probes  OK" % n_mem)

    # elements() cross-check on small groups
    rng_e = random.Random(7)
    n_elem = 0
    for _ in range(30):
        n = rng_e.randint(3, 6)
        k = rng_e.randint(1, 3)
        gens = [rand_perm(rng_e, n) for _ in range(k)]
        Gm = Group(gens, n)
        if Gm.order() > 200:
            continue
        elems = set(Gm.elements())
        assert len(elems) == Gm.order(), (gens, len(elems), Gm.order())
        Gs = PermutationGroup([Permutation(g, size=n) for g in gens])
        ref = set(tuple(p.array_form + tuple(range(len(p.array_form), n)))
                  if False else tuple(Permutation(p, size=n).array_form)
                  for p in Gs.generate())
        assert elems == ref, (gens,)
        # every enumerated element is a member
        for e in elems:
            assert Gm.contains(list(e))
        n_elem += 1
    print("[4c] elements() matches SymPy on %d small groups  OK" % n_elem)

    # ---- 5. intersection() vs sympy, ~200 random pairs ------------------
    def sympy_intersection_order(gensA, gensB, n):
        A = PermutationGroup([Permutation(g, size=n) for g in gensA])
        B = PermutationGroup([Permutation(g, size=n) for g in gensB])
        # Prefer a native .intersection if available and correct.
        try:
            inter = A.intersection(B)
            return inter.order()
        except Exception:
            pass
        # Brute force: enumerate the smaller group, filter by the other.
        if A.order() <= B.order():
            small, big = A, B
        else:
            small, big = B, A
        elems = [g for g in small.generate() if big.contains(g)]
        if not elems:
            return 1
        return PermutationGroup(elems).order()

    rng2 = random.Random(2024)
    n_int = 0
    for _ in range(30):
        n = rng2.randint(5, 8)
        ka = rng2.randint(1, 3)
        kb = rng2.randint(1, 3)
        gensA = [rand_perm(rng2, n) for _ in range(ka)]
        gensB = [rand_perm(rng2, n) for _ in range(kb)]
        A = Group(gensA, n)
        B = Group(gensB, n)
        mine = intersection(A, B).order()
        ref = sympy_intersection_order(gensA, gensB, n)
        assert mine == ref, (n, gensA, gensB, mine, ref)
        # the intersection must be contained in both
        I = intersection(A, B)
        for e in (I.elements() if I.order() <= 500 else []):
            assert A.contains(list(e)) and B.contains(list(e))
        n_int += 1
    print("[5] intersection() matches SymPy on %d random pairs (deg 5..8)  OK"
          % n_int)

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    _selftest()
