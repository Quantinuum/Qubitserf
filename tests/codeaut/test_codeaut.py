"""codeaut test suite -- one runnable file (``python tests/test_codeaut.py``).

Covers: the Leon single-code engine, the vendored permutation-group layer, the
Brouwer--Zimmermann low-weight enumerator (native + completeness), the CSS method ladder
(known orders, leon-dual vs joint-incidence cross-validation), and explicit engine selection
(``method=``).  Requires ``numpy`` and a built ``libcodeaut`` (auto-built on first import); the
CSS colored-graph solves need system ``nauty``/``dreadnaut`` on PATH.
"""

from __future__ import annotations

import math
import sys
import time
from types import SimpleNamespace

import numpy as np

import qubitserf.codeaut as codeaut
from qubitserf.codeaut import codes, gf2, graphaut, permgroup as pg, lowweight, leon, ward, invariants


def test_leon_single_code():
    G = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]], np.uint8)
    r = leon.automorphism_group(G)                           # simplex [7,3,4] -> GL(3,2)
    assert r.order == 168 and r.n == 7
    # AutResult.group() -> a permgroup.Group that lists gens and enumerates all elements
    grp = r.group()
    assert grp.order() == 168 and grp.gens() and grp.reduced_generators()
    assert grp.contains(list(range(7))) and sum(1 for _ in grp.elements()) == 168
    H = np.array([[1, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1]], np.uint8)
    assert leon.automorphism_group(H).order == 168           # Hamming [7,4,3] (dual) -> 168
    assert leon.automorphism_group(np.ones((1, 5), np.uint8)).order == 120   # rep[5,1,5] -> S_5
    print("  [leon] single-code automorphism orders OK (168, 168, 120)")


def test_leon_congruence_spanning_set():
    # The surface-code X side is a compact regression where the minimum-weight prefix needs
    # weights 2 and 4 (7 words), while the complete wt == 1 (mod 3) class is just the five
    # weight-4 words and already spans.  Both incidences must return the same exact group.
    G = codes.surface(3).Hx
    legacy = leon.automorphism_group(G, spanning_set="minweight")
    modular = leon.automorphism_group(G, spanning_set="congruence")
    hybrid = codeaut.classical_automorphisms(G, spanning_set="auto")
    assert legacy.order == modular.order == hybrid.order == 32
    assert legacy.num_codewords == 7 and legacy.weight_classes == [2, 4]
    assert modular.num_codewords == 5 and modular.weight_classes == [4]
    assert modular.spanning_set == hybrid.spanning_set == "congruence"
    assert (modular.modulus, modular.residue) == (3, 1)
    assert modular.enumeration_seconds >= 0 and modular.search_seconds >= 0

    # The selected set is the COMPLETE residue class, not a sample, and it spans C.
    B = gf2.row_basis_gf2(G); k = B.shape[0]
    idx = np.arange(1, 1 << k); U = ((idx[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    C = (U @ B) % 2; weights = C.sum(axis=1)
    selected = C[(weights % modular.modulus) == modular.residue]
    assert selected.shape[0] == modular.num_codewords
    assert gf2.rank_gf2(selected) == k

    # Restricting the search to m=2 makes the even-code residue class contain all 15 words;
    # hybrid mode correctly retains the smaller legacy prefix instead.
    mod2 = leon.automorphism_group(G, spanning_set="congruence", max_modulus=2)
    auto2 = leon.automorphism_group(G, spanning_set="auto", max_modulus=2)
    minimal = leon.automorphism_group(G, spanning_set="minimal")
    assert mod2.num_codewords == 15 and mod2.modulus == 2 and mod2.residue == 0
    assert auto2.spanning_set == "minweight" and auto2.num_codewords == 7
    assert minimal.spanning_set == "minimal" and minimal.num_codewords == 6
    assert minimal.num_incidences == 20 and minimal.order == legacy.order

    # Equality of exact orders plus mutual generator containment cross-checks group equality.
    A, M = legacy.group(), modular.group()
    assert all(A.contains(g) for g in M.gens()) and all(M.contains(g) for g in A.gens())
    for kwargs in ({"spanning_set": "nope"},
                   {"spanning_set": "congruence", "max_modulus": 1},
                   {"spanning_set": "congruence", "max_modulus": 2**32 + 2},
                   {"max_dim": 1.5}):
        try:
            leon.automorphism_group(G, **kwargs)
            assert False, ("expected ValueError", kwargs)
        except ValueError:
            pass
    # The native Gray counter is uint64.  Dimensions >=63 must fail explicitly rather than
    # overflow ``1ULL << k`` and silently return a truncated, wrong group.
    for selector in ("minweight", "minimal", "auto", "congruence"):
        try:
            leon.automorphism_group(np.eye(64, dtype=np.uint8), max_dim=64,
                                     spanning_set=selector)
            assert False, ("expected uint64 enumeration guard", selector)
        except ValueError as exc:
            assert "maximum 62" in str(exc)
    print("  [leon] congruence selector exact; surface side 7 -> 5 incidence words")


def test_ward_power_of_two_residues():
    # Ward inclusion--exclusion, BDD counts, and fibers agree with brute force.
    rng = np.random.default_rng(20260714)
    for modulus in (2, 4, 8, 16):
        for _ in range(12):
            G = rng.integers(0, 2, size=(5, 9), dtype=np.uint8)
            form = ward.ward_form(G, modulus)
            diagram = ward.WardDecisionDiagram(form)
            truth = [[] for _ in range(modulus)]
            for mask in range(1 << form.dim):
                message = np.array([(mask >> bit) & 1 for bit in range(form.dim)], np.uint8)
                residue = int(((message @ form.generator) % 2).sum()) % modulus
                assert form.evaluate(message) == residue
                truth[residue].append(mask)
            assert diagram.residue_counts() == [len(items) for items in truth]
            # Conditioned residue counts drive modular singleton/pair/triple signatures.  Check
            # two arbitrary message characters against the same brute-force truth table.
            condition_masks = [sum(int(form.generator[bit, column]) << bit
                                   for bit in range(form.dim))
                               for column in range(min(2, form.n))]
            conditioned = diagram.conditioned_residue_counts(condition_masks)
            brute_conditioned = [[0] * (1 << len(condition_masks)) for _ in range(modulus)]
            for current_residue, masks in enumerate(truth):
                for mask in masks:
                    state = 0
                    for index, linear_mask in enumerate(condition_masks):
                        state |= ((mask & linear_mask).bit_count() & 1) << index
                    brute_conditioned[current_residue][state] += 1
            assert conditioned == tuple(tuple(row) for row in brute_conditioned)
            for residue, masks in enumerate(truth):
                assert sorted(diagram.message_masks(residue)) == masks
                span = ward.residue_span(form, residue)
                brute_messages = np.array(
                    [[(mask >> bit) & 1 for bit in range(form.dim)] for mask in masks],
                    dtype=np.uint8).reshape((-1, form.dim))
                assert span.dimension == gf2.rank_gf2(brute_messages)

    # Power-of-two regression family: a=7=2^3-1 singleton rows plus a disjoint repetition
    # row.  The current prefix needs 2^a=128 words, whereas wt == 1 (mod 8) is a complete
    # 8-word spanning fiber.  Both constructions produce exactly S_7 x S_9.
    a = 7
    G = np.zeros((a + 1, 2 * a + 3), dtype=np.uint8)
    G[:a, :a] = np.eye(a, dtype=np.uint8)
    G[a, a:2 * a + 2] = 1
    legacy = leon.automorphism_group(G)
    modular = ward.automorphism_group(G, modulus=8)
    assert legacy.num_codewords == 1 << a
    assert modular.residues == (1,) and modular.num_codewords == a + 1
    assert modular.weight_classes == [1, a + 2]
    assert modular.order == legacy.order == math.factorial(a) * math.factorial(a + 2)
    A, M = legacy.group(), modular.group()
    assert all(A.contains(g) for g in M.gens()) and all(M.contains(g) for g in A.gens())

    # Support-minimal codewords are a competing exact invariant: on this direct-sum family they
    # recover the same eight-word graph without modular arithmetic.
    cocircuits = leon.automorphism_group(G, spanning_set="cocircuit")
    assert cocircuits.spanning_set == "minimal"
    assert cocircuits.num_codewords == a + 1 and cocircuits.num_incidences == 2 * (a + 1)
    assert cocircuits.order == legacy.order

    # A hard diagram guard is an exact Leon fallback, never a partial answer.
    fallback = ward.automorphism_group(G, modulus=8, max_bdd_nodes=1)
    assert not fallback.residues and fallback.order == legacy.order
    assert "fallback" in fallback.method
    for bad in (1, 3, 6, 4.5, "4", True):
        try:
            ward.ward_form(G, bad)
            assert False, ("expected power-of-two ValueError", bad)
        except ValueError:
            pass
    print("  [ward] mod-2^t form/BDD exact; power-8 incidence 128 -> 8 words")


def test_invariant_portfolio():
    # One code exercises every forced route.  Some compact relations are strict overgroups and
    # must advertise an exact fallback; direct certificates must agree with Leon by mutual
    # containment, not just by order.
    G = codes.surface(3).Hx
    reference = leon.automorphism_group(G).group()
    for method in invariants.available_methods():
        result = invariants.automorphism_group(
            G, method=method, max_dim=12, max_enumerated=1 << 12,
            timeout=15, nauty_timeout=2, traces_timeout=15)
        actual = result.group()
        assert result.exact and result.complete
        assert actual.order() == reference.order() == result.order
        assert all(reference.contains(generator) for generator in actual.gens())
        assert all(actual.contains(generator) for generator in reference.gens())

    lcd = invariants.automorphism_group(G, method="lcd")
    assert lcd.method == "lcd-projector" and not lcd.used_fallback
    assert invariants.orthogonal_projector(G) is not None

    simplex = codes.steane().Hx
    assert invariants.orthogonal_projector(simplex) is None
    lcd_fallback = invariants.automorphism_group(simplex, method="lcd")
    assert lcd_fallback.used_fallback and lcd_fallback.order == 168
    try:
        invariants.automorphism_group(simplex, method="lcd", fallback=False)
        assert False, "expected a non-LCD guard with fallback disabled"
    except invariants.InvariantLimitExceeded:
        pass
    zero_residue = invariants.automorphism_group(
        np.zeros((0, 5), dtype=np.uint8), method="residue", fallback=False)
    assert zero_residue.order == math.factorial(5) and not zero_residue.used_fallback

    # Exact component wreath product: four equivalent length-three repetition components.
    blocks = 4
    repeated = np.zeros((blocks, 3 * blocks), dtype=np.uint8)
    for block in range(blocks):
        repeated[block, 3 * block:3 * block + 3] = 1
    component = invariants.automorphism_group(repeated, method="components")
    assert component.method == "matroid-component-wreath"
    assert component.order == math.factorial(3) ** blocks * math.factorial(blocks)
    assert all(gf2.preserves_rowspace(repeated, generator)
               for generator in component.generators)

    small = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    stabilizer, orbit_size = invariants.rowspace_stabilizer(
        pg.symmetric_group(4), small, max_orbit=100)
    small_reference = leon.automorphism_group(small).group()
    assert orbit_size > 1 and stabilizer.order() == small_reference.order()
    assert all(stabilizer.contains(generator) for generator in small_reference.gens())
    assert all(small_reference.contains(generator) for generator in stabilizer.gens())

    # A failed/unparseable dreadnaut process must raise into the exact fallback.  Treating empty
    # output as an empty generating set would vacuously pass verification and silently return a
    # wrong trivial group.
    saved_run = graphaut.subprocess.run
    try:
        for fake in (SimpleNamespace(stdout="", stderr="synthetic failure", returncode=1),
                     SimpleNamespace(stdout="unparseable success", stderr="", returncode=0)):
            graphaut.subprocess.run = lambda *_args, _fake=fake, **_kwargs: _fake
            try:
                graphaut.relation_group([0, 0], [[0, 0], [0, 0]])
                assert False, "expected failed/unparseable dreadnaut process to raise"
            except RuntimeError:
                pass
    finally:
        graphaut.subprocess.run = saved_run

    # Public resource guards are validated before trivial-length shortcuts.  Zero values must
    # not leak inconsistent downstream ValueErrors or let an invalid Ward modulus appear valid.
    trivial = np.zeros((0, 1), dtype=np.uint8)
    bad_calls = [
        {"method": "components", "max_geometry_candidates": 0},
        {"method": "residue", "max_form_operations": 0},
        {"method": "residue", "max_residue_indicator_terms": 0},
        {"method": "modular", "max_bdd_nodes": 0},
        {"method": "combined", "max_bdd_states": 0},
        {"method": "minors", "max_minor_hull_work": 0},
        {"method": "auto", "max_bdd_states": 1.5},
        {"method": "lcd", "max_words": 0},
        {"method": "residue", "modulus": 3},
    ]
    for kwargs in bad_calls:
        try:
            invariants.automorphism_group(trivial, **kwargs)
            assert False, ("expected portfolio argument ValueError", kwargs)
        except ValueError:
            pass

    # The bounded route's BZ discovery remains output-sensitive above the exhaustive dimension
    # guard: the complete weight-three class spans 24 disjoint repetition components.
    blocks = 24
    high_dimensional = np.zeros((blocks, 3 * blocks), dtype=np.uint8)
    for block in range(blocks):
        high_dimensional[block, 3 * block:3 * block + 3] = 1
    bounded_cells, bounded_info = invariants._bounded_invariant_cells(
        high_dimensional, max_support_weight=3, max_bounded_subsets=1_000,
        max_bounded_bz_budget=1_000_000, max_bounded_class_size=10_000)
    assert [(cell["label"], len(cell["rows"])) for cell in bounded_cells] == [
        (("cocircuits", 3), blocks)]
    assert bounded_info["subsets_tested"] == 0
    assert bounded_info["bz"]["cocircuits"]["spans"]

    # Full shortened-hull weight enumerators distinguish coordinates that the former
    # (dimension, hull-dimension)-only signature merged.
    hull_fixture = np.array([[1, 1, 0, 1, 1], [0, 0, 1, 0, 1]], dtype=np.uint8)
    signature_kwargs = dict(
        max_hull_dimension=12, max_hull_words=1 << 14,
        max_schur_products=200_000)
    minor_zero = invariants._minor_invariant(hull_fixture, [0], **signature_kwargs)
    minor_two = invariants._minor_invariant(hull_fixture, [2], **signature_kwargs)
    assert tuple(part[:2] for part in minor_zero) == tuple(part[:2] for part in minor_two)
    assert minor_zero != minor_two

    # The conductor identity can reveal an LCD characteristic layer even when C and C^2 are
    # non-LCD; this fixture guards the expanded Schur portfolio.
    conductor_fixture = np.array(
        [[1, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1]], dtype=np.uint8)
    derived, _skipped = invariants._characteristic_codes(
        conductor_fixture, max_schur_products=200_000)
    conductor = dict(derived)["conductor-code-code"]
    assert len(conductor) == 2 and invariants.orthogonal_projector(conductor) is not None

    print("  [invariants] 14-route exact portfolio + guards/SSA/BZ/conductors OK")


def test_permgroup():
    for n in range(1, 8):
        assert pg.symmetric_group(n).order() == math.factorial(n)
    # cross-validate order against sympy if available (oracle only)
    try:
        from sympy.combinatorics import Permutation, PermutationGroup
        rng = np.random.default_rng(0)
        for _ in range(150):
            n = int(rng.integers(4, 9))
            gens = [list(rng.permutation(n)) for _ in range(int(rng.integers(1, 4)))]
            assert pg.Group(gens, n).order() == \
                PermutationGroup([Permutation(g) for g in gens]).order()
        # intersection
        for _ in range(80):
            n = int(rng.integers(4, 8))
            A = [list(rng.permutation(n)) for _ in range(2)]
            B = [list(rng.permutation(n)) for _ in range(2)]
            I = pg.intersection(pg.Group(A, n), pg.Group(B, n))
            SA = PermutationGroup([Permutation(g) for g in A])
            SB = PermutationGroup([Permutation(g) for g in B])
            small, other = (SA, SB) if SA.order() <= SB.order() else (SB, SA)
            ref = PermutationGroup(list(set(el for el in small.generate()
                                            if other.contains(el)))
                                   or [Permutation(list(range(n)))]).order()
            assert I.order() == ref
        print("  [permgroup] order + intersection cross-validated vs sympy")
    except ImportError:
        print("  [permgroup] symmetric-group orders OK (sympy not present; cross-val skipped)")


def test_bz_completeness():
    rng = np.random.default_rng(3)
    nclasses = 0
    for _ in range(40):
        k = int(rng.integers(8, 14)); n = int(rng.integers(k + 1, 2 * k + 5))
        B = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        if gf2.rank_gf2(B) != k:
            continue
        cls, info = lowweight.low_weight_classes(B, full_enum_max_dim=6, budget=10 ** 9)
        assert info["method"] == "bz"
        Bb = gf2.row_basis_gf2(B)
        idx = np.arange(1, 1 << k); U = ((idx[:, None] >> np.arange(k)) & 1).astype(np.uint8)
        C = (U @ Bb) % 2; w = C.sum(1)
        truth = {int(ww): set(r.tobytes() for r in C[w == ww]) for ww in np.unique(w) if ww > 0}
        for wt, rows in cls:                       # every returned class is certified-complete
            nclasses += 1
            assert set(r.tobytes() for r in rows) == truth.get(int(wt), set())
        if info["spans"]:
            assert gf2.rank_gf2(np.vstack([rows for _, rows in cls])) == k
    print(f"  [bz] {nclasses} certified classes match brute force; backends={codeaut.available_backends()}")


def test_css_known_orders():
    cases = {"steane": 168, "shor": 1296, "gross": 144}
    for name, expected in cases.items():
        r = codeaut.css_automorphism_group(codes.BUILTIN[name]())
        assert r.complete and r.verified and int(r.order) == expected, (name, r.order)
        # the returned generators really generate the reported order
        assert pg.Group(r.generators, r.n).order() == expected
    print(f"  [css] known orders OK: {cases}")


def test_leon_dual_vs_joint():
    from qubitserf.codeaut import css, joint
    for name in ("steane", "shor", "iceberg", "toric", "surface"):
        c = codes.toric(3) if name == "toric" else (
            codes.surface(3) if name == "surface" else codes.BUILTIN[name]())
        eff = css.effective_dims(c.Hx, c.Hz)
        rl = css._leon_dual_exact(c.Hx, c.Hz, c.n, 24, eff, time.time())
        rj = joint.joint_exact(c.Hx, c.Hz, max_dim=24)
        if rl is not None and rl["complete"] and rj["complete"]:
            assert rl["order"] == rj["order"], (name, rl["order"], rj["order"])
    print("  [css] leon-dual and joint-incidence exact paths agree")


def test_method_selection():
    # Each selectable engine ('leon', 'bz') is exact and agrees with the ladder on codes it can
    # solve.  Those two plus 'auto' are the whole public surface: the structural-subgroup floor is
    # internal to 'auto' and is not selectable.
    from qubitserf.codeaut import css
    st = codes.steane()
    exact = int(codeaut.css_automorphism_group(st).order)        # auto ladder -> 168
    for m in ("leon", "bz"):
        r = codeaut.css_automorphism_group(st, method=m)
        assert r.verified and r.complete and int(r.order) == exact, (m, r.order)
        assert pg.Group(r.generators, r.n).order() == exact
    assert css.METHODS == ("auto", "leon", "bz"), css.METHODS
    # gross [[144,12,12]]: BZ+graph is exact (144)
    g_bz = codeaut.css_automorphism_group(codes.gross(), method="bz")
    assert g_bz.verified and g_bz.complete and int(g_bz.order) == 144
    # gross has eff_dim=66 >> max_dim=24, so Leon's 2**eff_dim enumeration is infeasible: leon must
    # not silently return a bare order=1 but carry a note naming eff_dim and pointing at bz/max_dim.
    g_leon = codeaut.css_automorphism_group(codes.gross(), method="leon")
    assert not g_leon.complete and int(g_leon.order) == 1
    assert "eff_dim" in g_leon.method and ("max_dim" in g_leon.method or "bz" in g_leon.method), \
        g_leon.method
    # aliases resolve ('joint' -> bz); unknown and retired method names are rejected
    assert codeaut.css_automorphisms(st, method="joint").complete
    for bad in ("nope", "partial", "affine"):
        try:
            codeaut.css_automorphism_group(st, method=bad)
            assert False, f"expected ValueError for method={bad!r}"
        except ValueError:
            pass
    print(f"  [css] method selection OK (leon/bz exact={exact})")


def test_cli_pauli_stabilisers():
    from qubitserf.codeaut import cli
    # Steane as Pauli strings: X-rows -> Hx, Z-rows -> Hz; the group has order 168.
    text = "IIIXXXX\nIXXIIXX\nXIXIXIX\nIIIZZZZ\nIZZIIZZ\nZIZIZIZ\n"
    Hx, Hz = cli._css_from_pauli(text)
    assert Hx.shape == (3, 7) and Hz.shape == (3, 7)
    assert int(codeaut.css_automorphism_group((Hx, Hz)).order) == 168
    # non-CSS / malformed inputs all raise ValueError
    for bad in ("XXXXIII\nYZIIZZI\n",   # a Y
                "XXZIIII\nZZZZIII\n",    # a row mixing X and Z
                "XXXX\nZZZZIII\n",       # ragged lengths
                "IIIXXXX\nABCDEFG\n"):   # illegal characters
        try:
            cli._css_from_pauli(bad)
            assert False, ("expected ValueError", bad)
        except ValueError:
            pass
    print("  [cli] Pauli-stabiliser CSS parsing OK (Steane 168; non-CSS rejected)")


def main():
    tests = [test_leon_single_code, test_leon_congruence_spanning_set,
             test_ward_power_of_two_residues,
             test_invariant_portfolio,
             test_permgroup, test_bz_completeness,
             test_css_known_orders, test_leon_dual_vs_joint, test_method_selection,
             test_cli_pauli_stabilisers]
    print(f"codeaut {codeaut.version()} -- running {len(tests)} test groups")
    for t in tests:
        t()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
