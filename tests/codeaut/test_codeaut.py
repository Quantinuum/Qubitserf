"""codeaut test suite -- one runnable file (``python tests/test_codeaut.py``).

Covers: the Leon single-code engine, the vendored permutation-group layer, the
Brouwer--Zimmermann low-weight enumerator (native + completeness), the CSS method ladder
(known orders, leon-dual vs joint-incidence cross-validation), and explicit engine selection
(``method=``).  Requires ``numpy`` and a built ``libqubitserf`` (auto-built on first import); the
CSS colored-graph solves need system ``nauty``/``dreadnaut`` on PATH.
"""

from __future__ import annotations

import math
import sys
import time
from types import SimpleNamespace

import numpy as np

import qubitserf.codeaut as codeaut
from qubitserf.codeaut import codes, gf2, graphaut, lowweight, leon
from qubitserf.algebra import permgroup as pg


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
    hybrid = leon.automorphism_group(G, spanning_set="auto")
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


def test_classical_method_selection():
    # classical_automorphisms mirrors css_automorphisms' engine choice: 'leon', 'bz', 'auto'.
    # Both engines are exact, so on every code they must agree on the order AND on the group
    # itself (mutual containment of generators).
    simplex = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1],
                        [1, 0, 1, 0, 1, 0, 1]], np.uint8)
    rng = np.random.default_rng(7)
    random_code = rng.integers(0, 2, size=(5, 12), dtype=np.uint8)
    cases = [("steane Hx", codes.steane().Hx), ("surface(3) Hx", codes.surface(3).Hx),
             ("simplex [7,3,4]", simplex), ("random [12,<=5]", random_code)]
    for name, G in cases:
        rl = codeaut.classical_automorphisms(G, method="leon")
        rb = codeaut.classical_automorphisms(G, method="bz")
        assert rl.order == rb.order, (name, rl.order, rb.order)
        L, B = rl.group(), rb.group()
        assert all(L.contains(g) for g in B.gens()), name    # bz  subset of leon
        assert all(B.contains(g) for g in L.gens()), name    # leon subset of bz
        ra = codeaut.classical_automorphisms(G)              # 'auto' default, exact
        assert ra.order == rl.order, (name, ra.order)
    # the bz result carries its own dataclass with diagnostics
    rb = codeaut.classical_automorphisms(simplex, method="bz")
    assert isinstance(rb, codeaut.ClassicalAutResult)
    assert rb.order == 168 and rb.n == 7 and rb.dim == 3 and "exact" in rb.method
    # 'auto' uses Leon iff eff_dim = min(dim, n - dim) <= 20; every case above is small, so
    # auto returned Leon's AutResult.  'leon' itself runs on the cheaper of C / C^perp: a
    # high-dimensional code with a small dual must succeed without any max_dim escape hatch.
    big = rng.integers(0, 2, size=(9, 13), dtype=np.uint8)   # dim 9 > n - dim = 4
    ra = codeaut.classical_automorphisms(big)                # eff_dim = 4 -> Leon rung
    assert isinstance(ra, leon.AutResult)
    assert ra.order == codeaut.classical_automorphisms(big, method="bz").order
    # aliases resolve; unknown method names are rejected
    assert codeaut.classical_automorphisms(simplex, method="graph").order == 168
    for bad in ("nope", "joint", "partial"):
        try:
            codeaut.classical_automorphisms(simplex, method=bad)
            assert False, f"expected ValueError for method={bad!r}"
        except ValueError:
            pass
    print("  [classical] method selection OK (leon/bz/auto agree on "
          f"{len(cases)} codes; simplex=168)")


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
        cls, info = lowweight.low_weight_classes(B, full_enum_max_dim=6)
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
        grp = codeaut.css_automorphisms(codes.BUILTIN[name]())
        assert isinstance(grp, pg.Group), name
        assert grp.order() == expected, (name, grp.order())
        # the returned generators really generate the reported order
        assert pg.Group(grp.gens(), codes.BUILTIN[name]().n).order() == expected
    print(f"  [css] known orders OK: {cases}")


def test_leon_dual_vs_joint():
    from qubitserf.codeaut import css, joint
    for name in ("steane", "shor", "iceberg", "toric", "surface"):
        c = codes.toric(3) if name == "toric" else (
            codes.surface(3) if name == "surface" else codes.BUILTIN[name]())
        rl = css._leon_dual_exact(c.Hx, c.Hz, c.n, 24)
        rj = joint.joint_exact(c.Hx, c.Hz, max_dim=24)
        assert isinstance(rj["order"], int)                     # exact int, no strings
        if rl is not None:
            G, _how = rl
            assert G.order() == rj["order"], (name, G.order(), rj["order"])
    print("  [css] leon-dual and joint-incidence exact paths agree")


def test_method_selection():
    # Each selectable engine ('leon', 'bz') is exact-or-raise and agrees with the ladder on
    # codes it can solve.  Those two plus 'auto' are the whole public surface.
    from qubitserf.codeaut import css
    st = codes.steane()
    exact = codeaut.css_automorphism_group(st).order()           # auto ladder -> 168
    for m in ("leon", "bz"):
        grp = codeaut.css_automorphism_group(st, method=m)
        assert isinstance(grp, pg.Group) and grp.order() == exact, (m, grp.order())
    assert css.METHODS == ("auto", "leon", "bz"), css.METHODS
    # gross [[144,12,12]]: BZ+graph is exact (144)
    g_bz = codeaut.css_automorphism_group(codes.gross(), method="bz")
    assert g_bz.order() == 144
    # gross has eff_dim=66 >> max_dim=24, so Leon's 2**eff_dim enumeration is infeasible:
    # exact-or-raise means method='leon' must raise RuntimeError with an actionable message
    # (naming eff_dim / the leon stage, and pointing at method='bz').
    assert css.effective_dims(codes.gross().Hx, codes.gross().Hz)["eff_dim"] > css._MAX_DIM
    try:
        codeaut.css_automorphism_group(codes.gross(), method="leon")
        assert False, "expected RuntimeError for method='leon' on gross (eff_dim=66)"
    except RuntimeError as exc:
        msg = str(exc)
        assert "leon" in msg and "eff_dim" in msg and "bz" in msg, msg
    # aliases resolve ('joint' -> bz); unknown and retired method names are rejected
    assert codeaut.css_automorphisms(st, method="joint").order() == exact
    for bad in ("nope", "partial", "affine"):
        try:
            codeaut.css_automorphism_group(st, method=bad)
            assert False, f"expected ValueError for method={bad!r}"
        except ValueError:
            pass
    # a Group result feeds straight into group_intersection
    assert codeaut.group_intersection(
        codeaut.css_automorphisms(st), codeaut.css_automorphisms(st)).order() == exact
    print(f"  [css] method selection OK (leon/bz exact={exact}; "
          "leon on gross raises RuntimeError)")


def test_css_from_paulis():
    # Steane as Pauli strings: X-rows -> Hx, Z-rows -> Hz; the group has order 168.
    text = "IIIXXXX\nIXXIIXX\nXIXIXIX\nIIIZZZZ\nIZZIIZZ\nZIZIZIZ\n"
    Hx, Hz = codeaut.css_from_paulis(text)
    assert Hx.shape == (3, 7) and Hz.shape == (3, 7)
    assert codeaut.css_automorphism_group((Hx, Hz)).order() == 168
    # an iterable of lines parses identically to one newline-joined string
    Hx2, Hz2 = codeaut.css_from_paulis(text.splitlines())
    assert np.array_equal(Hx, Hx2) and np.array_equal(Hz, Hz2)
    # non-CSS / malformed inputs all raise ValueError
    for bad in ("XXXXIII\nYZIIZZI\n",   # a Y
                "XXZIIII\nZZZZIII\n",    # a row mixing X and Z
                "XXXX\nZZZZIII\n",       # ragged lengths
                "IIIXXXX\nABCDEFG\n"):   # illegal characters
        try:
            codeaut.css_from_paulis(bad)
            assert False, ("expected ValueError", bad)
        except ValueError:
            pass
    print("  [interop] css_from_paulis OK (Steane 168; non-CSS rejected)")


def test_verbose_progress():
    # verbose=True must print [codeaut] progress to stderr WITHOUT changing the result;
    # verbose=False (the default) must stay silent.
    import contextlib
    import io as _io
    simplex = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1],
                        [1, 0, 1, 0, 1, 0, 1]], np.uint8)
    quiet = codeaut.classical_automorphisms(simplex, method="bz")
    buf = _io.StringIO()
    with contextlib.redirect_stderr(buf):
        loud = codeaut.classical_automorphisms(simplex, method="bz", verbose=True)
    err = buf.getvalue()
    assert "[codeaut]" in err and err.count("\n") >= 2, err
    assert loud.order == quiet.order == 168
    assert loud.generators == quiet.generators

    st = codes.steane()
    quiet_css = codeaut.css_automorphisms(st)
    buf = _io.StringIO()
    with contextlib.redirect_stderr(buf):
        loud_css = codeaut.css_automorphisms(st, verbose=True)
    err_css = buf.getvalue()
    assert "[codeaut]" in err_css and "stage 1" in err_css and "done:" in err_css, err_css
    assert loud_css.order() == quiet_css.order() == 168

    # default (verbose=False) prints nothing to stderr
    buf = _io.StringIO()
    with contextlib.redirect_stderr(buf):
        codeaut.classical_automorphisms(simplex, method="bz")
        codeaut.css_automorphisms(st)
    assert buf.getvalue() == "", buf.getvalue()
    print("  [verbose] stderr progress emitted under verbose=True; silent by default")


def main():
    tests = [test_leon_single_code, test_leon_congruence_spanning_set,
             test_classical_method_selection,
             test_permgroup, test_bz_completeness,
             test_css_known_orders, test_leon_dual_vs_joint, test_method_selection,
             test_css_from_paulis, test_verbose_progress]
    print(f"codeaut {codeaut.version()} -- running {len(tests)} test groups")
    for t in tests:
        t()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
