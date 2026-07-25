"""CSS quantum-code automorphisms: the qubit-permutation group and the method ladder.

The qubit-permutation automorphism group of a CSS code is ``Aut(Hx) ∩ Aut(Hz)`` -- the
column permutations preserving both stabiliser rowspaces.  :func:`automorphism_group` runs a
method ladder whose stages can be selected explicitly with ``method``:

  * ``"auto"`` (default) -- the full ladder, cheapest exact route first (never slower than the
    engine that would originally have solved a given code);
  * ``"leon"`` -- **only** the exact Leon engine + dual-code trick (``Aut(Hx)`` and ``Aut(Hz)``
    each by Leon, then a permutation-group intersection).  Affordable when
    ``eff_dim = max_side min(rank, n-rank) <= max_dim`` (Leon enumerates ``2**eff_dim``);
  * ``"bz"`` -- **only** the joint Brouwer--Zimmermann + nauty/Traces **graph**-automorphism
    combination (the joint Hx+Hz incidence, and its single-side rescue).  Best for LDPC codes
    (quasi-cyclic / bivariate-bicycle / toric): it enumerates only the low-weight codeword
    classes instead of ``2**eff_dim``, staying exact and cheap at any ``eff_dim``.

Every rung re-verifies each generator over GF(2); the best verified result wins.  Under
``"auto"``, when no exact engine certifies the group, the ladder falls back to internal
structural subgroups so the caller still gets a verified lower bound rather than nothing; such
a result is always flagged by ``complete=False``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import gf2
from . import graphaut
from . import joint
from . import side as side_mod
from . import cyclic
from . import permgroup
from ._interop import as_css
from .leon import automorphism_group as _leon_aut


# --------------------------------------------------------------------------------- CSS code

class CSSCode:
    """A CSS code given by its X- and Z-stabiliser parity-check matrices ``Hx`` (``mx x n``)
    and ``Hz`` (``mz x n``) over GF(2).  Only the data the automorphism engines need."""

    __slots__ = ("Hx", "Hz", "n", "k")

    def __init__(self, Hx, Hz, *, k: Optional[int] = None):
        self.Hx = gf2.as_uint8(Hx)
        self.Hz = gf2.as_uint8(Hz)
        if self.Hx.shape[1] != self.Hz.shape[1]:
            raise ValueError("Hx and Hz must have the same number of columns (qubits)")
        self.n = int(self.Hx.shape[1])
        if k is None:
            k = self.n - gf2.rank_gf2(self.Hx) - gf2.rank_gf2(self.Hz)
        self.k = int(k)

    def __repr__(self):
        return f"CSSCode(n={self.n}, k={self.k}, mx={self.Hx.shape[0]}, mz={self.Hz.shape[0]})"

    def to_arrays(self):
        """``(Hx, Hz)`` -- the CSS interop array form (see :mod:`codeaut._interop`)."""
        return self.Hx, self.Hz


@dataclass
class CSSAutResult:
    """Best-effort qubit-permutation automorphism result for a CSS code.

    * ``complete=True``  -- ``generators`` generate the FULL (exact) group ``Aut(Hx) ∩ Aut(Hz)``.
    * ``complete=False`` -- ``generators`` generate a *verified subgroup* (a rigorous lower bound).

    Every generator is re-verified over GF(2) to preserve both ``Hx`` and ``Hz`` rowspaces
    (``verified``).  ``order`` is the exact group order as a **decimal string** (these orders can
    exceed 64 bits, and JSON has no big-integer type — parse with ``int(result.order)`` when you
    need an integer).  ``generators`` are 0-indexed image lists (``perm[i]`` = image of qubit ``i``).
    """
    order: str
    generators: list
    complete: bool
    verified: bool
    method: str
    seconds: float
    n: int
    eff: Optional[dict] = field(default=None)

    def group(self) -> "permgroup.Group":
        """The result as a :class:`codeaut.permgroup.Group` on the ``n`` qubits."""
        return permgroup.Group(self.generators, self.n)


# --------------------------------------------------------------------------------- diagnostics

def effective_dims(Hx, Hz) -> dict:
    """Per-side ranks and the binding ``eff_dim = max_side min(rank, n-rank)`` (dual-code trick:
    ``Aut(C) = Aut(C^perp)`` lets the cheaper side bound the Leon enumeration)."""
    Hx = gf2.as_uint8(Hx)
    Hz = gf2.as_uint8(Hz)
    n = Hx.shape[1]
    rx, rz = gf2.rank_gf2(Hx), gf2.rank_gf2(Hz)
    ex, ez = min(rx, n - rx), min(rz, n - rz)
    return {"n": int(n), "rank_hx": int(rx), "rank_hz": int(rz),
            "eff_x": int(ex), "eff_z": int(ez), "eff_dim": int(max(ex, ez))}


# ------------------------------------------------------------------------------- leon + dual

def _leon_dual_exact(Hx, Hz, n, max_dim, eff, t0, intersect_cap: int = 50_000,
                     allow_sympy: bool = False) -> Optional[dict]:
    """Exact ``Aut(Hx) ∩ Aut(Hz)`` via Leon on the cheaper of each side and its dual, then a
    permutation-group intersection.  Returns a record, or ``None`` if ``eff_dim > max_dim`` or
    the intersection would be too costly.

    The intersection enumerates the smaller of ``Aut(Hx)``, ``Aut(Hz)``; when that side's order
    exceeds ``intersect_cap`` this path is abandoned (returns ``None``) so the caller falls
    through to the joint-incidence route -- which computes the same group directly, without any
    generic group intersection, and is faster when a per-side group is large.  Leon returns the
    per-side orders, so this decision is made before building anything expensive.
    """
    Bx, _, _, ex = gf2.dual_basis(Hx)
    Bz, _, _, ez = gf2.dual_basis(Hz)
    if max(ex, ez) > max_dim:
        return None
    rx = _leon_aut(Bx, max_dim=max_dim)
    rz = _leon_aut(Bz, max_dim=max_dim)
    Gx = permgroup.Group(rx.generators, n)
    Gz = permgroup.Group(rz.generators, n)
    # intersection() enumerates the smaller group when small, else a backtracking subgroup
    # search (SymPy when available).  If neither is feasible it raises and the ladder falls
    # through to the joint-incidence route.
    G = permgroup.intersection(Gx, Gz, max_enumerate=intersect_cap, allow_sympy=allow_sympy)
    order = G.order()
    gens = G.gens()
    verified = all(gf2.preserves_rowspace(Hx, gp) and gf2.preserves_rowspace(Hz, gp)
                   for gp in gens)
    how = "SymPy backtrack" if allow_sympy else "enumerate"
    return {
        "order": str(order),
        "generators": gens,
        "complete": True,
        "verified": verified,
        "method": f"perm_aut(Hx) ∩ perm_aut(Hz) (Leon two-pass + dual trick, eff_dim={max(ex, ez)}, {how} ∩)",
        "seconds": round(time.time() - t0, 3),
        "n": int(n),
        "eff": eff,
    }


# ------------------------------------------------------------------------------- the ladder

def _order_int(rec) -> int:
    return int(rec["order"])


def _better(a, b) -> bool:
    """Is record ``a`` strictly better than ``b``?  Exact beats inexact; then larger order."""
    if bool(a["complete"]) != bool(b["complete"]):
        return bool(a["complete"])
    return _order_int(a) > _order_int(b)


def _trivial_record(n, eff, t0, *, note: str = "no verified non-identity automorphism found") -> dict:
    return {"order": "1", "generators": [], "complete": False,
            "verified": True, "method": f"trivial group ({note})",
            "seconds": round(time.time() - t0, 3), "n": int(n), "eff": eff}


def _to_result(rec) -> CSSAutResult:
    return CSSAutResult(order=rec["order"],
                        generators=rec["generators"], complete=rec["complete"],
                        verified=rec["verified"], method=rec["method"],
                        seconds=rec["seconds"], n=rec["n"], eff=rec.get("eff"))


# Canonical method names and their accepted aliases.  ``"auto"`` runs the whole ladder;
# ``"leon"`` / ``"bz"`` restrict it to that single engine (see module docstring).
METHODS = ("auto", "leon", "bz")
_METHOD_ALIASES = {
    "auto": "auto", "ladder": "auto", "full": "auto",
    "leon": "leon", "leon_dual": "leon", "dual": "leon", "intersection": "leon",
    "bz": "bz", "joint": "bz", "graph": "bz", "incidence": "bz", "bz_graph": "bz",
    "nauty": "bz", "traces": "bz",
}


def _normalize_method(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_")
    if key not in _METHOD_ALIASES:
        raise ValueError(f"unknown method {method!r}; choose one of {METHODS} "
                         "(or an alias such as 'joint'/'graph' for 'bz')")
    return _METHOD_ALIASES[key]


def automorphism_group(code, *, Hz=None, method: str = "auto", max_dim: int = 24,
                       budget: int = 60_000_000, allow_subgroup: bool = True,
                       backend: str = "auto", max_threads: Optional[int] = None) -> CSSAutResult:
    """Qubit-permutation automorphism group of a CSS code, via the method ladder.

    ``code`` is a :class:`CSSCode`, a ``(Hx, Hz)`` tuple, or any CSS-code-like object accepted by
    :func:`codeaut.as_css` (``lib.CSSCode``, ``qecdb.Code``, ...); or pass ``Hx`` as ``code`` and
    ``Hz=...``.

    ``method`` selects the engine (see the module docstring):

      * ``"auto"`` (default) -- the full ladder, cheapest exact route first;
      * ``"leon"`` -- only the exact Leon + dual-code-trick intersection (needs ``eff_dim <=
        max_dim``);
      * ``"bz"`` (aliases ``"joint"`` / ``"graph"``) -- only the joint Brouwer--Zimmermann +
        nauty/Traces graph-automorphism combination (and its single-side rescue); best for LDPC
        codes.

    ``max_dim`` bounds the Leon ``2**eff_dim`` enumeration; ``budget`` the Brouwer--Zimmermann
    combination budget.  ``backend`` (``"auto"`` / ``"cpu"`` / ``"gpu"``) selects the
    Brouwer--Zimmermann enumeration backend; ``"gpu"`` transparently falls back to the CPU backend
    when no GPU is detected.  ``max_threads`` caps the CPU backend's worker threads (``None`` =>
    all hardware cores; only the CPU backend is threaded).  Returns a :class:`CSSAutResult`.
    """
    method = _normalize_method(method)
    if not (Hz is None and isinstance(code, CSSCode)):
        code = CSSCode(*as_css(code, Hz))
    Hx, Hz, n = code.Hx, code.Hz, code.n
    t0 = time.time()
    eff = effective_dims(Hx, Hz)
    threads = 0 if max_threads is None else int(max_threads)

    best = _run_ladder(Hx, Hz, n, eff, max_dim, budget, allow_subgroup, backend, t0,
                       method=method, threads=threads)
    if best is None:
        if method == "leon" and eff["eff_dim"] > max_dim:
            note = (f"method='leon' but eff_dim={eff['eff_dim']} > max_dim={max_dim}; "
                    "raise max_dim, or use method='bz' (best for LDPC codes) / 'auto'")
        else:
            note = f"method={method!r} found no verified non-identity automorphism"
        best = _trivial_record(n, eff, t0, note=note)
    return _finish(best, t0)


def _finish(rec, t0) -> CSSAutResult:
    rec = dict(rec)
    rec["seconds"] = round(time.time() - t0, 3)
    return _to_result(rec)


# ------------------------------------------------------------------------------- the ladder

def _run_ladder(Hx, Hz, n, eff, max_dim, budget, allow_subgroup, backend, t0, *,
                method="auto", threads=0):
    """The method ladder, returning the best verified record (or ``None``).  With ``method="auto"``
    all stages run cheapest-first, so the slow last-resort intersection is only reached when nothing
    else solved the code; otherwise only the stages belonging to the chosen engine run:

      1. Leon + dual, **enumerate** intersection      (``leon``);
      2. joint BZ + nauty/Traces incidence            (``bz``);
      3. Leon + dual, **SymPy-backtrack** intersection (``leon``, general exact path);
      4. single-side rescue                            (``bz``);
      5. cyclic/affine structural subgroup             (``auto`` only, lower-bound floor);
      6. Tanner-graph subgroup                         (``auto`` only, lower-bound floor).

    Stages 5-6 are internal: they are not selectable engines, they run only under ``"auto"``
    once the exact routes have failed, and anything they return is flagged ``complete=False``.

    ``method`` is one of ``"auto"`` / ``"leon"`` / ``"bz"`` (already normalized)."""
    best = None

    def want(engine):
        return method in ("auto", engine)

    def consider(rec):
        nonlocal best
        if rec is not None and rec.get("verified", False) and (best is None or _better(rec, best)):
            best = rec

    small_eff = eff["eff_dim"] <= max_dim

    # Stage 1 -- Leon + dual, cheap (enumerate) intersection.
    if want("leon") and small_eff:
        try:
            rec = _leon_dual_exact(Hx, Hz, n, max_dim, eff, t0, allow_sympy=False)
            consider(rec)
            if rec is not None and rec["complete"] and rec["verified"]:
                return best
        except Exception:
            pass

    # Stage 2 -- joint BZ + nauty/Traces incidence.
    if want("bz") and (best is None or not best["complete"]):
        try:
            rec = joint.joint_exact(Hx, Hz, max_dim=max_dim, budget=budget,
                                    allow_subgroup_fallback=allow_subgroup, backend=backend,
                                    threads=threads)
            consider(rec)
            if best is not None and best["complete"]:
                return best
        except Exception:
            pass

    # Stage 3 -- Leon + dual, SymPy-backtrack intersection (general exact path).
    if want("leon") and (best is None or not best["complete"]) and small_eff:
        try:
            rec = _leon_dual_exact(Hx, Hz, n, max_dim, eff, t0, intersect_cap=200_000,
                                   allow_sympy=True)
            consider(rec)
            if rec is not None and rec["complete"] and rec["verified"]:
                return best
        except Exception:
            pass

    # Stage 4 -- single-side rescue.
    if want("bz") and (best is None or not best["complete"]):
        try:
            rec = side_mod.side_aut_subgroup(Hx, Hz, budget=budget, backend=backend,
                                             threads=threads)
            consider(rec)
            if best is not None and best["complete"]:
                return best
        except Exception:
            pass

    # Stage 5 -- cyclic / affine structural subgroup (internal lower-bound floor, "auto" only).
    if method == "auto" and (best is None or not best["complete"]):
        try:
            order, gens = cyclic.affine_automorphism_group(Hx, Hz, n)
            if gens:
                consider({"order": str(order), "generators": gens, "complete": False,
                          "verified": True, "method": "affine/cyclic structural subgroup (verified)",
                          "seconds": round(time.time() - t0, 3), "n": int(n), "eff": eff})
        except Exception:
            pass

    # Stage 6 -- type-preserving Tanner-graph subgroup floor (internal, "auto" only): a pure
    # last resort, reached only when no earlier stage found anything at all.
    if method == "auto" and best is None and allow_subgroup \
            and graphaut.nauty_binary() is not None:
        try:
            consider(joint._tanner_subgroup(n, Hx, Hz, eff, t0))
        except Exception:
            pass
    return best
