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

The contract is **exact-or-raise**: every stage either certifies the FULL group (each
generator GF(2)-re-verified to preserve both rowspaces) and the ladder returns it as a
:class:`qubitserf.algebra.permgroup.Group`, or the stage fails and the ladder moves on.  When
no stage certifies, :func:`automorphism_group` raises :class:`RuntimeError` listing what was
tried and why each attempt failed -- it NEVER returns a partial / lower-bound result.  A
certified trivial group (order 1) is a valid exact answer.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

from . import gf2
from . import joint
from . import side as side_mod
from ..algebra import permgroup
from ._interop import as_css
from .leon import automorphism_group as _leon_aut


def _log(msg: str) -> None:
    print(f"[codeaut] {msg}", file=sys.stderr, flush=True)


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

def _leon_dual_exact(Hx, Hz, n, max_dim, *, intersect_cap: int = 50_000,
                     allow_sympy: bool = False):
    """Exact ``Aut(Hx) ∩ Aut(Hz)`` via Leon on the cheaper of each side and its dual, then a
    permutation-group intersection.  Returns ``(Group, how)`` -- ``how`` a human-readable
    method string -- or ``None`` when ``eff_dim > max_dim`` (Leon enumerates ``2**eff_dim``).

    The intersection enumerates the smaller of ``Aut(Hx)``, ``Aut(Hz)``; when that is
    infeasible under ``intersect_cap`` / ``allow_sympy``, :func:`permgroup.intersection`
    raises and the ladder falls through to the joint-incidence route -- which computes the
    same group directly, without any generic group intersection, and is faster when a
    per-side group is large.  Every generator of the intersection is re-verified over GF(2);
    a verification failure raises (it would indicate an engine bug).
    """
    Bx, _, _, ex = gf2.dual_basis(Hx)
    Bz, _, _, ez = gf2.dual_basis(Hz)
    if max(ex, ez) > max_dim:
        return None
    rx = _leon_aut(Bx, max_dim=max_dim)
    rz = _leon_aut(Bz, max_dim=max_dim)
    Gx = permgroup.Group(rx.generators, n)
    Gz = permgroup.Group(rz.generators, n)
    G = permgroup.intersection(Gx, Gz, max_enumerate=intersect_cap, allow_sympy=allow_sympy)
    for gp in G.gens():
        if not (gf2.preserves_rowspace(Hx, gp) and gf2.preserves_rowspace(Hz, gp)):
            raise RuntimeError("leon+dual: an intersection generator failed GF(2) "
                               "re-verification (engine bug)")
    how = "SymPy backtrack" if allow_sympy else "enumerate"
    return G, (f"perm_aut(Hx) ∩ perm_aut(Hz) (Leon two-pass + dual trick, "
               f"eff_dim={max(ex, ez)}, {how} ∩)")


# ------------------------------------------------------------------------------- the ladder

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


# Fixed engine parameter (formerly the max_dim keyword argument): bounds Leon's 2**eff_dim
# enumeration (the native Gray-code enumerator hard-caps at 62 anyway); the BZ enumeration
# runs for as long as it needs (kill the process if it takes too long).
_MAX_DIM = 24


def automorphism_group(code, *, Hz=None, method: str = "auto",
                       backend: str = "auto", max_threads: Optional[int] = None,
                       verbose: bool = False) -> permgroup.Group:
    """Exact qubit-permutation automorphism group of a CSS code, via the method ladder.

    ``code`` is a :class:`CSSCode`, a ``(Hx, Hz)`` tuple, or any CSS-code-like object accepted by
    :func:`codeaut.as_css` (``lib.CSSCode``, ``qecdb.Code``, ...); or pass ``Hx`` as ``code`` and
    ``Hz=...``.

    ``method`` selects the engine (see the module docstring):

      * ``"auto"`` (default) -- the full ladder, cheapest exact route first;
      * ``"leon"`` -- only the exact Leon + dual-code-trick intersection (needs ``eff_dim <=
        24``, since Leon enumerates ``2**eff_dim``);
      * ``"bz"`` (aliases ``"joint"`` / ``"graph"``) -- only the joint Brouwer--Zimmermann +
        nauty/Traces graph-automorphism combination (and its single-side rescue); best for LDPC
        codes.

    ``backend`` (``"auto"`` / ``"cpu"`` / ``"gpu"``) selects the Brouwer--Zimmermann enumeration
    backend; ``"gpu"`` transparently falls back to the CPU backend when no GPU is detected.
    ``max_threads`` caps the CPU backend's worker threads (``None`` => all hardware cores; only
    the CPU backend is threaded).  ``verbose`` prints one-line ``[codeaut]`` ladder-progress
    messages to **stderr** (never stdout); it never changes the result.

    Returns the exact, certified group ``Aut(Hx) ∩ Aut(Hz)`` as a
    :class:`qubitserf.algebra.permgroup.Group` (``.order()`` is an exact Python int;
    ``.gens()`` are 0-indexed image lists).  Raises :class:`RuntimeError` when no engine can
    certify the full group -- the message lists which stages were tried, why each failed, and
    what to try instead.  Partial / lower-bound results are never returned.
    """
    method = _normalize_method(method)
    if not (Hz is None and isinstance(code, CSSCode)):
        code = CSSCode(*as_css(code, Hz))
    Hx, Hz, n = code.Hx, code.Hz, code.n
    t0 = time.time()
    eff = effective_dims(Hx, Hz)
    threads = 0 if max_threads is None else int(max_threads)

    G, how, failures = _run_ladder(Hx, Hz, n, eff, _MAX_DIM, backend, method=method,
                                   threads=threads, verbose=verbose)
    if G is None:
        msg = _failure_message(method, n, eff, failures)
        if verbose:
            _log(f"all stages failed after {round(time.time() - t0, 3)}s -- raising")
        raise RuntimeError(msg)
    if verbose:
        _log(f"done: order={G.order()}, method={how!r}, "
             f"seconds={round(time.time() - t0, 3)}")
    return G


def _failure_message(method, n, eff, failures) -> str:
    lines = [f"css automorphism group: no engine certified the exact Aut(Hx) ∩ Aut(Hz) "
             f"(method={method!r}, n={n}, eff_dim={eff['eff_dim']}); tried:"]
    lines += [f"  - {f}" for f in failures] or ["  - (no stage was applicable)"]
    if method == "leon":
        lines.append("try method='bz' (joint BZ + nauty/Traces incidence; exact at any "
                     "eff_dim, best for LDPC codes) or method='auto'")
    elif method == "bz":
        hint = ("try method='leon' (Leon + dual-code trick; needs eff_dim <= "
                f"{_MAX_DIM}) or method='auto'"
                if eff["eff_dim"] <= _MAX_DIM else
                "try method='auto'; check that system nauty/dreadnaut is installed "
                "and that the BZ backend is available")
        lines.append(hint)
    else:
        lines.append("all engines failed; check that system nauty/dreadnaut is installed, "
                     "or inspect the per-stage reasons above")
    return "\n".join(lines)


def _run_ladder(Hx, Hz, n, eff, max_dim, backend, *, method="auto", threads=0,
                verbose=False):
    """The method ladder.  Returns ``(Group, how, failures)``: the first stage to certify the
    exact group wins (``Group`` is ``None`` when every stage failed; ``failures`` collects one
    human-readable reason per failed/skipped stage).  With ``method="auto"`` all stages run
    cheapest-first; otherwise only the stages belonging to the chosen engine run:

      1. Leon + dual, **enumerate** intersection       (``leon``);
      2. joint BZ + nauty/Traces incidence             (``bz``);
      3. Leon + dual, **SymPy-backtrack** intersection (``leon``, general exact path);
      4. single-side rescue (exact routes only)        (``bz``).

    Every stage is exact: it either certifies the FULL group (GF(2)-re-verified generators)
    or contributes a failure reason.  ``method`` is one of ``"auto"`` / ``"leon"`` / ``"bz"``
    (already normalized)."""
    failures = []

    def log(msg):
        if verbose:
            _log(msg)

    def want(engine):
        return method in ("auto", engine)

    def fail(stage, reason):
        failures.append(f"{stage}: {reason}")
        log(f"{stage}: {reason}")

    small_eff = eff["eff_dim"] <= max_dim
    if want("leon") and not small_eff:
        fail("leon+dual (stages 1&3)",
             f"skipped -- eff_dim={eff['eff_dim']} > {max_dim} "
             "(Leon enumerates 2**eff_dim)")

    # Stage 1 -- Leon + dual, cheap (enumerate) intersection.
    if want("leon") and small_eff:
        stage = "stage 1 (leon+dual, enumerate ∩)"
        try:
            log(f"{stage}: starting (eff_dim={eff['eff_dim']} <= {max_dim})")
            res = _leon_dual_exact(Hx, Hz, n, max_dim, allow_sympy=False)
            if res is not None:
                G, how = res
                log(f"{stage}: certified exact, order={G.order()}")
                return G, how, failures
            fail(stage, "not applicable (eff_dim over cap)")
        except Exception as exc:
            fail(stage, f"{type(exc).__name__}: {exc}")

    # Stage 2 -- joint BZ + nauty/Traces incidence.
    if want("bz"):
        stage = "stage 2 (joint BZ + nauty/Traces incidence)"
        try:
            log(f"{stage}: starting (backend={backend})")
            rec = joint.joint_exact(Hx, Hz, max_dim=max_dim, backend=backend,
                                    threads=threads)
            G = permgroup.Group(rec["generators"], n)
            log(f"{stage}: certified exact, order={rec['order']}")
            return G, rec["method"], failures
        except Exception as exc:
            fail(stage, f"{type(exc).__name__}: {exc}")

    # Stage 3 -- Leon + dual, SymPy-backtrack intersection (general exact path).
    if want("leon") and small_eff:
        stage = "stage 3 (leon+dual, SymPy-backtrack ∩)"
        try:
            log(f"{stage}: starting")
            res = _leon_dual_exact(Hx, Hz, n, max_dim, intersect_cap=200_000,
                                   allow_sympy=True)
            if res is not None:
                G, how = res
                log(f"{stage}: certified exact, order={G.order()}")
                return G, how, failures
            fail(stage, "not applicable (eff_dim over cap)")
        except Exception as exc:
            fail(stage, f"{type(exc).__name__}: {exc}")

    # Stage 4 -- single-side rescue (exact routes only: all-gens-verify or small-overgroup
    # enumeration filter).
    if want("bz"):
        stage = "stage 4 (single-side rescue)"
        try:
            log(f"{stage}: starting")
            rec = side_mod.side_aut_subgroup(Hx, Hz, backend=backend, threads=threads)
            G = permgroup.Group(rec["generators"], n)
            log(f"{stage}: certified exact, order={rec['order']}")
            return G, rec["method"], failures
        except Exception as exc:
            fail(stage, f"{type(exc).__name__}: {exc}")

    return None, None, failures
