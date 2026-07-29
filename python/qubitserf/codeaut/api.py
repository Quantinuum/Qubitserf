"""codeaut -- the easy interface.

Three high-level convenience entry points that cover the common tasks, each accepting flexible
inputs:

* :func:`classical_automorphisms` -- ``Aut(C)`` of a binary linear code (pick the engine with
  ``method``: Leon's algorithm, or BZ low-weight classes + nauty/Traces);
* :func:`css_automorphisms`       -- ``Aut(Hx) ∩ Aut(Hz)`` of a CSS quantum code (method ladder;
  pick the engine with ``method``); exact-or-raise, returns a
  :class:`qubitserf.algebra.permgroup.Group`;
* :func:`group_intersection`      -- the intersection of two permutation groups.

These wrap the lower-level engines (``joint_exact``, ``side_aut_subgroup``,
``low_weight_classes``, the ``permgroup`` toolkit, ...); reach for those directly only when you
need a specific engine, backend, or diagnostic.  See the README for worked examples.
"""
from __future__ import annotations

import sys
import time

from . import gf2
from . import leon
from . import classical_bz as _classical_bz
from . import css as _css
from ..algebra import permgroup
from ._interop import as_css


def _log(msg: str) -> None:
    """One-line ``[codeaut]`` progress message on stderr (verbose mode only)."""
    print(f"[codeaut] {msg}", file=sys.stderr, flush=True)


# Canonical classical engine names and accepted aliases (mirrors css._METHOD_ALIASES).
_CLASSICAL_METHOD_ALIASES = {
    "auto": "auto", "ladder": "auto", "full": "auto",
    "leon": "leon",
    "bz": "bz", "graph": "bz", "incidence": "bz", "bz_graph": "bz",
    "nauty": "bz", "traces": "bz",
}


_LEON_EFF_DIM_MAX = 20     # "auto" uses Leon iff eff_dim = min(dim, n - dim) <= this


def classical_automorphisms(genmat, *, method: str = "auto",
                            backend: str = "auto", max_threads=None,
                            verbose: bool = False):
    """Permutation automorphism group ``Aut(C)`` of the binary linear code ``C =
    rowspace(genmat)``.

    ``genmat`` is an ``(m, n)`` array of 0/1 entries (any generating set; row-reduced
    internally).  Both engines are **exact**; the result has ``.generators`` (0-indexed image
    lists), the exact integer ``.order``, and ``.group()``.

    ``method`` selects the engine:

      * ``"auto"`` (default) -- Leon when ``eff_dim = min(dim(C), n - dim(C)) <= 20``
        (Leon's enumeration costs ``2**eff_dim`` via the dual-code trick,
        ``Aut(C) = Aut(C^perp)``), the ``bz`` route otherwise; if the chosen rung fails,
        the other is tried before raising.  Never approximates: if no engine can certify
        the exact group, it raises;
      * ``"leon"`` -- the exact Leon partition-backtracking engine
        (:func:`codeaut.code_automorphism_group`) on the cheaper of ``C`` / ``C^perp``,
        with ascending min-weight codeword classes.  Returns :class:`codeaut.AutResult`;
      * ``"bz"`` (aliases ``"graph"`` / ``"nauty"``) -- certified-complete Brouwer--Zimmermann
        low-weight classes of both ``C`` and ``C^perp``, solving the smaller certified
        incidence with nauty/Traces (:mod:`codeaut.classical_bz`); exact at any ``dim`` and
        best for LDPC-like codes; needs system ``nauty``.  Returns
        :class:`codeaut.classical_bz.ClassicalAutResult`.

    ``backend`` (``"auto"`` / ``"cpu"`` / ``"gpu"``) picks the BZ enumeration backend;
    ``max_threads`` caps the CPU backend's worker threads (``None`` => all hardware cores).
    ``verbose`` prints one-line ``[codeaut]`` progress messages to **stderr** (which rung
    is chosen and why, per-rung timing/order, auto-fallthrough reasons); it never changes
    the result.
    """
    key = str(method).strip().lower().replace("-", "_")
    if key not in _CLASSICAL_METHOD_ALIASES:
        raise ValueError(f"unknown method {method!r}; choose one of ('auto', 'leon', 'bz') "
                         "(or an alias such as 'graph'/'nauty' for 'bz')")
    method = _CLASSICAL_METHOD_ALIASES[key]
    threads = 0 if max_threads is None else int(max_threads)
    B, n, r, eff = gf2.dual_basis(gf2.as_uint8(genmat))

    def _leon():
        # the cheaper of C / C^perp (dual-code trick): the group is the same and Leon's
        # enumeration cost drops from 2**dim to 2**eff_dim.
        if verbose:
            _log(f"leon rung: Leon on the cheaper of C/C^perp "
                 f"(n={n}, dim={r}, eff_dim={eff}; enumerates 2**{eff})")
            t = time.time()
        res = leon.automorphism_group(B, max_dim=eff)
        if verbose:
            _log(f"leon rung: done in {time.time() - t:.3f}s, order={res.order}")
        return res

    def _bz():
        if verbose:
            _log(f"bz rung: BZ weight classes + nauty/Traces incidence "
                 f"(n={n}, dim={r}, backend={backend})")
            t = time.time()
        res = _classical_bz.automorphism_group(genmat, backend=backend, threads=threads,
                                               verbose=verbose)
        if verbose:
            _log(f"bz rung: done in {time.time() - t:.3f}s, order={res.order}")
        return res

    if method == "leon":
        return _leon()
    if method == "bz":
        return _bz()
    # "auto": Leon for eff_dim <= 20 (cost 2**eff_dim), bz above (both rungs are exact; a rung
    # either certifies the full group or raises, so falling through never loses exactness).
    first, second = (_leon, _bz) if eff <= _LEON_EFF_DIM_MAX else (_bz, _leon)
    if verbose:
        which = "leon" if first is _leon else "bz"
        _log(f"auto: eff_dim={eff} {'<=' if eff <= _LEON_EFF_DIM_MAX else '>'} "
             f"{_LEON_EFF_DIM_MAX} -> trying {which!r} first")
    try:
        return first()
    except Exception as exc:
        if verbose:
            _log(f"auto: first rung failed ({type(exc).__name__}: {exc}); "
                 "falling through to the other rung")
        return second()


def css_automorphisms(code, Hz=None, *, method="auto", backend="auto",
                      max_threads=None, verbose: bool = False) -> "permgroup.Group":
    """Exact qubit-permutation automorphism group ``Aut(Hx) ∩ Aut(Hz)`` of a CSS code, via the
    method ladder.

    ``code`` is any CSS-code-like object accepted by :func:`codeaut.as_css` -- a
    :class:`codeaut.CSSCode`, an ``(Hx, Hz)`` pair (pass ``Hz`` as the second positional
    argument or ``code`` as a 2-tuple), or anything exposing ``.Hx``/``.Hz`` or
    ``.to_arrays()``.

    ``method`` selects the engine:

      * ``"auto"`` (default) -- the full ladder, cheapest exact route first;
      * ``"leon"`` -- only the exact Leon + dual-code-trick intersection;
      * ``"bz"`` (aliases ``"joint"`` / ``"graph"``) -- only the joint Brouwer--Zimmermann +
        nauty/Traces graph-automorphism combination (and its single-side rescue); best for LDPC
        codes.

    ``backend`` (``"auto"`` / ``"cpu"`` / ``"gpu"``) selects the Brouwer--Zimmermann enumeration
    backend; ``"gpu"`` transparently falls back to the CPU backend when no GPU is detected (the
    result is identical).  ``max_threads`` caps the CPU backend's worker threads (``None`` => all
    hardware cores; only the CPU backend is threaded).  ``verbose`` prints one-line
    ``[codeaut]`` ladder-progress messages to **stderr** (each stage as it starts, which stage
    certified, and the final method + seconds); it never changes the result.

    Exact-or-raise: returns the certified full group as a
    :class:`qubitserf.algebra.permgroup.Group` (``.order()`` is an exact Python int), or raises
    :class:`RuntimeError` listing which engines were tried, why each failed, and what to try
    instead.  Partial results are never returned.
    """
    Hx, Hz = as_css(code, Hz)
    code = _css.CSSCode(Hx, Hz)
    return _css.automorphism_group(code, method=method, backend=backend,
                                   max_threads=max_threads, verbose=verbose)


def _as_group(g) -> "permgroup.Group":
    """Coerce a result/group-like object to a :class:`codeaut.permgroup.Group`."""
    if isinstance(g, permgroup.Group):
        return g
    grp = getattr(g, "group", None)          # AutResult.group()
    if callable(grp):
        return grp()
    gens = getattr(g, "generators", None)     # AutResult (image lists of length n)
    if gens is not None:
        degree = len(gens[0]) if gens else 0
        return permgroup.Group(gens, degree)
    if isinstance(g, (tuple, list)) and len(g) == 2:   # (generators, degree)
        return permgroup.Group(g[0], g[1])
    raise TypeError("expected a Group, an AutResult, or a (generators, degree) pair; got "
                    + type(g).__name__)


def group_intersection(g1, g2, **kwargs) -> "permgroup.Group":
    """Exact intersection ``g1 ∩ g2`` of two permutation groups on the same point set.

    Each argument may be a :class:`codeaut.permgroup.Group` (e.g. the return value of
    :func:`css_automorphisms`), a result from this library (:class:`codeaut.AutResult`), or a
    ``(generators, degree)`` pair.  Extra keyword args (``max_enumerate``, ``allow_sympy``)
    pass through to
    :func:`codeaut.permgroup.intersection`.  Returns a :class:`codeaut.permgroup.Group`.
    """
    return permgroup.intersection(_as_group(g1), _as_group(g2), **kwargs)
