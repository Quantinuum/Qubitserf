"""codeaut -- the easy interface.

Three high-level convenience entry points that cover the common tasks, each accepting flexible
inputs:

* :func:`classical_automorphisms` -- ``Aut(C)`` of a binary linear code (Leon's algorithm);
* :func:`css_automorphisms`       -- ``Aut(Hx) ∩ Aut(Hz)`` of a CSS quantum code (method ladder;
  pick the engine with ``method``);
* :func:`group_intersection`      -- the intersection of two permutation groups.

These wrap the lower-level engines (``joint_exact``, ``side_aut_subgroup``,
``affine_automorphism_group``, ``low_weight_classes``, the ``permgroup`` toolkit, ...); reach for
those directly only when you need a specific engine, backend, or diagnostic.  See the README for
worked examples.
"""
from __future__ import annotations

from . import leon
from . import css as _css
from . import permgroup
from ._interop import as_css


def classical_automorphisms(genmat, *, max_dim: int = 20, use_invariant: bool = True,
                            spanning_set: str = "minweight",
                            max_modulus=None) -> "leon.AutResult":
    """Permutation automorphism group ``Aut(C)`` of the binary linear code ``C =
    rowspace(genmat)``.

    ``genmat`` is an ``(m, n)`` array of 0/1 entries (any generating set; row-reduced
    internally).  Returns an :class:`codeaut.AutResult` with ``.generators`` (0-indexed image
    lists, a strong generating set) and the exact ``.order``.

    Codeword enumeration costs ``2**dim(C)``; ``max_dim`` caps the dimension.  ``spanning_set``
    is ``"minweight"`` (the legacy ascending-weight prefix), ``"congruence"`` (the smallest
    complete spanning weight-residue class), ``"auto"`` (cost-aware probing for large
    prefixes, using the residue class only when it shrinks the incidence), or ``"minimal"``
    (support-minimal/cocircuit filtering of the legacy prefix).  ``max_modulus`` bounds the
    residue search (default: ``n+1``).
    """
    return leon.automorphism_group(genmat, max_dim=max_dim, use_invariant=use_invariant,
                                   spanning_set=spanning_set, max_modulus=max_modulus)


def css_automorphisms(code, Hz=None, *, method="auto", backend="auto", max_threads=None,
                      **kwargs) -> "_css.CSSAutResult":
    """Qubit-permutation automorphism group ``Aut(Hx) ∩ Aut(Hz)`` of a CSS code, via the method
    ladder.

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
    hardware cores; only the CPU backend is threaded).  Extra keyword args (``max_dim``,
    ``budget``, ...) pass through to the engine.  Returns a :class:`codeaut.CSSAutResult`.
    """
    Hx, Hz = as_css(code, Hz)
    code = _css.CSSCode(Hx, Hz)
    return _css.automorphism_group(code, method=method, backend=backend,
                                   max_threads=max_threads, **kwargs)


def _as_group(g) -> "permgroup.Group":
    """Coerce a result/group-like object to a :class:`codeaut.permgroup.Group`."""
    if isinstance(g, permgroup.Group):
        return g
    grp = getattr(g, "group", None)          # CSSAutResult.group()
    if callable(grp):
        return grp()
    gens = getattr(g, "generators", None)     # AutResult (image lists of length n)
    if gens is not None:
        degree = len(gens[0]) if gens else 0
        return permgroup.Group(gens, degree)
    if isinstance(g, (tuple, list)) and len(g) == 2:   # (generators, degree)
        return permgroup.Group(g[0], g[1])
    raise TypeError("expected a Group, an Aut/CSSAutResult, or a (generators, degree) pair; got "
                    + type(g).__name__)


def group_intersection(g1, g2, **kwargs) -> "permgroup.Group":
    """Exact intersection ``g1 ∩ g2`` of two permutation groups on the same point set.

    Each argument may be a :class:`codeaut.permgroup.Group`, a result from this library
    (:class:`codeaut.AutResult` / :class:`codeaut.CSSAutResult`), or a ``(generators, degree)``
    pair.  Extra keyword args (``max_enumerate``, ``allow_sympy``) pass through to
    :func:`codeaut.permgroup.intersection`.  Returns a :class:`codeaut.permgroup.Group`.
    """
    return permgroup.intersection(_as_group(g1), _as_group(g2), **kwargs)
