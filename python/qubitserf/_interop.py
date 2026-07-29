"""CSS interop shim -- normalize any CSS-code-like object to ``(Hx, Hz)``.

Part of the Quirky ``products/`` CSS interoperability protocol: a *CSS code* is
anything that yields an ``(Hx, Hz)`` pair of GF(2) matrices via one of, in order:

  * an explicit ``(Hx, Hz)`` argument pair, or a 2-tuple/list of matrices;
  * a ``.to_arrays()`` method returning ``(Hx, Hz)``;
  * a ``.css_matrices()`` method returning ``(Hx, Hz)``  (qecdb ``Code``);
  * ``.Hx`` and ``.Hz`` attributes  (``lib.CSSCode``, ``codeaut.CSSCode``).

Within ``qubitserf`` the two engines (distfind / codeaut) share this single
canonical copy -- their ``_interop`` modules are thin re-export shims.  The
cross-PRODUCT duplication remains intentional: dependency-free (numpy only)
copies also live in qecdb_client / lib so the standalone packages never import
one another.  Keep those copies in sync with this one.
"""
from __future__ import annotations

import numpy as np


def _as_u8(m) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(m, dtype=np.uint8) & 1)
    if a.ndim != 2:
        raise ValueError("expected a 2D 0/1 matrix")
    return a


def as_css(obj, Hz=None):
    """Normalize ``obj`` (and optional ``Hz``) to ``(Hx, Hz)`` contiguous uint8
    GF(2) matrices.  See the module docstring for the accepted forms."""
    if Hz is not None:
        return _as_u8(obj), _as_u8(Hz)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return _as_u8(obj[0]), _as_u8(obj[1])
    for meth in ("to_arrays", "css_matrices"):
        f = getattr(obj, meth, None)
        if callable(f):
            hx, hz = f()
            return _as_u8(hx), _as_u8(hz)
    hx, hz = getattr(obj, "Hx", None), getattr(obj, "Hz", None)
    if hx is not None and hz is not None:
        return _as_u8(hx), _as_u8(hz)
    raise TypeError(
        "expected a CSS code as (Hx, Hz), a 2-tuple of matrices, an object with "
        ".to_arrays()/.css_matrices(), or one with .Hx/.Hz attributes; got "
        + type(obj).__name__)
