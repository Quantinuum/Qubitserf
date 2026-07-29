"""qubitserf -- a two-part toolkit for quantum & classical linear codes.

The package bundles two engines built on a shared native core: a single C++ shared
library ``libqubitserf`` (in ``qubitserf/_lib``, namespace ``qsf``) provides the
bit-packed GF(2) machinery and one two-level enumeration kernel (CPU / CUDA / Metal)
that serves both, alongside the shared Python plumbing (``_native`` loader, ``_interop``
CSS shim, ``_constructions`` code constructions).  Each engine keeps its own flat C ABI
and its own Python subpackage:

* :mod:`qubitserf.distfind` -- fast **exact minimum distance** of CSS quantum, general
  stabilizer, subsystem, and classical codes (Brouwer--Zimmermann, connected-cluster,
  meet-in-the-middle) with C++ / CUDA / Metal backends.
* :mod:`qubitserf.codeaut` -- **automorphism groups** of binary linear and CSS quantum
  codes (Leon's algorithm + Brouwer--Zimmermann + graph incidence) with C++ / CUDA /
  Metal backends.

The two submodules are imported lazily, so ``import qubitserf.distfind`` never pulls in
codeaut's dependencies (and vice versa)::

    from qubitserf import distfind
    d = distfind.css_distance(Hx, Hz)

    from qubitserf import codeaut
    G = codeaut.css_automorphism_group(codeaut.codes.steane())
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["distfind", "codeaut", "algebra", "__version__"]
__version__ = "0.1.0"

_SUBPACKAGES = frozenset({"distfind", "codeaut", "algebra"})


def __getattr__(name: str):
    """Lazily resolve ``qubitserf.distfind`` / ``qubitserf.codeaut`` (PEP 562)."""
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBPACKAGES)


if TYPE_CHECKING:  # help static analysers see the submodules
    from . import distfind as distfind
    from . import codeaut as codeaut
    from . import algebra as algebra
