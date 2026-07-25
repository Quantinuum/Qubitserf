"""qubitserf -- a two-part toolkit for quantum & classical linear codes.

The package bundles two independent, compartmentalized libraries that share a repo,
a build, and the small CSS interop shim, but nothing else:

* :mod:`qubitserf.distfind` -- fast **exact minimum distance** of CSS quantum, general
  stabilizer, subsystem, and classical codes (Brouwer--Zimmermann, connected-cluster,
  meet-in-the-middle) with C++ / CUDA / Metal backends.  CLI: ``distfind``.
* :mod:`qubitserf.codeaut` -- **automorphism groups** of binary linear and CSS quantum
  codes (Leon's algorithm + Brouwer--Zimmermann + graph incidence) with C++ / CUDA /
  Metal backends.  CLI: ``codeaut``.

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

__all__ = ["distfind", "codeaut", "__version__"]
__version__ = "0.1.0"

_SUBPACKAGES = frozenset({"distfind", "codeaut"})


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
