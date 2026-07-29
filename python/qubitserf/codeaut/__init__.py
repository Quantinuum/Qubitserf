"""codeaut -- automorphism groups of binary linear and CSS quantum codes.

A ``numpy`` + ``ctypes`` layer over the shared ``libqubitserf`` native library, computing:

* ``Aut(C)`` of a binary linear code, with the engine selectable via ``method``: ``"leon"``
  (**Leon's algorithm**, an optimized bit-packed C++ partition-backtracking engine -- two-pass,
  low memory), ``"bz"`` (certified-complete **Brouwer--Zimmermann** low-weight classes +
  nauty/Traces coloured-incidence solve; exact at any ``dim``, best for LDPC-like codes), or
  ``"auto"`` (cheapest exact route first);
* the qubit-permutation automorphism group ``Aut(Hx) ∩ Aut(Hz)`` of a **CSS quantum code**,
  via a method ladder whose engine you can select with ``method``: ``"leon"`` (Leon + dual-code
  trick), ``"bz"`` (joint **Brouwer--Zimmermann** + nauty/Traces graph incidence + single-side
  rescue; best for LDPC codes), or ``"auto"`` (the full ladder).  The CSS entry points are
  **exact-or-raise**: they return the certified full group as a
  :class:`qubitserf.algebra.permgroup.Group`, or raise ``RuntimeError`` -- never a partial
  result.

The Brouwer--Zimmermann low-weight enumeration has C++ / CUDA / Metal backends.  The
colored-graph solves use the system ``nauty``/Traces (``dreadnaut``) -- a documented system
dependency.

Quick start::

    import numpy as np
    from qubitserf import codeaut

    # classical: the [7,3,4] simplex code -> GL(3,2), order 168
    G = np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]], dtype=np.uint8)
    print(codeaut.classical_automorphisms(G).order)            # 168
    print(codeaut.classical_automorphisms(G, method="bz").order)   # 168 (BZ + nauty route)

    # CSS: the [[7,1,3]] Steane code -> a permgroup.Group of order 168 (exact)
    from qubitserf.codeaut import codes
    grp = codeaut.css_automorphisms(codes.steane())
    print(grp.order())                                         # 168 (exact Python int)

    # pick the engine explicitly (Leon + dual-code trick / joint BZ + graph incidence)
    grp = codeaut.css_automorphisms(codes.gross(), method="bz")
    print(grp.order())                                         # 144
"""

from __future__ import annotations

# submodules
from . import gf2
from . import graphaut
from . import lowweight
from . import matroid_pack
from . import joint
from . import side
from . import css
from . import codes
from . import classical_bz

# single-code Leon engine
from .leon import automorphism_group as code_automorphism_group
from .leon import code_automorphism_generators, AutResult

# single-code BZ + nauty/Traces engine
from .classical_bz import ClassicalAutResult

# CSS automorphisms (the method ladder; exact-or-raise, returns a permgroup.Group)
from .css import CSSCode, effective_dims, METHODS
from .css import automorphism_group as css_automorphism_group

# CSS interop shim (shared products/ protocol) + Pauli-stabiliser-string parser
from ._interop import as_css, css_from_paulis

# easy interface (the three convenience entry points -- see README)
from .api import (
    classical_automorphisms,
    css_automorphisms,
    group_intersection,
)

# engine entry points
from .joint import joint_exact
from .side import side_aut_subgroup
from .lowweight import low_weight_classes
from ._native import available_backends


def version() -> str:
    return __version__


__all__ = [
    # easy interface
    "classical_automorphisms", "css_automorphisms", "group_intersection",
    # single classical code
    "code_automorphism_group", "code_automorphism_generators", "AutResult",
    "ClassicalAutResult",
    # CSS
    "CSSCode", "css_automorphism_group", "METHODS",
    "effective_dims",
    # engines
    "joint_exact", "side_aut_subgroup",
    "low_weight_classes",
    # misc
    "available_backends", "version", "codes", "gf2",
    # CSS interop
    "as_css", "css_from_paulis",
]
__version__ = "0.1.0"
