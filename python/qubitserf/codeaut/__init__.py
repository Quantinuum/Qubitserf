"""codeaut -- automorphism groups of binary linear and CSS quantum codes.

A standalone library (``numpy`` + ``ctypes``) computing:

* ``Aut(C)`` of a binary linear code via **Leon's algorithm**, in an optimized bit-packed C++
  partition-backtracking engine (two-pass, low memory);
* the qubit-permutation automorphism group ``Aut(Hx) ∩ Aut(Hz)`` of a **CSS quantum code**,
  via a method ladder whose engine you can select with ``method``: ``"leon"`` (Leon + dual-code
  trick), ``"bz"`` (joint **Brouwer--Zimmermann** + nauty/Traces graph incidence + single-side
  rescue; best for LDPC codes), or ``"auto"`` (the full ladder).

The Brouwer--Zimmermann low-weight enumeration has C++ / CUDA / Metal backends.  The
colored-graph solves use the system ``nauty``/Traces (``dreadnaut``) -- a documented system
dependency.

Quick start::

    import numpy as np
    from qubitserf import codeaut

    # classical: the [7,3,4] simplex code -> GL(3,2), order 168
    G = np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]], dtype=np.uint8)
    print(codeaut.code_automorphism_group(G).order)            # 168

    # CSS: the [[7,1,3]] Steane code -> order 168 (exact)
    from qubitserf.codeaut import codes
    res = codeaut.css_automorphism_group(codes.steane())
    print(res.order, res.complete)                             # 168 True

    # pick the engine explicitly (Leon + dual-code trick / joint BZ + graph incidence)
    res = codeaut.css_automorphism_group(codes.gross(), method="bz")
    print(res.order, res.complete)                             # 144 True
"""

from __future__ import annotations

# submodules
from . import gf2
from . import permgroup
from . import graphaut
from . import lowweight
from . import matroid_pack
from . import joint
from . import side
from . import cyclic
from . import css
from . import codes
from . import ward
from . import invariants
from . import components

# single-code Leon engine
from .leon import automorphism_group as code_automorphism_group
from .leon import code_automorphism_generators, AutResult
from .ward import automorphism_group as ward_automorphism_group
from .ward import WardAutResult, WardDecisionDiagram, WardForm, ward_form
from .invariants import automorphism_group as invariant_automorphism_group
from .invariants import InvariantAutResult, InvariantLimitExceeded
from .invariants import METHODS as INVARIANT_METHODS
from .invariants import rowspace_stabilizer
from .components import component_automorphism_group, ComponentAutResult, ComponentGuardExceeded

# CSS automorphisms (the method ladder)
from .css import CSSCode, CSSAutResult, effective_dims, METHODS
from .css import automorphism_group as css_automorphism_group

# CSS interop shim (shared products/ protocol)
from ._interop import as_css

# easy interface (the three convenience entry points -- see README)
from .api import (
    classical_automorphisms,
    css_automorphisms,
    group_intersection,
)

# engine entry points
from .joint import joint_exact
from .side import side_aut_subgroup
from .cyclic import affine_automorphism_group
from .lowweight import low_weight_classes
from .permgroup import Group, intersection, symmetric_group
from ._native import available_backends


def version() -> str:
    return __version__


__all__ = [
    # easy interface
    "classical_automorphisms", "css_automorphisms", "group_intersection",
    # single classical code
    "code_automorphism_group", "code_automorphism_generators", "AutResult",
    "ward_automorphism_group", "WardAutResult", "WardDecisionDiagram", "WardForm",
    "ward_form", "invariant_automorphism_group", "InvariantAutResult",
    "InvariantLimitExceeded", "INVARIANT_METHODS",
    "rowspace_stabilizer",
    "component_automorphism_group", "ComponentAutResult", "ComponentGuardExceeded",
    # CSS
    "CSSCode", "CSSAutResult", "css_automorphism_group", "METHODS",
    "effective_dims",
    # engines
    "joint_exact", "side_aut_subgroup", "affine_automorphism_group",
    "low_weight_classes",
    # permutation groups
    "Group", "intersection", "symmetric_group",
    # misc
    "available_backends", "version", "codes", "gf2", "permgroup", "ward", "invariants",
    "components",
    # CSS interop
    "as_css",
]
__version__ = "0.1.0"
