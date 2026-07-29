"""qubitserf.algebra -- shared algebraic structures.

Currently one module: :mod:`qubitserf.algebra.permgroup`, exact permutation groups on
``0 .. degree-1`` (deterministic Schreier--Sims order/membership, intersection, and
element enumeration).  This is the return-type vocabulary of :mod:`qubitserf.codeaut`:
every automorphism result exposes ``.group() -> algebra.Group``.
"""
from .permgroup import Group, intersection, symmetric_group
from . import permgroup

__all__ = ["Group", "intersection", "symmetric_group", "permgroup"]
