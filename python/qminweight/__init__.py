"""qminweight: fast GPU + CPU quantum & classical code distance finding.

Deterministic algorithms (Brouwer-Zimmermann, meet-in-the-middle) accelerated on the
GPU via combinatorial-number-system work splitting.
"""
from .api import (
    Result,
    css_distance,
    classical_distance,
    available_backends,
    version,
)
from . import codes

__all__ = [
    "Result",
    "css_distance",
    "classical_distance",
    "available_backends",
    "version",
    "codes",
]
__version__ = "0.1.0"
