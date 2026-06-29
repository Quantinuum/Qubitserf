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
from ._interop import as_css
from . import codes

__all__ = [
    "Result",
    "css_distance",
    "classical_distance",
    "available_backends",
    "version",
    "as_css",
    "codes",
]
__version__ = "0.1.0"
