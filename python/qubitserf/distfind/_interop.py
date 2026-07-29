"""CSS interop shim for distfind -- re-exports the canonical copy at
:mod:`qubitserf._interop` (see its docstring for the accepted forms)."""
from __future__ import annotations

from .._interop import as_css, _as_u8

__all__ = ["as_css", "_as_u8"]
