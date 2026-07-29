"""ctypes bridge to the ``codeaut`` ABIs of the shared libqubitserf library.

The library exposes a flat ``extern "C"`` ABI (no pybind11 / Cython): the **Leon**
code-automorphism engine (``qaut_leon_*``) and the **Brouwer--Zimmermann** low-weight
enumerator (``codeaut_bz_*``), running on the same shared enumeration core as the
distfind engine.  The library itself is located/loaded (once, shared with distfind) by
:mod:`qubitserf._native` -- including the no-CMake CPU-only dev-tree build fallback;
this module attaches the codeaut signatures.

Override the library with ``QUBITSERF_LIB=/path/to/libqubitserf.so`` (the legacy
``CODEAUT_LIB_PATH`` is also honoured).
"""

from __future__ import annotations

import ctypes

from .._native import load as _load_shared

_lib = None


def _set_signatures(lib) -> None:
    c = ctypes
    vp = c.c_void_p
    u8 = c.POINTER(c.c_uint8)
    i32 = c.POINTER(c.c_int32)
    i64 = c.POINTER(c.c_int64)

    # --- Leon code-automorphism ABI ---
    lib.qaut_leon_run.restype = vp
    lib.qaut_leon_run.argtypes = [u8, c.c_int32, c.c_int32, c.c_int32, c.c_int32]
    if hasattr(lib, "qaut_leon_run_ex"):
        lib.qaut_leon_run_ex.restype = vp
        lib.qaut_leon_run_ex.argtypes = [
            u8, c.c_int32, c.c_int32, c.c_int32, c.c_int32,
            c.c_int32, c.c_int32,              # spanning-set selector, max modulus
        ]
    for name in ("qaut_leon_ok", "qaut_leon_dim", "qaut_leon_n", "qaut_leon_num_codewords",
                 "qaut_leon_num_classes", "qaut_leon_num_gens", "qaut_leon_num_factors"):
        getattr(lib, name).restype = c.c_int32
        getattr(lib, name).argtypes = [vp]
    for name in ("qaut_leon_selector", "qaut_leon_modulus", "qaut_leon_residue"):
        if hasattr(lib, name):
            getattr(lib, name).restype = c.c_int32
            getattr(lib, name).argtypes = [vp]
    for name in ("qaut_leon_enumeration_ns", "qaut_leon_search_ns"):
        if hasattr(lib, name):
            getattr(lib, name).restype = c.c_int64
            getattr(lib, name).argtypes = [vp]
    if hasattr(lib, "qaut_leon_num_incidences"):
        lib.qaut_leon_num_incidences.restype = c.c_int64
        lib.qaut_leon_num_incidences.argtypes = [vp]
    lib.qaut_leon_copy_gens.restype = None
    lib.qaut_leon_copy_gens.argtypes = [vp, i32]
    lib.qaut_leon_copy_factors.restype = None
    lib.qaut_leon_copy_factors.argtypes = [vp, i64]
    lib.qaut_leon_copy_weights.restype = None
    lib.qaut_leon_copy_weights.argtypes = [vp, i32]
    lib.qaut_leon_free.restype = None
    lib.qaut_leon_free.argtypes = [vp]

    # --- Brouwer-Zimmermann low-weight enumerator ABI (added in Phase 4) ---
    if hasattr(lib, "codeaut_bz_collect"):
        lib.codeaut_bz_collect.restype = vp
        lib.codeaut_bz_collect.argtypes = [
            u8, c.c_int32, c.c_int32,           # G (m x n), m, n
            c.c_int32, c.c_int32,               # p, keep_weight
            c.c_int32,                          # backend (0=cpu,1=gpu,2=auto)
            c.c_int32,                          # threads (<=0 => hardware concurrency)
        ]
        lib.codeaut_bz_ok.restype = c.c_int32
        lib.codeaut_bz_ok.argtypes = [vp]
        lib.codeaut_bz_count.restype = c.c_int64
        lib.codeaut_bz_count.argtypes = [vp]
        lib.codeaut_bz_combos.restype = c.c_int64
        lib.codeaut_bz_combos.argtypes = [vp]
        lib.codeaut_bz_copy_rows.restype = None
        lib.codeaut_bz_copy_rows.argtypes = [vp, u8]
        lib.codeaut_bz_backend.restype = c.c_char_p
        lib.codeaut_bz_backend.argtypes = [vp]
        lib.codeaut_bz_free.restype = None
        lib.codeaut_bz_free.argtypes = [vp]

    if hasattr(lib, "codeaut_backend_available"):
        lib.codeaut_backend_available.restype = c.c_int32
        lib.codeaut_backend_available.argtypes = [c.c_char_p]


def load():
    """Load (building if necessary) and return the configured ``ctypes.CDLL``."""
    global _lib
    if _lib is None:
        _lib = _load_shared()
        _set_signatures(_lib)
    return _lib


def available_backends() -> list[str]:
    """BZ enumeration backends usable in this build (always includes ``'cpu'``)."""
    lib = load()
    out = ["cpu"]
    if hasattr(lib, "codeaut_backend_available"):
        if lib.codeaut_backend_available(b"gpu"):
            out.append("gpu")
    return out
