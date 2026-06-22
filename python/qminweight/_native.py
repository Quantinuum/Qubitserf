"""ctypes binding to the compiled libqminweight shared library."""
from __future__ import annotations
import ctypes
import os
import sys
import glob

import numpy as np


class QMinWeightResult(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_int),
        ("lower_bound", ctypes.c_int),
        ("proven", ctypes.c_int),
        ("levels", ctypes.c_int),
        ("seconds", ctypes.c_double),
        ("backend", ctypes.c_char * 16),
    ]


def _find_library() -> str:
    override = os.environ.get("QMINWEIGHT_LIB")
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError("QMINWEIGHT_LIB=%s does not exist" % override)
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    libdir = os.path.join(here, "_lib")
    names = ["libqminweight.dylib", "libqminweight.so", "qminweight.dll"]
    for n in names:
        p = os.path.join(libdir, n)
        if os.path.exists(p):
            return p
    # also accept anything matching in _lib
    for p in glob.glob(os.path.join(libdir, "*qminweight*")):
        if p.endswith((".dylib", ".so", ".dll")):
            return p
    raise FileNotFoundError(
        "libqminweight not found in {}. Build it first: run ./build.sh".format(libdir)
    )


_lib = None


def lib():
    global _lib
    if _lib is not None:
        return _lib
    _lib = ctypes.CDLL(_find_library())

    u8p = ctypes.POINTER(ctypes.c_uint8)
    rp = ctypes.POINTER(QMinWeightResult)

    _lib.qminweight_css_distance.restype = ctypes.c_int
    _lib.qminweight_css_distance.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        u8p, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        rp,
    ]
    _lib.qminweight_classical_distance.restype = ctypes.c_int
    _lib.qminweight_classical_distance.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        rp,
    ]
    _lib.qminweight_backend_available.restype = ctypes.c_int
    _lib.qminweight_backend_available.argtypes = [ctypes.c_char_p]
    _lib.qminweight_version.restype = ctypes.c_char_p
    _lib.qminweight_version.argtypes = []
    return _lib


def _as_u8(mat) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(mat, dtype=np.uint8) & 1)
    if a.ndim != 2:
        raise ValueError("expected a 2D 0/1 matrix")
    return a


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def css_distance_raw(Hx, Hz, method, which, backend, threads, max_weight, verbose):
    L = lib()
    ax, az = _as_u8(Hx), _as_u8(Hz)
    out = QMinWeightResult()
    rc = L.qminweight_css_distance(
        _ptr(ax), ax.shape[0], ax.shape[1],
        _ptr(az), az.shape[0], az.shape[1],
        method.encode(), which.encode()[:1] or b"M", backend.encode(),
        int(threads), int(max_weight), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc != 0:
        raise RuntimeError("qminweight_css_distance failed (rc=%d)" % rc)
    return out


def classical_distance_raw(H, method, backend, threads, max_weight, verbose):
    L = lib()
    a = _as_u8(H)
    out = QMinWeightResult()
    rc = L.qminweight_classical_distance(
        _ptr(a), a.shape[0], a.shape[1],
        method.encode(), backend.encode(),
        int(threads), int(max_weight), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc != 0:
        raise RuntimeError("qminweight_classical_distance failed (rc=%d)" % rc)
    return out


def backend_available(name: str) -> bool:
    return bool(lib().qminweight_backend_available(name.encode()))


def version() -> str:
    return lib().qminweight_version().decode()
