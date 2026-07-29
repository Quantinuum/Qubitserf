"""ctypes binding to the ``distfind_*`` ABI of the shared libqubitserf library.

The library itself is located/loaded (once, shared with codeaut) by
:mod:`qubitserf._native`; this module attaches the distfind signatures and wraps the
raw calls.
"""
from __future__ import annotations
import ctypes

import numpy as np

from .._native import load as _load_shared


class DistfindResult(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_int),
        ("lower_bound", ctypes.c_int),
        ("proven", ctypes.c_int),
        ("levels", ctypes.c_int),
        ("seconds", ctypes.c_double),
        ("backend", ctypes.c_char * 16),
    ]


class DistfindOpResult(ctypes.Structure):
    _fields_ = [
        ("z_weight", ctypes.c_int),
        ("x_weight", ctypes.c_int),
        ("proven", ctypes.c_int),
        ("seconds", ctypes.c_double),
        ("backend", ctypes.c_char * 16),
    ]


_lib = None


def lib():
    global _lib
    if _lib is not None:
        return _lib
    _lib = _load_shared()

    u8p = ctypes.POINTER(ctypes.c_uint8)
    rp = ctypes.POINTER(DistfindResult)
    orp = ctypes.POINTER(DistfindOpResult)

    _lib.distfind_css_distance.restype = ctypes.c_int
    _lib.distfind_css_distance.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        u8p, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        rp,
    ]
    _lib.distfind_classical_distance.restype = ctypes.c_int
    _lib.distfind_classical_distance.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        rp,
    ]
    _lib.distfind_operator_weight.restype = ctypes.c_int
    _lib.distfind_operator_weight.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        u8p, ctypes.c_int, ctypes.c_int,
        u8p, u8p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        orp,
    ]
    _lib.distfind_subsystem_distance.restype = ctypes.c_int
    _lib.distfind_subsystem_distance.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        u8p, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        rp,
    ]
    _lib.distfind_stabilizer_distance.restype = ctypes.c_int
    _lib.distfind_stabilizer_distance.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        rp,
    ]
    _lib.distfind_subsystem_stabilizer_distance.restype = ctypes.c_int
    _lib.distfind_subsystem_stabilizer_distance.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        rp,
    ]
    _lib.distfind_stabilizer_operator_weight.restype = ctypes.c_int
    _lib.distfind_stabilizer_operator_weight.argtypes = [
        u8p, ctypes.c_int, ctypes.c_int,
        u8p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        rp,
    ]
    _lib.distfind_backend_available.restype = ctypes.c_int
    _lib.distfind_backend_available.argtypes = [ctypes.c_char_p]
    _lib.distfind_version.restype = ctypes.c_char_p
    _lib.distfind_version.argtypes = []
    return _lib


def _as_u8(mat) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(mat, dtype=np.uint8) & 1)
    if a.ndim != 2:
        raise ValueError("expected a 2D 0/1 matrix")
    return a


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def css_distance_raw(Hx, Hz, method, which, backend, threads, verbose):
    L = lib()
    ax, az = _as_u8(Hx), _as_u8(Hz)
    out = DistfindResult()
    rc = L.distfind_css_distance(
        _ptr(ax), ax.shape[0], ax.shape[1],
        _ptr(az), az.shape[0], az.shape[1],
        method.encode(), which.encode()[:1] or b"M", backend.encode(),
        int(threads), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc != 0:
        raise RuntimeError("distfind_css_distance failed (rc=%d)" % rc)
    return out


def classical_distance_raw(H, method, backend, threads, verbose):
    L = lib()
    a = _as_u8(H)
    out = DistfindResult()
    rc = L.distfind_classical_distance(
        _ptr(a), a.shape[0], a.shape[1],
        method.encode(), backend.encode(),
        int(threads), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc != 0:
        raise RuntimeError("distfind_classical_distance failed (rc=%d)" % rc)
    return out


def _as_u8_vec(vec) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(vec, dtype=np.uint8) & 1).reshape(-1)
    return a


def operator_weight_raw(Gx, Gz, z_op, x_op, method, backend, threads, verbose):
    L = lib()
    ax, az = _as_u8(Gx), _as_u8(Gz)
    z = _as_u8_vec(z_op)
    x = _as_u8_vec(x_op)
    n = ax.shape[1]
    if az.shape[1] != n:
        raise ValueError("Gx and Gz must have the same number of columns (n)")
    if z.shape[0] != n or x.shape[0] != n:
        raise ValueError("z_op and x_op must be length n = %d" % n)
    out = DistfindOpResult()
    rc = L.distfind_operator_weight(
        _ptr(ax), ax.shape[0], ax.shape[1],
        _ptr(az), az.shape[0], az.shape[1],
        _ptr(z), _ptr(x), n,
        method.encode(), backend.encode(),
        int(threads), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc != 0:
        raise RuntimeError("distfind_operator_weight failed (rc=%d)" % rc)
    return out


def subsystem_distance_raw(Gx, Gz, method, which, backend, threads, verbose):
    L = lib()
    ax, az = _as_u8(Gx), _as_u8(Gz)
    out = DistfindResult()
    rc = L.distfind_subsystem_distance(
        _ptr(ax), ax.shape[0], ax.shape[1],
        _ptr(az), az.shape[0], az.shape[1],
        method.encode(), which.encode()[:1] or b"M", backend.encode(),
        int(threads), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc == 3:
        raise ValueError("method 'cc' (connected cluster) is not supported for non-CSS "
                         "codes; use 'bz' or 'mitm'")
    if rc != 0:
        raise RuntimeError("distfind_subsystem_distance failed (rc=%d)" % rc)
    return out


def stabilizer_distance_raw(S, method, which, backend, threads, verbose):
    L = lib()
    a = _as_u8(S)
    if a.shape[1] % 2 != 0:
        raise ValueError("symplectic stabilizer matrix must have 2n (even) columns")
    out = DistfindResult()
    rc = L.distfind_stabilizer_distance(
        _ptr(a), a.shape[0], a.shape[1],
        method.encode(), which.encode()[:1] or b"M", backend.encode(),
        int(threads), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc == 3:
        raise ValueError("method 'cc' (connected cluster) is not supported for non-CSS "
                         "codes; use 'bz' or 'mitm'")
    if rc != 0:
        raise RuntimeError("distfind_stabilizer_distance failed (rc=%d)" % rc)
    return out


def subsystem_stabilizer_distance_raw(G, method, which, backend, threads, verbose):
    L = lib()
    a = _as_u8(G)
    if a.shape[1] % 2 != 0:
        raise ValueError("symplectic gauge matrix must have 2n (even) columns")
    out = DistfindResult()
    rc = L.distfind_subsystem_stabilizer_distance(
        _ptr(a), a.shape[0], a.shape[1],
        method.encode(), which.encode()[:1] or b"M", backend.encode(),
        int(threads), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc == 3:
        raise ValueError("method 'cc' (connected cluster) is not supported for non-CSS "
                         "codes; use 'bz' or 'mitm'")
    if rc != 0:
        raise RuntimeError("distfind_subsystem_stabilizer_distance failed (rc=%d)" % rc)
    return out


def stabilizer_operator_weight_raw(G, op, method, backend, threads, verbose):
    L = lib()
    a = _as_u8(G)
    o = _as_u8_vec(op)
    if a.shape[1] % 2 != 0:
        raise ValueError("symplectic gauge matrix must have 2n (even) columns")
    if o.shape[0] != a.shape[1]:
        raise ValueError("operator must be a length-2n symplectic [z|x] vector")
    out = DistfindResult()
    rc = L.distfind_stabilizer_operator_weight(
        _ptr(a), a.shape[0], a.shape[1],
        _ptr(o), o.shape[0],
        method.encode(), backend.encode(),
        int(threads), 1 if verbose else 0,
        ctypes.byref(out),
    )
    if rc != 0:
        raise RuntimeError("distfind_stabilizer_operator_weight failed (rc=%d)" % rc)
    return out


def backend_available(name: str) -> bool:
    return bool(lib().distfind_backend_available(name.encode()))


def version() -> str:
    return lib().distfind_version().decode()
