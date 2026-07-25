"""ctypes bridge to the native ``libcodeaut`` backend.

The shared library exposes a flat ``extern "C"`` ABI (no pybind11 / Cython): the **Leon**
code-automorphism engine (``qaut_leon_*``) and the **Brouwer--Zimmermann** low-weight
enumerator (``codeaut_bz_*``).  This module finds a prebuilt library, or builds the CPU
backend on first use with a direct compiler invocation (numpy + ctypes only -- no CMake
needed for the CPU path).  For the optional CUDA/Metal GPU backends, build with CMake
(``./build.sh`` or ``pip install .``) so the toolchains are detected.

Override the library with ``CODEAUT_LIB_PATH=/path/to/libcodeaut.so``.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
# qubitserf/codeaut  ->  python/qubitserf  ->  python  ->  <repo root>
_ROOT = _PKG_DIR.parent.parent.parent               # the merged qubitserf package root
_SRC = _ROOT / "src" / "codeaut"                    # codeaut native sources live under src/codeaut
_INC = _ROOT / "include"                            # headers at include/codeaut (see -I flags below)
_LIBDIR = _PKG_DIR / "_lib"

_IS_MAC = platform.system() == "Darwin"
_LIBNAME = "libcodeaut.dylib" if _IS_MAC else "libcodeaut.so"

_lib = None


def _candidate_paths():
    override = os.environ.get("CODEAUT_LIB_PATH")
    if override:
        yield Path(override)
    yield _LIBDIR / _LIBNAME
    yield _LIBDIR / "libcodeaut.so"
    yield _LIBDIR / "libcodeaut.dylib"


def _cpu_sources():
    """Top-level CPU translation units (cuda/metal are handled only by the CMake build)."""
    return sorted(str(p) for p in _SRC.glob("*.cpp"))


def _needs_build(lib_path: Path) -> bool:
    if not lib_path.exists():
        return True
    mtime = lib_path.stat().st_mtime
    watched = list(_SRC.glob("*.cpp")) + list(_INC.rglob("*.hpp")) + list(_INC.rglob("*.h"))
    return any(p.exists() and p.stat().st_mtime > mtime for p in watched)


def _build(lib_path: Path) -> None:
    """Compile the CPU backend directly (no CMake).  GPU backends require the CMake build."""
    sources = _cpu_sources()
    if not sources:
        raise RuntimeError(f"no C++ sources found in {_SRC}")
    _LIBDIR.mkdir(parents=True, exist_ok=True)
    cxx = os.environ.get("CXX", "clang++" if _IS_MAC else "c++")
    flags = ["-O3", "-std=c++17", "-fPIC", "-Wall", "-Wextra", "-pthread"]
    if _IS_MAC:
        link = ["-dynamiclib", "-install_name", f"@rpath/{lib_path.name}"]
    else:
        link = ["-shared"]
    cmd = [cxx, *flags, f"-I{_INC}", f"-I{_INC / 'codeaut'}", *link,
           "-o", str(lib_path), *sources]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("building libcodeaut failed:\n"
                           f"  command: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def _resolve_and_build() -> Path:
    # An explicit override or an up-to-date prebuilt library is used as-is.
    override = os.environ.get("CODEAUT_LIB_PATH")
    if override:
        return Path(override)
    for p in _candidate_paths():
        if p.exists() and not _needs_build(p):
            return p
    target = _LIBDIR / _LIBNAME
    _build(target)
    return target


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
            c.c_int64,                          # budget (combinations)
            c.c_int32,                          # backend (0=cpu,1=gpu,2=auto)
            c.c_int32,                          # threads (<=0 => hardware concurrency)
        ]
        lib.codeaut_bz_ok.restype = c.c_int32
        lib.codeaut_bz_ok.argtypes = [vp]
        lib.codeaut_bz_overflow.restype = c.c_int32
        lib.codeaut_bz_overflow.argtypes = [vp]
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
        lib_path = _resolve_and_build()
        if not lib_path.exists():
            raise RuntimeError(f"libcodeaut not found at {lib_path} "
                               "(set CODEAUT_LIB_PATH or run ./build.sh)")
        _lib = ctypes.CDLL(str(lib_path))
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
