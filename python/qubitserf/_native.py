"""Locate and load the single native library ``libqubitserf`` shared by both engines.

The library exposes two flat ``extern "C"`` ABIs -- ``distfind_*`` (distance finding)
and ``codeaut_bz_*`` / ``qaut_leon_*`` (automorphisms) -- over one shared enumeration
core.  Each subpackage's ``_native`` module attaches its own ctypes signatures to the
one ``CDLL`` returned by :func:`load`.

Resolution order:
  1. ``QUBITSERF_LIB`` (or the legacy ``DISTFIND_LIB`` / ``CODEAUT_LIB_PATH``) env override;
  2. a prebuilt, up-to-date ``qubitserf/_lib/libqubitserf.*`` (CMake dev build or wheel);
  3. dev-tree fallback: compile the CPU backend directly with the system compiler
     (numpy + ctypes only -- no CMake).  GPU backends require the CMake build
     (``./build.sh`` or ``pip install .``) so the toolchains are detected.
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent            # .../python/qubitserf
_ROOT = _PKG_DIR.parent.parent                        # the qubitserf package root
_SRC_DIRS = [_ROOT / "cpp" / "qsf", _ROOT / "cpp" / "distfind", _ROOT / "cpp" / "codeaut"]
_INC = _ROOT / "cpp"
_LIBDIR = _PKG_DIR / "_lib"

_IS_MAC = platform.system() == "Darwin"
_LIBNAMES = ["libqubitserf.dylib", "libqubitserf.so", "qubitserf.dll"]
_LIBNAME = "libqubitserf.dylib" if _IS_MAC else "libqubitserf.so"

_lib = None


def _candidate_paths():
    for var in ("QUBITSERF_LIB", "DISTFIND_LIB", "CODEAUT_LIB_PATH"):
        override = os.environ.get(var)
        if override:
            yield Path(override)
    for n in _LIBNAMES:
        yield _LIBDIR / n


def _cpu_sources():
    """Top-level CPU translation units (cuda/metal are handled only by the CMake build)."""
    out = []
    for d in _SRC_DIRS:
        out.extend(sorted(str(p) for p in d.glob("*.cpp")))
    return out


def _in_dev_tree() -> bool:
    return all(d.is_dir() for d in _SRC_DIRS) and _INC.is_dir()


def _needs_build(lib_path: Path) -> bool:
    if not lib_path.exists():
        return True
    if not _in_dev_tree():        # installed wheel: the shipped library is authoritative
        return False
    mtime = lib_path.stat().st_mtime
    watched = [p for d in _SRC_DIRS for p in d.glob("*.cpp")]
    watched += list(_INC.rglob("*.hpp")) + list(_INC.rglob("*.h"))
    return any(p.exists() and p.stat().st_mtime > mtime for p in watched)


def _build(lib_path: Path) -> None:
    """Compile the CPU backend directly (no CMake).  GPU backends require the CMake build."""
    sources = _cpu_sources()
    if not sources:
        raise RuntimeError(f"no C++ sources found under {_ROOT / 'src'}")
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
        raise RuntimeError("building libqubitserf failed:\n"
                           f"  command: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def find_library() -> Path:
    """Path of the library to load, building the CPU fallback if needed."""
    for var in ("QUBITSERF_LIB", "DISTFIND_LIB", "CODEAUT_LIB_PATH"):
        override = os.environ.get(var)
        if override:
            p = Path(override)
            if not p.exists():
                raise FileNotFoundError(f"{var}={override} does not exist")
            return p
    for p in _candidate_paths():
        if p.exists() and not _needs_build(p):
            return p
    if _in_dev_tree():
        target = _LIBDIR / _LIBNAME
        _build(target)
        return target
    raise FileNotFoundError(
        f"libqubitserf not found in {_LIBDIR}. Build it first: run ./build.sh "
        "(or set QUBITSERF_LIB)")


def load() -> ctypes.CDLL:
    """Load (building if necessary) and return the shared ``ctypes.CDLL`` singleton."""
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(find_library()))
    return _lib
