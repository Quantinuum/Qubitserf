"""Complete low-weight codeword enumeration via Brouwer--Zimmermann (BZ).

The exact ``Aut(C)`` of a binary linear code is the automorphism group of the coloured
coordinate<->codeword incidence holding the **complete** ascending weight classes that *span*
``C`` (Leon's reduction).  This module recovers those complete weight classes directly, by the
BZ information-set method with a rigorous **completeness certificate** -- never enumerating
``2**dim`` for structured families (whose spanning classes are just the minimum-weight class,
``~O(n)`` words).

The heavy ``C(k, <=p)`` subset enumeration runs in the native backend
(:func:`codeaut._native` -> ``codeaut_bz_collect``; CPU, or CUDA/Metal when built), with a
pure-numpy fallback if the native library is unavailable.

Public API
----------
``low_weight_classes(B, *, want_span=True, max_weight=None, backend="auto")
                     -> (classes, info)``
    ``classes`` is a list of ``(weight, rows)`` ascending, each ``rows`` a ``(count, n)``
    uint8 array of the **complete** weight class (only certified-complete classes returned).

Correctness (the crux)
----------------------
After enumerating, for each of ``m`` greedy max-disjoint systematic generators ``G_j``
(fresh-column count ``k_j``), all messages of Hamming weight ``<= p``, every codeword found by
*no* generator has weight ``>= LB(p) = sum_j max(0, p+1-(k-k_j))``.  Hence **all** codewords of
weight ``<= W_cert = LB(p)-1`` are captured, so every weight class ``<= W_cert`` is complete.
"""

from __future__ import annotations

import ctypes
import itertools
from typing import Optional

import numpy as np

from . import gf2
from . import matroid_pack
from . import _native

_BACKEND_CODE = {"cpu": 0, "gpu": 1, "auto": 2}

# Hard cap on the numpy ``2**dim`` full-enumeration fast path: beyond this dimension the dense
# ``(2**dim, n)`` codeword array is too costly (memory + time), so the bit-packed C++/GPU
# Brouwer--Zimmermann kernel is used instead (memory-bounded; it only materializes low-weight
# codewords).  Callers may pass a larger ``full_enum_max_dim`` but it is clamped to this.
_NUMPY_FULL_ENUM_CAP = 18


# ----------------------------------------------------------------------------------------
# native (or numpy) collection of XOR-of-subset codewords for ONE systematic generator
# ----------------------------------------------------------------------------------------

def _collect_one_native(G: np.ndarray, p: int, keep_weight: int, backend: int,
                        threads: int = 0):
    """All distinct codewords of weight ``1..keep_weight`` that are an XOR of a size-(1..p)
    subset of the rows of ``G``, via the native BZ kernel.  Returns ``(rows, combos)``.
    """
    lib = _native.load()
    G = np.ascontiguousarray(np.asarray(G, dtype=np.uint8) % 2)
    m, n = G.shape
    h = lib.codeaut_bz_collect(G.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                               m, n, int(p), int(keep_weight), int(backend),
                               int(threads))
    if not h or not lib.codeaut_bz_ok(h):
        if h:
            lib.codeaut_bz_free(h)
        raise RuntimeError("codeaut_bz_collect failed")
    try:
        cnt = int(lib.codeaut_bz_count(h))
        combos = int(lib.codeaut_bz_combos(h))
        rows = np.zeros((max(cnt, 1), n), dtype=np.uint8)
        if cnt:
            lib.codeaut_bz_copy_rows(h, rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        rows = rows[:cnt]
    finally:
        lib.codeaut_bz_free(h)
    return rows, combos


def _collect_one_numpy(G: np.ndarray, p: int, keep_weight: int, _backend: int,
                       _threads: int = 0):
    """Pure-numpy fallback for :func:`_collect_one_native` (used only if the native library is
    unavailable).  Same contract (single-threaded; ``_threads`` is ignored)."""
    G = np.asarray(G, dtype=np.uint8) % 2
    k, n = G.shape
    seen: set = set()
    combos = 0
    for sw in range(1, min(p, k) + 1):
        it = itertools.combinations(range(k), sw)
        while True:
            idx = list(itertools.islice(it, 200_000))
            if not idx:
                break
            ai = np.asarray(idx, dtype=np.int64)
            cw = np.bitwise_xor.reduce(G[ai], axis=1)
            combos += cw.shape[0]
            wts = cw.sum(axis=1)
            mask = (wts <= keep_weight) & (wts > 0)
            for row in cw[mask]:
                seen.add(row.tobytes())
    rows = (np.frombuffer(b"".join(sorted(seen)), dtype=np.uint8).reshape(-1, n).copy()
            if seen else np.zeros((0, n), np.uint8))
    return rows, combos


def _collect_one(G, p, keep_weight, backend, threads=0):
    try:
        return _collect_one_native(G, p, keep_weight, backend, threads)
    except Exception:
        return _collect_one_numpy(G, p, keep_weight, backend, threads)


# ----------------------------------------------------------------------------------------
# greedy max-disjoint systematic generators (BZ scaffolding) -- ported from orchestration
# ----------------------------------------------------------------------------------------

def _systematic_generator_on(B: np.ndarray, info, k: int, n: int):
    info = [int(c) for c in info]
    info_set = set(info)
    order = np.array(info + [c for c in range(n) if c not in info_set], dtype=np.int64)
    rref, piv_in_perm = gf2.rref_gf2(B[:, order])
    G = np.zeros((k, n), dtype=np.uint8)
    G[:, order] = rref
    return G, order[piv_in_perm]


def _matroid_packing(B: np.ndarray, k: int, n: int):
    try:
        infosets, _fresh = matroid_pack.pack_information_sets(B)
    except Exception:
        return None
    valid: list[list[int]] = []
    for info in infosets:
        cols = [int(c) for c in info]
        if len(set(cols)) != len(cols) or len(cols) != k:
            return None
        if gf2.rank_gf2(B[:, cols]) != k:
            return None
        valid.append(cols)
    return valid or None


def _greedy_cover(B: np.ndarray, k: int, n: int, used: np.ndarray, gens: list, fresh_counts: list):
    while not used.all():
        order = np.concatenate([np.flatnonzero(~used), np.flatnonzero(used)])
        rref, piv_in_perm = gf2.rref_gf2(B[:, order])
        pivots = order[piv_in_perm]
        fresh = int(np.count_nonzero(~used[pivots]))
        if fresh == 0:
            break
        G = np.zeros((k, n), dtype=np.uint8)
        G[:, order] = rref
        gens.append(G)
        fresh_counts.append(fresh)
        used[pivots] = True


def _systematic_generators(B: np.ndarray):
    B = gf2.row_basis_gf2(B)
    k, n = B.shape
    gens: list[np.ndarray] = []
    fresh_counts: list[int] = []
    used = np.zeros(n, dtype=bool)

    packing = _matroid_packing(B, k, n)
    if packing:
        for info in packing:
            fresh = int(np.count_nonzero(~used[info]))
            if fresh == 0:
                continue
            G, pivots = _systematic_generator_on(B, info, k, n)
            if set(int(p) for p in pivots) != set(info):
                gens, fresh_counts = [], []
                used = np.zeros(n, dtype=bool)
                break
            gens.append(G)
            fresh_counts.append(fresh)
            used[info] = True

    _greedy_cover(B, k, n, used, gens, fresh_counts)
    return gens, fresh_counts


def _bz_lower_bound(p: int, k: int, fresh_counts: list[int]) -> int:
    return sum(max(0, p + 1 - (k - kj)) for kj in fresh_counts)


def _active_generators(gens, fresh_counts, p: int, k: int):
    return [j for j, f in enumerate(fresh_counts) if (p + 1 - (k - f)) > 0]


# ----------------------------------------------------------------------------------------
# enumeration kernels
# ----------------------------------------------------------------------------------------

def _full_enumeration(B: np.ndarray):
    """All nonzero codewords of ``rowspace(B)`` bucketed by weight (trivially complete)."""
    Bb = gf2.row_basis_gf2(B)
    k, n = Bb.shape
    by_weight: dict[int, np.ndarray] = {}
    if k == 0:
        return by_weight, k, n
    idx = np.arange(1, 1 << k, dtype=np.int64)
    U = ((idx[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    C = (U @ Bb) % 2
    w = C.sum(axis=1).astype(int)
    for ww in np.unique(w):
        if ww == 0:
            continue
        by_weight[int(ww)] = C[w == ww].copy()
    return by_weight, k, n


def _bz_collect(gens, fresh_counts, p: int, k: int, keep_weight: int, n: int,
                backend: int, threads: int = 0):
    """Enumerate info-weight ``1..p`` over the **contributing** generators, keeping codewords of
    weight ``<= keep_weight``.  Returns ``(by_weight, combos_used)``."""
    seen: dict[int, set] = {}
    combos = 0
    for j in _active_generators(gens, fresh_counts, p, k):
        rows, c = _collect_one(gens[j], p, keep_weight, backend, threads)
        combos += c
        if rows.shape[0]:
            wts = rows.sum(axis=1).astype(int)
            for row, ww in zip(rows, wts):
                seen.setdefault(int(ww), set()).add(row.tobytes())
    return _finalize(seen, n), combos


def _finalize(seen: dict[int, set], n: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for w, s in seen.items():
        if not s:
            continue
        out[w] = np.frombuffer(b"".join(sorted(s)), dtype=np.uint8).reshape(-1, n).copy()
    return out


# ----------------------------------------------------------------------------------------
# spanning prefix
# ----------------------------------------------------------------------------------------

def _spanning_prefix(by_weight: dict[int, np.ndarray], k: int, max_weight_keep: Optional[int]):
    weights = sorted(w for w in by_weight if (max_weight_keep is None or w <= max_weight_keep))
    classes: list[tuple[int, np.ndarray]] = []
    stacked: list[np.ndarray] = []
    for w in weights:
        rows = by_weight[w]
        classes.append((w, rows))
        stacked.append(rows)
        if gf2.rank_gf2(np.vstack(stacked)) == k:
            return classes, True
    return classes, False


# ----------------------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------------------

def low_weight_classes(B, *, want_span: bool = True, max_weight: Optional[int] = None,
                       full_enum_max_dim: int = 20,
                       max_p: int = 12, max_class_size: int = 200_000, backend: str = "auto",
                       threads: int = 0):
    """Complete ascending low-weight codeword classes of ``C = rowspace(B)``.

    Returns ``(classes, info)`` -- ``classes`` a list of ``(weight, rows)`` ascending, each a
    ``(count, n)`` uint8 array of the **complete** weight class; ``info`` a dict reporting the
    method (``full_enum`` / ``bz`` / ``trivial``), spanning status, completeness certification,
    the BZ threshold ``W_cert`` and parameter ``p``, and per-class diagnostics.

    ``backend`` selects the native enumeration backend (``"cpu"`` / ``"gpu"`` / ``"auto"``) for
    the BZ branch; the full-enumeration fast path is pure numpy.  ``threads`` caps the CPU
    backend's worker threads (``<=0`` => hardware concurrency); ignored by the GPU backends.
    """
    be = _BACKEND_CODE.get(backend, 2)
    Bb = gf2.row_basis_gf2(B)
    k, n = Bb.shape

    info: dict = {
        "dim": int(k), "n": int(n), "method": None, "spans": False,
        "certified_all": False, "min_weight": None,
        "p": None, "W_cert": None, "num_infosets": None, "fresh_counts": None,
        "classes": [],
    }

    if k == 0:                                       # C = {0}: Aut = Sym(n)
        info["method"] = "trivial"
        info["spans"] = True
        info["certified_all"] = True
        info["W_cert"] = int(n)
        return [], info

    # ---- fast path: full 2**dim enumeration (trivially complete) -------------------------
    if k <= min(full_enum_max_dim, _NUMPY_FULL_ENUM_CAP):
        by_weight, k, n = _full_enumeration(Bb)
        classes, spans = _spanning_prefix(by_weight, k, max_weight)
        info["method"] = "full_enum"
        info["spans"] = bool(spans)
        info["certified_all"] = True
        info["min_weight"] = int(min(by_weight)) if by_weight else None
        info["W_cert"] = int(max(by_weight)) if by_weight else None
        info["num_infosets"] = 1
        info["classes"] = [{"weight": int(w), "count": int(r.shape[0]), "certified": True}
                           for w, r in classes]
        return classes, info

    # ---- BZ branch ----------------------------------------------------------------------
    gens, fresh_counts = _systematic_generators(Bb)
    info["method"] = "bz"
    info["num_infosets"] = len(gens)
    info["fresh_counts"] = list(map(int, fresh_counts))
    info["max_disjoint_bases"] = int(sum(1 for f in fresh_counts if f == k))

    best_classes: list[tuple[int, np.ndarray]] = []
    best_W_cert = -1
    for p in range(1, max_p + 1):
        W_cert = _bz_lower_bound(p, k, fresh_counts) - 1
        if W_cert < 1:
            continue
        keep = W_cert if max_weight is None else min(W_cert, max_weight)
        by_weight, combos = _bz_collect(gens, fresh_counts, p, k, keep, n, be, threads)
        if any(r.shape[0] > max_class_size for r in by_weight.values()):
            break
        best_W_cert = int(keep)
        classes, spans = _spanning_prefix(by_weight, k, keep)
        best_classes = classes
        info["min_weight"] = int(min(by_weight)) if by_weight else None
        info["p"] = int(p)
        info["W_cert"] = int(keep)
        if spans:
            info["spans"] = True
            break
        if max_weight is not None and W_cert >= max_weight:
            break
        if not want_span:
            break

    info["certified_all"] = best_W_cert >= 0
    info["classes"] = [{"weight": int(w), "count": int(r.shape[0]), "certified": True}
                       for w, r in best_classes]
    return best_classes, info
