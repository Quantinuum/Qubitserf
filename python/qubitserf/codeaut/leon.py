"""Leon's algorithm: the permutation automorphism group of a binary linear code.

For ``C = rowspace(generator_matrix)``,

    Aut(C) = { pi in S_n : pi . C = C }     (column permutations fixing the code)

-- the same group as MAGMA's ``AutomorphismGroup``, computed by an optimized, bit-packed C++
partition-backtracking engine (J. S. Leon, *Computing automorphism groups of error-correcting
codes*, IEEE Trans. Inf. Theory 28 (1982) 496-511).  The engine enumerates the minimum-weight
(and, if needed, higher) codeword classes that *span* ``C`` -- an ``Aut(C)``-invariant
structure -- then individualization-refinement backtracking on the coordinate<->codeword
incidence (a two-pass low-memory enumeration: peak memory is decoupled from ``2**dim``).
"""

from __future__ import annotations

import ctypes
import operator
from typing import Optional

import numpy as np

from . import _native


_SPANNING_SET_CODES = {"minweight": 0, "congruence": 1, "auto": 2,
                       "minimal": 3, "cocircuit": 3}
_ACTUAL_SPANNING_SETS = {0: "minweight", 1: "congruence", 3: "minimal"}
_INT32_MAX = (1 << 31) - 1


class AutResult:
    """Result of :func:`automorphism_group`.

    Attributes:
      * ``generators`` -- list of 0-indexed image lists (length ``n``) generating ``Aut(C)``
        as a permutation group of the ``n`` coordinates (a strong generating set);
      * ``order`` -- exact ``|Aut(C)|`` as a Python ``int``;
      * ``n`` -- code length (number of coordinates the permutations act on);
      * ``dim`` -- dimension of ``C``;
      * ``num_codewords`` / ``weight_classes`` -- the ``Aut``-invariant codeword structure used
        (number of codewords; the ascending weights whose classes were needed to span ``C``).
    """

    __slots__ = ("generators", "order", "n", "dim", "num_codewords", "num_incidences",
                 "weight_classes",
                 "spanning_set", "modulus", "residue", "enumeration_seconds",
                 "search_seconds")

    def __init__(self, generators, order, n, dim, num_codewords, num_incidences,
                 weight_classes, spanning_set, modulus, residue, enumeration_seconds,
                 search_seconds):
        self.generators = generators
        self.order = order
        self.n = n
        self.dim = dim
        self.num_codewords = num_codewords
        self.num_incidences = num_incidences
        self.weight_classes = weight_classes
        self.spanning_set = spanning_set
        self.modulus = modulus
        self.residue = residue
        self.enumeration_seconds = enumeration_seconds
        self.search_seconds = search_seconds

    def group(self):
        """This result as a :class:`codeaut.permgroup.Group` on the ``n`` coordinates, for
        membership tests and element enumeration."""
        from . import permgroup
        return permgroup.Group(self.generators, self.n)

    def __repr__(self):
        return (f"AutResult(order={self.order}, n={self.n}, dim={self.dim}, "
                f"num_gens={len(self.generators)}, spanning_set={self.spanning_set!r}, "
                f"weight_classes={self.weight_classes})")


def automorphism_group(generator_matrix, *, max_dim: int = 20,
                       use_invariant: bool = True, spanning_set: str = "minweight",
                       max_modulus: Optional[int] = None) -> AutResult:
    """Permutation automorphism group of ``C = rowspace(generator_matrix)``.

    ``generator_matrix`` is an ``(m, n)`` array of 0/1 entries (any generating set of ``C``;
    it is row-reduced internally).  Codeword enumeration costs ``2**dim(C)``; ``max_dim`` caps
    the dimension (raises :class:`ValueError` above it -- ``dim`` is the number of independent
    generators, so high-rate long codes are cheap).  ``spanning_set`` selects the complete
    Aut-invariant codeword structure used by Leon's incidence:

    * ``"minweight"`` (default) takes ascending exact-weight classes until they span;
    * ``"congruence"`` searches ``2 <= m <= max_modulus`` for the smallest spanning set
      ``{c != 0 : wt(c) == r (mod m)}``;
    * ``"auto"`` probes congruences only when the min-weight prefix is large (at least one
      quarter of the nonzero code or 512 words), and uses one only when it has fewer vertices.
    * ``"minimal"`` (alias ``"cocircuit"``) filters that prefix to support-minimal codewords;
      these still span at the same stopping weight and can only shrink the incidence graph.

    ``max_modulus=None`` searches through ``n+1`` (so an individual exact-weight class is a
    candidate).  Exact weights remain separate graph colours inside a congruence class.
    ``use_invariant`` toggles the engine's first-leaf refinement-invariant pruning (a no-op on
    the result).  Returns an :class:`AutResult` with the generators, exact order, selector
    diagnostics, and native enumeration/search timings.
    """
    G = np.ascontiguousarray(np.asarray(generator_matrix, dtype=np.uint8) % 2)
    if G.ndim != 2:
        raise ValueError("generator_matrix must be 2-D")
    m, n = G.shape
    try:
        selector_code = _SPANNING_SET_CODES[str(spanning_set).lower()]
    except KeyError:
        raise ValueError("spanning_set must be 'minweight', 'congruence', 'auto', or "
                         "'minimal'/'cocircuit'") from None
    if isinstance(max_dim, bool):
        raise ValueError("max_dim must be an integer in [0, 2**31 - 1]")
    try:
        max_dim_code = operator.index(max_dim)
    except TypeError:
        raise ValueError("max_dim must be an integer in [0, 2**31 - 1]") from None
    if not 0 <= max_dim_code <= _INT32_MAX:
        raise ValueError("max_dim must be an integer in [0, 2**31 - 1]")
    if max_modulus is not None:
        if isinstance(max_modulus, bool):
            raise ValueError("max_modulus must be an integer >= 2, or None")
        try:
            max_modulus_code = operator.index(max_modulus)
        except TypeError:
            raise ValueError("max_modulus must be an integer >= 2, or None") from None
        if not 2 <= max_modulus_code <= _INT32_MAX:
            raise ValueError("max_modulus must be an integer in [2, 2**31 - 1], or None")
    else:
        max_modulus_code = 0
    lib = _native.load()
    Gp = G.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    if hasattr(lib, "qaut_leon_run_ex"):
        h = lib.qaut_leon_run_ex(Gp, m, n, max_dim_code, 1 if use_invariant else 0,
                                 selector_code, max_modulus_code)
    else:
        if selector_code != 0 or max_modulus is not None:
            raise RuntimeError("loaded codeaut native library predates congruence spanning sets; "
                               "rebuild it or unset CODEAUT_LIB_PATH")
        h = lib.qaut_leon_run(Gp, m, n, max_dim_code, 1 if use_invariant else 0)
    if not h:
        raise RuntimeError("qaut_leon_run returned NULL")
    try:
        if not lib.qaut_leon_ok(h):
            dim = lib.qaut_leon_dim(h)
            if dim >= 63:
                raise ValueError(
                    f"dim(C) = {dim} is not representable by Leon's uint64 Gray-code "
                    "enumerator (maximum 62); use an invariant/projector/graph method")
            raise ValueError(f"dim(C) = {dim} exceeds max_dim = {max_dim}; raise max_dim "
                             "(cost is 2**dim) or use a graph/structural method for high-dim codes")
        ng = lib.qaut_leon_num_gens(h)
        nf = lib.qaut_leon_num_factors(h)
        nc = lib.qaut_leon_num_codewords(h)
        num_incidences = (int(lib.qaut_leon_num_incidences(h))
                          if hasattr(lib, "qaut_leon_num_incidences") else None)
        ncl = lib.qaut_leon_num_classes(h)
        dim = lib.qaut_leon_dim(h)

        gens_buf = np.zeros((max(ng, 1), n), dtype=np.int32)
        if ng:
            lib.qaut_leon_copy_gens(h, gens_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        generators = [gens_buf[i].astype(int).tolist() for i in range(ng)]

        fac_buf = np.zeros(max(nf, 1), dtype=np.int64)
        if nf:
            lib.qaut_leon_copy_factors(h, fac_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        order = 1
        for i in range(nf):
            order *= int(fac_buf[i])

        wt_buf = np.zeros(max(ncl, 1), dtype=np.int32)
        if ncl:
            lib.qaut_leon_copy_weights(h, wt_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        weight_classes = [int(wt_buf[i]) for i in range(ncl)]

        if hasattr(lib, "qaut_leon_selector"):
            actual_code = int(lib.qaut_leon_selector(h))
            actual_spanning_set = _ACTUAL_SPANNING_SETS.get(actual_code, "minweight")
            modulus_value = int(lib.qaut_leon_modulus(h))
            residue_value = int(lib.qaut_leon_residue(h))
            modulus = modulus_value if actual_code == 1 else None
            residue = residue_value if actual_code == 1 else None
            enumeration_seconds = float(lib.qaut_leon_enumeration_ns(h)) / 1e9
            search_seconds = float(lib.qaut_leon_search_ns(h)) / 1e9
        else:
            actual_spanning_set = "minweight"
            modulus = residue = None
            enumeration_seconds = search_seconds = None
    finally:
        lib.qaut_leon_free(h)

    return AutResult(generators, order, int(n), int(dim), int(nc), num_incidences, weight_classes,
                     actual_spanning_set, modulus, residue, enumeration_seconds, search_seconds)


# Familiar alias (mirrors the in-tree ``lib.native.leon`` name).
code_automorphism_generators = automorphism_group
