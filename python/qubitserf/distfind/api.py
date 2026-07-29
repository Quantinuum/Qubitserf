"""High-level distance-finding API.

Every entry point returns a plain ``int``. A distance is ``-1`` when it is not
defined for the input -- an empty code, or a code with no logical qubits, which
has no non-trivial operator to measure.
"""
from __future__ import annotations

import numpy as np

from . import _native
from . import io
from ._interop import as_css


_WHICH = {"min": "M", "m": "M", "z": "Z", "x": "X"}
_BACKENDS = {"auto", "cpu", "gpu"}


def _as_symplectic(S):
    """Coerce ``S`` to a contiguous (m, 2n) uint8 [z|x] symplectic matrix."""
    a = np.ascontiguousarray(np.asarray(S, dtype=np.uint8) & 1)
    if a.ndim != 2 or a.shape[1] % 2 != 0:
        raise ValueError("expected a 2D symplectic matrix with 2n (even) columns "
                         "in [z | x] order")
    return a


def _parse_symplectic_operator(operator, n):
    """Normalize ``operator`` to a length-2n [z|x] 0/1 array (n = qubits)."""
    if isinstance(operator, str):
        z, x = io.parse_operator(operator, n)
        return np.concatenate([z, x]).astype(np.uint8)
    if isinstance(operator, (tuple, list)) and len(operator) == 2:
        z = np.asarray(operator[0], dtype=np.uint8).reshape(-1) & 1
        x = np.asarray(operator[1], dtype=np.uint8).reshape(-1) & 1
        if z.shape[0] != n or x.shape[0] != n:
            raise ValueError("(z_vec, x_vec) must each have length n = %d" % n)
        return np.concatenate([z, x]).astype(np.uint8)
    a = np.asarray(operator, dtype=np.uint8).reshape(-1) & 1
    if a.shape[0] != 2 * n:
        raise ValueError("symplectic operator must have length 2n = %d" % (2 * n))
    return a.copy()


def _parse_operator(operator, n):
    """Normalize ``operator`` to a ``(z_vec, x_vec)`` pair of length-n 0/1 arrays.

    Accepts a Pauli string (chars I/X/Y/Z/._ ; a Y sets both Z and X), a 2-tuple
    ``(z_vec, x_vec)`` of 0/1 arrays, or a length-2n symplectic ``[z | x]`` array.
    """
    if isinstance(operator, str):
        return io.parse_operator(operator, n)
    if isinstance(operator, (tuple, list)) and len(operator) == 2:
        z = np.asarray(operator[0], dtype=np.uint8).reshape(-1) & 1
        x = np.asarray(operator[1], dtype=np.uint8).reshape(-1) & 1
        if z.shape[0] != n or x.shape[0] != n:
            raise ValueError("(z_vec, x_vec) must each have length n = %d" % n)
        return z, x
    a = np.asarray(operator, dtype=np.uint8).reshape(-1) & 1
    if a.shape[0] != 2 * n:
        raise ValueError("symplectic operator must have length 2n = %d" % (2 * n))
    return a[:n].copy(), a[n:].copy()


def _normalize_backend(backend) -> str:
    b = "auto" if backend is None else str(backend).lower()
    if b not in _BACKENDS:
        raise ValueError("backend must be one of auto/cpu/gpu")
    if b == "gpu" and not _native.backend_available("gpu"):
        raise RuntimeError("no GPU backend is available on this machine")
    return b


def css_distance(Hx, Hz=None, *, method="bz", which="min", backend="auto",
                 threads=0, verbose=False) -> int:
    """Distance of a CSS code given X- and Z-check matrices.

    The code may be given as two matrices ``css_distance(Hx, Hz)`` or as a single
    CSS-code-like object ``css_distance(code)`` -- anything accepted by
    :func:`as_css` (a ``(Hx, Hz)`` tuple, or an object exposing ``.to_arrays()`` /
    ``.css_matrices()`` / ``.Hx``+``.Hz``, e.g. ``lib.CSSCode``, ``codeaut.CSSCode``,
    ``qecdb.Code``).

    method:  "bz" (Brouwer-Zimmermann), "cc" (connected cluster), or "mitm".
    which:   "min" (= min(dX,dZ)), "z", or "x".
    backend: "auto", "cpu", or "gpu". "gpu" chooses the available accelerator
             for this machine.
    """
    Hx, Hz = as_css(Hx, Hz)
    w = _WHICH.get(str(which).lower())
    if w is None:
        raise ValueError("which must be one of min/z/x")
    backend = _normalize_backend(backend)
    r = _native.css_distance_raw(Hx, Hz, method, w, backend, threads, verbose)
    return int(r.distance)


def subsystem_css_distance(Gx, Gz=None, *, method="bz", which="min", backend="auto",
                           threads=0, verbose=False) -> int:
    """Dressed distance of a CSS *subsystem* code given its gauge generators.

    ``Gx`` (X-type) and ``Gz`` (Z-type) are the **gauge** generators (not the
    stabilizers); the stabilizer center is computed internally. The dressed
    distance is the minimum weight of an operator that commutes with every
    stabilizer but lies outside the gauge group.

    The gauge group may be given as two matrices ``subsystem_css_distance(Gx, Gz)``
    or as a single CSS-code-like object (anything :func:`as_css` accepts).

    method:  "bz", "cc" (uses the sparse stabilizer center), or "mitm".
    which:   "min" (= min(dX,dZ)), "z", or "x".
    backend: "auto", "cpu", or "gpu".
    """
    Gx, Gz = as_css(Gx, Gz)
    w = _WHICH.get(str(which).lower())
    if w is None:
        raise ValueError("which must be one of min/z/x")
    backend = _normalize_backend(backend)
    r = _native.subsystem_distance_raw(Gx, Gz, method, w, backend,
                                       threads, verbose)
    return int(r.distance)


def operator_weight(Gx, Gz, operator, *, method="bz", backend="auto",
                    threads=0, verbose=False) -> int:
    """Minimum weight of a Pauli ``operator`` modulo the group ``<Gx, Gz>``.

    ``Gx`` (X-type) and ``Gz`` (Z-type) are the generators of the multipliable
    group: the stabilizer generators for a stabilizer code, or the gauge
    generators for a subsystem code. The Z-part is minimized over its coset
    modulo rowspace(Gz) and the X-part over rowspace(Gx) independently; the
    returned weight is the larger of the two coset leaders.

    ``operator`` may be a Pauli string (length n; chars I/X/Y/Z/._ , a Y sets
    both the Z and X support), a 2-tuple ``(z_vec, x_vec)`` of 0/1 arrays, or a
    length-2n symplectic ``[z | x]`` array.

    method:  "bz" or "mitm" ("cc" falls back to bz). backend: "auto"/"cpu"/"gpu".
    """
    Gx, Gz = as_css(Gx, Gz)
    n = Gx.shape[1]
    if Gz.shape[1] != n:
        raise ValueError("Gx and Gz must have the same number of columns (n)")
    z_vec, x_vec = _parse_operator(operator, n)
    backend = _normalize_backend(backend)
    r = _native.operator_weight_raw(Gx, Gz, z_vec, x_vec, method, backend,
                                    threads, verbose)
    return max(int(r.z_weight), int(r.x_weight))


def stabilizer_distance(S, *, method="bz", which="min", backend="auto",
                        threads=0, verbose=False) -> int:
    """Distance of a general (possibly non-CSS) stabilizer code.

    ``S`` is a symplectic stabilizer matrix of shape ``(m, 2n)`` in ``[z | x]`` column
    order: row r is the Pauli with Z-support ``S[r, :n]`` and X-support ``S[r, n:]``.
    The distance is the minimum **symplectic** weight of an operator in ``C(S)`` (the
    normalizer) that is not itself a stabilizer.

    A CSS code (every row pure-X or pure-Z) is detected and routed to the fast CSS
    Hx/Hz solvers (honouring ``which``). For a genuinely non-CSS code:
      * ``"bz"`` (default) uses the weight-doubling isometry ``(a|b) -> (a|b|a^b)`` to
        reduce the symplectic-distance problem to an ordinary binary Hamming-distance
        problem of length ``3n`` and runs Brouwer-Zimmermann on it (symplectic distance
        = half the binary distance);
      * ``"mitm"`` uses the symplectic meet-in-the-middle search;
      * ``"cc"`` has no sound non-CSS form (it needs a sparse single-type CSS Tanner graph),
        so it is **rejected** for a non-CSS code -- raises ``ValueError`` (use bz or mitm).
    ``which`` is ignored for a non-CSS code (no separate Z/X distances).

    method:  "bz" (isometry, default) or "mitm"; "cc" is CSS-only (raises on a non-CSS code).
    which:   "min"/"z"/"x" (CSS only).
    backend: "auto", "cpu", or "gpu".
    """
    S = _as_symplectic(S)
    w = _WHICH.get(str(which).lower())
    if w is None:
        raise ValueError("which must be one of min/z/x")
    backend = _normalize_backend(backend)
    r = _native.stabilizer_distance_raw(S, method, w, backend, threads, verbose)
    return int(r.distance)


def subsystem_stabilizer_distance(G, *, method="bz", which="min", backend="auto",
                                  threads=0, verbose=False) -> int:
    """Dressed distance of a general (possibly non-CSS) subsystem code.

    ``G`` is the symplectic **gauge** matrix ``(m, 2n)`` in ``[z | x]`` order (the gauge
    generators may be non-commuting). The stabilizer center is computed internally; the
    dressed distance is the minimum symplectic weight of an operator that commutes with
    every stabilizer but lies outside the gauge group.

    Methods behave as in :func:`stabilizer_distance`: ``"bz"`` (default) via the
    weight-doubling isometry, or ``"mitm"``; ``"cc"`` is CSS-only and raises ``ValueError``
    on a non-CSS code.
    """
    G = _as_symplectic(G)
    w = _WHICH.get(str(which).lower())
    if w is None:
        raise ValueError("which must be one of min/z/x")
    backend = _normalize_backend(backend)
    r = _native.subsystem_stabilizer_distance_raw(G, method, w, backend,
                                                  threads, verbose)
    return int(r.distance)


def pauli_operator_weight(G, operator, *, method="bz", backend="auto",
                          threads=0, verbose=False) -> int:
    """Minimum symplectic weight of a general Pauli ``operator`` modulo the group ``<G>``.

    ``G`` is a symplectic ``(m, 2n)`` ``[z | x]`` matrix (stabilizer generators for a
    stabilizer code, gauge generators for a subsystem code). ``operator`` may be a Pauli
    string (length n; chars I/X/Y/Z/._ , a Y is X and Z on that qubit), a 2-tuple
    ``(z_vec, x_vec)`` of length-n 0/1 arrays, or a length-2n symplectic ``[z | x]``
    array. The returned weight is the minimum symplectic weight over the coset
    ``operator + rowspace(G)`` -- 0 exactly when ``operator`` is itself in ``rowspace(G)``.

    This is the non-CSS generalization of :func:`operator_weight` (which returns the
    independent Z/X coset weights for a CSS group).

    method:  "bz" (default; weight-doubling isometry + Brouwer-Zimmermann) or "mitm"
             (symplectic meet-in-the-middle). "cc" falls back to "mitm".
    """
    G = _as_symplectic(G)
    n = G.shape[1] // 2
    op = _parse_symplectic_operator(operator, n)
    backend = _normalize_backend(backend)
    r = _native.stabilizer_operator_weight_raw(G, op, method, backend,
                                               threads, verbose)
    return int(r.distance)


def classical_distance(H, *, method="bz", backend="auto",
                       threads=0, verbose=False) -> int:
    """Minimum distance of a classical linear code from its parity-check matrix H."""
    backend = _normalize_backend(backend)
    r = _native.classical_distance_raw(H, method, backend, threads, verbose)
    return int(r.distance)


def available_backends():
    """Return the list of usable backends."""
    out = ["cpu"]
    if _native.backend_available("gpu"):
        out.append("gpu")
    return out


def version() -> str:
    return _native.version()
