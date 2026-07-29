"""Exact CSS qubit-permutation automorphism group via the joint Hx+Hz incidence.

``Aut(Hx) ∩ Aut(Hz)`` -- the CSS qubit-permutation automorphism group -- is the automorphism
group of ONE coloured coordinate<->codeword incidence holding the **complete** spanning
ascending weight classes of BOTH sides (disjoint colours), solved by nauty/Traces.  Per side we
enumerate the cheaper of ``rowspace(H)`` and its dual (dual-code trick), and obtain the complete
classes by Brouwer--Zimmermann (:mod:`codeaut.lowweight`) -- never enumerating ``2**dim``.

This is the production route that converts the dominant hard family (quasi-cyclic / bivariate-
bicycle / toric codes) from "subgroup-only / infeasible" to EXACT and cheap: for those families
the minimum-weight class alone is complete and spans, with only ``~O(n)`` words.

Correctness
-----------
Only **certified-complete** weight classes (from :mod:`codeaut.lowweight`) ever enter the
incidence.  The result is exact iff every used class is certified-complete AND the classes span
BOTH sides AND every generator preserves both ``Hx`` and ``Hz`` rowspaces over GF(2); then the
incidence group equals ``Aut(Hx) ∩ Aut(Hz)`` exactly.  When any of these conditions fails,
:func:`joint_exact` **raises** :class:`RuntimeError` -- it never returns a partial or
unverified result.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import gf2
from . import graphaut


def _side_classes(H: np.ndarray, *, max_dim: int, full_enum_cap: int = 22,
                  backend: str = "auto", threads: int = 0):
    """Certified-complete ascending weight classes of the cheaper of ``rowspace(H)`` and its
    dual.  Returns ``(classes, side_info)``."""
    from . import lowweight
    B, n, r, eff = gf2.dual_basis(H)
    dualized = (n - r) < r
    classes, info = lowweight.low_weight_classes(
        B, want_span=True, full_enum_max_dim=max_dim, backend=backend,
        threads=threads)
    if (not info["spans"]) and info["method"] == "bz" and info["dim"] <= full_enum_cap:
        classes, info = lowweight.low_weight_classes(
            B, want_span=True, full_enum_max_dim=info["dim"], backend=backend,
            threads=threads)
    side = {
        "n": int(n), "rank": int(r), "eff": int(eff), "dualized": bool(dualized),
        "dim": int(info["dim"]), "method": info["method"], "spans": bool(info["spans"]),
        "certified_all": bool(info["certified_all"]),
        "min_weight": info["min_weight"], "p": info["p"], "W_cert": info["W_cert"],
        "num_classes": len(classes), "num_words": int(sum(rr.shape[0] for _, rr in classes)),
    }
    return classes, side


def joint_exact(Hx, Hz, *, max_dim: int = 20,
                nauty_timeout: Optional[float] = None,
                traces_timeout: Optional[float] = None,
                backend: str = "auto", threads: int = 0) -> dict:
    """Exact qubit-permutation automorphism group of the CSS code ``(Hx, Hz)`` via the joint
    BZ + nauty/Traces incidence.  Exact-or-raise: never returns a partial result.

    ``nauty_timeout`` (seconds): if set, the incidence solve tries dense nauty first and, on
    timeout, falls back to **Traces** (``At``) -- the BFS refiner that solves the residual
    GL(3,2)/A5 incidences nauty cannot finish.  Default ``None`` = nauty only.

    Returns a record with ``order`` (exact group order as a Python int), ``generators``
    (0-indexed image lists), ``method``, ``seconds``, ``n``, ``eff`` (per-side diagnostics).
    Raises :class:`RuntimeError` when the classes cannot be certified-complete and spanning on
    both sides, or when a generator fails GF(2) re-verification.
    """
    t0 = time.time()
    Hx = gf2.as_uint8(Hx)
    Hz = gf2.as_uint8(Hz)
    n = int(Hx.shape[1])
    if Hz.shape[1] != n:
        raise ValueError("Hx and Hz must have the same number of columns (qubits)")

    cx, sx = _side_classes(Hx, max_dim=max_dim, backend=backend, threads=threads)
    cz, sz = _side_classes(Hz, max_dim=max_dim, backend=backend, threads=threads)
    eff = {"x": sx, "z": sz, "eff_dim": max(sx["eff"], sz["eff"])}

    if not (sx["certified_all"] and sz["certified_all"]):
        raise RuntimeError(
            "joint_exact: could not certify-complete the low-weight classes "
            f"(x certified={sx['certified_all']}, z certified={sz['certified_all']})")
    if not (sx["spans"] and sz["spans"]):
        raise RuntimeError(
            "joint_exact: certified classes do not span both sides "
            f"(x spans={sx['spans']}, z spans={sz['spans']}); the incidence group would "
            "only be a verified subgroup, not the exact Aut(Hx) ∩ Aut(Hz)")

    G, V = graphaut.incidence_group(n, (cx, cz), nauty_timeout=nauty_timeout,
                                    traces_timeout=traces_timeout)
    gens = G.gens()
    for gp in gens:
        if not (gf2.preserves_rowspace(Hx, gp) and gf2.preserves_rowspace(Hz, gp)):
            raise RuntimeError("joint_exact: an incidence generator failed GF(2) "
                               "re-verification (engine bug)")
    return {
        "order": G.order(),
        "generators": gens,
        "method": (f"joint BZ+nauty incidence (exact); "
                   f"x:dim{sx['dim']}{'(dual)' if sx['dualized'] else ''} "
                   f"wt<={sx['W_cert']}, z:dim{sz['dim']}"
                   f"{'(dual)' if sz['dualized'] else ''} wt<={sz['W_cert']}"),
        "seconds": round(time.time() - t0, 3),
        "n": int(n),
        "eff": eff,
        "incidence_vertices": int(V),
    }
