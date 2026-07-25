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
incidence.  ``complete=True`` iff every used class is certified-complete AND the classes span
BOTH sides AND every generator preserves both ``Hx`` and ``Hz`` rowspaces over GF(2); then the
incidence group equals ``Aut(Hx) ∩ Aut(Hz)`` exactly.  Otherwise a **verified subgroup** is
returned with ``complete=False`` (a sound lower bound), GF(2)-reverified as the safety net.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

import numpy as np

from . import gf2
from . import graphaut


def _side_classes(H: np.ndarray, *, max_dim: int, budget: int, full_enum_cap: int = 22,
                  backend: str = "auto", threads: int = 0):
    """Certified-complete ascending weight classes of the cheaper of ``rowspace(H)`` and its
    dual.  Returns ``(classes, side_info)``."""
    from . import lowweight
    B, n, r, eff = gf2.dual_basis(H)
    dualized = (n - r) < r
    classes, info = lowweight.low_weight_classes(
        B, want_span=True, budget=budget, full_enum_max_dim=max_dim, backend=backend,
        threads=threads)
    if (not info["spans"]) and info["method"] == "bz" and info["dim"] <= full_enum_cap:
        classes, info = lowweight.low_weight_classes(
            B, want_span=True, budget=budget, full_enum_max_dim=info["dim"], backend=backend,
            threads=threads)
    side = {
        "n": int(n), "rank": int(r), "eff": int(eff), "dualized": bool(dualized),
        "dim": int(info["dim"]), "method": info["method"], "spans": bool(info["spans"]),
        "certified_all": bool(info["certified_all"]), "budget_hit": bool(info["budget_hit"]),
        "min_weight": info["min_weight"], "p": info["p"], "W_cert": info["W_cert"],
        "num_classes": len(classes), "num_words": int(sum(rr.shape[0] for _, rr in classes)),
    }
    return classes, side


def _group_record(G, n, Hx, Hz, *, complete, method, eff, t0, extra=None):
    order = G.order()
    gens = G.gens()
    verified = all(gf2.preserves_rowspace(Hx, gp) and gf2.preserves_rowspace(Hz, gp)
                   for gp in gens)
    rec = {
        "order": str(order),
        "generators": gens,
        "complete": bool(complete),
        "verified": bool(verified),
        "method": method,
        "seconds": round(time.time() - t0, 3),
        "n": int(n),
        "eff": eff,
    }
    if extra:
        rec.update(extra)
    return rec


def _tanner_subgroup(n, Hx, Hz, eff, t0, *, timeout=None) -> Optional[dict]:
    """Sound verified type-preserving Tanner-graph subgroup (universal fallback)."""
    if graphaut.nauty_binary() is None:
        return None
    G = graphaut.tanner_permutation_group(Hx, Hz, timeout=timeout)
    return _group_record(
        G, n, Hx, Hz, complete=False,
        method="type-preserving colored-Tanner-graph subgroup (nauty, lower bound)",
        eff=eff, t0=t0)


def joint_exact(Hx, Hz, *, max_dim: int = 20, budget: int = 60_000_000,
                allow_subgroup_fallback: bool = True,
                nauty_timeout: Optional[float] = None,
                traces_timeout: Optional[float] = None,
                backend: str = "auto", threads: int = 0) -> dict:
    """Exact (or best verified) qubit-permutation automorphism group of the CSS code
    ``(Hx, Hz)`` via the joint BZ + nauty/Traces incidence.

    ``nauty_timeout`` (seconds): if set, the incidence solve tries dense nauty first and, on
    timeout, falls back to **Traces** (``At``) -- the BFS refiner that solves the residual
    GL(3,2)/A5 incidences nauty cannot finish.  Default ``None`` = nauty only.

    Returns the worker-contract record: ``order`` (exact group order as a decimal string),
    ``generators`` (0-indexed image lists), ``complete``, ``verified``,
    ``method``, ``seconds``, ``n``, ``eff`` (per-side diagnostics).
    """
    t0 = time.time()
    Hx = gf2.as_uint8(Hx)
    Hz = gf2.as_uint8(Hz)
    n = int(Hx.shape[1])
    if Hz.shape[1] != n:
        raise ValueError("Hx and Hz must have the same number of columns (qubits)")

    cx, sx = _side_classes(Hx, max_dim=max_dim, budget=budget, backend=backend, threads=threads)
    cz, sz = _side_classes(Hz, max_dim=max_dim, budget=budget, backend=backend, threads=threads)
    eff = {"x": sx, "z": sz, "eff_dim": max(sx["eff"], sz["eff"])}

    x_usable = sx["certified_all"]
    z_usable = sz["certified_all"]
    both_span = sx["spans"] and sz["spans"]

    if x_usable and z_usable:
        G, V = graphaut.incidence_group(n, (cx, cz), nauty_timeout=nauty_timeout,
                                        traces_timeout=traces_timeout)
        rec = _group_record(G, n, Hx, Hz, complete=False, method="pending", eff=eff, t0=t0,
                            extra={"incidence_vertices": int(V)})
        if both_span and rec["verified"]:
            rec["complete"] = True
            rec["method"] = (f"joint BZ+nauty incidence (exact); "
                             f"x:dim{sx['dim']}{'(dual)' if sx['dualized'] else ''} "
                             f"wt<={sx['W_cert']}, z:dim{sz['dim']}"
                             f"{'(dual)' if sz['dualized'] else ''} wt<={sz['W_cert']}")
            return rec
        if rec["verified"]:
            rec["complete"] = False
            rec["method"] = ("joint BZ+nauty incidence (complete classes, non-spanning -> "
                             "verified subgroup)")
            if not allow_subgroup_fallback:
                return rec
            tan = _tanner_subgroup(n, Hx, Hz, eff, t0)
            if tan is not None and int(tan["order"]) > int(rec["order"]):
                return tan
            return rec
        if not allow_subgroup_fallback:
            rec["complete"] = False
            rec["method"] = "joint incidence superset (UNVERIFIED -- do not use)"
            return rec

    if not allow_subgroup_fallback:
        raise RuntimeError("joint_exact: could not certify-complete spanning classes "
                           f"(x_usable={x_usable}, z_usable={z_usable}) and fallback disabled")
    tan = _tanner_subgroup(n, Hx, Hz, eff, t0)
    if tan is None:
        raise RuntimeError("joint_exact: subgroup fallback unavailable (no nauty?)")
    return tan
