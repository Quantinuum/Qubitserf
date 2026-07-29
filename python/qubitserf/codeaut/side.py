"""Single-side automorphism rescue for joint-incidence-hard codes.

For codes where the JOINT Hx+Hz min-weight incidence is too large for nauty (the residual
group-algebra family: GL(3,2), Coxeter, A5, ...), the automorphism group of ONE side's
incidence is often far more tractable, and frequently yields the EXACT qubit-permutation group.

Let ``G_e = Aut(rowspace(H_easy))`` be the exact automorphism group of the cheaper side
(from that side's certified-complete SPANNING classes).  Write ``T = Aut(Hx) ∩ Aut(Hz)``.
Always ``T ⊆ G_e``.

  * If **every generator** of ``G_e`` preserves ``rowspace(H_other)`` over GF(2), then the whole
    of ``G_e`` does, so ``G_e ⊆ T``; with ``T ⊆ G_e`` this gives ``T = G_e`` -- **EXACT**.
  * Else if ``|G_e|`` is small, enumerate ``G_e`` and keep the elements preserving ``H_other`` --
    this is exactly ``T`` -- **EXACT**.
  * Else this route cannot certify the exact group: :func:`side_aut_subgroup` **raises**
    :class:`RuntimeError` (exact-or-raise; partial lower bounds are never returned).
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

import numpy as np

from . import gf2
from . import graphaut
from ..algebra import permgroup


def _record(order, gens, n, Hx, Hz, *, method, eff, t0):
    for gp in gens:
        if not (gf2.preserves_rowspace(Hx, gp) and gf2.preserves_rowspace(Hz, gp)):
            raise RuntimeError("side_aut_subgroup: a generator failed GF(2) "
                               "re-verification (engine bug)")
    return {
        "order": int(order),
        "generators": gens,
        "method": method,
        "seconds": round(time.time() - t0, 3),
        "n": int(n),
        "eff": eff,
    }


def side_aut_subgroup(Hx, Hz, *, enum_cap: int = 50_000, full_enum_max_dim: int = 20,
                      side_timeout: Optional[float] = 120.0,
                      traces_timeout: Optional[float] = None,
                      backend: str = "auto", threads: int = 0) -> dict:
    """Exact ``Aut(Hx) ∩ Aut(Hz)`` via a single side's incidence, or raise.

    The result is exact in two cases: every generator of the single-side overgroup ``G_e``
    verifies against the other side (``T = G_e``), or ``|G_e| <= enum_cap`` and the elements
    of ``G_e`` preserving the other side are filtered by enumeration (``T`` exactly).  When
    neither side gives a certified-spanning, in-time incidence whose overgroup exactifies,
    raises :class:`RuntimeError` with the per-side reasons.  Returns a record with ``order``
    (exact Python int), ``generators``, ``method``, ``seconds``, ``n``, ``eff``.
    """
    from . import lowweight
    t0 = time.time()
    Hx = gf2.as_uint8(Hx)
    Hz = gf2.as_uint8(Hz)
    n = int(Hx.shape[1])

    sides = []
    for tag, Hself, Hother in (("x", Hx, Hz), ("z", Hz, Hx)):
        B, _n, r, eff = gf2.dual_basis(Hself)
        cl, info = lowweight.low_weight_classes(B, want_span=True,
                                                full_enum_max_dim=full_enum_max_dim,
                                                backend=backend, threads=threads)
        nwords = sum(rows.shape[0] for _, rows in cl)
        sides.append({"tag": tag, "Hself": Hself, "Hother": Hother, "classes": cl,
                      "info": info, "nwords": nwords, "eff": eff})
    eff_dim = max(s["eff"] for s in sides)
    eff = {"eff_dim": int(eff_dim),
           "x": {"eff": sides[0]["eff"], "spans": sides[0]["info"]["spans"],
                 "nwords": sides[0]["nwords"]},
           "z": {"eff": sides[1]["eff"], "spans": sides[1]["info"]["spans"],
                 "nwords": sides[1]["nwords"]}}

    sides.sort(key=lambda s: s["nwords"])               # cheaper (fewer vertices) side first
    reasons = []
    for s in sides:
        info = s["info"]
        if not (info["spans"] and info["certified_all"]):
            reasons.append(f"side[{s['tag']}]: classes not certified-spanning "
                           f"(spans={info['spans']}, certified={info['certified_all']})")
            continue
        try:
            G, V = graphaut.incidence_group(n, (s["classes"],), nauty_timeout=side_timeout,
                                            traces_timeout=traces_timeout)
        except subprocess.TimeoutExpired:
            reasons.append(f"side[{s['tag']}]: incidence solve timed out "
                           f"(side_timeout={side_timeout}s)")
            continue
        gens = G.gens()
        order = G.order()
        all_ok = all(gf2.preserves_rowspace(s["Hother"], gp) for gp in gens)
        if all_ok:
            return _record(order, gens, n, Hx, Hz,
                           method=f"single-side[{s['tag']}] Aut(rowspace) all-gens-verify "
                                  f"(exact; V={V}, words={s['nwords']})", eff=eff, t0=t0)
        if order <= enum_cap:
            keep = [list(el) for el in G if gf2.preserves_rowspace(s["Hother"], list(el))]
            sub = permgroup.Group(keep, n)
            return _record(sub.order(), sub.reduced_generators(), n, Hx, Hz,
                           method=f"single-side[{s['tag']}] overgroup enum-filter "
                                  f"(exact; |G_e|={order})", eff=eff, t0=t0)
        reasons.append(f"side[{s['tag']}]: overgroup |G_e|={order} > enum_cap={enum_cap} "
                       "and not all generators verify (cannot exactify)")
    raise RuntimeError("side_aut_subgroup: no side certified the exact group -- "
                       + "; ".join(reasons))
