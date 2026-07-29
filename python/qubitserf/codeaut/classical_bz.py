"""Classical ``Aut(C)`` via Brouwer--Zimmermann low-weight enumeration + nauty/Traces.

The exact automorphism group of a binary linear code ``C`` is the automorphism group of the
coloured coordinate<->codeword incidence built over any Aut-invariant, **complete**, spanning
family of codeword classes (Leon's reduction).  This engine obtains certified-complete
ascending weight classes of **both** ``C`` and ``C^perp`` (``Aut(C) = Aut(C^perp)``, so either
side's incidence gives the same group) from :mod:`codeaut.lowweight`, then solves the smaller
certified incidence with nauty/Traces (:mod:`codeaut.graphaut`) -- never enumerating
``2**dim``.  Both sides are enumerated because dimension is a bad cost proxy for BZ: the cost
driver is the weight distribution (how large the BZ parameter ``p`` must grow before the
certified classes span, and how many words they hold), not ``dim``.  An LDPC-style generating
set has its low-weight words on the primal side while its (low-dimensional) dual basis is
dense -- picking by dimension would send BZ to the expensive side.  Choosing by actual
certified word count mirrors :mod:`codeaut.side` (the CSS single-side route).
This is the classical single-code analogue of the CSS ``method="bz"`` route (the single-side
case of :mod:`codeaut.side`, with no "other side" constraint).

Correctness (the crux)
----------------------
When every class is certified complete and the classes together span, the incidence group *is*
``Aut(C)``: every automorphism of ``C`` permutes each complete weight class (hence extends to a
graph automorphism), and conversely a coordinate permutation preserving each class setwise
preserves their span -- which is ``C`` (or ``C^perp``; same group).  Each returned generator is
nevertheless re-verified to preserve ``rowspace(genmat)`` over GF(2) as a safety net; a
verification failure would mean the completeness certificate was violated, so the engine
**raises** instead of returning a wrong group.  Likewise, when no certified spanning class set
exists it raises -- it never returns an unverified overgroup or a silent subgroup.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional

from . import gf2
from . import graphaut


def _log(msg: str) -> None:
    print(f"[codeaut] {msg}", file=sys.stderr, flush=True)


@dataclass
class ClassicalAutResult:
    """Exact ``Aut(C)`` from the classical BZ + nauty/Traces engine (:func:`automorphism_group`).

    The classical-``bz`` counterpart of :class:`codeaut.AutResult` (which the Leon engine
    returns): ``order`` is the exact ``|Aut(C)|`` as a Python ``int``; ``generators`` are
    0-indexed image lists on the ``n`` coordinates; ``dim`` is ``dim(C)``.
    ``num_codewords`` / ``weight_classes`` describe the certified incidence actually used --
    of ``C`` itself, or of ``C^perp`` when ``dualized`` (the dual-code trick; the group is the
    same).  ``method`` records the route and diagnostics; ``seconds`` the wall time.
    """
    order: int
    generators: list
    n: int
    dim: int
    num_codewords: int
    weight_classes: list
    dualized: bool
    method: str
    seconds: float

    def group(self):
        """This result as a :class:`codeaut.permgroup.Group` on the ``n`` coordinates."""
        from ..algebra import permgroup
        return permgroup.Group(self.generators, self.n)

    def __repr__(self):
        return (f"ClassicalAutResult(order={self.order}, n={self.n}, dim={self.dim}, "
                f"num_gens={len(self.generators)}, dualized={self.dualized}, "
                f"weight_classes={self.weight_classes})")


def automorphism_group(genmat, *, full_enum_max_dim: int = 20,
                       backend: str = "auto", threads: int = 0,
                       nauty_timeout: Optional[float] = None,
                       traces_timeout: Optional[float] = None,
                       verbose: bool = False) -> ClassicalAutResult:
    """Exact ``Aut(C)`` of ``C = rowspace(genmat)`` via BZ weight classes + nauty/Traces.

    Enumerates the certified-complete ascending weight classes of **both** ``C`` and ``C^perp``
    (:func:`codeaut.low_weight_classes` with ``want_span=True``; ``Aut(C) = Aut(C^perp)``),
    then solves the coloured coordinate<->codeword incidence of whichever certified side has
    fewer words with nauty (falling back to Traces when ``nauty_timeout`` is set and expires).
    ``backend`` (``"auto"`` / ``"cpu"`` / ``"gpu"``) selects the native enumeration backend;
    ``threads`` caps the CPU backend's worker threads (``<= 0`` => all hardware cores).

    Always exact: raises :class:`RuntimeError` when neither side yields a certified-complete
    spanning class set, or when a returned generator fails GF(2) re-verification (never
    returns an unverified overgroup or a silent subgroup).  ``verbose`` prints one-line
    ``[codeaut]`` progress messages to **stderr**; it never changes the result.
    """
    t0 = time.time()
    H = gf2.as_uint8(genmat)
    if H.ndim != 2:
        raise ValueError("genmat must be 2-D")
    from . import lowweight
    Bp = gf2.row_basis_gf2(H)
    n = int(H.shape[1])
    r = int(Bp.shape[0])
    eff = min(r, n - r)
    Bd = gf2.nullspace_basis_gf2(H)
    sides = []
    for is_dual, B in ((False, Bp), (True, Bd)):
        if verbose:
            _log(f"classical bz: enumerating {'dual' if is_dual else 'primal'} side "
                 f"(dim={B.shape[0]}, n={n}, backend={backend})")
            te = time.time()
        classes, info = lowweight.low_weight_classes(B, want_span=True,
                                                     full_enum_max_dim=full_enum_max_dim,
                                                     backend=backend, threads=threads)
        if info["spans"] and info["certified_all"]:
            nwords = sum(rows.shape[0] for _, rows in classes)
            sides.append((nwords, is_dual, classes, info))
            if verbose:
                _log(f"classical bz: {'dual' if is_dual else 'primal'} side certified in "
                     f"{time.time() - te:.3f}s ({len(classes)} weight classes, "
                     f"{nwords} words)")
        elif verbose:
            _log(f"classical bz: {'dual' if is_dual else 'primal'} side not certified "
                 f"(spans={info['spans']}, certified_all={info['certified_all']}) "
                 f"after {time.time() - te:.3f}s")
    if not sides:
        raise RuntimeError(
            f"classical bz engine could not certify a complete spanning weight-class set on "
            f"either side (dim={r}, dual dim={n - r}, n={n}); use "
            f"method='leon' if 2**dim is affordable")
    sides.sort(key=lambda s: s[0])            # fewer incidence words first (as codeaut.side)
    _, dualized, classes, info = sides[0]
    if verbose:
        _log(f"classical bz: solving {'dual' if dualized else 'primal'} incidence with "
             f"nauty/Traces ({sides[0][0]} words)")
        ts = time.time()
    G, V = graphaut.incidence_group(n, (classes,), nauty_timeout=nauty_timeout,
                                    traces_timeout=traces_timeout)
    if verbose:
        _log(f"classical bz: nauty solve done in {time.time() - ts:.3f}s "
             f"(V={V}); verifying generators over GF(2)")
    gens = G.gens()
    bad = [gp for gp in gens if not gf2.preserves_rowspace(H, gp)]
    if bad:
        # Impossible when the classes are complete and spanning (see module docstring); a
        # failure means the certificate was violated, so refuse rather than approximate.
        raise RuntimeError(
            f"classical bz engine internal error: {len(bad)} of {len(gens)} incidence-graph "
            f"generators do not preserve rowspace(genmat) although the weight classes were "
            f"certified complete and spanning -- refusing to return an unverified group")
    nwords = sum(rows.shape[0] for _, rows in classes)
    method = (f"bz+nauty incidence on {'dual ' if dualized else ''}weight classes "
              f"(exact; dim={r}, eff={eff}, V={V}, words={nwords}, enum={info['method']})")
    return ClassicalAutResult(order=G.order(), generators=gens, n=int(n), dim=int(r),
                              num_codewords=int(nwords),
                              weight_classes=[int(w) for w, _ in classes],
                              dualized=bool(dualized), method=method,
                              seconds=round(time.time() - t0, 3))
