"""Subprocess entry point that runs the reference BZ distance finder.

This is invoked by ``benchmark.py`` as a *separate process* so that we can
impose a hard wall-clock timeout (the reference ``codeDistance`` can hang or run
for minutes on larger / degenerate inputs).  It prints a single JSON line to
stdout:

    {"ok": true, "n": .., "k": .., "d": .., "seconds": ..}
    {"ok": false, "error": "..."}

Usage (internal):

    python _ref_runner.py css   <Hx.npy> <Hz.npy>
    python _ref_runner.py class <H.npy>          # H is a *parity-check* matrix

For the classical case we treat the parity-check matrix exactly as qubitserf
does (code = ker(H)) and feed it to the reference's single-block path.

The reference package (`codedistance`) has a circular-import quirk and pulls in
optional heavy deps (gurobipy) at import time; both are worked around here.
"""
from __future__ import annotations

import json
import sys
import time
import types


def _load_reference():
    """Import codedistance robustly and return its `distance` module.

    * Shims `gurobipy` (license-gated, only used by the unused GurobiDist path).
    * Repairs the codedistance circular import by copying names from
      `code_library` into the `distance` module namespace (CSSCode etc.).
    """
    if "gurobipy" not in sys.modules:
        g = types.ModuleType("gurobipy")
        g.GRB = types.SimpleNamespace()
        g.Model = None
        sys.modules["gurobipy"] = g

    import codedistance  # noqa: F401  (triggers package __init__)
    from codedistance import distance as D
    from codedistance import code_library as CL

    for nm in dir(CL):
        if not nm.startswith("__") and not hasattr(D, nm):
            setattr(D, nm, getattr(CL, nm))
    return D


def run_css(hx_path: str, hz_path: str, method: str = "BZDistMW") -> dict:
    import numpy as np

    D = _load_reference()
    Hx = np.load(hx_path).astype(np.int8)
    Hz = np.load(hz_path).astype(np.int8)
    params = {"LOCheck": False}
    t0 = time.perf_counter()
    rz = D.CSScodeDistance(Hx, Hz, method=method, component="Z", params=dict(params))
    rx = D.CSScodeDistance(Hx, Hz, method=method, component="X", params=dict(params))
    dt = time.perf_counter() - t0
    return {
        "ok": True,
        "n": int(rz["n"]),
        "k": int(rz["k"]),
        "d": int(min(rz["d"], rx["d"])),
        "dz": int(rz["d"]),
        "dx": int(rx["d"]),
        "seconds": dt,
    }


def run_classical(h_path: str) -> dict:
    import numpy as np

    D = _load_reference()
    H = np.load(h_path).astype(np.int8)
    params = {"LOCheck": False}
    t0 = time.perf_counter()
    # Single-block path; reference treats H as a stabiliser/parity matrix and
    # enumerates the code ker(H) -> same semantics as qubitserf.classical_distance.
    res = D.codeDistance(H, None, tB=1, method="BZDistMW", params=dict(params))
    dt = time.perf_counter() - t0
    return {"ok": True, "n": int(res["n"]), "k": int(res["k"]),
            "d": int(res["d"]), "seconds": dt}


def main() -> int:
    mode = sys.argv[1]
    try:
        if mode == "css":
            method = sys.argv[4] if len(sys.argv) > 4 else "BZDistMW"
            out = run_css(sys.argv[2], sys.argv[3], method)
        elif mode == "class":
            out = run_classical(sys.argv[2])
        else:
            out = {"ok": False, "error": f"unknown mode {mode!r}"}
    except Exception as exc:  # noqa: BLE001 - report everything cleanly
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
