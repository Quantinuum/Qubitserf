"""The IBM 'gross code' [[144,12,12]] — a deliberately hard case for BZ.

Bivariate-bicycle codes are the regime where Webster et al. recommend connected-cluster
or Gurobi rather than Brouwer-Zimmermann: here the BZ search dimension K = n - rank(Hx)
= 78 exceeds n/2 = 72, so only ONE disjoint full information set fits and the BZ lower
bound rises only ~1 per weight level. qubitserf still *finds* the distance (12) instantly
via the random-information-set seed, and brackets it rigorously, but certifying the lower
bound requires enumerating to d=10 (C(78,10) ~ 1.3e12 combinations) -- which the
tuned Metal GPU backend now certifies in under 5 minutes on an M4 laptop (both dZ and
dX proven = 12 with --zx, no symmetry assumptions).

    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python bench/gross_code.py
"""
from __future__ import annotations
import time

import qubitserf.distfind as df
from qubitserf.distfind import codes


def main():
    Hx, Hz = codes.gross_code()
    n = Hx.shape[1]
    backend = "gpu" if "gpu" in df.available_backends() else "cpu"
    print(f"gross code: n={n}, the IBM [[144,12,12]] bivariate bicycle code")
    print(f"backend={backend}\n")
    print("enumeration depth -> rigorous bracket on the distance:")
    print(f"{'to d':>6} {'lower':>6} {'upper':>6} {'proven':>7} {'time':>9}")
    for cap in (5, 6, 7, 8):
        t0 = time.perf_counter()
        r = df.css_distance(Hx, Hz, backend=backend, max_weight=cap)
        dt = time.perf_counter() - t0
        print(f"{cap:>6} {r.lower_bound:>6} {r.distance:>6} {str(r.proven):>7} {dt:>8.2f}s")
    print("\nThe upper bound 12 (the true, known distance) is found immediately by the "
          "RIS seed; the lower bound needs d=10 to reach 12 (~2 min/side on an M4 GPU; "
          "run without max_weight, or `qubitserf --zx`, for the full proof).")


if __name__ == "__main__":
    raise SystemExit(main())
