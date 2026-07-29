"""The IBM 'gross code' [[144,12,12]] — a deliberately hard case for BZ.

Bivariate-bicycle codes are the regime where Webster et al. recommend connected-cluster
or Gurobi rather than Brouwer-Zimmermann: here the BZ search dimension K = n - rank(Hx)
= 78 exceeds n/2 = 72, so only ONE disjoint full information set fits and the BZ lower
bound rises only ~1 per weight level.  qubitserf still *finds* the distance (12) instantly
via the random-information-set seed, but the lower bound requires enumerating to d=10
(C(78,10) ~ 1.3e12 combinations) -- which the tuned Metal GPU backend does in under 5
minutes on an M4 laptop, while connected cluster does the whole job in well under a second.

This script times ``css_distance`` on the gross code per method and backend.

    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python bench/distfind/gross_code.py
"""
from __future__ import annotations
import time

import qubitserf.distfind as df
from qubitserf.distfind import codes


def run(Hx, Hz, method, backend):
    t0 = time.perf_counter()
    d = df.css_distance(Hx, Hz, method=method, backend=backend)
    return d, time.perf_counter() - t0


def main():
    Hx, Hz = codes.gross_code()
    print(f"gross code: n={Hx.shape[1]}, the IBM [[144,12,12]] bivariate bicycle code")
    print(f"backends: {df.available_backends()}\n")
    print(f"{'method':>10} {'backend':>8} {'d':>4} {'time':>10}")
    jobs = [("cc", "cpu")] + [("bz", b) for b in df.available_backends()]
    for method, backend in jobs:
        d, dt = run(Hx, Hz, method, backend)
        print(f"{method:>10} {backend:>8} {d:>4} {dt:>9.2f}s")
    print("\nBZ has to enumerate to weight 10 before its lower bound reaches 12; "
          "connected cluster exploits the sparsity instead.")


if __name__ == "__main__":
    raise SystemExit(main())
