"""GPU-vs-CPU benchmark for the Brouwer-Zimmermann backend.

The GPU backend is the intended tool for the regime where the BZ enumeration is
the actual cost; on sub-millisecond codes the whole solve is dominated by the
shared random-information-set seed and per-dispatch latency, so the CPU and GPU
necessarily tie (the GPU path even falls back to the CPU there). This benchmark
therefore reports the cpu/gpu speedup over the codes whose CPU solve takes more
than a threshold (default 10 ms) -- the codes for which "use the GPU" is the
documented advice -- and verifies the GPU distance equals the CPU distance on
every code it runs.

    PYTHONPATH=python MPLBACKEND=Agg \
      /opt/miniconda3/envs/sage_env/bin/python bench/gpu_vs_cpu.py

Env:
    REPEAT       runs per (code, backend); the min is reported (default 3)
    WHICH        CSS component: min / z / x (default min)
    MIN_MS       cpu-time cutoff in ms for the headline median (default 10)
"""
from __future__ import annotations
import os, time, statistics, sys
import numpy as np
import qubitserf.distfind as df
from qubitserf.distfind import codes

REPEAT = int(os.environ.get("REPEAT", "3"))
WHICH = os.environ.get("WHICH", "min")
MIN_MS = float(os.environ.get("MIN_MS", "10"))


def dense_classical(n, m, seed):
    """A random dense [n, n-rank] classical code's parity-check matrix."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(m, n), dtype=np.uint8)


def build():
    """List of (name, kind, payload). kind in {'css','classical'}."""
    out = []
    for L in (6, 7, 8):
        Hx, Hz = codes.toric(L)
        out.append((f"toric L={L}", "css", (Hx, Hz)))
    for L in (7, 8):
        Hx, Hz = codes.surface(L)
        out.append((f"surface L={L}", "css", (Hx, Hz)))
    out.append(("rand[60,~35]", "classical", dense_classical(60, 25, 1)))
    out.append(("rand[64,~40]", "classical", dense_classical(64, 24, 2)))
    return out


def timeit(kind, payload, backend):
    best = float("inf"); dist = None
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        if kind == "css":
            r = df.css_distance(*payload, method="bz", which=WHICH, backend=backend)
        else:
            r = df.classical_distance(payload, method="bz", backend=backend)
        best = min(best, time.perf_counter() - t0)
        dist = r.distance
    return dist, best


def fmt_t(t):
    return f"{t*1e3:.1f}ms" if t < 1 else f"{t:.2f}s"


def main():
    print("backends:", df.available_backends(), "| which:", WHICH,
          "| repeat:", REPEAT, "| cutoff:", MIN_MS, "ms")
    print(f"{'code':>16} {'n':>4} {'d':>3} {'cpu':>9} {'gpu':>9} {'cpu/gpu':>8}  used")
    print("-" * 64)
    big = []
    for name, kind, payload in build():
        n = (payload[0] if kind == "css" else payload).shape[1]
        dc, tc = timeit(kind, payload, "cpu")
        dg, tg = timeit(kind, payload, "gpu")
        assert dc == dg, f"{name}: cpu {dc} != gpu {dg}"
        r = tc / tg if tg > 0 else float("nan")
        used = tc * 1e3 >= MIN_MS
        if used:
            big.append(r)
        print(f"{name:>16} {n:>4} {dc:>3} {fmt_t(tc):>9} {fmt_t(tg):>9} "
              f"{r:>7.2f}x  {'YES' if used else 'no'}")
    print("-" * 64)
    if big:
        med = statistics.median(big)
        print(f">{MIN_MS:.0f}ms codes: n={len(big)}  median cpu/gpu = {med:.2f}x  "
              f"min={min(big):.2f}x max={max(big):.2f}x")
        print("GOAL (median >= 2x):", "PASS" if med >= 2.0 else "FAIL")
    else:
        print("no codes above the cutoff")
    return 0


if __name__ == "__main__":
    sys.exit(main())
