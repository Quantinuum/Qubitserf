"""GPU scaling benchmark for qubitserf (no reference comparison).

Shows qubitserf solving codes far beyond where the reference BZ implementation is
feasible (it times out above ~n=100). Reports CPU and GPU wall-clock times and
the speedup, and verifies the GPU result matches the CPU result.

    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python bench/scaling.py

Writes bench/scaling.md.
"""
from __future__ import annotations
import os
import time

import qubitserf.distfind as df
from qubitserf.distfind import codes

HERE = os.path.dirname(os.path.abspath(__file__))
SCALING_MD = os.path.join(HERE, "scaling.md")

# (family, L, run_cpu?) — skip CPU once it gets too slow to keep the script quick.
PLAN = [
    ("toric", 7, True), ("toric", 8, True), ("toric", 9, False),
    ("toric", 10, False), ("toric", 11, False),
    ("surface", 7, True), ("surface", 8, True), ("surface", 9, False),
]


def fmt(t):
    if t is None:
        return "-"
    return f"{t*1e3:.0f}ms" if t < 1 else f"{t:.2f}s"


def time_backend(Hx, Hz, backend):
    t0 = time.perf_counter()
    d = df.css_distance(Hx, Hz, backend=backend)
    return d, time.perf_counter() - t0


def main():
    have_gpu = "gpu" in df.available_backends()
    rows = []
    print(f"{'code':>12} {'n':>5} {'d':>4} {'cpu':>9} {'gpu':>9} {'speedup':>8}")
    for fam, L, run_cpu in PLAN:
        Hx, Hz = getattr(codes, fam)(L)
        n = Hx.shape[1]
        dg, tg = (None, None)
        if have_gpu:
            dg, tg = time_backend(Hx, Hz, "gpu")
        dc, tc = (None, None)
        if run_cpu:
            dc, tc = time_backend(Hx, Hz, "cpu")
        d = dg if dg is not None else dc
        if dc is not None and dg is not None:
            assert dc == dg, f"{fam}{L}: cpu {dc} != gpu {dg}"
        sp = (tc / tg) if (tc and tg) else None
        rows.append((f"{fam} L={L}", n, d, tc, tg, sp))
        print(f"{fam+' L='+str(L):>12} {n:>5} {str(d):>4} {fmt(tc):>9} {fmt(tg):>9} "
              f"{(f'{sp:.1f}x' if sp else '-'):>8}")

    with open(SCALING_MD, "w") as f:
        f.write("# qubitserf GPU scaling\n\n")
        f.write("qubitserf on codes beyond the reach of the reference BZ implementation "
                "(`codedistance.BZDistMW` times out above ~n=100). GPU result verified "
                "equal to CPU where both ran.\n\n")
        f.write("| code | n | d | t_cpu | t_gpu | cpu/gpu |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, n, d, tc, tm, sp in rows:
            f.write(f"| {name} | {n} | {d} | {fmt(tc)} | {fmt(tm)} | "
                    f"{(f'{sp:.1f}x' if sp else '-')} |\n")
    print(f"\nWrote {SCALING_MD}")


if __name__ == "__main__":
    raise SystemExit(main())
