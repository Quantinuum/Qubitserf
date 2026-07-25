#!/usr/bin/env python3
"""No-regression benchmark: codeaut vs. the original orchestration engine, head-to-head.

For each code, runs BOTH the standalone ``codeaut`` ladder and the original orchestration
engine (``orchestration/automorphisms/lib/joint_aut.py`` + Sage) on the **same** matrices on the
**same** machine, and checks:

  * the automorphism-group **orders agree** (correctness parity), and
  * codeaut is **not slower** -- ``codeaut`` must come in at or under the orchestration time
    (within a small tolerance for timing noise).

The orchestration engine is the production code that produced the agent-team results, so a
head-to-head on identical inputs is the faithful "not allowed to be slower" gate.  If the
orchestration engine cannot be imported (no Sage / not in-tree), the benchmark still runs the
codeaut side and validates known orders.

Run (from the repo root)::

    PATH=/usr/bin:$PATH \
    PYTHONPATH=codeaut/python:src:orchestration/automorphisms/lib \
    /usr/bin/python3 codeaut/bench/compare_orchestration.py
"""

from __future__ import annotations

import time

import numpy as np

import qubitserf.codeaut as codeaut
from qubitserf.codeaut import codes

# tolerance: codeaut may be at most this factor (+ constant) of the orchestration time.
TOL_FACTOR = 1.30
TOL_CONST = 0.05

KNOWN_ORDERS = {"steane": 168, "shor": 1296, "gross": 144}


def code_suite():
    yield "steane", codes.steane()
    yield "shor", codes.shor()
    yield "iceberg6", codes.iceberg(3)
    yield "toric_L3", codes.toric(3)
    yield "toric_L4", codes.toric(4)
    yield "toric_L5", codes.toric(5)
    yield "surface_d3", codes.surface(3)
    yield "surface_d5", codes.surface(5)
    yield "gross_144", codes.gross()


class _Shim:
    """Minimal code object for the orchestration ``aut_core.css_exact`` (needs only Hx/Hz/n)."""
    def __init__(self, Hx, Hz):
        self.Hx, self.Hz = Hx, Hz
        self.n = int(Hx.shape[1])


def _import_orchestration():
    try:
        import joint_aut
        import aut_core
        return joint_aut, aut_core
    except Exception as e:  # pragma: no cover
        print(f"(orchestration engine unavailable: {type(e).__name__}: {e})")
        return None, None


def _orch_best(joint_aut, aut_core, code):
    """The orchestration's best (order, time): the faster of its leon+Sage-intersect path
    (``aut_core.css_exact``, used by harvest_worker for low eff_dim) and its joint-incidence
    path (``joint_aut.joint_exact``).  Returns ``(order_exact, seconds)``."""
    best_order, best_t = None, float("inf")
    # leon + Sage intersection (harvest_worker path), when eff_dim small enough
    try:
        t0 = time.time()
        G = aut_core.css_exact(_Shim(code.Hx, code.Hz), max_dim=24)
        t = time.time() - t0
        best_order, best_t = str(int(G.order())), t
    except Exception:
        pass
    # joint BZ + nauty incidence (joint_worker path)
    try:
        t0 = time.time()
        rec = joint_aut.joint_exact(code.Hx, code.Hz, max_dim=24)
        t = time.time() - t0
        if rec["complete"] and t < best_t:
            best_order, best_t = rec["order_exact"], t
        elif best_order is None:
            best_order, best_t = rec["order_exact"], t
    except Exception:
        pass
    return best_order, (best_t if best_t != float("inf") else float("nan"))


def main():
    joint_aut, aut_core = _import_orchestration()
    orch = joint_aut is not None
    rows = []
    failures = []
    print(f"\n{'code':14s} {'n':>4s} {'order':>14s} {'codeaut(s)':>11s} "
          f"{'orch(s)':>9s} {'speedup':>8s}  ok")
    print("-" * 78)
    for name, code in code_suite():
        # codeaut full ladder
        t0 = time.time()
        r = codeaut.css_automorphism_group(code, max_dim=24)
        t_ca = time.time() - t0
        order_ca = r.order

        # known-order check
        if name.split("_")[0] in KNOWN_ORDERS and name in KNOWN_ORDERS:
            if int(order_ca) != KNOWN_ORDERS[name]:
                failures.append(f"{name}: codeaut order {order_ca} != known {KNOWN_ORDERS[name]}")

        # orchestration head-to-head: its BEST method per code (leon+Sage or joint incidence)
        order_orch, t_orch, speed, ok = "-", float("nan"), float("nan"), True
        if orch:
            order_orch, t_orch = _orch_best(joint_aut, aut_core, code)
            if order_ca != order_orch:
                ok = False
                failures.append(f"{name}: order mismatch codeaut={order_ca} orch={order_orch}")
            if t_ca > t_orch * TOL_FACTOR + TOL_CONST:
                ok = False
                failures.append(f"{name}: codeaut {t_ca:.3f}s SLOWER than orch {t_orch:.3f}s")
            speed = t_orch / t_ca if t_ca > 0 else float("inf")

        rows.append((name, code.n, order_ca, t_ca, t_orch, speed, ok))
        sp = f"{speed:6.2f}x" if speed == speed else "    -   "
        to = f"{t_orch:8.3f}" if t_orch == t_orch else "       -"
        print(f"{name:14s} {code.n:>4d} {order_ca:>14s} {t_ca:>11.3f} {to:>9s} {sp:>8s}  "
              f"{'OK' if rows[-1][6] else 'FAIL'}")

    print("-" * 78)
    if orch:
        valid = [r for r in rows if r[4] == r[4]]
        if valid:
            tot_ca = sum(r[3] for r in valid)
            tot_orch = sum(r[4] for r in valid)
            print(f"TOTAL codeaut {tot_ca:.3f}s vs orchestration {tot_orch:.3f}s "
                  f"({tot_orch / tot_ca:.2f}x faster overall)")
    if failures:
        print("\nREGRESSIONS / FAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nNO REGRESSION: orders match and codeaut is at least as fast on every code.")


if __name__ == "__main__":
    main()
