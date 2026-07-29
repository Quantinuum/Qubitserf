#!/usr/bin/env python3
"""Validate codeaut against the real qecdb codes: correctness + speed.

Samples CSS codes from the live qumba database (stratified by effective dimension), runs
``codeaut`` on each, and compares to the orchestration's RECORDED result:

  * **correctness** -- ``codeaut.css_automorphism_group`` is exact-or-raise: when it returns a
    group, its order MUST equal the recorded ``aut_order_exact`` (a different order is a hard
    bug).  When it raises RuntimeError on a code the orchestration solved, that is a coverage
    regression ("downgrade"), not an incorrectness.
  * **speed** -- codeaut's wall time vs the recorded ``aut_meta.seconds``.

Run (from the repo root, with the qecdb SSH tunnel up)::

    PATH=/usr/bin:$PATH \
    PYTHONPATH=codeaut/python:src:orchestration/automorphisms/lib \
    /usr/bin/python3 codeaut/bench/compare_qecdb.py [N_per_bucket] [per_code_timeout_s]
"""

from __future__ import annotations

import sys
import time

import numpy as np

import qubitserf.codeaut as codeaut
from db import codes_collection
from lib.codes import qecdb


def sample(col, match, size):
    return list(col.aggregate([{"$match": {"css": True, **match}}, {"$sample": {"size": size}}]))


def main():
    n_bucket = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    per_timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    col = codes_collection()

    base = {"aut_complete": True, "aut_order_exact": {"$exists": True}}
    buckets = [
        ("eff<=12", {**base, "aut_meta.eff_dim": {"$lte": 12}}),
        ("eff 13-24", {**base, "aut_meta.eff_dim": {"$gte": 13, "$lte": 24}}),
        ("eff>24", {**base, "aut_meta.eff_dim": {"$gt": 24}}),
    ]
    docs = []
    for label, match in buckets:
        s = sample(col, match, n_bucket)
        print(f"sampled {len(s):3d} codes with {label}")
        docs += s
    print(f"\nrunning codeaut on {len(docs)} qecdb codes (per-code timeout {per_timeout}s)\n")

    n_ok = n_downgrade = n_bug = n_err = 0
    slower = []
    ratios = []
    print(f"{'[[n,k,d]]':16s} {'eff':>4s} {'recorded':>22s} {'codeaut':>22s} "
          f"{'rec(s)':>8s} {'ca(s)':>8s}  status")
    print("-" * 110)
    for doc in docs:
        try:
            qc = qecdb.css_code_from_doc(doc)
            code = codeaut.CSSCode(np.asarray(qc.Hx, np.uint8), np.asarray(qc.Hz, np.uint8))
        except Exception as e:
            n_err += 1
            continue
        rec_order = str(doc["aut_order_exact"])
        rec_s = float(doc.get("aut_meta", {}).get("seconds", float("nan")))
        eff = doc.get("aut_meta", {}).get("eff_dim", "?")
        t0 = time.time()
        try:
            # exact-or-raise: a RuntimeError on a code the orchestration solved is a coverage
            # regression (a "downgrade"), counted separately from hard errors.
            grp = codeaut.css_automorphism_group(code)
        except RuntimeError:
            ca_s = time.time() - t0
            n_downgrade += 1
            print(f"[[{doc['n']},{doc['k']},{doc.get('d')}]]".ljust(16) +
                  f" {str(eff):>4s} {rec_order:>22s} {'RAISED (no exact)':>22s} "
                  f"{rec_s:>8.3f} {ca_s:>8.3f}  DOWNGRADE")
            continue
        except Exception as e:
            n_err += 1
            print(f"[[{doc['n']},{doc['k']},{doc.get('d')}]]".ljust(16) +
                  f" {str(eff):>4s} {rec_order:>22s} {'ERROR: '+type(e).__name__:>22s}")
            continue
        ca_s = time.time() - t0
        ca_order = str(grp.order())

        if ca_order == rec_order:
            status, n_ok = "MATCH", n_ok + 1
        else:
            status, n_bug = "BUG(order!=)", n_bug + 1

        if rec_s == rec_s:
            ratios.append(ca_s / rec_s if rec_s > 0 else 1.0)
            if ca_s > rec_s * 1.5 + 0.1:
                slower.append((doc["n"], doc["k"], eff, rec_s, ca_s, status))
        print(f"[[{doc['n']},{doc['k']},{doc.get('d')}]]".ljust(16) +
              f" {str(eff):>4s} {rec_order:>22s} {ca_order:>22s} {rec_s:>8.3f} {ca_s:>8.3f}  {status}")

    print("-" * 110)
    print(f"\nCORRECTNESS: {n_ok} match, {n_downgrade} downgrade (raised where the recorded "
          f"run was exact), {n_bug} BUG (wrong order), {n_err} errors")
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        print(f"SPEED: median codeaut/recorded = {med:.2f}x  (max {max(ratios):.2f}x); "
              f"{len(slower)} codes >1.5x+0.1s slower")
        for n, k, eff, rs, cs, st in slower[:20]:
            print(f"   slower: [[{n},{k}]] eff={eff} recorded={rs:.3f}s codeaut={cs:.3f}s ({st})")
    if n_bug:
        print("\nFAIL: codeaut reported a WRONG exact order on some code.")
        raise SystemExit(1)
    print("\nNo correctness bugs (every exact result matches the recorded order).")


if __name__ == "__main__":
    main()
