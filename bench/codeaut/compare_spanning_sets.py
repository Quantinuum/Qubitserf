#!/usr/bin/env python3
"""Benchmark Leon's legacy min-weight prefix against weight-congruence spanning sets.

The two selectors feed the same native incidence-graph solver and both enumerate all ``2**k``
codewords twice.  This benchmark separates native selection time from graph-search time and
reports the actual graph footprint, so a smaller incidence is not mistaken for a faster whole
algorithm when the selector scan itself dominates.

Run from the codeaut repository root::

    PYTHONPATH=python python bench/compare_spanning_sets.py --repeats 5

Use ``--json`` for machine-readable records.  Every non-legacy result is checked against the
legacy exact group by order and mutual generator containment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np

from qubitserf.codeaut import codes, gf2, leon


def _full_rank_random(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        G = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        if gf2.rank_gf2(G) == k:
            return G


def _singleton_plus_repetition(a: int) -> np.ndarray:
    """Adversarial ``[2a+2,a+1]`` family for the ascending-prefix selector.

    The first ``a`` rows are singleton coordinates and the final row repeats over ``a+2``
    disjoint coordinates.  The legacy prefix retains ``2**a`` words before it reaches the last
    generator.  In contrast ``wt == 1 (mod a+1)`` contains the ``a`` units and the repetition word,
    exactly ``a+1`` independent words.
    """
    n = 2 * a + 2
    G = np.zeros((a + 1, n), dtype=np.uint8)
    G[:a, :a] = np.eye(a, dtype=np.uint8)
    G[a, a:] = 1
    return G


def _cases(random_per_shape: int):
    steane = codes.steane()
    shor = codes.shor()
    surface = codes.surface(3)
    toric = codes.toric(3)
    out = [
        ("simplex[7,3]", steane.Hx),
        ("hamming[7,4]", gf2.nullspace_basis_gf2(steane.Hx)),
        ("shor-x[9,2]", shor.Hx),
        ("shor-z[9,6]", shor.Hz),
        ("surface3-x[9,4]", surface.Hx),
        ("surface3-z[9,4]", surface.Hz),
        ("toric3-x[18,8]", toric.Hx),
        ("toric3-z[18,8]", toric.Hz),
    ]
    for a in (8, 10, 12):
        out.append((f"unit+rep[a={a}]", _singleton_plus_repetition(a)))
    # Includes small instances where a genuinely multi-weight residue class wins, plus a
    # dimension ladder where the selector overhead becomes visible.
    shapes = ((12, 6), (16, 8), (20, 10), (24, 12), (28, 14), (32, 16))
    for n, k in shapes:
        for sample in range(random_per_shape):
            seed = 20_260_714 + 1000 * n + 10 * k + sample
            out.append((f"random[{n},{k}]#{sample}", _full_rank_random(n, k, seed)))
    return out


def _graph_mib(n: int, num_codewords: int) -> float:
    vertices = n + num_codewords
    return vertices * ((vertices + 63) // 64) * 8 / (1024 ** 2)


def _median_result(G, selector: str, repeats: int, max_modulus):
    runs = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        current = leon.automorphism_group(
            G, max_dim=max(20, gf2.rank_gf2(G)), spanning_set=selector,
            max_modulus=max_modulus)
        wall = time.perf_counter() - t0
        if result is not None:
            assert current.order == result.order
            assert current.num_codewords == result.num_codewords
            assert current.weight_classes == result.weight_classes
        result = current
        runs.append((wall, current.enumeration_seconds, current.search_seconds))
    return result, {
        "total_seconds": statistics.median(x[0] for x in runs),
        "enumeration_seconds": statistics.median(x[1] for x in runs),
        "search_seconds": statistics.median(x[2] for x in runs),
    }


def _same_group(a, b) -> bool:
    if a.order != b.order:
        return False
    A, B = a.group(), b.group()
    return all(A.contains(g) for g in B.gens()) and all(B.contains(g) for g in A.gens())


def run(args):
    selectors = [x.strip() for x in args.selectors.split(",") if x.strip()]
    bad = [x for x in selectors if x not in ("minweight", "congruence", "auto")]
    if bad or "minweight" not in selectors:
        raise ValueError("--selectors must contain minweight and only minweight/congruence/auto")

    # Build/load before timing the first case.
    leon.automorphism_group(codes.steane().Hx, spanning_set="minweight")

    records = []
    for name, G in _cases(args.random_per_shape):
        B = gf2.row_basis_gf2(G)
        baseline = None
        case_records = []
        for selector in selectors:
            result, timing = _median_result(B, selector, args.repeats, args.max_modulus)
            if selector == "minweight":
                baseline = result
            elif not _same_group(baseline, result):
                raise AssertionError(f"group mismatch for {name}: minweight vs {selector}")
            rec = {
                "case": name,
                "n": int(B.shape[1]),
                "k": int(B.shape[0]),
                "requested": selector,
                "selected": result.spanning_set,
                "modulus": result.modulus,
                "residue": result.residue,
                "weights": result.weight_classes,
                "num_codewords": result.num_codewords,
                "num_incidences": result.num_incidences,
                "graph_mib": _graph_mib(result.n, result.num_codewords),
                "order": str(result.order),
                **timing,
            }
            case_records.append(rec)
            records.append(rec)
        base_time = case_records[0]["total_seconds"]
        base_words = case_records[0]["num_codewords"]
        for rec in case_records:
            rec["speedup"] = base_time / rec["total_seconds"] if rec["total_seconds"] else math.inf
            rec["word_ratio"] = rec["num_codewords"] / base_words if base_words else 1.0
    return records


def _print_table(records):
    print("case                         sel->actual       m:r   words  edges   graphMiB  "
          "enum_ms search_ms total_ms speedup weights")
    print("-" * 126)
    for r in records:
        mr = "-" if r["modulus"] is None else f'{r["modulus"]}:{r["residue"]}'
        label = f'{r["requested"]}->{r["selected"]}'
        print(f'{r["case"]:<28} {label:<17} {mr:>5} {r["num_codewords"]:7d} '
              f'{r["num_incidences"]:6d} {r["graph_mib"]:10.3f} '
              f'{1e3*r["enumeration_seconds"]:8.3f} {1e3*r["search_seconds"]:9.3f} '
              f'{1e3*r["total_seconds"]:8.3f} {r["speedup"]:7.2f}x {r["weights"]}')

    for selector in sorted(set(r["requested"] for r in records) - {"minweight"}):
        rows = [r for r in records if r["requested"] == selector]
        wins = sum(r["speedup"] > 1.0 for r in rows)
        shrinks = sum(r["word_ratio"] < 1.0 for r in rows)
        multi = sum(r["selected"] == "congruence" and len(r["weights"]) > 1 for r in rows)
        print(f"\n{selector}: median speedup {statistics.median(r['speedup'] for r in rows):.2f}x; "
              f"faster on {wins}/{len(rows)}; smaller incidence on {shrinks}/{len(rows)}; "
              f"multi-weight residue winners {multi}/{len(rows)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed repetitions per case and selector (median reported)")
    ap.add_argument("--random-per-shape", type=int, default=2,
                    help="deterministic random matrices per (n,k) shape")
    ap.add_argument("--selectors", default="minweight,congruence,auto",
                    help="comma-separated subset containing minweight")
    ap.add_argument("--max-modulus", type=int, default=None,
                    help="largest searched modulus (default n+1)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.repeats < 1 or args.random_per_shape < 0:
        ap.error("--repeats must be >=1 and --random-per-shape must be >=0")
    records = run(args)
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        _print_table(records)


if __name__ == "__main__":
    main()
