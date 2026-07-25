#!/usr/bin/env python3
"""Benchmark power-of-two Ward fibers against codeaut's current min-weight prefix.

The Ward route constructs ``wt(xG) mod 2**t`` by truncated inclusion--exclusion, compiles it to
a reduced decision diagram, and materializes only a small complete spanning residue cover.  The
legacy route enumerates codewords in native C++ and takes ascending exact-weight classes until
they span.  Two controls use the same native solver: support-minimal/cocircuit filtering of the
legacy prefix, and an exhaustive-congruence search that scans all ``2**k`` messages.  All routes
return the exact same permutation automorphism group; every case checks this by order and mutual
generator containment.  Table labels are L=legacy, K=cocircuit, C=full-scan congruence, W=Ward.

Run from the codeaut repository root::

    PYTHONPATH=python python bench/compare_ward.py --repeats 5

The two dimension-16 structured cases are deliberately slow under the legacy algorithm, so they
are measured once unless ``--slow-repeats`` is supplied. Use ``--json`` for machine-readable
records.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np

from qubitserf.codeaut import codes, gf2, leon, ward


def _full_rank_random(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        generator = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        if gf2.rank_gf2(generator) == k:
            return generator


def _singleton_plus_repetition(a: int) -> np.ndarray:
    """The power-of-two separation family, with ``a = 2**t - 1``.

    Singleton rows occupy the first ``a`` coordinates; the final row repeats on the next
    ``a+2`` coordinates; one zero coordinate makes the length ``2a+3``.  The min-weight prefix
    contains all ``2**a`` singleton combinations before the last row enters its span.  Modulo
    ``a+1=2**t``, residue one consists of the ``a`` units and the repetition word, so it has only
    ``a+1`` elements and spans the code.
    """
    generator = np.zeros((a + 1, 2 * a + 3), dtype=np.uint8)
    generator[:a, :a] = np.eye(a, dtype=np.uint8)
    generator[a, a:2 * a + 2] = 1
    return generator


def _connected_parity_bridge(modulus: int) -> np.ndarray:
    """Connected-matroid control with the same sparse residue-one fiber.

    Adding ``modulus`` copies of the all-message parity column leaves weights unchanged modulo
    ``modulus`` but puts every basis-column type in one matroid circuit.  Residue one still has
    exactly the ``modulus`` basis messages, while the legacy prefix first passes through the
    nonzero even-message subspace.
    """
    m = int(modulus)
    generator = np.zeros((m, 3 * m), dtype=np.uint8)
    generator[:m - 1, :m - 1] = np.eye(m - 1, dtype=np.uint8)
    generator[m - 1, m - 1:2 * m] = 1
    generator[:, 2 * m:] = 1
    return generator


def _cases(random_per_shape: int):
    steane = codes.steane()
    surface = codes.surface(3)
    shor = codes.shor()
    toric = codes.toric(3)
    cases = [
        ("simplex[7,3]", steane.Hx, 8, False),
        ("hamming[7,4]", gf2.nullspace_basis_gf2(steane.Hx), 8, False),
        ("surface3-x[9,4]", surface.Hx, 8, False),
        ("surface3-z[9,4]", surface.Hz, 8, False),
        ("shor-z[9,6]", shor.Hz, 8, False),
        ("toric3-x[18,8]", toric.Hx, 8, False),
        ("unit+rep[7]", _singleton_plus_repetition(7), 8, False),
        ("unit+rep[15]", _singleton_plus_repetition(15), 16, True),
        ("connected-bridge[7]", _connected_parity_bridge(8), 8, False),
        ("connected-bridge[15]", _connected_parity_bridge(16), 16, True),
    ]
    for n, k in ((12, 6), (16, 8), (20, 10), (24, 12)):
        for sample in range(random_per_shape):
            seed = 20_260_714 + 1000 * n + 10 * k + sample
            cases.append((f"random[{n},{k}]#{sample}",
                          _full_rank_random(n, k, seed), 8, False))
    return cases


def _same_group(left, right) -> bool:
    if left.order != right.order:
        return False
    a, b = left.group(), right.group()
    return (all(a.contains(generator) for generator in b.gens()) and
            all(b.contains(generator) for generator in a.gens()))


def _median_run(function, repeats: int):
    runs = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        current = function()
        elapsed = time.perf_counter() - started
        if result is not None:
            assert current.order == result.order
        result = current
        runs.append((elapsed, current))
    wall = statistics.median(item[0] for item in runs)
    representative = min(runs, key=lambda item: abs(item[0] - wall))[1]
    phases = {}
    for attribute in ("form_seconds", "bdd_seconds", "enumeration_seconds", "search_seconds"):
        values = [getattr(item[1], attribute) for item in runs
                  if getattr(item[1], attribute, None) is not None]
        if values:
            phases[attribute] = statistics.median(values)
    return wall, representative, phases


def run(args):
    # Build/load the native library before the first timed case.
    leon.automorphism_group(codes.steane().Hx)
    records = []
    for name, matrix, modulus, slow in _cases(args.random_per_shape):
        generator = gf2.row_basis_gf2(matrix)
        repeats = args.slow_repeats if slow else args.repeats
        base_seconds, baseline, base_phases = _median_run(
            lambda: leon.automorphism_group(generator, max_dim=max(24, generator.shape[0]),
                                            spanning_set="minweight"), repeats)
        minimal_seconds, minimal, minimal_phases = _median_run(
            lambda: leon.automorphism_group(generator, max_dim=max(24, generator.shape[0]),
                                            spanning_set="minimal"), repeats)
        scan_seconds, scanned, scan_phases = _median_run(
            lambda: leon.automorphism_group(generator, max_dim=max(24, generator.shape[0]),
                                            spanning_set="congruence",
                                            max_modulus=modulus), repeats)
        ward_seconds, modular, ward_phases = _median_run(
            lambda: ward.automorphism_group(generator, modulus=modulus,
                                            max_dim=max(24, generator.shape[0]),
                                            max_words=args.max_words), repeats)
        if (not _same_group(baseline, minimal) or not _same_group(baseline, scanned) or
                not _same_group(baseline, modular)):
            raise AssertionError(f"group mismatch for {name}")
        legacy_enum = base_phases.get("enumeration_seconds", 0.0)
        legacy_search = base_phases.get("search_seconds", 0.0)
        scan_enum = scan_phases.get("enumeration_seconds", 0.0)
        scan_search = scan_phases.get("search_seconds", 0.0)
        minimal_enum = minimal_phases.get("enumeration_seconds", 0.0)
        minimal_search = minimal_phases.get("search_seconds", 0.0)
        ward_accounted = sum(ward_phases.get(key, 0.0) for key in
                             ("form_seconds", "bdd_seconds", "enumeration_seconds",
                              "search_seconds"))
        records.append({
            "case": name,
            "n": int(generator.shape[1]),
            "k": int(generator.shape[0]),
            "modulus": modulus,
            "ward_residues": list(modular.residues),
            "legacy_words": baseline.num_codewords,
            "minimal_words": minimal.num_codewords,
            "scan_words": scanned.num_codewords,
            "ward_words": modular.num_codewords,
            "legacy_edges": baseline.num_incidences,
            "minimal_edges": minimal.num_incidences,
            "scan_edges": scanned.num_incidences,
            "ward_edges": modular.num_incidences,
            "scan_modulus": scanned.modulus,
            "scan_residue": scanned.residue,
            "ward_bdd_nodes": modular.bdd_nodes,
            "legacy_seconds": base_seconds,
            "minimal_seconds": minimal_seconds,
            "scan_seconds": scan_seconds,
            "ward_seconds": ward_seconds,
            "legacy_enumeration_seconds": legacy_enum,
            "legacy_search_seconds": legacy_search,
            "legacy_unaccounted_seconds": max(0.0, base_seconds - legacy_enum - legacy_search),
            "minimal_enumeration_seconds": minimal_enum,
            "minimal_search_seconds": minimal_search,
            "minimal_unaccounted_seconds": max(
                0.0, minimal_seconds - minimal_enum - minimal_search),
            "scan_enumeration_seconds": scan_enum,
            "scan_search_seconds": scan_search,
            "scan_unaccounted_seconds": max(0.0, scan_seconds - scan_enum - scan_search),
            "ward_form_seconds": ward_phases.get("form_seconds", 0.0),
            "ward_bdd_seconds": ward_phases.get("bdd_seconds", 0.0),
            "ward_enumeration_seconds": ward_phases.get("enumeration_seconds", 0.0),
            "ward_search_seconds": ward_phases.get("search_seconds", 0.0),
            "ward_unaccounted_seconds": max(0.0, ward_seconds - ward_accounted),
            "speedup": base_seconds / ward_seconds if ward_seconds else math.inf,
            "scan_speedup": base_seconds / scan_seconds if scan_seconds else math.inf,
            "minimal_speedup": base_seconds / minimal_seconds if minimal_seconds else math.inf,
            "word_ratio": modular.num_codewords / baseline.num_codewords,
            "ward_used": bool(modular.residues),
            "order": str(modular.order),
            "method": modular.method,
            "repeats": repeats,
        })
    return records


def _print_table(records):
    print("case                    [n,k]  m WardR words(L/K/C/W)        edges(L/K/C/W)       "
          "ms(L/K/C/W)                         W/L")
    print("-" * 150)
    for row in records:
        residues = ",".join(str(x) for x in row["ward_residues"]) or "fallback"
        print(f'{row["case"]:<23} [{row["n"]:2d},{row["k"]:2d}] '
              f'{row["modulus"]:2d} {residues:>5} '
              f'{row["legacy_words"]:5d}/{row["minimal_words"]:<5d}/'
              f'{row["scan_words"]:<5d}/{row["ward_words"]:<5d} '
              f'{row["legacy_edges"]:5d}/{row["minimal_edges"]:<5d}/'
              f'{row["scan_edges"]:<5d}/{row["ward_edges"]:<5d} '
              f'{1e3 * row["legacy_seconds"]:7.2f}/{1e3 * row["minimal_seconds"]:<7.2f}/'
              f'{1e3 * row["scan_seconds"]:<7.2f}/{1e3 * row["ward_seconds"]:<7.2f} '
              f'{row["speedup"]:7.2f}x')
    ward_rows = [row for row in records if row["ward_used"]]
    print(f"\nWard used on {len(ward_rows)}/{len(records)} cases; "
          f"faster on {sum(row['speedup'] > 1 for row in records)}/{len(records)}; "
          f"fewer word vertices on "
          f"{sum(row['ward_words'] < row['legacy_words'] for row in records)}/{len(records)}; "
          f"fewer incidence edges on "
          f"{sum(row['ward_edges'] < row['legacy_edges'] for row in records)}/{len(records)}; "
          f"median speedup {statistics.median(row['speedup'] for row in records):.2f}x.")
    print(f"Cocircuit selector: faster on "
          f"{sum(row['minimal_speedup'] > 1 for row in records)}/{len(records)}, graph no larger "
          f"on {sum(row['minimal_words'] <= row['legacy_words'] for row in records)}/"
          f"{len(records)}, median speedup "
          f"{statistics.median(row['minimal_speedup'] for row in records):.2f}x.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--slow-repeats", type=int, default=1)
    parser.add_argument("--random-per-shape", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=200_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if min(args.repeats, args.slow_repeats, args.max_words) < 1 or args.random_per_shape < 0:
        parser.error("repeat counts/max-words must be positive and random-per-shape nonnegative")
    records = run(args)
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        _print_table(records)


if __name__ == "__main__":
    main()
