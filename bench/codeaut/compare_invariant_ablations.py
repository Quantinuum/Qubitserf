#!/usr/bin/env python3
"""Benchmark targeted invariant-portfolio guard ablations against Leon.

This driver complements :mod:`compare_invariant_portfolio`: it keeps that benchmark's
deterministic corpus, timing normalization, and exact group validation, but forces a small set
of deliberately different public configurations:

* bounded circuit/cocircuit support through weights 2, 4, and 8;
* coarse versus fully expanded hull-minor weight-enumerator/hull-square signatures;
* Schur characteristic-code construction with basic, medium, and full product budgets; and
* the default combined relation as the portfolio reference point.

Every returned exact group is checked against the legacy min-weight group by order and mutual
generator containment.  The table and JSON retain exact-hit/fallback status, phase timings,
candidate overgroup order, row-space orbit index, and graph sizes.

Run from the standalone ``products/codeaut`` repository root::

    PYTHONPATH=python python bench/compare_invariant_ablations.py
    PYTHONPATH=python python bench/compare_invariant_ablations.py \
        --json /tmp/codeaut-invariant-ablations.json

The default is a seven-case deterministic structural corpus.  Use ``--cases '*'`` for the full
portfolio corpus and ``--list-variants`` to inspect the exact public guard settings.
"""

from __future__ import annotations

import argparse
import fnmatch
import statistics
import sys
from dataclasses import dataclass
from typing import Any

from qubitserf.codeaut import invariants

import compare_invariant_portfolio as portfolio


FOCUSED_CASES = ",".join((
    "builtin-simplex*",
    "builtin-surface3-x*",
    "target-hull1*",
    "target-simplex-design*",
    "target-sparse-cocircuit*",
    "target-even-weight*",
    "random[[]16?8[]]*",
))


@dataclass(frozen=True)
class Ablation:
    name: str
    family: str
    method: str
    settings: dict[str, int]
    note: str


def _ablations() -> list[Ablation]:
    """Return the fixed, publication-comparable ablation matrix.

    ``max_hull_* = 1`` makes every nontrivial minor hull use its invariant coarse marker
    whenever expansion would exceed the guard.  The full configuration comfortably expands
    every minor in the focused corpus.  Schur budgets 16 and 64 expose the first useful and
    intermediate characteristic-code layers on these small codes; 200,000 is the production
    default.
    """
    return [
        Ablation(
            f"bounded-b{weight}", "bounded", "bounded",
            {
                "max_support_weight": weight,
                "max_bounded_subsets": 2_000_000,
                "max_bounded_bz_budget": 10_000_000,
                "max_bounded_class_size": 200_000,
            },
            f"force fixed-support/BZ circuits and cocircuits through weight {weight}")
        for weight in (2, 4, 8)
    ] + [
        Ablation(
            "hull-coarse-minors", "hull", "hull",
            {
                "max_hull_dimension": 1,
                "max_hull_words": 1,
                "max_schur_products": 1,
            },
            "force guarded coarse minor-hull and Schur markers"),
        Ablation(
            "hull-full-we-schur", "hull", "hull",
            {
                "max_hull_dimension": 12,
                "max_hull_words": 1 << 14,
                "max_schur_products": 200_000,
            },
            "expand minor-hull weight enumerators and hull-square signatures on the corpus"),
        Ablation(
            "schur-basic", "schur", "schur", {"max_schur_products": 16},
            "basic characteristic-code product budget"),
        Ablation(
            "schur-medium", "schur", "schur", {"max_schur_products": 64},
            "medium characteristic-code product budget"),
        Ablation(
            "schur-full", "schur", "schur", {"max_schur_products": 200_000},
            "production-default characteristic-code product budget"),
        Ablation(
            "combined-full", "combined", "combined",
            {
                "max_support_weight": 8,
                "max_bounded_subsets": 2_000_000,
                "max_bounded_bz_budget": 10_000_000,
                "max_bounded_class_size": 200_000,
                "max_hull_dimension": 12,
                "max_hull_words": 1 << 14,
                "max_schur_products": 200_000,
            },
            "force the default combined relation, including bounded hyperedges and residue "
            "projectors"),
    ]


def _select_ablations(patterns: str) -> list[Ablation]:
    wanted = [item.strip() for item in patterns.split(",") if item.strip()]
    variants = _ablations()
    if not wanted:
        return variants
    return [variant for variant in variants
            if any(fnmatch.fnmatch(variant.name, pattern) for pattern in wanted)]


def _runner(variant: Ablation):
    def run(generator, case: portfolio.BenchmarkCase, limits: portfolio.Limits):
        kwargs: dict[str, Any] = {
            "method": variant.method,
            "modulus": case.modulus,
            "max_dim": limits.max_dim,
            "max_words": limits.max_words,
            "max_enumerated": limits.max_enumerated,
            "max_geometry_rank": limits.max_geometry_rank,
            "max_geometry_candidates": limits.max_geometry_candidates,
            "max_moment_triples": limits.max_moment_triples,
            "max_stabilizer_orbit": limits.max_stabilizer_orbit,
            "timeout": limits.timeout,
            "fallback": not limits.no_fallback,
            **variant.settings,
        }
        return portfolio._invoke(invariants.automorphism_group, generator, kwargs)

    return run


def _method_specs(variants: list[Ablation]) -> list[portfolio.MethodSpec]:
    baseline = portfolio._builtin_specs()["legacy"]
    return [baseline] + [
        portfolio.MethodSpec(
            variant.name,
            f"codeaut.invariants:{variant.method}",
            _runner(variant),
            exact_by_contract=True,
        )
        for variant in variants
    ]


def _annotate(records: list[dict[str, Any]], variants: list[Ablation]) -> None:
    lookup = {variant.name: variant for variant in variants}
    for record in records:
        variant = lookup.get(record["requested_method"])
        record["ablation_family"] = "baseline" if variant is None else variant.family
        record["forced_method"] = "legacy" if variant is None else variant.method
        record["variant_settings"] = {} if variant is None else dict(variant.settings)
        record["variant_note"] = "Leon min-weight baseline" if variant is None else variant.note


def _milliseconds(value) -> str:
    return "-" if value is None else f"{1e3 * value:.2f}"


def _value(value) -> str:
    return "-" if value is None else str(value)


def _print_table(records: list[dict[str, Any]]) -> None:
    print("case                         variant                 hit/fallback   val       "
          "candidate   orbit words verts edges  pre_ms search_ms stab_ms total_ms speedup")
    print("-" * 170)
    for row in records:
        speedup = "-" if row.get("speedup") is None else f"{row['speedup']:.2f}x"
        print(
            f"{row['case']:<28} {row['requested_method']:<23} {row['status']:<14} "
            f"{row['validation']:<9} {_value(row['candidate_group_order']):>11} "
            f"{_value(row['orbit_index']):>7} {_value(row['num_codewords']):>5} "
            f"{_value(row['num_vertices']):>5} {_value(row['num_incidences']):>5} "
            f"{_milliseconds(row['preprocessing_seconds']):>7} "
            f"{_milliseconds(row['search_seconds']):>9} "
            f"{_milliseconds(row['stabilizer_seconds']):>7} "
            f"{_milliseconds(row.get('total_seconds')):>8} {speedup:>7}")

    print("\nPer-variant summary")
    for name in dict.fromkeys(row["requested_method"] for row in records):
        rows = [row for row in records if row["requested_method"] == name]
        hits = sum(row["status"] in ("baseline", "exact_hit") for row in rows)
        fallbacks = sum(row["status"] == "exact_fallback" for row in rows)
        guarded = sum(row["status"] == "guarded" for row in rows)
        bad = sum(row["status"] in ("error", "invalid_exact") for row in rows)
        times = [row["total_seconds"] for row in rows
                 if row.get("total_seconds") is not None]
        speedups = [row["speedup"] for row in rows if row.get("speedup") is not None]
        median_ms = "-" if not times else f"{1e3 * statistics.median(times):.2f}"
        median_speedup = "-" if not speedups else f"{statistics.median(speedups):.2f}x"
        print(f"  {name:<23} hits {hits:>2}/{len(rows):<2}  fallbacks {fallbacks:>2}  "
              f"guarded {guarded:>2}  errors {bad:>2}  median_ms {median_ms:>8}  "
              f"speedup {median_speedup}")


def _print_variants(variants: list[Ablation]) -> None:
    for variant in variants:
        settings = ", ".join(f"{key}={value}" for key, value in variant.settings.items())
        print(f"{variant.name:<24} method={variant.method:<8} {settings or '(defaults)'}")
        print(f"  {variant.note}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1,
                        help="timed repetitions per case/variant (median reported)")
    parser.add_argument("--cases", default=FOCUSED_CASES,
                        help="comma-separated shell globs; default is the focused corpus")
    parser.add_argument("--tags", default="",
                        help="retain cases matching at least one comma-separated tag")
    parser.add_argument("--exclude-tags", default="",
                        help="omit cases with any comma-separated tag")
    parser.add_argument("--random-per-shape", type=int, default=1,
                        help="random controls added to the reusable corpus")
    parser.add_argument("--variants", default="",
                        help="comma-separated variant-name globs; default is all ablations")
    parser.add_argument("--max-dim", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=200_000)
    parser.add_argument("--max-enumerated", type=int, default=1 << 18)
    parser.add_argument("--max-geometry-rank", type=int, default=6)
    parser.add_argument("--max-geometry-candidates", type=int, default=100_000)
    parser.add_argument("--max-moment-triples", type=int, default=20_000)
    parser.add_argument("--max-stabilizer-orbit", type=int, default=5_000)
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="per-method timeout hint; 0 disables it")
    parser.add_argument("--no-fallback", action="store_true",
                        help="expose failed/guarded forced certificates instead of Leon fallback")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--json", nargs="?", const="-", metavar="PATH")
    parser.add_argument("--csv", metavar="PATH")
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    limits_to_check = (
        args.max_dim, args.max_words, args.max_enumerated,
        args.max_geometry_rank, args.max_geometry_candidates,
        args.max_moment_triples,
    )
    if args.repeats < 1 or args.random_per_shape < 0:
        parser.error("--repeats must be positive and --random-per-shape nonnegative")
    if min(limits_to_check) < 1 or args.max_stabilizer_orbit < 0 or args.timeout < 0:
        parser.error("resource limits must be positive; stabilizer orbit and timeout may be zero")

    variants = _select_ablations(args.variants)
    cases = portfolio._select_cases(
        portfolio._corpus(args.random_per_shape), args.cases, args.tags, args.exclude_tags)
    if args.list_variants:
        _print_variants(variants)
        return 0
    if args.list_cases:
        portfolio._print_cases(cases)
        return 0
    if not variants:
        parser.error("variant filters selected no ablations")
    if not cases:
        parser.error("case filters selected no benchmark cases")

    limits = portfolio.Limits(
        args.max_dim,
        args.max_words,
        args.max_enumerated,
        args.max_geometry_rank,
        args.max_geometry_candidates,
        args.max_moment_triples,
        args.max_stabilizer_orbit,
        None if args.timeout == 0 else args.timeout,
        60_000_000,
        args.no_fallback,
    )
    records, mismatches = portfolio.run_benchmark(
        cases, _method_specs(variants), limits, args.repeats, args.verbose)
    _annotate(records, variants)
    if args.json != "-":
        _print_table(records)
    if args.json:
        portfolio._write_json(args.json, records)
    if args.csv:
        portfolio._write_csv(args.csv, records)
    return 0 if args.allow_mismatch or mismatches == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
