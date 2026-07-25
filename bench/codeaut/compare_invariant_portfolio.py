#!/usr/bin/env python3
"""Benchmark codeaut's invariant portfolio against the exact Leon baseline.

The benchmark is deliberately an adapter rather than a second portfolio implementation.  It
loads :mod:`codeaut.invariants` lazily, discovers advertised methods, calls each method through
the common dispatcher (or a registered method-specific callable), and normalises result records
for comparison.  This lets the benchmark land before every experimental invariant does.

Every result declared exact is checked against Leon's min-weight-prefix result by exact order and
mutual generator containment.  Guarded, unavailable, overgroup-only, and exact-fallback outcomes
remain visible instead of aborting the whole corpus.  Timings distinguish invariant construction
from graph/group search whenever the method reports those phases; total time is always an outer
wall-clock median.

Run from the standalone ``products/codeaut`` repository root::

    PYTHONPATH=python python bench/compare_invariant_portfolio.py

Use ``--repeats 3`` (or more) for a publication timing sweep; the one-repeat default keeps the
full cross-product of methods and structural cases practical during development.

Useful discovery and output modes::

    PYTHONPATH=python python bench/compare_invariant_portfolio.py --list-methods
    PYTHONPATH=python python bench/compare_invariant_portfolio.py --methods auto
    PYTHONPATH=python python bench/compare_invariant_portfolio.py --methods all
    PYTHONPATH=python python bench/compare_invariant_portfolio.py --tags lcd,hull
    PYTHONPATH=python python bench/compare_invariant_portfolio.py --json results.json \
        --csv results.csv

Omitting ``--methods`` (or selecting ``all``) runs the full discovered suite.  Selecting
``auto`` benchmarks the portfolio's actual automatic dispatcher, alongside the legacy baseline.

The preferred ``codeaut.invariants`` contract is::

    available_methods() -> iterable[str] or mapping[str, metadata]
    automorphism_group(G, *, method, max_dim, timeout, max_words, modulus) -> result

where ``result`` exposes ``generators``, ``order``, ``exact``/``complete``, ``method``,
``used_fallback``, phase timings, graph sizes, and optionally ``group()``.  ``METHODS`` and
``METHOD_REGISTRY`` registries and method-specific callables are also understood.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import importlib
import inspect
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from qubitserf.codeaut import codes, gf2, leon, permgroup, ward


PORTFOLIO_METHOD_HINTS = (
    "auto",
    "bounded",
    "combined",
    "components",
    "compressed",
    "fingerprint",
    "geometry",
    "hull",
    "lcd",
    "minors",
    "modular",
    "moments",
    "residue",
    "schur",
)

GENERIC_DISPATCH_NAMES = (
    "automorphism_group",
    "invariant_automorphism_group",
    "run_method",
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    generator: np.ndarray
    tags: tuple[str, ...]
    modulus: int = 8
    note: str = ""


@dataclass(frozen=True)
class MethodSpec:
    name: str
    source: str
    runner: Optional[Callable]
    exact_by_contract: Optional[bool] = None
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class Limits:
    max_dim: int
    max_words: int
    max_enumerated: int
    max_geometry_rank: int
    max_geometry_candidates: int
    max_moment_triples: int
    max_stabilizer_orbit: int
    timeout: Optional[float]
    budget: int
    no_fallback: bool


# ---------------------------------------------------------------------------
# Deterministic benchmark corpus.
# ---------------------------------------------------------------------------


def _full_rank_random(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        generator = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        if gf2.rank_gf2(generator) == k:
            return generator


def _systematic_with_hull(n: int, k: int, hull_dim: int, seed: int) -> np.ndarray:
    """Find a deterministic systematic code with the requested hull dimension."""
    if not (0 <= hull_dim <= k <= n):
        raise ValueError("invalid [n,k] or hull dimension")
    rng = np.random.default_rng(seed)
    identity = np.eye(k, dtype=np.uint8)
    for _ in range(100_000):
        tail = rng.integers(0, 2, size=(k, n - k), dtype=np.uint8)
        generator = np.ascontiguousarray(np.hstack((identity, tail)))
        gram_rank = gf2.rank_gf2((generator @ generator.T) % 2)
        if k - gram_rank == hull_dim:
            return generator
    raise RuntimeError(f"could not construct deterministic hull-{hull_dim} [{n},{k}] code")


def _simplex(dimension: int) -> np.ndarray:
    """Binary simplex generator: every nonzero projective point is a column."""
    columns = [[(value >> bit) & 1 for value in range(1, 1 << dimension)]
               for bit in range(dimension)]
    return np.asarray(columns, dtype=np.uint8)


def _rm1(variables: int) -> np.ndarray:
    """Evaluation generator of RM(1, variables), an affine-geometry target."""
    points = np.arange(1 << variables, dtype=np.uint64)
    rows = [np.ones(1 << variables, dtype=np.uint8)]
    rows.extend(((points >> bit) & 1).astype(np.uint8) for bit in range(variables))
    return np.ascontiguousarray(np.vstack(rows))


def _singleton_plus_repetition(a: int) -> np.ndarray:
    """Sparse cocircuit / Ward-residue separation family, ``a+1`` a power of two."""
    generator = np.zeros((a + 1, 2 * a + 3), dtype=np.uint8)
    generator[:a, :a] = np.eye(a, dtype=np.uint8)
    generator[a, a:2 * a + 2] = 1
    return generator


def _connected_parity_bridge(modulus: int) -> np.ndarray:
    """Connected matroid with a sparse residue-one fiber modulo ``modulus``."""
    m = int(modulus)
    generator = np.zeros((m, 3 * m), dtype=np.uint8)
    generator[:m - 1, :m - 1] = np.eye(m - 1, dtype=np.uint8)
    generator[m - 1, m - 1:2 * m] = 1
    generator[:, 2 * m:] = 1
    return generator


def _repeated_components() -> np.ndarray:
    """Direct sum of repetition blocks, with repeats and isomorphic components."""
    block_lengths = (2, 2, 3, 3, 4, 5)
    generator = np.zeros((len(block_lengths), sum(block_lengths)), dtype=np.uint8)
    offset = 0
    for row, length in enumerate(block_lengths):
        generator[row, offset:offset + length] = 1
        offset += length
    return generator


def _even_weight_code(length: int) -> np.ndarray:
    """A small graphic/cographic target with many bounded-weight words."""
    generator = np.zeros((length - 1, length), dtype=np.uint8)
    generator[:, :length - 1] = np.eye(length - 1, dtype=np.uint8)
    generator[:, length - 1] = 1
    return generator


def _corpus(random_per_shape: int) -> list[BenchmarkCase]:
    steane = codes.steane()
    surface = codes.surface(3)
    shor = codes.shor()
    toric = codes.toric(3)
    out = [
        BenchmarkCase(
            "builtin-simplex[7,3]", steane.Hx,
            ("builtin", "hull", "design", "moments", "geometry"),
            note="self-orthogonal projective-plane code"),
        BenchmarkCase(
            "builtin-hamming[7,4]", gf2.nullspace_basis_gf2(steane.Hx),
            ("builtin", "hull", "design", "moments")),
        BenchmarkCase(
            "builtin-surface3-x[9,4]", surface.Hx,
            ("builtin", "lcd", "sparse-cocircuits")),
        BenchmarkCase(
            "builtin-shor-z[9,6]", shor.Hz,
            ("builtin", "lcd", "components", "repeated-columns")),
        BenchmarkCase(
            "builtin-toric3-x[18,8]", toric.Hx,
            ("builtin", "hull", "sparse-cocircuits")),
        BenchmarkCase(
            "target-lcd[14,7]", _systematic_with_hull(14, 7, 0, 20_260_714),
            ("lcd", "projector", "structured-random")),
        BenchmarkCase(
            "target-hull1[14,7]", _systematic_with_hull(14, 7, 1, 20_260_715),
            ("hull", "small-hull", "shortening", "structured-random")),
        BenchmarkCase(
            "target-repeated-components[19,6]", _repeated_components(),
            ("components", "repeated-columns", "parallel-columns")),
        BenchmarkCase(
            "target-sparse-cocircuit[17,8]", _singleton_plus_repetition(7),
            ("sparse-cocircuits", "ward-residues", "components"), modulus=8),
        BenchmarkCase(
            "target-connected-residue[24,8]", _connected_parity_bridge(8),
            ("ward-residues", "sparse-cocircuits", "connected-matroid"), modulus=8),
        BenchmarkCase(
            "target-simplex-design[15,4]", _simplex(4),
            ("design", "moments", "geometry", "schur")),
        BenchmarkCase(
            "target-rm1[8,4]", _rm1(3),
            ("geometry", "moments", "schur", "affine")),
        BenchmarkCase(
            "target-even-weight[10,9]", _even_weight_code(10),
            ("bounded-circuits", "minors", "geometry")),
    ]
    for n, k in ((12, 6), (16, 8)):
        for sample in range(random_per_shape):
            seed = 20_260_714 + 1000 * n + 10 * k + sample
            out.append(BenchmarkCase(
                f"random[{n},{k}]#{sample}", _full_rank_random(n, k, seed),
                ("random", "fingerprint", "control")))
    return out


def _select_cases(cases: list[BenchmarkCase], patterns: str, tags: str,
                  exclude_tags: str) -> list[BenchmarkCase]:
    wanted_patterns = [item.strip() for item in patterns.split(",") if item.strip()]
    wanted_tags = {item.strip() for item in tags.split(",") if item.strip()}
    excluded = {item.strip() for item in exclude_tags.split(",") if item.strip()}
    selected = []
    for case in cases:
        case_tags = set(case.tags)
        if wanted_patterns and not any(fnmatch.fnmatch(case.name, pattern)
                                       for pattern in wanted_patterns):
            continue
        if wanted_tags and not (wanted_tags & case_tags):
            continue
        if excluded & case_tags:
            continue
        selected.append(case)
    return selected


# ---------------------------------------------------------------------------
# Portfolio discovery and call adaptation.
# ---------------------------------------------------------------------------


def _normalise_method_name(value: Any) -> Optional[str]:
    if isinstance(value, str):
        name = value.strip()
        return name or None
    if isinstance(value, Mapping):
        return _normalise_method_name(value.get("name") or value.get("method"))
    return _normalise_method_name(getattr(value, "name", None))


def _advertised_entries(value: Any) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    if value is None:
        return entries
    if isinstance(value, Mapping):
        for key, metadata in value.items():
            name = _normalise_method_name(key) or _normalise_method_name(metadata)
            if name:
                entries[name] = metadata
        return entries
    if isinstance(value, str):
        return {value: None}
    if isinstance(value, Iterable):
        for item in value:
            name = _normalise_method_name(item)
            if name:
                entries[name] = item
    return entries


def _registry_callable(metadata: Any) -> Optional[Callable]:
    if callable(metadata):
        return metadata
    if isinstance(metadata, Mapping):
        for key in ("callable", "function", "runner", "compute"):
            candidate = metadata.get(key)
            if callable(candidate):
                return candidate
    for key in ("run", "compute", "automorphism_group"):
        candidate = getattr(metadata, key, None)
        if callable(candidate):
            return candidate
    return None


def _metadata_exact(metadata: Any) -> Optional[bool]:
    if isinstance(metadata, Mapping):
        for key in ("exact", "complete"):
            if key in metadata and metadata[key] is not None:
                return bool(metadata[key])
    for key in ("exact", "complete"):
        value = getattr(metadata, key, None)
        if value is not None and not callable(value):
            return bool(value)
    return None


def _unexpected_keyword(error: TypeError) -> Optional[str]:
    match = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", str(error))
    return match.group(1) if match else None


def _invoke(function: Callable, generator: np.ndarray, kwargs: dict[str, Any]):
    """Call an evolving experimental API while removing only proven-unsupported keywords."""
    call_kwargs = dict(kwargs)
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()):
        call_kwargs = {key: value for key, value in call_kwargs.items()
                       if key in signature.parameters}

    while True:
        try:
            return function(generator, **call_kwargs)
        except TypeError as error:
            keyword = _unexpected_keyword(error)
            if keyword is None or keyword not in call_kwargs:
                raise
            del call_kwargs[keyword]


def _method_kwargs(name: str, case: BenchmarkCase, limits: Limits,
                   generic_dispatch: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_dim": limits.max_dim,
        "max_words": limits.max_words,
        "max_enumerated": limits.max_enumerated,
        "max_geometry_rank": limits.max_geometry_rank,
        "max_geometry_candidates": limits.max_geometry_candidates,
        "max_moment_triples": limits.max_moment_triples,
        "max_stabilizer_orbit": limits.max_stabilizer_orbit,
        "timeout": limits.timeout,
        "budget": limits.budget,
        "modulus": case.modulus,
    }
    if generic_dispatch:
        kwargs["method"] = name
    if limits.no_fallback:
        kwargs["fallback"] = False
    return kwargs


def _make_runner(function: Callable, name: str, generic_dispatch: bool) -> Callable:
    def run(generator: np.ndarray, case: BenchmarkCase, limits: Limits):
        return _invoke(function, generator, _method_kwargs(name, case, limits,
                                                            generic_dispatch))
    return run


def _direct_method_callable(module, name: str, metadata: Any) -> Optional[Callable]:
    registered = _registry_callable(metadata)
    if registered is not None:
        return registered
    slug = re.sub(r"[^0-9A-Za-z_]", "_", name)
    for attribute in (
            f"{slug}_automorphism_group", f"automorphism_group_{slug}",
            f"{slug}_group", slug):
        candidate = getattr(module, attribute, None)
        if callable(candidate):
            return candidate
    if name == "portfolio":
        candidate = getattr(module, "invariant_portfolio_group", None)
        if callable(candidate):
            return candidate
    return None


def _discover_portfolio() -> tuple[dict[str, MethodSpec], Optional[str]]:
    try:
        module = importlib.import_module("codeaut.invariants")
    except Exception as error:
        return {}, f"codeaut.invariants unavailable: {type(error).__name__}: {error}"

    entries: dict[str, Any] = {}
    discovery_errors = []
    available = getattr(module, "available_methods", None)
    if callable(available):
        try:
            entries.update(_advertised_entries(available()))
        except Exception as error:  # discovery must not hide independently registered methods
            discovery_errors.append(f"available_methods failed: {type(error).__name__}: {error}")
    elif available is not None:
        entries.update(_advertised_entries(available))
    for attribute in ("METHODS", "METHOD_REGISTRY", "REGISTRY"):
        entries.update(_advertised_entries(getattr(module, attribute, None)))

    generic = None
    for attribute in GENERIC_DISPATCH_NAMES:
        candidate = getattr(module, attribute, None)
        if callable(candidate):
            generic = candidate
            break

    if not entries:
        portfolio = getattr(module, "invariant_portfolio_group", None)
        if callable(portfolio):
            entries["portfolio"] = portfolio
        elif generic is not None:
            # Transitional modules sometimes land the dispatcher before their registry.
            entries.update((name, None) for name in PORTFOLIO_METHOD_HINTS)

    specs: dict[str, MethodSpec] = {}
    for name, metadata in entries.items():
        if name in ("legacy", "minweight"):
            continue
        direct = _direct_method_callable(module, name, metadata)
        function = direct or generic
        if function is None:
            specs[name] = MethodSpec(
                name, "codeaut.invariants", None, _metadata_exact(metadata),
                "advertised method has neither a registered callable nor a generic dispatcher")
            continue
        specs[name] = MethodSpec(
            name=name,
            source="codeaut.invariants",
            runner=_make_runner(function, name, generic_dispatch=(direct is None)),
            exact_by_contract=_metadata_exact(metadata),
        )
    error = "; ".join(discovery_errors) if discovery_errors else None
    return specs, error


def _builtin_specs() -> dict[str, MethodSpec]:
    def legacy_runner(generator: np.ndarray, _case: BenchmarkCase, limits: Limits):
        return leon.automorphism_group(
            generator, max_dim=limits.max_dim, spanning_set="minweight")

    def cocircuit_runner(generator: np.ndarray, _case: BenchmarkCase, limits: Limits):
        return leon.automorphism_group(
            generator, max_dim=limits.max_dim, spanning_set="minimal")

    def ward_runner(generator: np.ndarray, case: BenchmarkCase, limits: Limits):
        kwargs = {
            "modulus": case.modulus,
            "max_dim": limits.max_dim,
            "max_words": limits.max_words,
        }
        if limits.timeout is not None:
            kwargs.update(timeout=limits.timeout, nauty_timeout=min(limits.timeout, 5.0),
                          traces_timeout=limits.timeout)
        return ward.automorphism_group(generator, **kwargs)

    return {
        "legacy": MethodSpec("legacy", "codeaut.leon", legacy_runner, True),
        "cocircuit": MethodSpec("cocircuit", "codeaut.leon", cocircuit_runner, True),
        "ward": MethodSpec("ward", "codeaut.ward", ward_runner, True),
    }


def _choose_methods(requested: Optional[str]) -> tuple[list[MethodSpec], list[str]]:
    builtin = _builtin_specs()
    portfolio, discovery_error = _discover_portfolio()
    available = {**builtin, **{name: spec for name, spec in portfolio.items()
                               if name not in builtin}}
    notices = [discovery_error] if discovery_error else []

    tokens = ([] if requested is None else
              [item.strip() for item in requested.split(",") if item.strip()])
    if not tokens or tokens == ["all"]:
        names = ["legacy", "cocircuit", "ward", *sorted(portfolio)]
    elif tokens == ["auto"]:
        names = ["legacy", "auto"]
    else:
        names = ["legacy", *[item for item in tokens if item != "legacy"]]

    seen = set()
    specs = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if name in available:
            specs.append(available[name])
        else:
            reason = discovery_error or f"method {name!r} was not advertised by codeaut.invariants"
            specs.append(MethodSpec(name, "unavailable", None, None, reason))
    return specs, notices


# ---------------------------------------------------------------------------
# Result normalisation and exact group validation.
# ---------------------------------------------------------------------------


def _source_objects(result: Any) -> list[Any]:
    roots = []
    if isinstance(result, tuple) and len(result) == 2:
        roots.extend((result[1], result[0]))
    else:
        roots.append(result)
    for root in list(roots):
        for name in ("stats", "diagnostics", "certificate"):
            nested = root.get(name) if isinstance(root, Mapping) else getattr(root, name, None)
            if nested is not None:
                roots.append(nested)
    return roots


def _read_from(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    value = getattr(source, name, None)
    return None if callable(value) else value


def _first(result: Any, *names: str) -> Any:
    for source in _source_objects(result):
        for name in names:
            value = _read_from(source, name)
            if value is not None:
                return value
    return None


def _seconds(result: Any, *names: str) -> Optional[float]:
    value = _first(result, *names)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _exact_flag(result: Any, contract: Optional[bool]) -> Optional[bool]:
    explicit = []
    for name in ("exact", "complete", "is_exact"):
        value = _first(result, name)
        if value is not None:
            explicit.append(bool(value))
    if explicit:
        return all(explicit)
    proof = str(_first(result, "proof_status", "status") or "").lower()
    if proof in ("exact", "exact_spanning", "exact_sum", "certified", "complete"):
        return True
    if proof in ("overgroup", "subgroup", "incomplete", "lower_bound"):
        return False
    return contract


def _group_from_result(result: Any, degree: int):
    roots = _source_objects(result)
    for root in roots:
        if hasattr(root, "contains") and hasattr(root, "gens"):
            return root
        group_value = root.get("group") if isinstance(root, Mapping) else getattr(root, "group", None)
        if callable(group_value):
            try:
                return group_value()
            except TypeError:
                pass
        elif group_value is not None and hasattr(group_value, "contains"):
            return group_value
    generators = _first(result, "generators", "gens")
    if generators is not None:
        return permgroup.Group(generators, degree)
    return None


def _order_from_result(result: Any, degree: int) -> Optional[int]:
    value = _first(result, "order", "group_order")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    group = _group_from_result(result, degree)
    if group is not None:
        order = getattr(group, "order", None)
        if callable(order):
            return int(order())
        if order is not None:
            return int(order)
    return None


def _fallback_fields(result: Any, requested: str) -> tuple[bool, Optional[str]]:
    reason = _first(result, "fallback_reason", "guard_reason")
    explicit = _first(result, "used_fallback", "is_fallback")
    actual = str(_first(result, "method", "selected_method", "actual_method",
                        "spanning_set") or "")
    fallback = bool(explicit) if explicit is not None else False
    if reason:
        fallback = True
    actual_lower = actual.lower()
    if "fallback" in actual_lower:
        fallback = True
    if requested not in ("legacy", "cocircuit") and actual_lower in ("minweight", "legacy"):
        fallback = True
    return fallback, None if reason is None else str(reason)


def _diagnostics(result: Any) -> Any:
    value = _first(result, "diagnostics")
    return _jsonable(value) if value is not None else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _normalise_result(result: Any, spec: MethodSpec, case: BenchmarkCase,
                      generator: np.ndarray) -> dict[str, Any]:
    n = int(generator.shape[1])
    order = _order_from_result(result, n)
    exact = _exact_flag(result, spec.exact_by_contract)
    fallback, fallback_reason = _fallback_fields(result, spec.name)

    preprocessing = _seconds(result, "preprocessing_seconds", "preprocess_seconds",
                             "construction_seconds")
    if preprocessing is None:
        pieces = [_seconds(result, name) for name in
                  ("form_seconds", "bdd_seconds", "enumeration_seconds")]
        present = [value for value in pieces if value is not None]
        preprocessing = sum(present) if present else None
    search = _seconds(result, "search_seconds", "group_seconds", "solver_seconds")
    if search is None:
        search_parts = [_seconds(result, name) for name in
                        ("graph_seconds", "stabilizer_seconds")]
        present = [value for value in search_parts if value is not None]
        search = sum(present) if present else None
    stabilizer = _seconds(result, "stabilizer_seconds", "exactification_seconds")
    reported_total = _seconds(result, "seconds", "total_seconds", "wall_seconds")

    words = _first(result, "num_codewords", "word_vertices", "num_words")
    vertices = _first(result, "num_vertices", "graph_vertices")
    edges = _first(result, "num_incidences", "graph_edges", "num_edges")
    words = _optional_int(words)
    vertices = _optional_int(vertices)
    edges = _optional_int(edges)
    if vertices is None and words is not None:
        vertices = n + words

    actual_method = _first(result, "method", "selected_method", "actual_method",
                           "spanning_set", "fallback_method")
    candidate_order = _first(result, "candidate_group_order", "overgroup_order",
                             "candidate_order")
    orbit_index = _first(result, "orbit_index", "rowspace_orbit", "object_orbit_size")
    status = ("exact_fallback" if fallback else "exact_hit") if exact else (
        "inexact_fallback" if fallback else "inexact" if exact is False else "unclassified")
    return {
        "case": case.name,
        "tags": list(case.tags),
        "note": case.note,
        "n": n,
        "k": int(generator.shape[0]),
        "hull_dim": int(generator.shape[0] -
                        gf2.rank_gf2((generator @ generator.T) % 2)),
        "column_types": len({bytes(generator[:, column])
                             for column in range(generator.shape[1])}),
        "modulus": case.modulus,
        "requested_method": spec.name,
        "actual_method": None if actual_method is None else str(actual_method),
        "source": spec.source,
        "status": status,
        "exact": exact,
        "used_fallback": fallback,
        "fallback_reason": fallback_reason,
        "validation": "pending" if exact else "not-exact",
        "order": None if order is None else str(order),
        "candidate_group_order": None if candidate_order is None else str(candidate_order),
        "orbit_index": _jsonable(orbit_index),
        "num_codewords": words,
        "num_vertices": vertices,
        "num_incidences": edges,
        "preprocessing_seconds": preprocessing,
        "search_seconds": search,
        "stabilizer_seconds": stabilizer,
        "reported_total_seconds": reported_total,
        "diagnostics": _diagnostics(result),
    }


def _classify_exception(error: Exception) -> str:
    name = type(error).__name__.lower()
    text = str(error).lower()
    if isinstance(error, (TimeoutError,)) or "timeout" in name or "timed out" in text:
        return "guarded"
    if ("limit" in name or "guard" in name or "budget" in text or "max_dim" in text or
            "exceeds" in text):
        return "guarded"
    if isinstance(error, (ImportError, NotImplementedError)):
        return "unavailable"
    if "unknown method" in text or "unsupported method" in text or "not available" in text:
        return "unavailable"
    return "error"


def _error_record(spec: MethodSpec, case: BenchmarkCase, generator: np.ndarray,
                  status: str, message: str, error_type: Optional[str] = None) -> dict[str, Any]:
    return {
        "case": case.name,
        "tags": list(case.tags),
        "note": case.note,
        "n": int(generator.shape[1]),
        "k": int(generator.shape[0]),
        "hull_dim": int(generator.shape[0] -
                        gf2.rank_gf2((generator @ generator.T) % 2)),
        "column_types": len({bytes(generator[:, column])
                             for column in range(generator.shape[1])}),
        "modulus": case.modulus,
        "requested_method": spec.name,
        "actual_method": None,
        "source": spec.source,
        "status": status,
        "exact": None,
        "used_fallback": False,
        "fallback_reason": None,
        "validation": status,
        "order": None,
        "candidate_group_order": None,
        "orbit_index": None,
        "num_codewords": None,
        "num_vertices": None,
        "num_incidences": None,
        "preprocessing_seconds": None,
        "search_seconds": None,
        "stabilizer_seconds": None,
        "reported_total_seconds": None,
        "total_seconds": None,
        "unaccounted_seconds": None,
        "speedup": None,
        "diagnostics": None,
        "error_type": error_type,
        "error": message,
    }


def _median_optional(values: list[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def _run_method(spec: MethodSpec, case: BenchmarkCase, generator: np.ndarray,
                limits: Limits, repeats: int) -> tuple[dict[str, Any], Any]:
    if spec.runner is None:
        return (_error_record(spec, case, generator, "unavailable",
                              spec.unavailable_reason or "method unavailable"), None)

    runs = []
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            result = spec.runner(generator, case, limits)
            elapsed = time.perf_counter() - started
            runs.append((elapsed, result, _normalise_result(result, spec, case, generator)))
            if len(runs) > 1 and runs[-1][2]["order"] != runs[0][2]["order"]:
                raise RuntimeError("method returned inconsistent group orders across repetitions")
    except Exception as error:
        return (_error_record(spec, case, generator, _classify_exception(error), str(error),
                              type(error).__name__), None)

    wall = statistics.median(run[0] for run in runs)
    representative = min(runs, key=lambda run: abs(run[0] - wall))
    record = representative[2]
    record["preprocessing_seconds"] = _median_optional(
        [run[2]["preprocessing_seconds"] for run in runs])
    record["search_seconds"] = _median_optional([run[2]["search_seconds"] for run in runs])
    record["stabilizer_seconds"] = _median_optional(
        [run[2]["stabilizer_seconds"] for run in runs])
    record["reported_total_seconds"] = _median_optional(
        [run[2]["reported_total_seconds"] for run in runs])
    record["total_seconds"] = wall
    accounted = sum(value for value in
                    (record["preprocessing_seconds"], record["search_seconds"],
                     record["stabilizer_seconds"])
                    if value is not None)
    record["unaccounted_seconds"] = max(0.0, wall - accounted)
    record["speedup"] = None
    record["error_type"] = None
    record["error"] = None
    record["repeats"] = repeats
    return record, representative[1]


def _validate_exact(record: dict[str, Any], result: Any,
                    baseline_record: dict[str, Any], baseline_result: Any,
                    degree: int) -> bool:
    if record["exact"] is not True:
        return True
    if record["order"] is None or baseline_record["order"] is None:
        record["validation"] = "order-unavailable"
        record["status"] = "invalid_exact"
        return False
    if int(record["order"]) != int(baseline_record["order"]):
        record["validation"] = "order-mismatch"
        record["status"] = "invalid_exact"
        return False
    left = _group_from_result(baseline_result, degree)
    right = _group_from_result(result, degree)
    if left is None or right is None:
        record["validation"] = "group-unavailable"
        record["status"] = "invalid_exact"
        return False
    try:
        same = (all(left.contains(generator) for generator in right.gens()) and
                all(right.contains(generator) for generator in left.gens()))
    except Exception as error:
        record["validation"] = f"containment-error:{type(error).__name__}"
        record["status"] = "invalid_exact"
        return False
    record["validation"] = "equal" if same else "generator-mismatch"
    if not same:
        record["status"] = "invalid_exact"
    return same


# ---------------------------------------------------------------------------
# Benchmark execution and output.
# ---------------------------------------------------------------------------


def run_benchmark(cases: list[BenchmarkCase], methods: list[MethodSpec], limits: Limits,
                  repeats: int, verbose: bool = False) -> tuple[list[dict[str, Any]], int]:
    # Pay native-library loading/build cost before the first timed sample.
    leon.automorphism_group(codes.steane().Hx, max_dim=max(3, limits.max_dim),
                            spanning_set="minweight")
    records = []
    mismatches = 0
    for case in cases:
        generator = gf2.row_basis_gf2(case.generator)
        baseline_spec = next(spec for spec in methods if spec.name == "legacy")
        baseline_record, baseline_result = _run_method(
            baseline_spec, case, generator, limits, repeats)
        if baseline_result is None:
            baseline_record["validation"] = "baseline-failed"
            records.append(baseline_record)
            mismatches += 1
            continue
        baseline_record["validation"] = "baseline"
        baseline_record["status"] = "baseline"
        baseline_record["speedup"] = 1.0
        records.append(baseline_record)
        baseline_seconds = baseline_record["total_seconds"]

        for spec in methods:
            if spec.name == "legacy":
                continue
            record, result = _run_method(spec, case, generator, limits, repeats)
            if record.get("total_seconds") is not None:
                record["speedup"] = (baseline_seconds / record["total_seconds"]
                                     if record["total_seconds"] else math.inf)
            if result is not None and not _validate_exact(
                    record, result, baseline_record, baseline_result, generator.shape[1]):
                mismatches += 1
            records.append(record)
            if verbose and record["status"] in ("error", "guarded", "invalid_exact"):
                print(f"[{case.name}/{spec.name}] {record['status']}: "
                      f"{record.get('error') or record.get('validation')}", file=sys.stderr)
    return records, mismatches


def _milliseconds(value: Optional[float]) -> str:
    return "-" if value is None else f"{1e3 * value:.2f}"


def _integer(value: Any) -> str:
    return "-" if value is None else str(value)


def _print_table(records: list[dict[str, Any]]) -> None:
    print("case                             method               status          val       "
          " words   verts    edges   pre_ms search_ms  stab_ms total_ms speedup")
    print("-" * 149)
    for row in records:
        speedup = "-" if row.get("speedup") is None else f"{row['speedup']:.2f}x"
        print(f"{row['case']:<32} {row['requested_method']:<20} {row['status']:<15} "
              f"{row['validation']:<10} {_integer(row['num_codewords']):>6} "
              f"{_integer(row['num_vertices']):>7} {_integer(row['num_incidences']):>8} "
              f"{_milliseconds(row['preprocessing_seconds']):>8} "
              f"{_milliseconds(row['search_seconds']):>9} "
              f"{_milliseconds(row['stabilizer_seconds']):>8} "
              f"{_milliseconds(row.get('total_seconds')):>8} {speedup:>7}")

    print("\nPer-method summary")
    for method in dict.fromkeys(row["requested_method"] for row in records):
        rows = [row for row in records if row["requested_method"] == method]
        hits = sum(row["status"] in ("baseline", "exact_hit") for row in rows)
        fallbacks = sum(row["status"] == "exact_fallback" for row in rows)
        unavailable = sum(row["status"] == "unavailable" for row in rows)
        guarded = sum(row["status"] == "guarded" for row in rows)
        bad = sum(row["status"] in ("error", "invalid_exact") for row in rows)
        speedups = [row["speedup"] for row in rows if row.get("speedup") is not None]
        median = "-" if not speedups else f"{statistics.median(speedups):.2f}x"
        print(f"  {method:<20} exact hits {hits:>2}/{len(rows):<2}  fallbacks {fallbacks:>2}  "
              f"guarded {guarded:>2}  unavailable {unavailable:>2}  errors {bad:>2}  "
              f"median speedup {median}")


def _write_json(path: str, records: list[dict[str, Any]]) -> None:
    payload = json.dumps(_jsonable(records), indent=2, sort_keys=False)
    if path == "-":
        print(payload)
    else:
        Path(path).write_text(payload + "\n", encoding="utf-8")


def _write_csv(path: str, records: list[dict[str, Any]]) -> None:
    if not records:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(records[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {}
            for key in fields:
                value = record.get(key)
                row[key] = (json.dumps(_jsonable(value), sort_keys=True)
                            if isinstance(value, (dict, list, tuple)) else value)
            writer.writerow(row)


def _print_cases(cases: list[BenchmarkCase]) -> None:
    for case in cases:
        basis = gf2.row_basis_gf2(case.generator)
        hull = basis.shape[0] - gf2.rank_gf2((basis @ basis.T) % 2)
        print(f"{case.name:<34} [{basis.shape[1]},{basis.shape[0]}] hull={hull:<2} "
              f"m={case.modulus:<2} tags={','.join(case.tags)}")


def _print_methods(methods: list[MethodSpec], notices: list[str]) -> None:
    for spec in methods:
        availability = "available" if spec.runner is not None else "unavailable"
        exact = "?" if spec.exact_by_contract is None else str(spec.exact_by_contract).lower()
        print(f"{spec.name:<24} {availability:<11} exact={exact:<5} source={spec.source}")
        if spec.unavailable_reason:
            print(f"  {spec.unavailable_reason}")
    for notice in notices:
        print(f"notice: {notice}", file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1,
                        help="timed repetitions per available case/method (median reported)")
    parser.add_argument("--random-per-shape", type=int, default=1,
                        help="deterministic random matrices per [n,k] control shape")
    parser.add_argument("--methods", default=None,
                        help="comma-separated method names; omit or use 'all' for the full "
                             "suite, or use 'auto' for the portfolio auto dispatcher "
                             "(legacy is always added)")
    parser.add_argument("--cases", default="",
                        help="comma-separated shell globs selecting case names")
    parser.add_argument("--tags", default="",
                        help="comma-separated tags; retain cases matching at least one")
    parser.add_argument("--exclude-tags", default="",
                        help="comma-separated case tags to omit")
    parser.add_argument("--max-dim", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=200_000)
    parser.add_argument("--max-enumerated", type=int, default=1 << 18,
                        help="portfolio codeword-enumeration guard")
    parser.add_argument("--max-geometry-rank", type=int, default=6,
                        help="largest projective rank attempted by geometry methods")
    parser.add_argument("--max-geometry-candidates", type=int, default=100_000,
                        help="projective/component canonicalization candidate guard")
    parser.add_argument("--max-moment-triples", type=int, default=20_000,
                        help="triple-moment escalation guard")
    parser.add_argument("--max-stabilizer-orbit", type=int, default=5_000,
                        help="largest row-space orbit exactified before exact fallback")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="per-method timeout hint; 0 disables the hint")
    parser.add_argument("--budget", type=int, default=60_000_000,
                        help="generic invariant/graph work-budget hint")
    parser.add_argument("--no-fallback", action="store_true",
                        help="ask portfolio methods to expose guards instead of falling back")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--json", nargs="?", const="-", metavar="PATH",
                        help="write JSON to PATH, or stdout when PATH is omitted")
    parser.add_argument("--csv", metavar="PATH", help="write flat CSV records to PATH")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="return success even if a result declared exact fails validation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.random_per_shape < 0:
        parser.error("--repeats must be positive and --random-per-shape nonnegative")
    integer_limits = (args.max_dim, args.max_words, args.max_enumerated,
                      args.max_geometry_rank, args.max_geometry_candidates,
                      args.max_moment_triples, args.max_stabilizer_orbit, args.budget)
    if min(integer_limits) < 1 or args.timeout < 0:
        parser.error("resource limits must be positive (timeout may be zero)")

    cases = _select_cases(_corpus(args.random_per_shape), args.cases, args.tags,
                          args.exclude_tags)
    methods, notices = _choose_methods(args.methods)
    if args.list_cases:
        _print_cases(cases)
        return 0
    if args.list_methods:
        _print_methods(methods, notices)
        return 0
    if not cases:
        parser.error("case filters selected no benchmark cases")

    limits = Limits(args.max_dim, args.max_words, args.max_enumerated,
                    args.max_geometry_rank, args.max_geometry_candidates,
                    args.max_moment_triples, args.max_stabilizer_orbit,
                    None if args.timeout == 0 else args.timeout,
                    args.budget, args.no_fallback)
    records, mismatches = run_benchmark(cases, methods, limits, args.repeats, args.verbose)
    if args.json != "-":
        _print_table(records)
    if args.json:
        _write_json(args.json, records)
    if args.csv:
        _write_csv(args.csv, records)
    for notice in notices:
        print(f"notice: {notice}", file=sys.stderr)
    return 0 if args.allow_mismatch or mismatches == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
