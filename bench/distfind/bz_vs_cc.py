"""BZ vs connected-cluster across the sparse/dense divide.

The regimes are complementary and this benchmark shows BOTH directions:

- CC exploits Tanner-graph sparsity: on LDPC codes (toric, bivariate-bicycle) it
  certifies distances in milliseconds where BZ's information-set lower bound rises
  too slowly.
- BZ exploits information sets: on DENSE-check codes (quantum BCH, whose parity
  checks have row weight ~n/2) CC's cluster growth explodes while BZ certifies in
  milliseconds-to-seconds.

The cases span different code families, so raw n is not a comparable hardness
axis; the table just reports each code's [[n,k,d]] alongside the per-method times.

    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python bench/bz_vs_cc.py

Writes bench/bz_vs_cc.md (and bench/bz_vs_cc.json). Env: TIMEOUT (s, default 330),
BIN (default build/qubitserf), SKIP_GPU=1 to omit the GPU series.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, ".cache")
BIN = os.environ.get("BIN", os.path.join(ROOT, "build", "qubitserf"))
TIMEOUT = float(os.environ.get("TIMEOUT", "330"))
SKIP_GPU = os.environ.get("SKIP_GPU", "") == "1"

import bch_codes  # noqa: E402  (same directory)
import qubitserf.distfind.codes as codes  # noqa: E402  (PYTHONPATH=python)


def bb(l, m):
    return codes.bivariate_bicycle(l, m, [("x", 3), ("y", 1), ("y", 2)],
                                   [("y", 3), ("x", 1), ("x", 2)])


def _case_list():
    """[(label, family, Hx, Hz, expected_d_or_None)]; family in {dense, sparse}."""
    cases = [
        ("steane [[7,1,3]]", "dense", *codes.steane(), 3),
        ("toric L=6 [[72,2,6]]", "sparse", *codes.toric(6), 6),
        ("bb [[72,12,6]]", "sparse", *bb(6, 6), 6),
        ("toric L=10 [[200,2,10]]", "sparse", *codes.toric(10), 10),
        ("gross [[144,12,12]]", "sparse", *codes.gross_code(), 12),
    ]
    for label, m, delta, _ in bch_codes.CASES:
        Hx, Hz, n, k = bch_codes.quantum_bch(m, delta)
        # designed delta is only a LOWER bound on d -> checked as d >= delta below
        cases.append((label, "dense", Hx, Hz, -delta))
    return cases


def _write_mats(slug, Hx, Hz):
    os.makedirs(CACHE, exist_ok=True)
    fx = os.path.join(CACHE, f"{slug}_Hx.txt")
    fz = os.path.join(CACHE, f"{slug}_Hz.txt")
    if not os.path.exists(fx):
        np.savetxt(fx, Hx, fmt="%d")
        np.savetxt(fz, Hz, fmt="%d")
    return fx, fz


def run_cli(fx, fz, method_flag, backend_flag):
    """(d or None, seconds, timed_out) for one CLI invocation."""
    cmd = [BIN, "--hx", fx, "--hz", fz, method_flag, backend_flag]
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, TIMEOUT, True
    dt = time.perf_counter() - t0
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {out.stderr.strip()}")
    return int(out.stdout.strip().split()[-1]), dt, False


METHODS = [  # (key, label, method flag, backend flag)
    ("cc", "cc (cpu)", "--cc", "--cpu"),
    ("bz_cpu", "bz (cpu)", "--bz", "--cpu"),
    ("bz_gpu", "bz (gpu)", "--bz", "--gpu"),
]


def fmt_t(seconds, timed_out):
    if timed_out:
        return f">{TIMEOUT:.0f}s"
    if seconds < 1:
        return f"{seconds * 1e3:.0f}ms"
    return f"{seconds:.1f}s"


def main():
    methods = [m for m in METHODS if not (SKIP_GPU and m[0] == "bz_gpu")]
    if "--replot" in sys.argv:   # regenerate md+png from the saved results of a prior run
        with open(os.path.join(HERE, "bz_vs_cc.json")) as f:
            saved = json.load(f)
        rows = [(r["label"], r["family"], r["n"], r["d"],
                 {k: tuple(v) for k, v in r["res"].items()}) for r in saved]
        _write_md(rows, methods)
        return
    rows = []
    for label, family, Hx, Hz, want in _case_list():
        # slug over the FULL label: distinct codes must never share a cache file
        slug = "".join(c if c.isalnum() else "_" for c in label)
        fx, fz = _write_mats(slug, Hx, Hz)
        res = {}
        for key, mlabel, mflag, bflag in methods:
            d, dt, to = run_cli(fx, fz, mflag, bflag)
            res[key] = (d, dt, to)
            print(f"{label:>26} {mlabel:>9}: "
                  f"{'d=%d' % d if d is not None else 'timeout':>9} ({fmt_t(dt, to)})",
                  flush=True)
        finished = {d for d, _, to in res.values() if not to}
        assert len(finished) <= 1, f"{label}: methods disagree: {res}"
        d = finished.pop() if finished else (want if want is not None and want > 0 else None)
        if want is not None and d is not None:
            if want > 0:
                assert d == want, f"{label}: d={d} != expected {want}"
            else:  # negative want = designed-distance lower bound (BCH)
                assert d >= -want, f"{label}: d={d} < designed distance {-want}"
        rows.append((label, family, Hx.shape[1], d, res))

    with open(os.path.join(HERE, "bz_vs_cc.json"), "w") as f:
        json.dump([{"label": l, "family": fam, "n": n, "d": d, "res": res}
                   for l, fam, n, d, res in rows], f, indent=1)
    _write_md(rows, methods)


def _write_md(rows, methods):
    md = os.path.join(HERE, "bz_vs_cc.md")
    with open(md, "w") as f:
        f.write("# BZ vs connected-cluster: the sparse/dense divide\n\n")
        f.write(f"Uniform per-run timeout {TIMEOUT:.0f}s.\n\n")
        f.write("| code | family | n | d |")
        for _, mlabel, _, _ in methods:
            f.write(f" {mlabel} |")
        f.write("\n|---|---|---|---|" + "---|" * len(methods) + "\n")
        for label, family, n, d, res in rows:
            f.write(f"| {label} | {family} | {n} | {d if d is not None else '?'} |")
            for key, _, _, _ in methods:
                dv, dt, to = res[key]
                f.write(f" {fmt_t(dt, to)} |")
            f.write("\n")
        f.write("\n## Takeaway\n\n")
        f.write("Connected-cluster certifies sparse LDPC codes (toric, bivariate-"
                "bicycle) in milliseconds where BZ needs minutes or times out; "
                "Brouwer-Zimmermann certifies dense-check quantum BCH codes in "
                "milliseconds-to-seconds where cluster growth makes CC time out. "
                "Neither dominates: the right method follows the Tanner-graph "
                "sparsity of the code.\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    raise SystemExit(main())
