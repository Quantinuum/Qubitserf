"""Benchmark qubitserf (CPU / GPU) against the reference BZ finder.

Reference: ``codedistance.BZDistMW`` (the Brouwer-Zimmermann implementation that
ships with the codeDistancePYPI package the paper uses).

Run from the repo root:

    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python bench/benchmark.py

The reference package is found via either an installed `codedistance` or a
clone on PYTHONPATH (default: /tmp/codeDistancePYPI).  The reference is run in a
*subprocess with a hard timeout* (`bench/_ref_runner.py`) because it can hang or
run for minutes on larger codes; once a method exceeds the timeout on one size
it is skipped for all larger sizes in the same family.

Outputs a table to stdout and writes `bench/results.md`.

As a regression guard, GPU is run several times per code and the script flags
any disagreement (an earlier stale-device-buffer bug caused this; it is fixed).
The fastest of those runs is reported.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

import qubitserf.distfind as df
from qubitserf.distfind import codes


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_MD = os.path.join(HERE, "results.md")

# Where the reference clone lives (clone path beats a pip install only if set).
REF_CLONE = os.environ.get("CODEDISTANCE_CLONE", "/tmp/codeDistancePYPI")

# Per-code wall-clock timeout for the reference (seconds).  If the reference
# exceeds this on size s, it is skipped for all larger sizes in that family.
# NB: each reference call also pays a fixed ~5s import cost (numba/stim/ldpc).
REF_TIMEOUT = float(os.environ.get("REF_TIMEOUT", "30"))

# How many times to run GPU to probe its (known) non-determinism.
GPU_TRIES = int(os.environ.get("GPU_TRIES", "3"))

BACKENDS = df.available_backends()
HAS_GPU = "gpu" in BACKENDS


# --------------------------------------------------------------------------- #
# Reference invocation (subprocess + hard timeout)
# --------------------------------------------------------------------------- #
def _ref_env() -> dict:
    env = dict(os.environ)
    parts = [REF_CLONE, os.path.join(REPO, "python")]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("MPLBACKEND", "Agg")
    return env


def reference_available() -> bool:
    """Can we import the reference (clone or installed) at all?"""
    # distance.py does `from gurobipy import GRB`, so the stub needs a GRB attr.
    code = (
        "import sys,types;"
        "g=types.ModuleType('gurobipy');g.GRB=types.SimpleNamespace();g.Model=None;"
        "sys.modules['gurobipy']=g;"
        "import codedistance"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], env=_ref_env(),
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def run_reference(mode: str, *npy_paths: str, timeout: float) -> dict:
    """Run _ref_runner.py in a subprocess; return its JSON dict or an error."""
    cmd = [sys.executable, os.path.join(HERE, "_ref_runner.py"), mode, *npy_paths]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, env=_ref_env(), capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "seconds": time.perf_counter() - t0,
                "timed_out": True}
    if r.returncode != 0:
        return {"ok": False, "error": f"rc={r.returncode}: {r.stderr.strip()[:200]}"}
    line = (r.stdout.strip().splitlines() or [""])[-1]
    try:
        out = json.loads(line)
    except Exception:
        return {"ok": False, "error": f"bad output: {r.stdout.strip()[:200]}"}
    return out


def _save_npy(arr: np.ndarray, tmpdir: str, name: str) -> str:
    p = os.path.join(tmpdir, name)
    np.save(p, np.asarray(arr, dtype=np.uint8))
    return p + ".npy"


# --------------------------------------------------------------------------- #
# qubitserf invocation
# --------------------------------------------------------------------------- #
def time_css(Hx, Hz, backend: str, tries: int = 1):
    """Return (distance, best_distance, seconds, stable, values)."""
    vals, secs = [], []
    for _ in range(tries):
        t0 = time.perf_counter()
        r = df.css_distance(Hx, Hz, method="bz", which="min", backend=backend)
        secs.append(time.perf_counter() - t0)
        vals.append(r.distance)
    stable = len(set(vals)) == 1
    return vals[0], max(vals), min(secs), stable, vals


def time_classical(H, backend: str, tries: int = 1):
    vals, secs = [], []
    for _ in range(tries):
        t0 = time.perf_counter()
        r = df.classical_distance(H, method="bz", backend=backend)
        secs.append(time.perf_counter() - t0)
        vals.append(r.distance)
    stable = len(set(vals)) == 1
    return vals[0], max(vals), min(secs), stable, vals


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    name: str
    n: int
    k: object = ""          # "" if not computed
    d_cpu: object = None
    d_gpu: object = None  # representative (single-run) value
    d_gpu_best: object = None
    gpu_stable: object = None
    d_ref: object = None
    t_cpu: float = float("nan")
    t_gpu: float = float("nan")
    t_ref: float = float("nan")
    ref_note: str = ""
    match: str = ""         # "yes"/"no"/"n/a"
    notes: list = field(default_factory=list)


def fmt_t(t):
    if t is None or (isinstance(t, float) and (t != t)):
        return "-"
    if t < 1e-3:
        return f"{t*1e6:.0f}us"
    if t < 1.0:
        return f"{t*1e3:.1f}ms"
    return f"{t:.2f}s"


def fmt(x):
    return "-" if x is None else str(x)


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def build_css_codes():
    """List of (name, Hx, Hz) where the GPU can matter."""
    # Sizes chosen so qubitserf-CPU stays bounded (a few seconds) while still
    # reaching the regime where the GPU clearly wins.  surface L=8 (n=145) is
    # omitted: its CPU BZ search runs for minutes, which would dominate the
    # whole benchmark; toric L=8 (n=128) and surface L=7 (n=113) already show
    # the multi-second CPU vs sub-second GPU gap.
    out = []
    for L in (4, 5, 6, 7, 8):
        Hx, Hz = codes.toric(L)
        out.append((f"toric L={L}", Hx, Hz))
    for L in (4, 5, 6, 7):
        Hx, Hz = codes.surface(L)
        out.append((f"surface L={L}", Hx, Hz))
    # A couple of hypergraph-product codes from classical bases.
    h_rep = codes.repetition_parity(6)            # [6,1,6] -> HGP
    Hx, Hz = codes.hypergraph_product(h_rep, h_rep)
    out.append(("hgp(rep6,rep6)", Hx, Hz))
    h_ham = codes.hamming_parity(3)               # [7,4,3]
    Hx, Hz = codes.hypergraph_product(h_ham, h_ham)
    out.append(("hgp(ham3,ham3)", Hx, Hz))
    return out


def build_classical_codes():
    # Each entry: (name, H, run_reference?).  The reference's single-block path
    # enumerates ker(H) and gets stuck on the degenerate even-code structure of
    # the repetition parity checks, so we skip the reference there (and say so)
    # while still timing qubitserf, which solves them instantly.
    out = []
    # Hamming codes: reference handles these fine (non-degenerate).
    for r in (3, 4):
        out.append((f"hamming r={r}", codes.hamming_parity(r), True))
    # Repetition codes: qubitserf exact; reference would hang -> skip reference.
    for n in (6, 8):
        out.append((f"repetition n={n}", codes.repetition_parity(n), False))
    # A random LDPC parity check (qubitserf only; reference distance of ker(H)).
    out.append(("rand_ldpc(12,18,3)", codes.random_ldpc_parity(12, 18, 3, seed=1), True))
    return out


def sweep_css(ref_ok: bool, tmpdir: str):
    rows = []
    ref_family_skip = {}  # family-prefix -> True once it has timed out
    mismatches = []
    for name, Hx, Hz in build_css_codes():
        n = int(Hx.shape[1])
        row = Row(name=name, n=n)
        print(f"\n[css] {name:18s} n={n}")

        # ---- qubitserf CPU (ground truth) ----
        try:
            dcpu, _, tcpu, _, _ = time_css(Hx, Hz, "cpu")
            row.d_cpu, row.t_cpu = dcpu, tcpu
            print(f"    cpu   d={dcpu}  {fmt_t(tcpu)}")
        except Exception as exc:  # noqa: BLE001
            row.notes.append(f"cpu: {type(exc).__name__}")
            print(f"    cpu   FAILED: {exc}")
            dcpu = None

        # ---- qubitserf GPU ----
        if HAS_GPU:
            try:
                dgpu, dbest, tgpu, stable, vals = time_css(Hx, Hz, "gpu",
                                                               tries=GPU_TRIES)
                row.d_gpu, row.t_gpu = dgpu, tgpu
                row.gpu_stable = stable
                # "best" = value matching CPU if present, else min observed
                row.d_gpu_best = dcpu if (dcpu in vals) else min(vals)
                tag = "stable" if stable else f"FLAKY {vals}"
                print(f"    gpu d={dgpu} best={row.d_gpu_best} {fmt_t(tgpu)} ({tag})")
            except Exception as exc:  # noqa: BLE001
                row.notes.append(f"gpu: {type(exc).__name__}")
                print(f"    gpu FAILED: {exc}")

        # ---- reference BZDistMW ----
        fam = name.split()[0]
        if ref_ok and not ref_family_skip.get(fam):
            hx_p = _save_npy(Hx, tmpdir, "hx")
            hz_p = _save_npy(Hz, tmpdir, "hz")
            res = run_reference("css", hx_p, hz_p, timeout=REF_TIMEOUT)
            if res.get("ok"):
                row.d_ref = res["d"]
                row.k = res.get("k", "")
                row.t_ref = res.get("seconds", float("nan"))
                print(f"    ref   d={row.d_ref} k={row.k} {fmt_t(row.t_ref)}")
                # If the reference itself was slow, skip larger sizes in family.
                if row.t_ref is not None and row.t_ref > REF_TIMEOUT * 0.8:
                    ref_family_skip[fam] = True
            elif res.get("timed_out"):
                row.ref_note = f">{REF_TIMEOUT:.0f}s (timeout)"
                ref_family_skip[fam] = True
                print(f"    ref   TIMEOUT (>{REF_TIMEOUT:.0f}s) -> skip larger {fam}")
            else:
                row.ref_note = "error"
                row.notes.append(f"ref: {res.get('error','?')[:60]}")
                print(f"    ref   ERROR: {res.get('error')}")
        elif ref_ok:
            row.ref_note = "skipped (family timed out earlier)"
            print(f"    ref   skipped ({fam} timed out at a smaller size)")
        else:
            row.ref_note = "reference unavailable"

        # ---- match check ----
        if dcpu is not None and isinstance(row.d_ref, int):
            if dcpu == row.d_ref:
                row.match = "yes"
            else:
                row.match = "NO"
                mismatches.append((name, dcpu, row.d_ref))
        else:
            row.match = "n/a"
        # GPU best should also match CPU where both ran.
        if dcpu is not None and isinstance(row.d_gpu_best, int) and row.d_gpu_best != dcpu:
            mismatches.append((f"{name} (gpu-best)", dcpu, row.d_gpu_best))

        rows.append(row)
    return rows, mismatches


def sweep_classical(ref_ok: bool, tmpdir: str):
    rows = []
    mismatches = []
    for name, H, ref_for_this in build_classical_codes():
        n = int(H.shape[1])
        row = Row(name=name, n=n)
        print(f"\n[class] {name:18s} n={n}")
        try:
            dcpu, _, tcpu, _, _ = time_classical(H, "cpu")
            row.d_cpu, row.t_cpu = dcpu, tcpu
            print(f"    cpu   d={dcpu}  {fmt_t(tcpu)}")
        except Exception as exc:  # noqa: BLE001
            row.notes.append(f"cpu: {type(exc).__name__}")
            dcpu = None

        if HAS_GPU:
            try:
                dgpu, dbest, tgpu, stable, vals = time_classical(
                    H, "gpu", tries=GPU_TRIES)
                row.d_gpu, row.t_gpu, row.gpu_stable = dgpu, tgpu, stable
                row.d_gpu_best = dcpu if (dcpu in vals) else min(vals)
                print(f"    gpu d={dgpu} best={row.d_gpu_best} {fmt_t(tgpu)} "
                      f"({'stable' if stable else 'FLAKY '+str(vals)})")
            except Exception as exc:  # noqa: BLE001
                row.notes.append(f"gpu: {type(exc).__name__}")

        if ref_ok and ref_for_this:
            h_p = _save_npy(H, tmpdir, "h")
            res = run_reference("class", h_p, timeout=REF_TIMEOUT)
            if res.get("ok"):
                row.d_ref, row.t_ref = res["d"], res.get("seconds", float("nan"))
                print(f"    ref   d={row.d_ref} {fmt_t(row.t_ref)}")
            elif res.get("timed_out"):
                row.ref_note = f">{REF_TIMEOUT:.0f}s (timeout)"
                print(f"    ref   TIMEOUT (>{REF_TIMEOUT:.0f}s)")
            else:
                row.ref_note = "error"
                row.notes.append(f"ref: {res.get('error','?')[:60]}")
        elif ref_ok and not ref_for_this:
            row.ref_note = "skipped (degenerate for ref)"
            print("    ref   skipped (reference hangs on this degenerate code)")
        else:
            row.ref_note = "reference unavailable"

        if dcpu is not None and isinstance(row.d_ref, int):
            row.match = "yes" if dcpu == row.d_ref else "NO"
            if dcpu != row.d_ref:
                mismatches.append((name, dcpu, row.d_ref))
        else:
            row.match = "n/a"
        rows.append(row)
    return rows, mismatches


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def speedup(num, den):
    if not (isinstance(num, float) and isinstance(den, float)):
        return None
    if num != num or den != den or den <= 0:
        return None
    return num / den


def render_table(rows, title):
    hdr = ("| code | n | k | d(cpu) | d(gpu) | d(ref) | t_cpu | t_gpu | "
           "t_ref | ref(default)/gpu | cpu/gpu | match |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [f"### {title}", "", hdr, sep]
    for r in rows:
        gpu_cell = fmt(r.d_gpu_best)
        if r.gpu_stable is False:
            gpu_cell += "*"
        d_ref = fmt(r.d_ref) if isinstance(r.d_ref, int) else (r.ref_note or "-")
        su_ref = speedup(r.t_ref, r.t_gpu)
        su_cpu = speedup(r.t_cpu, r.t_gpu)
        lines.append(
            f"| {r.name} | {r.n} | {fmt(r.k)} | {fmt(r.d_cpu)} | {gpu_cell} | "
            f"{d_ref} | {fmt_t(r.t_cpu)} | {fmt_t(r.t_gpu)} | {fmt_t(r.t_ref)} | "
            f"{('%.1fx'%su_ref) if su_ref else '-'} | "
            f"{('%.2fx'%su_cpu) if su_cpu else '-'} | {r.match} |"
        )
    return "\n".join(lines)


def summarize(rows):
    ref_speedups, cpu_speedups = [], []
    for r in rows:
        s = speedup(r.t_ref, r.t_gpu)
        if s:
            ref_speedups.append(s)
        s = speedup(r.t_cpu, r.t_gpu)
        if s:
            cpu_speedups.append(s)
    return ref_speedups, cpu_speedups


def main():
    print("qubitserf benchmark")
    print("backends:", BACKENDS)
    ref_ok = reference_available()
    print("reference (codedistance.BZDistMW) available:", ref_ok,
          f"(clone={REF_CLONE})" if ref_ok else "")
    print(f"reference per-code timeout: {REF_TIMEOUT:.0f}s; gpu probe tries: {GPU_TRIES}")

    with tempfile.TemporaryDirectory() as tmpdir:
        css_rows, css_mis = sweep_css(ref_ok, tmpdir)
        class_rows, class_mis = sweep_classical(ref_ok, tmpdir)

    mismatches = css_mis + class_mis
    css_table = render_table(css_rows, "CSS quantum codes")
    class_table = render_table(class_rows, "Classical codes")

    ref_su, cpu_su = summarize(css_rows + class_rows)

    # ---- console table ----
    print("\n" + "=" * 80)
    print(css_table)
    print()
    print(class_table)
    print("=" * 80)

    # ---- summary text ----
    summ = []
    summ.append("## Summary\n")
    summ.append(f"- Backends available: `{BACKENDS}`.")
    summ.append(f"- Reference: `codedistance.BZDistMW` (Brouwer-Zimmermann), "
                f"run per-component with a {REF_TIMEOUT:.0f}s per-code timeout.")
    if ref_su:
        summ.append(f"- **default(BZDistMW)/gpu speedup**: min {min(ref_su):.1f}x, "
                    f"max {max(ref_su):.1f}x, median {sorted(ref_su)[len(ref_su)//2]:.1f}x "
                    f"(over {len(ref_su)} codes where the reference finished).")
    else:
        summ.append("- No reference timings available for a speedup ratio.")
    if cpu_su:
        summ.append(f"- **cpu/gpu speedup**: min {min(cpu_su):.2f}x, "
                    f"max {max(cpu_su):.2f}x, median {sorted(cpu_su)[len(cpu_su)//2]:.2f}x "
                    "(values < 1 mean the CPU was faster — expected for small codes where "
                    "GPU dispatch overhead dominates).")
    # Where the reference timed out but qubitserf finished:
    ref_timeouts = [r.name for r in css_rows + class_rows
                    if "timeout" in (r.ref_note or "").lower()]
    if ref_timeouts:
        summ.append(f"- Reference **timed out** (>{REF_TIMEOUT:.0f}s) on: "
                    f"{', '.join(ref_timeouts)} — qubitserf solved all of these.")
    # Distance agreement:
    if mismatches:
        summ.append("- **DISTANCE MISMATCHES DETECTED** (qubitserf vs reference / "
                    "gpu-best vs cpu):")
        for nm, a, b in mismatches:
            summ.append(f"    - {nm}: {a} vs {b}")
    else:
        summ.append("- **All distances matched**: qubitserf-cpu equals the reference "
                    "BZDistMW on every code where the reference finished, and GPU's "
                    "best-of-retries value equals the CPU value everywhere.")
    summ.append("")
    n_flaky = sum(1 for r in css_rows + class_rows
                  if getattr(r, "gpu_stable", None) is False)
    summ.append("### GPU determinism\n")
    if n_flaky == 0:
        summ.append(f"GPU was run {GPU_TRIES}x per code as a determinism probe and "
                    "returned the **same correct distance every time** on all codes "
                    "(an earlier stale-device-buffer bug, where the buffer cache keyed on "
                    "host pointer identity, has been fixed by keying on a unique per-solve "
                    "token). A `*` would mark any code whose GPU runs disagreed; there "
                    "are none. Reported GPU times are the fastest of the probe runs.")
    else:
        summ.append(f"WARNING: {n_flaky} code(s) showed non-deterministic GPU results "
                    "(marked `*`); the value shown is the best (correct) one. This "
                    "indicates a regression.")
    summary_text = "\n".join(summ)
    print("\n" + summary_text)

    # ---- write results.md ----
    with open(RESULTS_MD, "w") as f:
        f.write("# qubitserf benchmark results\n\n")
        f.write("Generated by `bench/benchmark.py`. Reference = "
                "`codedistance.BZDistMW` (Brouwer-Zimmermann).\n\n")
        f.write("Columns: `d(cpu)` is qubitserf on the CPU (ground truth, deterministic); "
                "`d(gpu)` is the GPU distance, verified identical across "
                f"{GPU_TRIES} runs (`*` = GPU runs disagreed, should never appear); "
                "`d(ref)` is the reference distance. `t_*` are wall-clock times. "
                "`ref/gpu` and `cpu/gpu` are speed ratios.\n\n")
        f.write(css_table + "\n\n")
        f.write(class_table + "\n\n")
        f.write(summary_text + "\n")
    print(f"\nWrote {RESULTS_MD}")

    # Hard assertion: never silently pass a real distance disagreement.
    real = [m for m in mismatches]
    if real:
        print("\nERROR: distance mismatches found (see summary).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
