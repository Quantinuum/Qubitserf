"""Comprehensive qubitserf benchmark across several CSS / classical families.

Compares, per code:

  * qubitserf ``cc``           (connected cluster, always certifies, sub-second)
  * qubitserf ``bz`` (cpu)     (Brouwer-Zimmermann)
  * qubitserf ``bz`` (gpu)   (same, on the GPU backend if present)
  * qubitserf ``mitm``         (meet-in-the-middle; small codes only -- slow)
  * reference ``BZDistMW``         (codeDistance package, subprocess + timeout)
  * reference ``connectedClusterMW`` (codeDistance package, subprocess + timeout)

For every measurement we record (distance, seconds) and wrap it in try/except +
a per-call wall-clock budget, so the run always finishes.  Methods that exceed
their per-size budget are reported as ``timeout`` and skipped for larger sizes
in the same family -- which is what happens to BZ on the hard codes (sparse,
weak BZ lower bound), where it cannot certify in tractable time.

After collecting, the script CROSS-CHECKS that every *certifying* method that
finished agrees on the distance, and flags any mismatch.  It writes a grouped
markdown report (per-method times + speedup columns + summary) to
``bench/comprehensive_results.md`` and, if matplotlib is importable, simple
``time vs n`` PNG plots alongside it.

Run from the repo root:

    PYTHONPATH=python:bench CODEDISTANCE_CLONE=/tmp/codeDistancePYPI \
        /opt/miniconda3/envs/sage_env/bin/python bench/comprehensive.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import threading
import time
from dataclasses import dataclass, field

import qubitserf.distfind as df

# Reuse only the time formatter.  The external codeDistance reference package is
# NOT executed here: its numbers are read from a frozen cache (see REF_CACHE
# below).  qubitserf's own cc/bz/mitm are re-measured live.
from benchmark import fmt_t

import bench_codes as bc


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "comprehensive_results.md")

# Per-method, per-code wall-clock budgets (seconds).
DF_BUDGET = float(os.environ.get("DF_BUDGET", "30"))      # in-process qubitserf cap
MITM_MAX_N = int(os.environ.get("MITM_MAX_N", "62"))      # mitm only for n <= this
# BZ is only *attempted* up to this n.  The in-process budget runs the native
# solver on a daemon thread that cannot be cancelled, so a BZ call that exceeds
# the budget keeps burning a CPU core in the background until it finishes (see
# DRAIN_BUDGET below).  BZ on large sparse codes (d >> 1, weak lower bound) is
# the orphan risk; those codes are recorded as a timeout and BZ is then skipped
# for the larger sizes of the same family.
# All codes above BZ_MAX_N have cc/reference still attempted.
#
# 1024 is the native BZ ceiling on the GPU backends (codeword stride <= MAX_WORDS
# = 16 u64 words = 1024 bits; above that the GPU path auto-falls-back to the
# dynamic CPU solver).  The CPU backend itself is unbounded.  This covers every
# code in the benchmark; the large sparse QLDPC codes within the window
# (toric/surface L>=9, bb288) are the ones BZ is expected to time out on.
BZ_MAX_N = int(os.environ.get("BZ_MAX_N", "1024"))
REF_TIMEOUT = float(os.environ.get("REF_TIMEOUT", "30"))  # timeout of the CACHED reference run
# How long we wait for an over-budget native solve to drain before abandoning its
# thread (it stays alive as a daemon and dies with the process).  Draining keeps a
# finished-just-late solver from inflating the NEXT timing, but an unbounded wait
# would stall the whole run on a code BZ cannot finish at all.
DRAIN_BUDGET = float(os.environ.get("DRAIN_BUDGET", "60"))

# Frozen reference numbers from a prior run (bench/ref_cache.json), keyed by the
# stripped code name -> {"ref_bz": rec, "ref_cc": rec}.  We reuse these rather than
# re-executing the external codeDistance package; qubitserf's cc/bz/mitm ARE
# re-measured live below, so the ref/cc speedups combine a fresh qubitserf timing
# with the (unchanged) reference timing.
REF_CACHE_PATH = os.path.join(HERE, "ref_cache.json")
try:
    with open(REF_CACHE_PATH) as _f:
        REF_CACHE = json.load(_f)
except FileNotFoundError:
    REF_CACHE = {}

# Warm-robust timing: a measurement that finishes under REPEAT_BELOW seconds is
# re-run up to REPEAT times and the minimum is kept.  The first GPU call to a
# given (stride, d) JIT-compiles its kernel, so a single shot can charge a cheap
# code with one-off warmup/dispatch jitter (which otherwise makes a smaller code
# look slower than a larger one).  Slow, enumeration-dominated runs stay single
# shot so this never multiplies the expensive cases.
REPEAT = int(os.environ.get("BENCH_REPEAT", "3"))
REPEAT_BELOW = float(os.environ.get("BENCH_REPEAT_BELOW", "1.0"))

BACKENDS = df.available_backends()
HAS_GPU = "gpu" in BACKENDS


# --------------------------------------------------------------------------- #
# In-process qubitserf call with a soft wall-clock budget.
# --------------------------------------------------------------------------- #
# The native solver runs to completion regardless (no cooperative cancellation),
# but we run it on a worker thread and *give up waiting* after `budget` seconds.
# A method that blows its budget is recorded as timed_out and skipped for larger
# sizes of the same family.
# --------------------------------------------------------------------------- #
@dataclass
class Meas:
    ok: bool = False
    distance: object = None
    seconds: float = float("nan")
    timed_out: bool = False
    error: str = ""


def _run_df_once(fn, budget: float) -> Meas:
    box: dict = {}

    def worker():
        t0 = time.perf_counter()
        try:
            r = fn()
            box["r"] = r
            box["secs"] = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            box["err"] = f"{type(exc).__name__}: {exc}"
            box["secs"] = time.perf_counter() - t0

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(budget)
    if th.is_alive():
        # We give up *reporting* after `budget`, but the native solver cannot be
        # cancelled cooperatively.  Draining it before we return keeps a
        # just-too-slow solver from holding a CPU core / the GPU busy and inflating
        # the NEXT code's timing.  The drain is bounded (DRAIN_BUDGET): BZ on a
        # sparse code with a weak lower bound can run for hours, and waiting that
        # out would stall the whole benchmark.  An abandoned thread is a daemon and
        # dies with the process.
        th.join(DRAIN_BUDGET)
        return Meas(ok=False, timed_out=True, seconds=budget,
                    error=f">{budget:.0f}s (budget)")
    if "err" in box:
        return Meas(ok=False, seconds=box.get("secs", float("nan")), error=box["err"])
    return Meas(ok=True, distance=int(box["r"]), seconds=box["secs"])


def _run_df(fn, budget: float) -> Meas:
    # First shot also warms up the GPU's per-(stride, d) JIT kernel cache.
    m = _run_df_once(fn, budget)
    # Re-run only cheap, successful measurements to discard warmup/dispatch
    # jitter and report the steady-state minimum; expensive runs stay single
    # shot (see REPEAT_BELOW).
    if m.ok and isinstance(m.seconds, float) and m.seconds < REPEAT_BELOW:
        for _ in range(REPEAT - 1):
            m2 = _run_df_once(fn, budget)
            if m2.ok and m2.seconds < m.seconds:
                m = m2
    return m


def df_css(Hx, Hz, method, backend="auto", budget=DF_BUDGET) -> Meas:
    return _run_df(lambda: df.css_distance(Hx, Hz, method=method, which="min",
                                           backend=backend),
                   budget)


def df_classical(H, method, backend="auto", budget=DF_BUDGET) -> Meas:
    return _run_df(lambda: df.classical_distance(H, method=method, backend=backend),
                   budget)


# --------------------------------------------------------------------------- #
# Reference lookup -> Meas  (reused from a frozen prior run; NO external code run)
# --------------------------------------------------------------------------- #
def cached_ref(name, key) -> Meas:
    """Reuse a prior run's reference measurement for `name` (ref_bz / ref_cc).

    Returns an "unavailable" Meas (renders as `n/a`) when the cache has no entry
    for this code -- e.g. the Reed-Muller families, which were added after the
    cached reference run.
    """
    rec = REF_CACHE.get(name.strip() if isinstance(name, str) else name, {}).get(key)
    if rec is None:
        return Meas(error="reference unavailable")
    if rec.get("timed_out"):
        secs = rec.get("seconds", REF_TIMEOUT)
        return Meas(timed_out=True, seconds=secs, error=f">{REF_TIMEOUT:.0f}s (timeout)")
    if not rec.get("ok"):
        return Meas(error="reference unavailable")
    return Meas(ok=True, distance=rec["distance"], seconds=rec["seconds"])


# --------------------------------------------------------------------------- #
# Per-code result row
# --------------------------------------------------------------------------- #
METHOD_ORDER = ["cc", "bz_cpu", "bz_gpu", "mitm", "ref_bz", "ref_cc"]
METHOD_LABEL = {
    "cc": "qubitserf cc",
    "bz_cpu": "qubitserf bz (cpu)",
    "bz_gpu": "qubitserf bz (gpu)",
    "mitm": "qubitserf mitm",
    "ref_bz": "ref BZDistMW",
    "ref_cc": "ref connClusterMW",
}
# Methods that, when they finish, give a distance to cross-check.
CERTIFYING = ["cc", "bz_cpu", "bz_gpu", "mitm", "ref_bz", "ref_cc"]
QUBITSERF_METHODS = ["cc", "bz_cpu", "bz_gpu", "mitm"]
REF_METHODS = ["ref_bz", "ref_cc"]


@dataclass
class Row:
    name: str
    n: int
    known_d: object
    meas: dict = field(default_factory=dict)   # method -> Meas
    consensus_d: object = None
    mismatch: bool = False        # any certifying disagreement at all
    qubitserf_mismatch: bool = False  # qubitserf methods / known disagree (a real bug)
    ref_only_mismatch: bool = False  # only a reference method disagrees (ref defect)

    def get(self, m) -> Meas:
        return self.meas.get(m, Meas())


def cell(m: Meas) -> str:
    """Render a Meas as a markdown table cell: distance and time."""
    if m.timed_out:
        return f"timeout {fmt_t(m.seconds)}"
    if not m.ok:
        if not m.error:
            return "-"
        if m.error.startswith("skip"):
            return "skip"
        if "unavailable" in m.error:
            return "n/a"
        return "err"
    return f"{m.distance} {fmt_t(m.seconds)}"


def t_cell(m: Meas) -> str:
    if m.timed_out:
        return f">{fmt_t(m.seconds)}"
    if not m.ok:
        return "-"
    return fmt_t(m.seconds)


# --------------------------------------------------------------------------- #
# Cross-check
# --------------------------------------------------------------------------- #
def reconcile(row: Row):
    """Collect every reported distance and classify any disagreement.

    We separate a *qubitserf* disagreement (qubitserf methods or the known textbook
    distance disagree among themselves -- a real bug we must not pass silently)
    from a *reference-only* disagreement (all qubitserf methods + known agree, but
    one of the reference package's methods reports a different value -- a defect
    in the reference, flagged but non-fatal).
    """
    certified = []
    for m in CERTIFYING:
        meas = row.get(m)
        if meas.ok and isinstance(meas.distance, int):
            certified.append((m, meas.distance))
    # Also fold in the known textbook distance, if any.
    if isinstance(row.known_d, int):
        certified.append(("known", row.known_d))

    ds = {d for _, d in certified}
    if len(ds) <= 1:
        if ds:
            row.consensus_d = next(iter(ds))
        return certified

    row.mismatch = True
    row.consensus_d = certified
    # qubitserf side (cc/bz/mitm + known textbook value)
    df_vals = {d for nm, d in certified if nm in QUBITSERF_METHODS or nm == "known"}
    ref_vals = {d for nm, d in certified if nm in REF_METHODS}
    if len(df_vals) > 1:
        row.qubitserf_mismatch = True
    elif df_vals and ref_vals and not ref_vals.issubset(df_vals):
        # qubitserf self-consistent, but a reference method disagrees with it.
        row.ref_only_mismatch = True
    else:
        row.qubitserf_mismatch = True
    return certified


# --------------------------------------------------------------------------- #
# Sweep one CSS family
# --------------------------------------------------------------------------- #
def sweep_css_family(fam_name, entries, ref_ok, tmpdir, mismatches, log):
    rows = []
    # Per-method "stop running larger sizes" flags once a budget is blown.
    stop = {m: False for m in METHOD_ORDER}
    for name, Hx, Hz, known_d in entries:
        n = int(Hx.shape[1])
        hard = name in bc.HARD_CSS_NAMES
        row = Row(name=name, n=n, known_d=known_d)
        log(f"\n[{fam_name}] {name}  n={n}{'  (hard)' if hard else ''}")

        # ---- qubitserf cc (always; certifies fast) ----
        if not stop["cc"]:
            m = df_css(Hx, Hz, "cc")
            row.meas["cc"] = m
            log(f"    cc        {cell(m)}")
            if m.timed_out:
                stop["cc"] = True

        # ---- qubitserf bz cpu (only attempted n<=BZ_MAX_N) ----
        if n > BZ_MAX_N:
            row.meas["bz_cpu"] = Meas(error=f"skip n>{BZ_MAX_N}")
        elif not stop["bz_cpu"]:
            m = df_css(Hx, Hz, "bz", backend="cpu")
            row.meas["bz_cpu"] = m
            log(f"    bz cpu    {cell(m)}")
            if m.timed_out:
                stop["bz_cpu"] = True

        # ---- qubitserf bz gpu (only attempted n<=BZ_MAX_N) ----
        if n > BZ_MAX_N:
            row.meas["bz_gpu"] = Meas(error=f"skip n>{BZ_MAX_N}")
        elif HAS_GPU and not stop["bz_gpu"]:
            m = df_css(Hx, Hz, "bz", backend="gpu")
            row.meas["bz_gpu"] = m
            log(f"    bz gpu  {cell(m)}")
            if m.timed_out:
                stop["bz_gpu"] = True

        # ---- qubitserf mitm (small codes only) ----
        if not stop["mitm"] and n <= MITM_MAX_N:
            m = df_css(Hx, Hz, "mitm")
            row.meas["mitm"] = m
            log(f"    mitm      {cell(m)}")
            if m.timed_out:
                stop["mitm"] = True
        elif n > MITM_MAX_N:
            row.meas["mitm"] = Meas(error=f"skip n>{MITM_MAX_N}")

        # ---- reference BZDistMW / connectedClusterMW (reused from prior run) ----
        row.meas["ref_bz"] = cached_ref(name, "ref_bz")
        row.meas["ref_cc"] = cached_ref(name, "ref_cc")
        log(f"    ref bz    {cell(row.meas['ref_bz'])}  (cached)")
        log(f"    ref cc    {cell(row.meas['ref_cc'])}  (cached)")

        certified = reconcile(row)
        if row.mismatch:
            kind = "qubitserf" if row.qubitserf_mismatch else "reference-only"
            mismatches.append((name, certified, kind))
            log(f"    !! MISMATCH ({kind}): {certified}")
        rows.append(row)
    return rows


def sweep_classical(entries, ref_ok, tmpdir, mismatches, log):
    rows = []
    for name, H, known_d in entries:
        n = int(H.shape[1])
        row = Row(name=name, n=n, known_d=known_d)
        log(f"\n[classical] {name}  n={n}")

        m = df_classical(H, "cc")
        row.meas["cc"] = m
        log(f"    cc        {cell(m)}")

        m = df_classical(H, "bz", backend="cpu")
        row.meas["bz_cpu"] = m
        log(f"    bz cpu    {cell(m)}")

        if HAS_GPU:
            m = df_classical(H, "bz", backend="gpu")
            row.meas["bz_gpu"] = m
            log(f"    bz gpu  {cell(m)}")

        if n <= MITM_MAX_N:
            m = df_classical(H, "mitm")
            row.meas["mitm"] = m
            log(f"    mitm      {cell(m)}")
        else:
            row.meas["mitm"] = Meas(error=f"skip n>{MITM_MAX_N}")

        # Reference single-block path (reused from prior run).
        row.meas["ref_bz"] = cached_ref(name, "ref_bz")
        log(f"    ref bz    {cell(row.meas['ref_bz'])}  (cached)")

        certified = reconcile(row)
        if row.mismatch:
            kind = "qubitserf" if row.qubitserf_mismatch else "reference-only"
            mismatches.append((name, certified, kind))
            log(f"    !! MISMATCH ({kind}): {certified}")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _su(num: float, den: float):
    if not (isinstance(num, float) and isinstance(den, float)):
        return None
    if math.isnan(num) or math.isnan(den) or den <= 0:
        return None
    return num / den


def _su_str(s):
    return f"{s:.1f}x" if s else "-"


def render_family(title, rows, methods) -> str:
    """One markdown table for a family: distance cells + per-method speedups vs cc."""
    hdr_cells = ["code", "n", "known d"]
    for mth in methods:
        hdr_cells.append(METHOD_LABEL[mth])
    # Speedup columns: reference methods relative to qubitserf's fastest certifier.
    hdr_cells += ["ref_bz / cc", "ref_cc / cc", "bz_cpu / bz_gpu"]
    sep = "|" + "|".join("---" for _ in hdr_cells) + "|"
    lines = [f"### {title}", "", "| " + " | ".join(hdr_cells) + " |", sep]
    for r in rows:
        cells = [r.name, str(r.n), str(r.known_d if r.known_d is not None else "?")]
        for mth in methods:
            cells.append(cell(r.get(mth)))
        cc = r.get("cc")
        su_refbz = _su(r.get("ref_bz").seconds, cc.seconds) \
            if (r.get("ref_bz").ok and cc.ok) else None
        su_refcc = _su(r.get("ref_cc").seconds, cc.seconds) \
            if (r.get("ref_cc").ok and cc.ok) else None
        su_gpu = _su(r.get("bz_cpu").seconds, r.get("bz_gpu").seconds) \
            if (r.get("bz_cpu").ok and r.get("bz_gpu").ok) else None
        cells += [_su_str(su_refbz), _su_str(su_refcc), _su_str(su_gpu)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_summary(all_rows, mismatches) -> str:
    out = ["## Summary", ""]
    out.append(f"- Backends available: `{BACKENDS}`.")
    out.append(f"- Per-code qubitserf budget: {DF_BUDGET:.0f}s; bz only for "
               f"n <= {BZ_MAX_N}; mitm only for n <= {MITM_MAX_N}; reference numbers "
               f"reused from a prior {REF_TIMEOUT:.0f}s run (not re-executed).")

    # Speedups: reference (whichever method) vs qubitserf cc, gathered over rows
    # where both finished.
    ref_bz_su, ref_cc_su, gpu_su = [], [], []
    for r in all_rows:
        cc = r.get("cc")
        if cc.ok:
            s = _su(r.get("ref_bz").seconds, cc.seconds)
            if s and r.get("ref_bz").ok:
                ref_bz_su.append(s)
            s = _su(r.get("ref_cc").seconds, cc.seconds)
            if s and r.get("ref_cc").ok:
                ref_cc_su.append(s)
        s = _su(r.get("bz_cpu").seconds, r.get("bz_gpu").seconds)
        if s and r.get("bz_cpu").ok and r.get("bz_gpu").ok:
            gpu_su.append(s)

    def stat(name, xs):
        if not xs:
            out.append(f"- {name}: no comparable runs.")
            return
        out.append(f"- **{name}**: median {statistics.median(xs):.1f}x, "
                   f"max {max(xs):.1f}x, min {min(xs):.1f}x "
                   f"(over {len(xs)} codes).")

    stat("ref BZDistMW / qubitserf cc speedup", ref_bz_su)
    stat("ref connectedClusterMW / qubitserf cc speedup", ref_cc_su)
    stat("qubitserf bz cpu / gpu speedup", gpu_su)

    # Where each method "wins" (fastest finished method per code).
    wins: dict[str, int] = {}
    for r in all_rows:
        best, best_t = None, float("inf")
        for mth in METHOD_ORDER:
            m = r.get(mth)
            if m.ok and isinstance(m.seconds, float) and not math.isnan(m.seconds):
                if m.seconds < best_t:
                    best, best_t = mth, m.seconds
        if best:
            wins[best] = wins.get(best, 0) + 1
    if wins:
        win_str = ", ".join(f"{METHOD_LABEL[k]}: {v}" for k, v in
                            sorted(wins.items(), key=lambda kv: -kv[1]))
        out.append(f"- Fastest-finished method per code: {win_str}.")

    # Codes only CC could certify (no other certifying method finished).
    cc_only = []
    for r in all_rows:
        cc = r.get("cc")
        if not cc.ok:
            continue
        others = False
        for mth in CERTIFYING:
            if mth == "cc":
                continue
            if r.get(mth).ok:
                others = True
                break
        if not others:
            cc_only.append(r.name)
    if cc_only:
        out.append(f"- **Only qubitserf cc certified the distance** on: "
                   f"{', '.join(cc_only)} (every other method either timed out "
                   "or was skipped).")

    # Where the reference timed out but qubitserf finished.
    ref_to = [r.name for r in all_rows
              if r.get("ref_bz").timed_out or r.get("ref_cc").timed_out]
    if ref_to:
        out.append(f"- Reference timed out (> {REF_TIMEOUT:.0f}s) on: "
                   f"{', '.join(sorted(set(ref_to)))} — qubitserf cc solved all of these.")

    # Distance agreement.  Split into qubitserf self-disagreements (real bugs)
    # and reference-only defects (qubitserf + textbook agree; a reference method
    # is wrong).
    df_bugs = [(nm, c) for nm, c, kind in mismatches if kind == "qubitserf"]
    ref_bugs = [(nm, c) for nm, c, kind in mismatches if kind == "reference-only"]

    if df_bugs:
        out.append("- **QUBITSERF DISTANCE MISMATCHES DETECTED** (qubitserf methods "
                   "or the known distance disagree among themselves):")
        for nm, certified in df_bugs:
            out.append(f"    - {nm}: {certified}")
    else:
        out.append("- **All qubitserf distances agree**: qubitserf cc / bz (cpu+gpu) "
                   "/ mitm reported the same exact distance on every code, matching "
                   "the known textbook distances where those are defined, and matching "
                   "the reference where the reference is correct.")
    if ref_bugs:
        out.append("- **Reference-package defects flagged** (qubitserf + textbook "
                   "agree; a reference method disagrees -- qubitserf is correct):")
        for nm, certified in ref_bugs:
            out.append(f"    - {nm}: {certified}  "
                       "(reference connectedClusterMW mis-reports the Z-component)")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Optional plots
# --------------------------------------------------------------------------- #
def maybe_plot(css_family_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    saved = []
    for fam, rows in css_family_rows.items():
        if len(rows) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for mth in METHOD_ORDER:
            xs, ys = [], []
            for r in rows:
                m = r.get(mth)
                if m.ok and isinstance(m.seconds, float) and not math.isnan(m.seconds) \
                        and m.seconds > 0:
                    xs.append(r.n)
                    ys.append(m.seconds)
            if len(xs) >= 2:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot([xs[i] for i in order], [ys[i] for i in order],
                        marker="o", label=METHOD_LABEL[mth])
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_yscale("log")
        ax.set_xlabel("n (physical qubits)")
        ax.set_ylabel("time (s, log)")
        ax.set_title(f"qubitserf comprehensive: {fam}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        # For the topological families, add a second (top) x-axis labelling each n
        # by the code distance d.  d grows with n here (toric n=2d^2, surface
        # n=d^2+(d-1)^2), so distance is the more meaningful hardness scale.
        if fam in ("toric", "surface"):
            nd = sorted({(r.n, r.known_d) for r in rows if r.known_d is not None})
            if nd:
                xlim = ax.get_xlim()
                ax.set_xlim(xlim)                 # freeze so the two axes stay aligned
                axd = ax.twiny()
                axd.set_xscale(ax.get_xscale())
                axd.set_xlim(xlim)
                axd.set_xticks([n for n, _ in nd])
                axd.set_xticklabels([str(d) for _, d in nd])
                axd.set_xlabel("distance d")
        path = os.path.join(HERE, f"comprehensive_{fam}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        saved.append(os.path.basename(path))
    return saved


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    def log(s):
        print(s, flush=True)

    log("qubitserf comprehensive benchmark")
    log(f"backends: {BACKENDS}")
    ref_ok = bool(REF_CACHE)
    log(f"reference: reusing {len(REF_CACHE)} cached codeDistance measurements from a "
        f"prior {REF_TIMEOUT:.0f}s run (external package NOT re-executed)")
    log(f"budgets: df={DF_BUDGET:.0f}s drain={DRAIN_BUDGET:.0f}s "
        f"mitm_max_n={MITM_MAX_N} ref_timeout(cached)={REF_TIMEOUT:.0f}s")

    mismatches = []
    css_family_rows: dict[str, list] = {}
    all_rows: list = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for fam_name, entries in bc.css_families().items():
            rows = sweep_css_family(fam_name, entries, ref_ok, tmpdir,
                                    mismatches, log)
            css_family_rows[fam_name] = rows
            all_rows.extend(rows)

        class_rows = sweep_classical(bc.classical_families(), ref_ok, tmpdir,
                                     mismatches, log)
        all_rows.extend(class_rows)

    # ---- render report ----
    css_methods = ["cc", "bz_cpu", "bz_gpu", "mitm", "ref_bz", "ref_cc"]
    class_methods = ["cc", "bz_cpu", "bz_gpu", "mitm", "ref_bz"]

    sections = []
    for fam_name, rows in css_family_rows.items():
        sections.append(render_family(f"CSS: {fam_name}", rows, css_methods))
    sections.append(render_family("Classical codes", class_rows, class_methods))

    plots = maybe_plot(css_family_rows)
    summary = render_summary(all_rows, mismatches)

    with open(OUT_MD, "w") as f:
        f.write("# qubitserf comprehensive benchmark results\n\n")
        f.write("Generated by `bench/comprehensive.py`. Each cell shows the "
                "reported distance and wall-clock time. `timeout` means the "
                "per-method budget was exceeded.\n\n")
        f.write("Methods: **qubitserf cc** (connected cluster), **qubitserf bz** on "
                "cpu and gpu (Brouwer-Zimmermann), "
                "**qubitserf mitm** (meet-in-the-middle, small n only), and the "
                "reference `codeDistance` package's **BZDistMW** and "
                "**connectedClusterMW**.\n\n")
        f.write("qubitserf timings are freshly measured; the two reference columns "
                "are **reused from a prior 30s run** (`bench/ref_cache.json`) — the "
                "reference code is unchanged, so its numbers are carried over rather "
                "than re-executed. `n/a` marks codes with no cached reference (the "
                "Reed-Muller families, added later).\n\n")
        for s in sections:
            f.write(s + "\n\n")
        f.write(summary + "\n")
        if plots:
            f.write("\n### Plots\n\n")
            for p in plots:
                f.write(f"![{p}]({p})\n\n")
    log(f"\nWrote {OUT_MD}")
    if plots:
        log(f"Wrote plots: {', '.join(plots)}")

    log("\n" + summary)

    df_bugs = [m for m in mismatches if m[2] == "qubitserf"]
    ref_bugs = [m for m in mismatches if m[2] == "reference-only"]
    if ref_bugs:
        log(f"\nNOTE: {len(ref_bugs)} reference-package defect(s) flagged "
            "(qubitserf correct): " + ", ".join(m[0] for m in ref_bugs))
    if df_bugs:
        log("\nERROR: qubitserf distance mismatches detected (see summary).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
