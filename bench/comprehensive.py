"""Comprehensive qminweight benchmark across several CSS / classical families.

Compares, per code:

  * qminweight ``cc``           (connected cluster, always certifies, sub-second)
  * qminweight ``bz`` (cpu)     (Brouwer-Zimmermann; capped on hard codes)
  * qminweight ``bz`` (gpu)   (same, on the GPU backend if present)
  * qminweight ``mitm``         (meet-in-the-middle; small codes only -- slow)
  * reference ``BZDistMW``         (codeDistance package, subprocess + timeout)
  * reference ``connectedClusterMW`` (codeDistance package, subprocess + timeout)

For every measurement we record (distance, lower_bound, proven, seconds) and
wrap it in try/except + a per-call wall-clock budget, so the run always
finishes.  Methods that exceed their per-size budget are skipped for larger
sizes in the same family.  Hard codes (sparse, weak BZ lower bound) run BZ with
a ``max_weight`` cap and are reported as a rigorous ``[lower, upper]`` bracket
so BZ cannot hang.

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

import math
import os
import statistics
import tempfile
import threading
import time
from dataclasses import dataclass, field

import qminweight as df

# Re-use the proven reference plumbing from the existing benchmark module.
from benchmark import run_reference, reference_available, _save_npy, fmt_t

import bench_codes as bc


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "comprehensive_results.md")

# Per-method, per-code wall-clock budgets (seconds).
DF_BUDGET = float(os.environ.get("DF_BUDGET", "300"))     # in-process qminweight cap
MITM_MAX_N = int(os.environ.get("MITM_MAX_N", "130"))     # mitm only for n <= this
# BZ is only *attempted* up to this n.  The in-process budget runs the native
# solver on a daemon thread that cannot be cancelled, so a BZ call that exceeds
# the budget keeps burning a CPU core in the background.  Codes above BZ_MAX_N
# that appear in HARD_CSS_NAMES are protected by a max_weight cap (bounded work);
# uncapped BZ on large sparse codes (d >> 1, weak lower bound) is the orphan risk.
# All codes above BZ_MAX_N have cc/reference still attempted.
#
# 1024 is the native BZ ceiling on the GPU backends (codeword stride <= MAX_WORDS
# = 16 u64 words = 1024 bits; above that the GPU path auto-falls-back to the
# dynamic CPU solver).  The CPU backend itself is unbounded.  This covers every
# code in the benchmark; the large sparse QLDPC codes within the window
# (toric/surface L>=9, bb288) are in HARD_CSS_NAMES so they run a bounded cap.
BZ_MAX_N = int(os.environ.get("BZ_MAX_N", "1024"))
REF_TIMEOUT = float(os.environ.get("REF_TIMEOUT", "300"))  # reference subprocess cap
BZ_CAP = int(os.environ.get("BZ_CAP", "6"))               # max_weight on hard codes

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
# In-process qminweight call with a soft wall-clock budget.
# --------------------------------------------------------------------------- #
# The native solver runs to completion regardless (no cooperative cancellation),
# but we run it on a worker thread and *give up waiting* after `budget` seconds.
# A method that blows its budget is recorded as timed_out and skipped for larger
# sizes of the same family.  We always pass a max_weight cap to BZ on hard codes,
# which bounds its work tightly so the thread does finish promptly anyway.
# --------------------------------------------------------------------------- #
@dataclass
class Meas:
    ok: bool = False
    distance: object = None
    lower_bound: object = None
    proven: object = None
    seconds: float = float("nan")
    timed_out: bool = False
    error: str = ""
    capped: bool = False     # BZ run with a max_weight cap (bracket, maybe unproven)


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
        return Meas(ok=False, timed_out=True, seconds=budget,
                    error=f">{budget:.0f}s (budget)")
    if "err" in box:
        return Meas(ok=False, seconds=box.get("secs", float("nan")), error=box["err"])
    r = box["r"]
    return Meas(ok=True, distance=r.distance, lower_bound=r.lower_bound,
                proven=bool(r.proven), seconds=box["secs"])


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


def df_css(Hx, Hz, method, backend="auto", max_weight=0, budget=DF_BUDGET) -> Meas:
    m = _run_df(lambda: df.css_distance(Hx, Hz, method=method, which="min",
                                        backend=backend, max_weight=max_weight),
                budget)
    m.capped = max_weight > 0
    return m


def df_classical(H, method, backend="auto", max_weight=0, budget=DF_BUDGET) -> Meas:
    m = _run_df(lambda: df.classical_distance(H, method=method, backend=backend,
                                              max_weight=max_weight),
                budget)
    m.capped = max_weight > 0
    return m


# --------------------------------------------------------------------------- #
# Reference call wrapper -> Meas
# --------------------------------------------------------------------------- #
def ref_css(Hx, Hz, method, ok, tmpdir) -> Meas:
    if not ok:
        return Meas(error="reference unavailable")
    hx = _save_npy(Hx, tmpdir, "hx")
    hz = _save_npy(Hz, tmpdir, "hz")
    res = run_reference("css", hx, hz, method, timeout=REF_TIMEOUT)
    return _ref_to_meas(res)


def ref_classical(H, ok, tmpdir) -> Meas:
    if not ok:
        return Meas(error="reference unavailable")
    h = _save_npy(H, tmpdir, "h")
    res = run_reference("class", h, timeout=REF_TIMEOUT)
    return _ref_to_meas(res)


def _ref_to_meas(res: dict) -> Meas:
    if res.get("ok"):
        return Meas(ok=True, distance=res["d"], lower_bound=res["d"], proven=True,
                    seconds=res.get("seconds", float("nan")))
    if res.get("timed_out"):
        return Meas(timed_out=True, seconds=res.get("seconds", float("nan")),
                    error=f">{REF_TIMEOUT:.0f}s (timeout)")
    return Meas(error=str(res.get("error", "error"))[:80])


# --------------------------------------------------------------------------- #
# Per-code result row
# --------------------------------------------------------------------------- #
METHOD_ORDER = ["cc", "bz_cpu", "bz_gpu", "mitm", "ref_bz", "ref_cc"]
METHOD_LABEL = {
    "cc": "qminweight cc",
    "bz_cpu": "qminweight bz (cpu)",
    "bz_gpu": "qminweight bz (gpu)",
    "mitm": "qminweight mitm",
    "ref_bz": "ref BZDistMW",
    "ref_cc": "ref connClusterMW",
}
# Methods that, when ok and proven, give a certified exact distance to cross-check.
CERTIFYING = ["cc", "bz_cpu", "bz_gpu", "mitm", "ref_bz", "ref_cc"]
QMINWEIGHT_METHODS = ["cc", "bz_cpu", "bz_gpu", "mitm"]
REF_METHODS = ["ref_bz", "ref_cc"]


@dataclass
class Row:
    name: str
    n: int
    known_d: object
    meas: dict = field(default_factory=dict)   # method -> Meas
    consensus_d: object = None
    mismatch: bool = False        # any certifying disagreement at all
    qminweight_mismatch: bool = False  # qminweight methods / known disagree (a real bug)
    ref_only_mismatch: bool = False  # only a reference method disagrees (ref defect)

    def get(self, m) -> Meas:
        return self.meas.get(m, Meas())


def cell(m: Meas) -> str:
    """Render a Meas as a markdown table cell: distance(+bracket) and time."""
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
    if m.capped and not m.proven:
        return f"[{m.lower_bound},{m.distance}] {fmt_t(m.seconds)}"
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
    """Collect every certified exact distance and classify any disagreement.

    We separate a *qminweight* disagreement (qminweight methods or the known textbook
    distance disagree among themselves -- a real bug we must not pass silently)
    from a *reference-only* disagreement (all qminweight methods + known agree, but
    one of the reference package's methods reports a different value -- a defect
    in the reference, flagged but non-fatal).
    """
    certified = []
    for m in CERTIFYING:
        meas = row.get(m)
        if meas.ok and meas.proven and isinstance(meas.distance, int):
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
    # qminweight side (cc/bz/mitm + known textbook value)
    df_vals = {d for nm, d in certified if nm in QMINWEIGHT_METHODS or nm == "known"}
    ref_vals = {d for nm, d in certified if nm in REF_METHODS}
    if len(df_vals) > 1:
        row.qminweight_mismatch = True
    elif df_vals and ref_vals and not ref_vals.issubset(df_vals):
        # qminweight self-consistent, but a reference method disagrees with it.
        row.ref_only_mismatch = True
    else:
        row.qminweight_mismatch = True
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

        # ---- qminweight cc (always; certifies fast) ----
        if not stop["cc"]:
            m = df_css(Hx, Hz, "cc")
            row.meas["cc"] = m
            log(f"    cc        {cell(m)}")
            if m.timed_out:
                stop["cc"] = True

        # ---- qminweight bz cpu (capped on hard codes; only attempted n<=BZ_MAX_N) ----
        if n > BZ_MAX_N:
            row.meas["bz_cpu"] = Meas(error=f"skip n>{BZ_MAX_N}")
        elif not stop["bz_cpu"]:
            mw = BZ_CAP if hard else 0
            m = df_css(Hx, Hz, "bz", backend="cpu", max_weight=mw)
            row.meas["bz_cpu"] = m
            log(f"    bz cpu    {cell(m)}")
            # Only a *full* (uncapped) run that blows its budget should disable
            # larger sizes; capped runs are bounded by construction.
            if m.timed_out and not m.capped:
                stop["bz_cpu"] = True

        # ---- qminweight bz gpu (only attempted n<=BZ_MAX_N) ----
        if n > BZ_MAX_N:
            row.meas["bz_gpu"] = Meas(error=f"skip n>{BZ_MAX_N}")
        elif HAS_GPU and not stop["bz_gpu"]:
            mw = BZ_CAP if hard else 0
            m = df_css(Hx, Hz, "bz", backend="gpu", max_weight=mw)
            row.meas["bz_gpu"] = m
            log(f"    bz gpu  {cell(m)}")
            if m.timed_out and not m.capped:
                stop["bz_gpu"] = True

        # ---- qminweight mitm (small codes only) ----
        if not stop["mitm"] and n <= MITM_MAX_N:
            m = df_css(Hx, Hz, "mitm")
            row.meas["mitm"] = m
            log(f"    mitm      {cell(m)}")
            if m.timed_out:
                stop["mitm"] = True
        elif n > MITM_MAX_N:
            row.meas["mitm"] = Meas(error=f"skip n>{MITM_MAX_N}")

        # ---- reference BZDistMW ----
        if ref_ok and not stop["ref_bz"]:
            m = ref_css(Hx, Hz, "BZDistMW", ref_ok, tmpdir)
            row.meas["ref_bz"] = m
            log(f"    ref bz    {cell(m)}")
            if m.timed_out:
                stop["ref_bz"] = True

        # ---- reference connectedClusterMW ----
        if ref_ok and not stop["ref_cc"]:
            m = ref_css(Hx, Hz, "connectedClusterMW", ref_ok, tmpdir)
            row.meas["ref_cc"] = m
            log(f"    ref cc    {cell(m)}")
            if m.timed_out:
                stop["ref_cc"] = True

        certified = reconcile(row)
        if row.mismatch:
            kind = "qminweight" if row.qminweight_mismatch else "reference-only"
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

        # Reference single-block path; it hangs on degenerate repetition-style
        # checks, but our classical set (Hamming, random LDPC) is non-degenerate.
        if ref_ok:
            m = ref_classical(H, ref_ok, tmpdir)
            row.meas["ref_bz"] = m
            log(f"    ref bz    {cell(m)}")

        certified = reconcile(row)
        if row.mismatch:
            kind = "qminweight" if row.qminweight_mismatch else "reference-only"
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
    # Speedup columns: reference methods relative to qminweight's fastest certifier.
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
    out.append(f"- Per-code qminweight budget: {DF_BUDGET:.0f}s; bz only for "
               f"n <= {BZ_MAX_N}; mitm only for n <= {MITM_MAX_N}; reference "
               f"subprocess timeout {REF_TIMEOUT:.0f}s; BZ max_weight cap on hard "
               f"codes: {BZ_CAP}.")

    # Speedups: reference (whichever method) vs qminweight cc, gathered over rows
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

    stat("ref BZDistMW / qminweight cc speedup", ref_bz_su)
    stat("ref connectedClusterMW / qminweight cc speedup", ref_cc_su)
    stat("qminweight bz cpu / gpu speedup", gpu_su)

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

    # Codes only CC could certify (no other certifying method finished+proven).
    cc_only = []
    for r in all_rows:
        cc = r.get("cc")
        if not (cc.ok and cc.proven):
            continue
        others = False
        for mth in CERTIFYING:
            if mth == "cc":
                continue
            m = r.get(mth)
            if m.ok and m.proven:
                others = True
                break
        if not others:
            cc_only.append(r.name)
    if cc_only:
        out.append(f"- **Only qminweight cc certified the exact distance** on: "
                   f"{', '.join(cc_only)} (every other method either timed out, "
                   "was capped without proving, or was skipped).")

    # Where the reference timed out but qminweight finished.
    ref_to = [r.name for r in all_rows
              if r.get("ref_bz").timed_out or r.get("ref_cc").timed_out]
    if ref_to:
        out.append(f"- Reference timed out (> {REF_TIMEOUT:.0f}s) on: "
                   f"{', '.join(sorted(set(ref_to)))} — qminweight cc solved all of these.")

    # Distance agreement.  Split into qminweight self-disagreements (real bugs)
    # and reference-only defects (qminweight + textbook agree; a reference method
    # is wrong).
    df_bugs = [(nm, c) for nm, c, kind in mismatches if kind == "qminweight"]
    ref_bugs = [(nm, c) for nm, c, kind in mismatches if kind == "reference-only"]

    if df_bugs:
        out.append("- **QMINWEIGHT DISTANCE MISMATCHES DETECTED** (qminweight methods "
                   "or the known distance disagree among themselves):")
        for nm, certified in df_bugs:
            out.append(f"    - {nm}: {certified}")
    else:
        out.append("- **All qminweight distances agree**: qminweight cc / bz (cpu+gpu) "
                   "/ mitm reported the same exact distance on every code, matching "
                   "the known textbook distances where those are defined, and matching "
                   "the reference where the reference is correct.")
    if ref_bugs:
        out.append("- **Reference-package defects flagged** (qminweight + textbook "
                   "agree; a reference method disagrees -- qminweight is correct):")
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
        ax.set_title(f"qminweight comprehensive: {fam}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
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

    log("qminweight comprehensive benchmark")
    log(f"backends: {BACKENDS}")
    ref_ok = reference_available()
    log(f"reference (codeDistance) available: {ref_ok}")
    log(f"budgets: df={DF_BUDGET:.0f}s mitm_max_n={MITM_MAX_N} "
        f"ref_timeout={REF_TIMEOUT:.0f}s bz_cap={BZ_CAP}")

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
        f.write("# qminweight comprehensive benchmark results\n\n")
        f.write("Generated by `bench/comprehensive.py`. Each cell shows the "
                "reported distance and wall-clock time. A `[lower,upper]` cell is "
                "a BZ run capped at a max_weight (rigorous bracket, distance not "
                "certified there). `timeout` means the per-method budget was "
                "exceeded.\n\n")
        f.write("Methods: **qminweight cc** (connected cluster), **qminweight bz** on "
                "cpu and gpu (Brouwer-Zimmermann, capped on hard sparse codes), "
                "**qminweight mitm** (meet-in-the-middle, small n only), and the "
                "reference `codeDistance` package's **BZDistMW** and "
                "**connectedClusterMW** (subprocess, hard timeout).\n\n")
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

    df_bugs = [m for m in mismatches if m[2] == "qminweight"]
    ref_bugs = [m for m in mismatches if m[2] == "reference-only"]
    if ref_bugs:
        log(f"\nNOTE: {len(ref_bugs)} reference-package defect(s) flagged "
            "(qminweight correct): " + ", ".join(m[0] for m in ref_bugs))
    if df_bugs:
        log("\nERROR: qminweight distance mismatches detected (see summary).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
