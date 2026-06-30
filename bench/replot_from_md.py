"""Re-render the comprehensive benchmark PNGs with **qubitserf** legends WITHOUT
recomputing any distances.

The expensive benchmark data (GPU + a reference package) is *not* recomputed
here.  Instead we parse the already-computed numbers out of
``bench/comprehensive_results.md`` (one markdown table per code family) and
re-draw each ``comprehensive_<family>.png`` using exactly the same plot style as
``comprehensive.py`` -- only the ``METHOD_LABEL`` legend strings differ, now
reading "qubitserf ..." instead of the old "qminweight ...".

Two families -- ``reed_muller_r1`` / ``reed_muller_r2`` -- have orphan PNGs that
were produced by a separate experimental run and were never written into any
markdown table (the md's Plots section does not reference them either).  Their
already-computed numbers are recovered below from the captured benchmark run log
(authoritative for the points it holds) plus the few tail points that only the
prior PNGs retained; see ``REED_MULLER`` for provenance.  Nothing is recomputed.

Run from the package root:

    PATH=/opt/miniconda3/envs/sage_env/bin:$PATH PYTHONPATH=python MPLBACKEND=Agg \
        /opt/miniconda3/envs/sage_env/bin/python bench/replot_from_md.py
"""
from __future__ import annotations

import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
IN_MD = os.path.join(HERE, "comprehensive_results.md")

# --------------------------------------------------------------------------- #
# Plot vocabulary -- copied verbatim from comprehensive.py (post-rename), so the
# legends read "qubitserf ...".  Kept inline (rather than importing
# comprehensive) so this script never imports qubitserf / touches a backend.
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
LABEL_TO_METHOD = {v: k for k, v in METHOD_LABEL.items()}


# --------------------------------------------------------------------------- #
# Markdown parsing
# --------------------------------------------------------------------------- #
_TIME_RE = re.compile(r"^([0-9]*\.?[0-9]+)(us|ms|s)$")
_UNIT = {"us": 1e-6, "ms": 1e-3, "s": 1.0}


def parse_time(cell: str):
    """Return seconds for a result cell, or None if it has no plottable time.

    A cell is ``"<distance> <time>"`` (e.g. ``"3 375us"``), a capped bracket
    ``"[lower,upper] <time>"`` (e.g. ``"[8,12] 4.36s"`` -- use the time), or one
    of ``skip`` / ``timeout ...`` / ``-`` / ``n/a`` / ``err`` (no point).  The
    time is always the last whitespace token; ``timeout`` cells are dropped to
    match comprehensive.py's plotter, which only plots finished (``ok``) runs.
    """
    cell = cell.strip()
    if not cell or cell.startswith(("timeout", "skip", "n/a", "err")) or cell == "-":
        return None
    tok = cell.split()[-1]
    m = _TIME_RE.match(tok)
    if not m:
        return None
    return float(m.group(1)) * _UNIT[m.group(2)]


def parse_md(path: str) -> dict:
    """Parse comprehensive_results.md -> {family: [(n, {method: seconds}), ...]}.

    Only ``### CSS: <family>`` sections become families (matching the families
    comprehensive.py plots); the Classical / Summary / Plots sections are
    skipped, as comprehensive.py does not emit a classical PNG.
    """
    fams: dict[str, list] = {}
    cur = None          # current family name or None
    cols = None         # list of method keys (or None) per table column
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.startswith("### "):
            title = line[4:].strip()
            if title.startswith("CSS: "):
                cur = title[len("CSS: "):].strip()
                fams[cur] = []
            else:
                cur = None       # Classical codes / other -> not plotted
            cols = None
            continue
        if cur is None or not line.startswith("|"):
            cols = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("- "):
            continue             # separator row "|---|---|..."
        if cols is None:
            # Header row: map each column label to a method key (or None).
            cols = [LABEL_TO_METHOD.get(c) for c in cells]
            continue
        # Data row.  Column 1 is n (col 0 = code, col 2 = known d).
        try:
            n = int(cells[1])
        except (ValueError, IndexError):
            continue
        meas: dict[str, float] = {}
        for c, key in zip(cells, cols):
            if key is None:
                continue
            secs = parse_time(c)
            if secs is not None and secs > 0:
                meas[key] = secs
        fams[cur].append((n, meas))
    return fams


# --------------------------------------------------------------------------- #
# Reed-Muller families -- not in any markdown table (orphan PNGs).
#
# Provenance: the per-(n, method) numbers below are the *already-computed*
# benchmark results, NOT a recomputation.  Points marked (log) are taken
# verbatim from the captured comprehensive.py run log; points marked (png) are
# the tail points that survived only in the prior PNGs (the captured log was
# truncated at reed_muller_r2 n=128), read back from those plots.  ``d`` is the
# fixed distance shown in the original plot title suffix "(d=N)".
# --------------------------------------------------------------------------- #
REED_MULLER = {
    "reed_muller_r1": {
        "d": 4,
        "rows": [
            # n,   cc,      bz_cpu,  bz_gpu,  mitm,    ref_bz,   ref_cc
            (16,  {"cc": 547e-6, "bz_cpu": 254e-6, "bz_gpu": 248e-6,        # (log)
                   "mitm": 932e-6, "ref_bz": 2.3e-3, "ref_cc": 1.2e-3}),
            (32,  {"cc": 582e-6, "bz_cpu": 433e-6, "bz_gpu": 378e-6,        # (log)
                   "mitm": 1.7e-3, "ref_bz": 6.3e-3, "ref_cc": 6.0e-3}),
            (64,  {"cc": 691e-6, "bz_cpu": 863e-6, "bz_gpu": 877e-6,        # (log)
                   "mitm": 9.4e-3, "ref_bz": 98.2e-3, "ref_cc": 45.1e-3}),
            (128, {"cc": 1.5e-3, "bz_cpu": 2.6e-3, "bz_gpu": 2.4e-3,        # (log)
                   "mitm": 132.2e-3, "ref_bz": 3.13, "ref_cc": 368.1e-3}),
            (256, {"cc": 6.3e-3, "bz_cpu": 9.7e-3, "bz_gpu": 9.5e-3,        # (log)
                   "ref_bz": 102.84, "ref_cc": 2.93}),
            (512, {"bz_cpu": 0.072, "bz_gpu": 0.105}),                      # (png tail)
        ],
    },
    "reed_muller_r2": {
        "d": 8,
        "rows": [
            (64,  {"cc": 135.61, "bz_cpu": 62.4e-3, "bz_gpu": 12.8e-3,      # (log)
                   "mitm": 6.00, "ref_bz": 23.46}),
            (128, {"bz_cpu": 40.0, "bz_gpu": 0.8}),                         # (png tail)
            (256, {"bz_gpu": 200.0}),                                       # (png tail)
        ],
    },
}


# --------------------------------------------------------------------------- #
# Rendering -- faithful replica of comprehensive.py:maybe_plot's per-family draw
# (figsize, log-y, markers, axis labels, title, legend, grid, dpi, tight_layout).
# The only difference: METHOD_LABEL now says "qubitserf"; reed_muller titles
# carry the original "(d=N)" suffix.
# --------------------------------------------------------------------------- #
# Per-family plot titles. The four below are set explicitly; any other family falls back
# to "qubitserf <fam> benchmark".
TITLES = {
    "toric": "qubitserf toric benchmark",
    "surface": "qubitserf surface benchmark",
    "reed_muller_r1": "qubitserf distance 4 RM",
    "reed_muller_r2": "qubitserf distance 8 RM",
}


def plot_family(fam: str, rows: list, title_suffix: str = "",
                min_points: int = 2) -> str | None:
    # comprehensive.py:maybe_plot draws a method only with >= 2 points; the 5 md
    # families were produced that way, so they keep min_points=2.  The separate
    # reed_muller plotter (the one that added the "(d=N)" title) also drew
    # single-point series as dots, so those families pass min_points=1.
    if len(rows) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for mth in METHOD_ORDER:
        xs, ys = [], []
        for n, meas in rows:
            secs = meas.get(mth)
            if isinstance(secs, float) and not math.isnan(secs) and secs > 0:
                xs.append(n)
                ys.append(secs)
        if len(xs) >= min_points:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order],
                    marker="o", label=METHOD_LABEL[mth])
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_yscale("log")
    ax.set_xlabel("n (physical qubits)")
    ax.set_ylabel("time (s, log)")
    ax.set_title(TITLES.get(fam, f"qubitserf {fam} benchmark{title_suffix}"))
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(HERE, f"comprehensive_{fam}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return os.path.basename(path)


def main() -> int:
    print("Legend labels used:")
    for mth in METHOD_ORDER:
        print(f"  {mth:7s} -> {METHOD_LABEL[mth]}")

    saved, empty = [], []

    fams = parse_md(IN_MD)
    for fam, rows in fams.items():
        # A row whose every method cell was unplottable contributes no point.
        if not any(meas for _, meas in rows):
            empty.append(f"CSS: {fam} (no parseable data)")
            continue
        name = plot_family(fam, rows)
        (saved if name else empty).append(name or f"CSS: {fam} (<2 plottable rows)")

    for fam, spec in REED_MULLER.items():
        name = plot_family(fam, spec["rows"], title_suffix=f" (d={spec['d']})",
                           min_points=1)
        (saved if name else empty).append(name or f"{fam} (<2 plottable rows)")

    print("\nRegenerated PNGs:")
    for n in saved:
        print(f"  {n}")
    if empty:
        print("\nNo plottable data for:")
        for e in empty:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
