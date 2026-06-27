"""Compute distances of qecdb CSS codes with unknown distance, via GPU Brouwer-Zimmermann.

For every CSS code in the local `qumba` DB with `d: null`, parse its stabilisers
into (Hx, Hz), then run the BZ distance for the Z- and X-components in separate
subprocesses (each hard-timeout'd). A component is EXACT when BZ proves it;
otherwise the verbose trace still yields an improved [lower, upper] bracket.

    PYTHONPATH=src:research/distance/qminweight/python MPLBACKEND=Agg \
      /opt/miniconda3/envs/sage_env/bin/python \
      research/distance/qminweight/bench/qecdb_unknown_bz.py [--timeout S] [--limit N] [--query JSON]

Results are written to bench/qecdb_unknown_bz_results.jsonl (one row per code) and
printed as a table. Nothing is written back to the database.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile, time
import numpy as np
import pymongo

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from lib.codes.qecdb import _extract_css_basis          # noqa: E402
from lib.utility import gf2                              # noqa: E402

WORKER = os.path.join(HERE, "qecdb_bz_worker.py")
OUT = os.path.join(HERE, "qecdb_unknown_bz_results.jsonl")
URI = os.environ.get("QECDB_URI", "mongodb://localhost:27017")
PY = sys.executable
_LINE = re.compile(r"\[bz \w+\] d=(\d+) upper=(\d+) lower=(\d+)")


def solve_component(Hx, Hz, which, timeout, maxw):
    """Run one component in a subprocess; return dict with exact value or a bracket."""
    with tempfile.TemporaryDirectory() as td:
        hx, hz = os.path.join(td, "hx.npy"), os.path.join(td, "hz.npy")
        np.save(hx, Hx.astype(np.uint8)); np.save(hz, Hz.astype(np.uint8))
        cmd = [PY, WORKER, hx, hz, which, str(maxw)]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(REPO, "src"), os.path.join(HERE, "..", "python"),
             env.get("PYTHONPATH", "")])
        env.setdefault("MPLBACKEND", "Agg")
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, env=env)
        except subprocess.TimeoutExpired as e:
            stderr = e.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return _bracket_from_trace(stderr, "timeout", time.perf_counter() - t0)
        if p.returncode != 0:
            return {"status": "error", "err": p.stderr.strip()[-200:],
                    "seconds": time.perf_counter() - t0}
        line = (p.stdout.strip().splitlines() or [""])[-1]
        try:
            j = json.loads(line)
            return {"status": "exact" if j["proven"] else "capped",
                    "distance": j["distance"], "lower": j["lower_bound"],
                    "upper": j["distance"], "seconds": j["seconds"]}
        except Exception:
            return _bracket_from_trace(p.stderr, "parse_fail",
                                       time.perf_counter() - t0)


def _bracket_from_trace(stderr, status, seconds):
    """Recover the best (lower, upper) seen in the verbose trace (e.g. after a kill)."""
    lo, up = 0, None
    for m in _LINE.finditer(stderr or ""):
        up = int(m.group(2)); lo = max(lo, int(m.group(3)))
    return {"status": status, "lower": lo, "upper": up, "seconds": seconds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-weight", type=int, default=0)
    ap.add_argument("--query", default='{"css": true, "d": null}')
    args = ap.parse_args()

    col = pymongo.MongoClient(URI)["qumba"]["codes"]
    docs = list(col.find(json.loads(args.query)))
    # easiest first: smallest known upper bound, then smallest n
    docs.sort(key=lambda d: (d.get("d_upper_bound") or 9999, d.get("n", 0)))
    if args.limit:
        docs = docs[:args.limit]
    print(f"codes: {len(docs)} | timeout {args.timeout}s/component | uri {URI}")
    print(f"{'#':>3} {'n':>4} {'k':>3} {'old':>9} {'dz':>9} {'dx':>9} {'d':>6} {'t':>7}")

    fout = open(OUT, "w")
    n_exact = 0
    for i, doc in enumerate(docs):
        n = int(doc["n"])
        oid = str(doc["_id"])
        try:
            Hx, Hz = _extract_css_basis(doc.get("H", "").split(), n)
        except Exception as e:
            print(f"{i:>3} {n:>4}  parse error: {e}")
            continue
        k = n - gf2.rank_gf2(Hx) - gf2.rank_gf2(Hz)
        old = (doc.get("d_lower_bound"), doc.get("d_upper_bound"))
        z = solve_component(Hx, Hz, "z", args.timeout, args.max_weight)
        x = solve_component(Hx, Hz, "x", args.timeout, args.max_weight)

        def cell(c):
            if c["status"] == "exact":
                return str(c["distance"])
            lo = c.get("lower") or 0; up = c.get("upper")
            return f"{lo}-{up if up is not None else '?'}*"
        # overall distance = min of the two; exact only if both exact
        d_str = "?"
        if z["status"] == "exact" and x["status"] == "exact":
            d_val = min(z["distance"], x["distance"]); d_str = str(d_val)
            n_exact += 1
        secs = z.get("seconds", 0) + x.get("seconds", 0)
        print(f"{i:>3} {n:>4} {k:>3} {str(old):>9} {cell(z):>9} {cell(x):>9} "
              f"{d_str:>6} {secs:>6.1f}s")
        fout.write(json.dumps({"_id": oid, "n": n, "k": k, "old_bounds": old,
                               "z": z, "x": x, "d": d_str}) + "\n")
        fout.flush()
    fout.close()
    print(f"\nEXACT distances resolved: {n_exact}/{len(docs)}  ->  {OUT}")


if __name__ == "__main__":
    sys.exit(main())
