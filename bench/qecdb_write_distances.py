"""Write computed distances for the previously-unknown CSS codes back to qumba.

Sources the n=89 results from bench/qecdb_unknown_bz_results.jsonl (GPU BZ, each
component proven) and recomputes the 4 bivariate-bicycle codes' dz/dx via `cc`.
For each code it sets d / dx / dz and collapses d_lower_bound = d_upper_bound = d.

Safety:
  * only touches the exact _id list it computed, and only where the stored d is
    still null (never clobbers a value someone else set);
  * --dry-run (default) just reports the match/skip counts per target DB;
  * --apply performs the $set. Targets are chosen with --targets local,live.

    python qecdb_write_distances.py --dry-run --targets local,live
    python qecdb_write_distances.py --apply   --targets local,live
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pymongo
from bson import ObjectId

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import qminweight as df  # noqa: E402

LOCAL = os.environ.get("QECDB_LOCAL_URI", "mongodb://localhost:27017")
# Live droplet readWrite URI is a secret: provide it via QECDB_LIVE_URI (never hardcode).
# e.g. export QECDB_LIVE_URI="$(grep -o 'mongodb://serban[^ ]*' secrets/qec_db/secrets.local.md | head -1)"
LIVE = os.environ.get("QECDB_LIVE_URI", "")
JSONL = os.path.join(HERE, "qecdb_unknown_bz_results.jsonl")


def bb_components():
    """Recompute dz/dx for the 4 bivariate-bicycle codes via cc (from sparse split)."""
    col = pymongo.MongoClient(LOCAL)["qumba"]["codes"]
    out = []
    for doc in col.find({"css": True, "d": None, "n": {"$in": [108, 126, 144]}}):
        rows = doc["H"].split(); n = doc["n"]
        Hx, Hz = [], []
        for r in rows:
            xv = [1 if c in "XY" else 0 for c in r]; zv = [1 if c in "ZY" else 0 for c in r]
            (Hx if any(xv) and not any(zv) else Hz).append(xv if any(xv) and not any(zv) else zv)
        Hx, Hz = np.array(Hx, np.uint8), np.array(Hz, np.uint8)
        dz = df.css_distance(Hx, Hz, method="cc", which="z").distance
        dx = df.css_distance(Hx, Hz, method="cc", which="x").distance
        out.append({"_id": str(doc["_id"]), "dz": dz, "dx": dx, "d": min(dz, dx)})
    return out


def updates():
    ups = []
    for line in open(JSONL):
        r = json.loads(line)
        if r["z"]["status"] != "exact" or r["x"]["status"] != "exact":
            continue
        dz, dx = r["z"]["distance"], r["x"]["distance"]
        ups.append({"_id": r["_id"], "dz": dz, "dx": dx, "d": min(dz, dx)})
    ups += bb_components()
    return ups


def run(uri, label, ups, apply):
    cl = pymongo.MongoClient(uri, serverSelectionTimeoutMS=8000)
    col = cl["qumba"]["codes"]
    matched = skip_set = skip_missing = wrote = 0
    for u in ups:
        oid = ObjectId(u["_id"])
        doc = col.find_one({"_id": oid}, {"d": 1})
        if doc is None:
            skip_missing += 1; continue
        if doc.get("d") is not None:
            skip_set += 1; continue
        matched += 1
        if apply:
            res = col.update_one(
                {"_id": oid, "d": None},
                {"$set": {"d": int(u["d"]), "dx": int(u["dx"]), "dz": int(u["dz"]),
                          "d_lower_bound": int(u["d"]), "d_upper_bound": int(u["d"])}})
            wrote += res.modified_count
    cl.close()
    print(f"[{label}] eligible(d=null & present)={matched}  already-set={skip_set}  "
          f"missing={skip_missing}  written={wrote if apply else '(dry-run)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--targets", default="local")
    args = ap.parse_args()
    ups = updates()
    print(f"prepared {len(ups)} distance updates from {os.path.basename(JSONL)} + cc(BB)")
    for t in args.targets.split(","):
        uri, label = (LOCAL, "local") if t.strip() == "local" else (LIVE, "live")
        if not uri:
            print(f"[{label}] skipped: set QECDB_LIVE_URI (no hardcoded secret)")
            continue
        try:
            run(uri, label, ups, args.apply)
        except Exception as e:
            print(f"[{label}] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
