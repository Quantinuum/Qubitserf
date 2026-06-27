"""One-code BZ distance worker (run as a subprocess so the driver can hard-timeout it).

    python qecdb_bz_worker.py HX.npy HZ.npy {z|x|min} [max_weight]

Prints one JSON line to stdout on success; streams the per-weight-level verbose
trace (`[bz ...] d=.. upper=.. lower=..`) to stderr so the driver can still
recover an improved lower/upper bound if the process is killed on timeout.
"""
import sys, json, time
import numpy as np
import qminweight as df

hx, hz, which = sys.argv[1], sys.argv[2], sys.argv[3]
maxw = int(sys.argv[4]) if len(sys.argv) > 4 else 0
Hx = np.load(hx); Hz = np.load(hz)
t0 = time.perf_counter()
r = df.css_distance(Hx, Hz, method="bz", which=which, backend="gpu",
                    verbose=True, max_weight=maxw)
print(json.dumps({"distance": r.distance, "lower_bound": r.lower_bound,
                  "proven": bool(r.proven), "seconds": time.perf_counter() - t0,
                  "backend": r.backend}))
