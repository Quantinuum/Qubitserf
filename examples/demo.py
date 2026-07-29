#!/usr/bin/env python3
"""codeaut demo -- run with:  PYTHONPATH=../python python demo.py  (or after `pip install .`)."""

import numpy as np

from qubitserf import codeaut
from qubitserf.codeaut import codes

print(f"codeaut {codeaut.version()}  |  BZ backends: {codeaut.available_backends()}\n")

# 1) classical code: the [7,3,4] simplex -> GL(3,2), order 168
G = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]], np.uint8)
print("simplex [7,3,4]      Aut(C) order:", codeaut.classical_automorphisms(G).order)

# 2) CSS codes via the method ladder (exact-or-raise: returns a permgroup.Group)
import time

for name in ("steane", "shor", "toric", "surface", "gross"):
    c = codes.toric(3) if name == "toric" else codes.surface(3) if name == "surface" \
        else codes.BUILTIN[name]()
    eff = codeaut.effective_dims(c.Hx, c.Hz)
    t0 = time.time()
    grp = codeaut.css_automorphisms(c)
    print(f"{name:8s} n={c.n:3d} eff_dim={eff['eff_dim']:2d}  "
          f"|Aut(Hx)∩Aut(Hz)|={grp.order():>6}  ({time.time() - t0:.3f}s)")

# 3) pick the engine explicitly on the gross [[144,12,12]] code
for method in ("auto", "bz"):
    t0 = time.time()
    grp = codeaut.css_automorphisms(codes.gross(), method=method)
    print(f"\ngross [[144,12,12]] method={method:10s}: order={grp.order()} "
          f"({time.time() - t0:.3f}s)")
