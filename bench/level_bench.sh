#!/usr/bin/env bash
# Per-level GPU BZ kernel benchmark on the Gross [[144,12,12]] code (Z side).
# Reports the enumeration time for each weight level d (the profiler's per-level ms),
# which is the quantity the GPU kernel work optimizes. d=8 (~2s) is the fast iteration
# proxy; d=9 (~18s) the confirmation proxy; d=10 is the level that actually proves d=12.
#
# Usage:  bench/level_bench.sh [MAXW]   (default MAXW=8)
set -euo pipefail
cd "$(dirname "$0")/.."
MAXW="${1:-8}"
BIN="${BIN:-./build/qubitserf}"
BACKEND="${BACKEND:---gpu}"   # BACKEND=--cpu for the CPU kernel
DATA="${DATA:-bench/.cache}"

# Generate the Gross code's parity-check matrices once (bivariate bicycle, l=12 m=6,
# A = x^3+y+y^2, B = y^3+x+x^2; Hx=[A|B], Hz=[B^T|A^T]).
if [ ! -f "$DATA/gross_Hx.txt" ]; then
  mkdir -p "$DATA"
  python3 - "$DATA" <<'PY'
import sys, numpy as np
l, m = 12, 6
Sl = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
Sm = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
x = np.kron(Sl, np.eye(m, dtype=np.uint8))
y = np.kron(np.eye(l, dtype=np.uint8), Sm)
mp = lambda M, p: np.linalg.matrix_power(M.astype(np.int64), p) % 2
A = (mp(x, 3) + mp(y, 1) + mp(y, 2)) % 2
B = (mp(y, 3) + mp(x, 1) + mp(x, 2)) % 2
np.savetxt(sys.argv[1] + '/gross_Hx.txt', np.hstack([A, B]), fmt='%d')
np.savetxt(sys.argv[1] + '/gross_Hz.txt', np.hstack([B.T, A.T]), fmt='%d')
PY
fi

QUBITSERF_PROFILE=1 "$BIN" --hx "$DATA/gross_Hx.txt" --hz "$DATA/gross_Hz.txt" \
    --bz "$BACKEND" --z --max-weight "$MAXW" -v 2>&1 \
  | grep -E '^\[prof' \
  | sed -E 's/.*d=([0-9]+).*work=([0-9]+) -> ([0-9.]+) ms/d=\1  work=\2  \3 ms/'
