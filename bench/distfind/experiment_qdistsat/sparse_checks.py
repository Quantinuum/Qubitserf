"""Explore equivalent sparse check presentations on QDistSAT inputs."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


DATA = Path(__file__).resolve().parents[1] / "qdistsat_data"


def simplify(matrix, mode):
    words = [int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
             for row in matrix]
    order_mode = "integer"
    if ":" in mode:
        mode, order_mode = mode.split(":")
    if mode.startswith("augment"):
        bound = mode.removeprefix("augment")
        weights = [word.bit_count() for word in words]
        def acceptable(left, right):
            left_weight, right_weight = left.bit_count(), right.bit_count()
            if bound == "max":
                limit = max(left_weight, right_weight)
            elif bound == "min":
                limit = min(left_weight, right_weight)
            elif bound == "strict":
                limit = max(left_weight, right_weight) - 1
            elif bound == "mean":
                limit = (sum(weights) + len(weights) - 1) // len(weights)
            else:
                limit = int(bound)
            return (left ^ right).bit_count() <= limit
        pairs = [left ^ right
                 for index, left in enumerate(words) for right in words[:index]
                 if acceptable(left, right)]
        if order_mode == "append":
            words = list(dict.fromkeys(words + pairs))
        elif order_mode == "weight":
            words = sorted(set(words + pairs), key=lambda word: (word.bit_count(), word))
        else:
            words = sorted(set(words + pairs))
    elif mode == "sorted":
        words.sort()
    elif mode.startswith("reduce"):
        order = np.arange(len(words))
        random = np.random.default_rng(int(mode.removeprefix("reduce")))
        for _ in range(100):
            random.shuffle(order)
            changes = 0
            for index in order:
                current = words[index]
                best = min((current ^ other for other_index, other in enumerate(words)
                            if other_index != index), key=int.bit_count)
                if best.bit_count() < current.bit_count():
                    words[index] = best
                    changes += 1
            if not changes:
                break
    return np.asarray([[(word >> column) & 1 for column in range(matrix.shape[1])]
                       for word in words], dtype=np.uint8)


def worker(mode, case, which):
    from qubitserf.distfind import _native
    from qubitserf.codeaut.gf2 import rank_gf2

    original = [np.loadtxt(DATA / f"{case}_{side}.txt", dtype=np.uint8)
                for side in ("Hx", "Hz")]
    changed = [simplify(matrix, mode) for matrix in original]
    for before, after in zip(original, changed):
        assert rank_gf2(before) == rank_gf2(after) == rank_gf2(np.vstack([before, after]))
    print(json.dumps({"mode": mode, "shape": [matrix.shape for matrix in changed],
                      "weights": [dict(Counter(map(int, matrix.sum(axis=1)))) for matrix in changed],
                      "degrees": [dict(Counter(map(int, matrix.sum(axis=0)))) for matrix in changed]}), flush=True)
    _native.lib()
    start = time.perf_counter()
    result = _native.css_distance_raw(*changed, "cc", which, "cpu", 1, True)
    print(json.dumps({"mode": mode, "elapsed": time.perf_counter() - start,
                      "distance": result.distance, "proven": bool(result.proven)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", default="original")
    parser.add_argument("--case", default="TN_250_10_15")
    parser.add_argument("--which", default="M")
    parser.add_argument("--seconds", type=float, default=10)
    options = parser.parse_args()
    if options.worker:
        worker(options.mode, options.case, options.which)
        return
    for mode in options.mode.split(","):
        command = [sys.executable, __file__, "--worker", "--mode", mode,
                   "--case", options.case, "--which", options.which]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=options.seconds, env=os.environ)
            print(result.stdout, flush=True)
            print(result.stderr[-1600:], flush=True)
        except subprocess.TimeoutExpired as error:
            print(mode, "TIMEOUT", error.stdout, error.stderr[-1600:], flush=True)


if __name__ == "__main__":
    main()
