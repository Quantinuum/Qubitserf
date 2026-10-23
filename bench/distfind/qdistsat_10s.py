"""Benchmark 13 pinned QDistSAT CSS codes with an exact native-call deadline.

Run from the package root: python bench/distfind/qdistsat_10s.py --output results
Repeat --case to select names or shell-style patterns. Input validation and
library loading precede the timer; all native preprocessing remains inside it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("qdistsat_data")
DATA_COMMIT = "6f110fadfffd6904c950f5da904ddc495176060a"
CASES = (
    "BB_108_8_10", "BB_144_12_12", "LP_238_44_6", "LP_340_56_8",
    "LP_442_68_10", "TN_108_2_12", "TN_200_10_10", "GB_144_12_12",
    "LP_544_80_12", "TN_144_2_13", "BB_144_14_14", "BB_288_12_18",
    "TN_250_10_15",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
}


def deadline_clock():
    """Use a suspend-inclusive wall clock on Linux and macOS."""
    for name in ("CLOCK_BOOTTIME", "CLOCK_MONOTONIC_RAW"):
        if hasattr(time, name):
            return time.clock_gettime(getattr(time, name))
    return time.monotonic()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cases(patterns):
    if not patterns:
        return list(CASES)
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(case, pattern) for case in CASES):
            raise ValueError(f"case pattern matches no inputs: {pattern}")
    return [case for case in CASES
            if any(fnmatch.fnmatchcase(case, pattern) for pattern in patterns)]


def verify_inputs(cases):
    manifest = json.loads((DATA / "input_manifest.json").read_text())
    selected = {}
    for case in cases:
        for side in ("Hx", "Hz"):
            name = f"{case}_{side}.txt"
            actual = sha256(DATA / name)
            if actual != manifest[name]["sha256"]:
                raise ValueError(f"input hash mismatch: {name}")
            selected[name] = manifest[name]
    return selected


def load_code(case):
    import numpy as np
    from qubitserf.codeaut.gf2 import rank_gf2

    checks = [np.loadtxt(DATA / f"{case}_{side}.txt", dtype=np.int64, ndmin=2)
              for side in ("Hx", "Hz")]
    _, n_text, k_text, _ = case.split("_")
    n_qubits, n_logicals = int(n_text), int(k_text)
    if any(matrix.shape[1] != n_qubits or not np.isin(matrix, [0, 1]).all()
           for matrix in checks):
        raise ValueError(f"invalid binary check matrices for {case}")
    hx, hz = [np.ascontiguousarray(matrix, dtype=np.uint8) for matrix in checks]
    ranks = [int(rank_gf2(matrix)) for matrix in (hx, hz)]
    if np.any((hx @ hz.T) & 1) or n_qubits - sum(ranks) != n_logicals:
        raise ValueError(f"CSS commutation or dimension mismatch for {case}")
    return hx, hz, {"ranks": ranks, "shapes": [list(hx.shape), list(hz.shape)]}


def worker(args):
    verify_inputs([args.worker])
    from qubitserf.distfind import _native

    hx, hz, evidence = load_code(args.worker)
    native = _native.lib()
    evidence["native_version"] = native.distfind_version().decode()
    print(json.dumps({"ready": True, "case": args.worker, **evidence}), flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise RuntimeError("missing parent start signal")
    cpu_start = time.process_time()
    wall_start = deadline_clock()
    result = _native.css_distance_raw(hx, hz, args.method, "M", "cpu", 1, True)
    elapsed = deadline_clock() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    print(json.dumps({
        "distance": result.distance, "lower_bound": result.lower_bound,
        "proven": bool(result.proven), "levels": result.levels,
        "native_seconds": result.seconds, "call_wall_seconds": elapsed,
        "call_cpu_seconds": cpu_elapsed, "backend": result.backend.decode(),
    }), flush=True)


def read_event(child, timeout, output):
    """Bound even partial-line startup output and retain it for diagnostics."""
    deadline = deadline_clock() + timeout
    message = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - deadline_clock()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("worker event deadline exceeded")
            chunk = os.read(child.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("worker exited before emitting a complete event")
            output.write(chunk)
            output.flush()
            message.extend(chunk)
            if len(message) > 65536:
                raise ValueError("worker event exceeds 64 KiB")
            if b"\n" in message:
                return json.loads(message.decode())


def classify_report(report, expected_distance, timeout):
    elapsed = report["call_wall_seconds"]
    if not math.isfinite(elapsed) or elapsed < 0:
        return "error", "invalid measured call duration"
    if elapsed > timeout:
        return "timeout", "native call exceeded the requested wall-time budget"
    if not report["proven"]:
        return "error", "native result is not an exact proof"
    if report["distance"] != expected_distance:
        return "error", f"expected distance {expected_distance}, got {report['distance']}"
    if report["backend"] != "cc" and not report["backend"].startswith("cpu"):
        return "error", f"unexpected backend: {report['backend']}"
    return "completed", None


def run_case(case, command, env, output, timeout, startup_timeout, repetition=1):
    stem = case if repetition == 1 else f"{case}.repeat{repetition}"
    record = {"case": case, "repetition": repetition,
              "expected_distance": int(case.rsplit("_", 1)[1]),
              "stdout": f"{stem}.stdout", "stderr": f"{stem}.stderr"}
    child = None
    phase = "startup"
    with (output / record["stdout"]).open("wb") as stdout, \
            (output / record["stderr"]).open("wb") as stderr:
        try:
            child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
            ready = read_event(child, startup_timeout, stdout)
            if not ready.get("ready") or ready.get("case") != case:
                raise ValueError("invalid worker readiness event")
            record["input"] = ready
            phase = "call"
            watchdog_start = deadline_clock()
            child.stdin.write(b"GO\n")
            child.stdin.close()
            report = read_event(child, max(0, timeout - (deadline_clock() - watchdog_start)), stdout)
            record["observed_wall_seconds"] = deadline_clock() - watchdog_start
            record.update(report)
            status, error = classify_report(report, record["expected_distance"], timeout)
            record["status"] = status
            if error:
                record["error"] = error
            phase = "shutdown"
            child.wait(timeout=5)
            if child.returncode:
                record.update(status="error", error=f"worker exited with status {child.returncode}")
        except TimeoutError as error:
            record.update(status="timeout" if phase == "call" else "error",
                          error=str(error), phase=phase)
            if phase == "call":
                record["observed_wall_seconds"] = deadline_clock() - watchdog_start
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError,
                subprocess.TimeoutExpired) as error:
            record.update(status="error", error=str(error), phase=phase)
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait()
                record["returncode"] = child.returncode
                if child.stdin and not child.stdin.closed:
                    child.stdin.close()
                if child.stdout:
                    stdout.write(child.stdout.read())
                    child.stdout.close()
    return record


def command_output(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def metadata(args, inputs):
    source_paths = sorted(path for directory in (ROOT / "cpp", ROOT / "python")
                          for path in directory.rglob("*")
                          if path.suffix in {".cpp", ".hpp", ".h", ".cu", ".metal", ".py"})
    source_paths += [ROOT / "CMakeLists.txt", ROOT / "build.sh", Path(__file__).resolve()]
    build = {}
    cache = ROOT / "build/CMakeCache.txt"
    if cache.is_file():
        build["cmake_cache_sha256"] = sha256(cache)
        for line in cache.read_text().splitlines():
            if line.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CXX_COMPILER:",
                                "CMAKE_CXX_FLAGS", "QUBITSERF_", "DISTFIND_")) and "=" in line:
                key, value = line.split("=", 1)
                build[key] = value
    diff = command_output(["git", "diff", "HEAD", "--", "cpp", "python", "CMakeLists.txt"])
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    elif Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "qubitserf_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "build": build, "library": str(args.library), "library_sha256": sha256(args.library),
        "qdist_sat_commit": DATA_COMMIT, "inputs": inputs,
        "input_manifest_sha256": sha256(DATA / "input_manifest.json"),
        "method": args.method, "which": "M", "backend": "cpu", "threads": 1,
        "thread_environment": THREAD_ENV, "timeout_seconds": args.timeout,
        "startup_timeout_seconds": args.startup_timeout,
        "timeout_excludes_input_and_library_loading": True,
        "timeout_includes_native_preprocessing": True,
        "watchdog_starts_at_parent_go_signal": True,
        "platform": platform.platform(), "cpu": cpu, "python": sys.version,
        "python_executable": sys.executable, "repetitions": args.repeat,
    }


def save_results(output, info, results):
    payload = {"metadata": info, "results": results,
               "summary": {status: sum(record["status"] == status for record in results)
                           for status in ("completed", "timeout", "error")}}
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output / "results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="case name or quoted glob; repeatable")
    parser.add_argument("--method", choices=("cc", "bz"), default="cc")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--startup-timeout", type=float, default=30)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("timeout", "startup_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.worker:
        worker(args)
        return 0
    try:
        cases = select_cases(args.case)
        inputs = verify_inputs(cases)
        if args.library is None:
            override = next((os.environ[name] for name in
                             ("QUBITSERF_LIB", "DISTFIND_LIB", "CODEAUT_LIB_PATH")
                             if os.environ.get(name)), None)
            library_name = "libqubitserf.dylib" if sys.platform == "darwin" else "libqubitserf.so"
            args.library = Path(override) if override else ROOT / "python/qubitserf/_lib" / library_name
        args.library = args.library.resolve(strict=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        args.output = Path(__file__).with_name("qdistsat_results") / f"{stamp}-{args.method}"
    args.output = args.output.resolve()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **THREAD_ENV, PYTHONPATH=str(ROOT / "python"),
               QUBITSERF_LIB=str(args.library))
    info = metadata(args, inputs)
    results = []
    save_results(args.output, info, results)
    for repetition in range(1, args.repeat + 1):
        for case in cases:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", case,
                       "--method", args.method]
            record = run_case(case, command, env, args.output, args.timeout,
                              args.startup_timeout, repetition)
            if sha256(args.library) != info["library_sha256"]:
                record.update(status="error", error="native library changed during benchmark")
            results.append(record)
            save_results(args.output, info, results)
            print(json.dumps(record), flush=True)
    print(f"Results: {args.output / 'results.json'}", flush=True)
    return int(any(record["status"] == "error" for record in results))


if __name__ == "__main__":
    raise SystemExit(main())
