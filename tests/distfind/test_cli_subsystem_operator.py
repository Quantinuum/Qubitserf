"""CLI coverage for the new --operator (operator weight) and --subsystem
(dressed distance) flags. Driven via subprocess so stdout parseability is
exercised end-to-end.

Run from the repo root:
    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from qubitserf.distfind import codes


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY_DIR = os.path.join(_PKG_DIR, "python")


def _run(args, *, stdin=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = _PY_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "qubitserf.distfind", *args],
        input=stdin, capture_output=True, text=True, env=env,
    )


def _write_matrix(tmp_path, name, M):
    p = tmp_path / name
    np.savetxt(p, np.asarray(M, dtype=np.uint8), fmt="%d")
    return str(p)


# --------------------------------------------------------------------------- #
# --operator: operator weight from a Pauli string.
# --------------------------------------------------------------------------- #
def test_cli_operator_max():
    r = _run(["--builtin", "steane", "--operator", "ZZZZZZZ"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "3"


def test_cli_operator_zx():
    r = _run(["--builtin", "steane", "--operator", "YYYYYYY", "--zx"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "3 3"


def test_cli_operator_stabilizer_zero():
    # The first Steane Z-stabilizer (Hamming row 0) as a Z-only operator -> 0.
    Hx, Hz = codes.steane()
    row = Hz[0]
    s = "".join("Z" if b else "I" for b in row)
    r = _run(["--builtin", "steane", "--operator", s, "--zx"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "0 0"


def test_cli_operator_json():
    r = _run(["--builtin", "steane", "--operator", "ZZZZZZZ", "--json"])
    assert r.returncode == 0, r.stderr
    obj = json.loads(r.stdout)
    assert obj["z_weight"] == 3 and obj["x_weight"] == 0 and obj["weight"] == 3
    assert obj["proven"] is True


# --------------------------------------------------------------------------- #
# --subsystem: dressed distance from gauge generators.
# --------------------------------------------------------------------------- #
def test_cli_subsystem_bacon_shor(tmp_path):
    Gx, Gz = codes.bacon_shor(3)
    hx = _write_matrix(tmp_path, "gx.txt", Gx)
    hz = _write_matrix(tmp_path, "gz.txt", Gz)
    r = _run(["--hx", hx, "--hz", hz, "--subsystem"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "3"


def test_cli_subsystem_zx(tmp_path):
    Gx, Gz = codes.bacon_shor(3)
    hx = _write_matrix(tmp_path, "gx.txt", Gx)
    hz = _write_matrix(tmp_path, "gz.txt", Gz)
    r = _run(["--hx", hx, "--hz", hz, "--subsystem", "--zx"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "3 3"


def test_cli_subsystem_rejected_for_classical(tmp_path):
    H = _write_matrix(tmp_path, "h.txt", codes.repetition_parity(5))
    r = _run(["--classical", H, "--subsystem"])
    assert r.returncode == 1
    assert "CSS" in r.stderr
