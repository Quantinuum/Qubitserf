"""Shared pytest fixtures/helpers for the qubitserf test suite.

Run from the repo root with:

    PYTHONPATH=python /opt/miniconda3/envs/sage_env/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import pytest

import qubitserf as df


# Public backends that are actually usable on this machine (e.g. ['cpu', 'gpu']).
AVAILABLE_BACKENDS = df.available_backends()
HAS_GPU = "gpu" in AVAILABLE_BACKENDS


# A reusable marker so the whole GPU block skips cleanly when no accelerator is available.
requires_gpu = pytest.mark.skipif(
    not HAS_GPU, reason="gpu backend not available on this machine"
)


def best_of(fn, *, tries: int = 1):
    """Call ``fn`` up to ``tries`` times and return (values_seen, best_min).

    The GPU backend is (as of this build) *non-deterministic* on larger codes:
    it occasionally reports a distance below the true value.  A true minimum
    distance is a lower bound that the search can only over-estimate when it
    misses the optimal codeword, but here the native kernel can also report a
    *too-small* value, so we collect every value seen across retries and let the
    caller decide how to interpret them.  Returns the set of distances seen.
    """
    seen = []
    for _ in range(max(1, tries)):
        seen.append(fn())
    return seen
