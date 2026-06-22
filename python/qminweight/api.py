"""High-level distance-finding API."""
from __future__ import annotations
from dataclasses import dataclass
from . import _native


@dataclass
class Result:
    distance: int           # best upper bound found (the distance, when proven)
    lower_bound: int        # proven lower bound (== distance when proven)
    proven: bool
    levels: int
    seconds: float
    backend: str

    @classmethod
    def _from(cls, r):
        backend = r.backend.decode()
        if backend in ("metal", "cuda"):
            backend = "gpu"
        return cls(
            distance=r.distance,
            lower_bound=r.lower_bound,
            proven=bool(r.proven),
            levels=r.levels,
            seconds=r.seconds,
            backend=backend,
        )


_WHICH = {"min": "M", "m": "M", "z": "Z", "x": "X"}
_BACKENDS = {"auto", "cpu", "gpu"}


def _normalize_backend(backend) -> str:
    b = "auto" if backend is None else str(backend).lower()
    if b not in _BACKENDS:
        raise ValueError("backend must be one of auto/cpu/gpu")
    if b == "gpu" and not _native.backend_available("gpu"):
        raise RuntimeError("no GPU backend is available on this machine")
    return b


def css_distance(Hx, Hz, *, method="bz", which="min", backend="auto",
                 threads=0, max_weight=0, verbose=False) -> Result:
    """Exact distance of a CSS code given X- and Z-check matrices.

    method:  "bz" (Brouwer-Zimmermann), "cc" (connected cluster), or "mitm".
    which:   "min" (= min(dX,dZ)), "z", or "x".
    backend: "auto", "cpu", or "gpu". "gpu" chooses the available accelerator
             for this machine.
    """
    w = _WHICH.get(str(which).lower())
    if w is None:
        raise ValueError("which must be one of min/z/x")
    backend = _normalize_backend(backend)
    r = _native.css_distance_raw(Hx, Hz, method, w, backend, threads, max_weight, verbose)
    return Result._from(r)


def classical_distance(H, *, method="bz", backend="auto",
                       threads=0, max_weight=0, verbose=False) -> Result:
    """Minimum distance of a classical linear code from its parity-check matrix H."""
    backend = _normalize_backend(backend)
    r = _native.classical_distance_raw(H, method, backend, threads, max_weight, verbose)
    return Result._from(r)


def available_backends():
    """Return the list of usable backends."""
    out = ["cpu"]
    if _native.backend_available("gpu"):
        out.append("gpu")
    return out


def version() -> str:
    return _native.version()
