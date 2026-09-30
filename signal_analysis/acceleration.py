"""Compute-backend discovery and explicit GPU execution policy.

CUDA is optional.  A requested ``gpu`` backend never silently falls back to
CPU; ``auto`` uses CUDA only for stages with a validated implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComputeBackend(Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class GPUBackendUnavailableError(RuntimeError):
    """Raised when a caller explicitly requires GPU DSP before it exists."""


@dataclass(frozen=True)
class GPUCapability:
    available: bool
    device_name: str | None
    reason: str


def detect_gpu_capability() -> GPUCapability:
    """Probe CUDA through CuPy without making it a package requirement."""
    try:
        import cupy  # type: ignore[import-not-found]
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        if device_count < 1:
            return GPUCapability(False, None, "CuPy found no CUDA devices")
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        name = properties["name"].decode() if isinstance(properties["name"], bytes) else str(properties["name"])
        return GPUCapability(
            True,
            name,
            "CUDA device available; spectral, receiver decisions, K=7 Viterbi, "
            "and correlation kernels are available. RS correction remains CPU-native.",
        )
    except Exception as exc:
        return GPUCapability(False, None, f"CUDA/CuPy unavailable: {type(exc).__name__}: {exc}")


def require_supported_backend(requested: str | ComputeBackend = ComputeBackend.CPU) -> GPUCapability:
    """Validate a backend choice without silently falling back from GPU."""
    try:
        backend = requested if isinstance(requested, ComputeBackend) else ComputeBackend(str(requested).lower())
    except ValueError as exc:
        raise ValueError("compute backend must be cpu, gpu, or auto") from exc
    capability = detect_gpu_capability()
    if backend is ComputeBackend.GPU and not capability.available:
        raise GPUBackendUnavailableError(f"GPU execution was requested but is unavailable: {capability.reason}")
    return capability


def should_use_gpu(requested: str | ComputeBackend = ComputeBackend.CPU) -> bool:
    """Return whether an optional CUDA stage should execute on GPU.

    ``auto`` is deliberately conservative when CUDA cannot be probed.  An
    explicit GPU request is validated by :func:`require_supported_backend`.
    """
    try:
        backend = requested if isinstance(requested, ComputeBackend) else ComputeBackend(str(requested).lower())
    except ValueError as exc:
        raise ValueError("compute backend must be cpu, gpu, or auto") from exc
    capability = require_supported_backend(backend)
    return backend is ComputeBackend.GPU or (backend is ComputeBackend.AUTO and capability.available)
