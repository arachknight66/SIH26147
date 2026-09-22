"""Explicit adapter for the optional SIH26147 native compute extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from signal_analysis import _native
except ImportError as exc:
    _native = None
    _NATIVE_IMPORT_ERROR: ImportError | None = exc
else:
    _NATIVE_IMPORT_ERROR = None


class NativeUnavailableError(RuntimeError):
    """Raised when native execution was requested but the extension is unavailable."""


@dataclass(frozen=True)
class NativeRuntimeInfo:
    available: bool
    version: str | None
    api_version: int | None
    import_error: str | None


def runtime_info() -> NativeRuntimeInfo:
    """Return availability without activating a Python compute fallback."""
    if _native is None:
        return NativeRuntimeInfo(False, None, None, str(_NATIVE_IMPORT_ERROR))
    return NativeRuntimeInfo(True, str(_native.__version__), int(_native.API_VERSION), None)


def require_native() -> Any:
    if _native is None:
        raise NativeUnavailableError(
            "The SIH26147 native extension is not installed. Build/install the project "
            "before requesting native execution."
        ) from _NATIVE_IMPORT_ERROR
    return _native


def new_cancellation_token() -> Any:
    return require_native().CancellationToken()


def new_progress_state() -> Any:
    return require_native().ProgressState()


def decode_viterbi_k7_r12(
    llrs: np.ndarray,
    *,
    cancellation: Any | None = None,
    progress: Any | None = None,
) -> Any:
    """Run the native fixed-code pilot; input must already be 1-D C float32."""
    if not isinstance(llrs, np.ndarray):
        raise TypeError("llrs must be a NumPy array")
    if llrs.dtype != np.float32:
        raise TypeError("llrs must have dtype float32; implicit conversion is disabled")
    if llrs.ndim != 1:
        raise ValueError("llrs must be one-dimensional")
    if not llrs.flags.c_contiguous:
        raise ValueError("llrs must be C-contiguous; implicit copies are disabled")
    return require_native().decode_viterbi_k7_r12(llrs, cancellation, progress)


def analyze_window(samples: np.ndarray, *, config: Any | None = None) -> Any:
    """Run native parameter estimation and ranked modulation analysis."""
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy array")
    if samples.dtype != np.complex64:
        raise TypeError("samples must have dtype complex64; implicit conversion is disabled")
    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if not samples.flags.c_contiguous:
        raise ValueError("samples must be C-contiguous; implicit copies are disabled")
    native = require_native()
    return native.analyze_window(samples, config or native.AnalysisConfig())


def demodulate(samples: np.ndarray, *, config: Any) -> Any:
    """Run the native receiver with strict zero-copy input requirements."""
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy array")
    if samples.dtype != np.complex64:
        raise TypeError("samples must have dtype complex64; implicit conversion is disabled")
    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if not samples.flags.c_contiguous:
        raise ValueError("samples must be C-contiguous; implicit copies are disabled")
    return require_native().demodulate(samples, config)


def _bit_array(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8; implicit conversion is disabled")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be one-dimensional and C-contiguous")
    return value


def _llr_array(value: np.ndarray, name: str = "llrs") -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32; implicit conversion is disabled")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be one-dimensional and C-contiguous")
    return value


def deinterleave(bits: np.ndarray, llrs: np.ndarray, *, config: Any) -> Any:
    return require_native().deinterleave(_bit_array(bits, "bits"), _llr_array(llrs), config)


def interleave_bits(bits: np.ndarray, *, config: Any) -> np.ndarray:
    return np.asarray(require_native().interleave_bits(_bit_array(bits, "bits"), config), dtype=np.uint8)


def interleave_llrs(llrs: np.ndarray, *, config: Any) -> np.ndarray:
    return np.asarray(require_native().interleave_llrs(_llr_array(llrs), config), dtype=np.float32)


def decode_reed_solomon(bytes_: np.ndarray, *, config: Any) -> Any:
    return require_native().decode_reed_solomon(_bit_array(bytes_, "bytes"), config)


def encode_reed_solomon(message: np.ndarray, *, config: Any) -> np.ndarray:
    return np.asarray(require_native().encode_reed_solomon(_bit_array(message, "message"), config), dtype=np.uint8)


def decode_ldpc(llrs: np.ndarray, *, matrix: Any, config: Any | None = None) -> Any:
    native = require_native()
    return native.decode_ldpc(_llr_array(llrs), matrix, config or native.LdpcConfig())


__all__ = [
    "NativeRuntimeInfo",
    "NativeUnavailableError",
    "decode_viterbi_k7_r12",
    "analyze_window",
    "demodulate",
    "deinterleave",
    "interleave_bits",
    "interleave_llrs",
    "decode_reed_solomon",
    "encode_reed_solomon",
    "decode_ldpc",
    "new_cancellation_token",
    "new_progress_state",
    "require_native",
    "runtime_info",
]
