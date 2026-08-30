from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from app.exceptions import InvalidEndianError, InvalidSampleCountError, UnsupportedIQDatatypeError
from app.models.metadata import Diagnostic, DiagnosticSeverity, MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import Endian, IQOrder, SignalRecording, SourceFormat

_DTYPES = {"complex64": "c8", "float32": "f4", "int8": "i1", "int16": "i2", "uint8": "u1"}

@dataclass(frozen=True)
class RawIQConfig:
    dtype: str
    iq_order: IQOrder = IQOrder.IQ
    endian: Endian = Endian.LITTLE
    sample_rate_hz: float | None = None
    center_frequency_hz: float | None = None
    compute_hash: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES: raise UnsupportedIQDatatypeError(f"Unsupported IQ datatype '{self.dtype}'. Supported: {', '.join(_DTYPES)}.")
        if self.endian not in (Endian.LITTLE, Endian.BIG): raise InvalidEndianError("Endian must be little or big.")
        if self.iq_order not in (IQOrder.IQ, IQOrder.QI): raise ValueError("Raw IQ order must be IQ or QI.")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0: raise ValueError("Sample rate must be positive when provided.")
        if self.center_frequency_hz is not None and self.center_frequency_hz < 0: raise ValueError("Center frequency cannot be negative when provided.")

def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

class RawIQReader:
    def __init__(self, path: str | Path, config: RawIQConfig):
        self.path, self.config = Path(path), config
        if not self.path.is_file(): raise FileNotFoundError(f"Raw IQ file does not exist: {self.path}")
        if self.path.stat().st_size == 0: raise InvalidSampleCountError("Raw IQ file is empty; no samples can be interpreted.")
        self._scalar_dtype = np.dtype(("<" if config.endian == Endian.LITTLE else ">") + _DTYPES[config.dtype])
        factor = 1 if config.dtype == "complex64" else 2
        if self.path.stat().st_size % (self._scalar_dtype.itemsize * factor):
            raise InvalidSampleCountError(f"Invalid raw-IQ interpretation: {config.dtype} requires complete {'complex' if factor == 1 else 'I/Q scalar pairs'}; file size is incompatible. Possible causes: wrong datatype or truncated recording.")

    @property
    def sample_count(self) -> int:
        return self.path.stat().st_size // self._scalar_dtype.itemsize // (1 if self.config.dtype == "complex64" else 2)

    def read_chunk(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > self.sample_count: raise ValueError("Chunk lies outside the recording.")
        offset = start * (1 if self.config.dtype == "complex64" else 2)
        items = count * (1 if self.config.dtype == "complex64" else 2)
        raw = np.memmap(self.path, dtype=self._scalar_dtype, mode="r", offset=offset * self._scalar_dtype.itemsize, shape=(items,))
        if self.config.dtype == "complex64": return np.asarray(raw, dtype=np.complex64)
        pairs = np.asarray(raw).reshape(-1, 2).astype(np.float32)
        if self.config.iq_order == IQOrder.QI: pairs = pairs[:, ::-1]
        return (pairs[:, 0] + 1j * pairs[:, 1]).astype(np.complex64)

    def read(self) -> SignalRecording:
        samples = self.read_chunk(0, self.sample_count)
        source = MetadataSource.USER_INPUT
        sr = MetadataValue(self.config.sample_rate_hz, source, MetadataStatus.ASSUMED, 1.0, "Explicit user-provided value") if self.config.sample_rate_hz is not None else MetadataValue.unknown("Raw IQ contains no intrinsic absolute sampling-rate information.")
        cf = MetadataValue(self.config.center_frequency_hz, source, MetadataStatus.ASSUMED, 1.0, "Explicit user-provided value") if self.config.center_frequency_hz is not None else MetadataValue.unknown("Raw IQ contains no intrinsic RF center-frequency information.")
        diagnostics = [Diagnostic(DiagnosticSeverity.WARNING, "MISSING_SAMPLE_RATE", "Absolute sample rate is unavailable.", sr.evidence)] if sr.value is None else []
        if cf.value is None: diagnostics.append(Diagnostic(DiagnosticSeverity.WARNING, "MISSING_CENTER_FREQUENCY", "Center frequency is unavailable.", cf.evidence))
        non_finite = int(np.size(samples) - np.count_nonzero(np.isfinite(samples)))
        if non_finite:
            diagnostics.append(Diagnostic(DiagnosticSeverity.ERROR, "NON_FINITE_SAMPLES", f"Recording contains {non_finite} NaN or infinite complex samples.", "Correct the input capture or apply an explicit sanitization step before analysis."))
        return SignalRecording(samples=samples, source_format=SourceFormat.RAW_IQ, original_dtype=self.config.dtype, channels=2, semantic_type="complex_iq", iq_order=self.config.iq_order, endian=self.config.endian, sample_rate_hz=sr, center_frequency_hz=cf, diagnostics=diagnostics, provenance={"source_path": str(self.path), "file_size": self.path.stat().st_size, "sha256": _file_hash(self.path) if self.config.compute_hash else None, "loader": "RawIQReader", "conversion": "interleaved scalar I/Q converted to complex64 without amplitude scaling", "input_configuration": {"dtype": self.config.dtype, "iq_order": self.config.iq_order.value, "endian": self.config.endian.value, "sample_rate_hz": self.config.sample_rate_hz, "center_frequency_hz": self.config.center_frequency_hz}})
