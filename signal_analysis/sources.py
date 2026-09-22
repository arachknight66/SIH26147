"""Bounded native recording sources and spectral analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .loaders import RawIQConfig
from .native import require_native

DEFAULT_CHUNK_SAMPLES = 262_144
DEFAULT_PREVIEW_SAMPLES = 2_097_152


class AnalysisMode(Enum):
    PREVIEW = "preview"
    FULL = "full"


class SourceAnalysisStatus(Enum):
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class SourceDescription:
    path: Path
    source_format: str
    sample_count: int
    channel_count: int
    scalar_format: str
    semantic_type: str
    sample_rate_hz: float | None
    sample_rate_source: str
    sample_rate_status: str
    center_frequency_hz: float | None
    center_frequency_source: str
    center_frequency_status: str
    amplitude_normalized: bool
    conversion: str


@dataclass(frozen=True)
class SourceChunk:
    samples: np.ndarray
    start_sample: int
    source_sample_count: int

    @property
    def end_sample_exclusive(self) -> int:
        return self.start_sample + int(self.samples.size)


@dataclass(frozen=True)
class ProcessedCoverage:
    mode: AnalysisMode
    source_samples: int
    processed_input_samples: int
    first_sample: int
    end_sample_exclusive: int
    complete: bool


@dataclass(frozen=True)
class SourceSpectralResult:
    execution_status: SourceAnalysisStatus
    frequencies: np.ndarray
    psd: np.ndarray
    times: np.ndarray
    stft_power: np.ndarray
    frequency_unit: str
    time_unit: str
    power_unit: str
    welch_segments: int
    dropped_stft_frames: int
    coverage: ProcessedCoverage
    statistics: Any


class RecordingSource:
    """Python ownership wrapper around a native chunk source."""

    def __init__(self, native_source: Any, description: SourceDescription) -> None:
        self._native = native_source
        self.description = description

    def read_chunk(self, start: int, count: int) -> np.ndarray:
        return self.read_chunk_with_metadata(start, count).samples

    def read_chunk_with_metadata(self, start: int, count: int) -> SourceChunk:
        if start < 0 or count < 0:
            raise ValueError("chunk start and count must be nonnegative")
        chunk = self._native.read_chunk(start, count)
        return SourceChunk(
            samples=chunk.samples,
            start_sample=int(chunk.start_sample),
            source_sample_count=int(chunk.source_sample_count),
        )

    def iter_chunks(
        self,
        *,
        start: int = 0,
        count: int | None = None,
        chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
    ) -> Iterator[np.ndarray]:
        if start < 0 or start > self.description.sample_count:
            raise ValueError("start is outside the recording")
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        available = self.description.sample_count - start
        if count is not None and count < 0:
            raise ValueError("count must be nonnegative")
        remaining = available if count is None else min(available, count)
        offset = start
        while remaining:
            chunk = self.read_chunk(offset, min(chunk_samples, remaining))
            if chunk.size == 0:
                raise RuntimeError("native source made no progress before end of recording")
            yield chunk
            offset += int(chunk.size)
            remaining -= int(chunk.size)


def _raw_scalar_format(dtype: str) -> Any:
    native = require_native()
    mapping = {
        "complex64": native.ScalarFormat.COMPLEX_F32,
        "float32": native.ScalarFormat.F32,
        "int8": native.ScalarFormat.S8,
        "uint8": native.ScalarFormat.U8,
        "int16": native.ScalarFormat.S16,
        "int24": native.ScalarFormat.S24,
        "int32": native.ScalarFormat.S32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported raw IQ dtype: {dtype}") from exc


def open_raw_iq_source(path: str | Path, config: RawIQConfig) -> RecordingSource:
    native = require_native()
    if config.iq_order not in {"IQ", "QI"}:
        raise ValueError("raw IQ order must be IQ or QI")
    if config.endian not in {"little", "big"}:
        raise ValueError("raw IQ byte order must be little or big")
    if config.center_frequency_hz is not None and not np.isfinite(config.center_frequency_hz):
        raise ValueError("center frequency must be finite when supplied")
    native_source = native.RawIQSource(
        str(path),
        _raw_scalar_format(config.dtype),
        native.IQOrder.IQ if config.iq_order == "IQ" else native.IQOrder.QI,
        native.ByteOrder.LITTLE if config.endian == "little" else native.ByteOrder.BIG,
        config.sample_rate_hz,
    )
    info = native_source.info
    return RecordingSource(
        native_source,
        SourceDescription(
            path=Path(path),
            source_format="RAW_IQ",
            sample_count=int(info.sample_count),
            channel_count=int(info.channel_count),
            scalar_format=info.scalar_format.name,
            semantic_type="complex_iq",
            sample_rate_hz=info.sample_rate_hz,
            sample_rate_source="user_input" if info.sample_rate_hz is not None else "missing",
            sample_rate_status="KNOWN" if info.sample_rate_hz is not None else "MISSING",
            center_frequency_hz=config.center_frequency_hz,
            center_frequency_source=(
                "user_input" if config.center_frequency_hz is not None else "missing"
            ),
            center_frequency_status=(
                "KNOWN" if config.center_frequency_hz is not None else "MISSING"
            ),
            amplitude_normalized=bool(info.normalized),
            conversion=info.conversion,
        ),
    )


def open_wav_source(path: str | Path, mode: str | None = None) -> RecordingSource:
    native = require_native()
    modes = {
        "mono": native.WavChannelMode.MONO,
        "channel_0": native.WavChannelMode.CHANNEL_0,
        "channel_1": native.WavChannelMode.CHANNEL_1,
        "stereo_iq": native.WavChannelMode.STEREO_IQ,
    }
    if mode is None:
        # Try the unambiguous mono interpretation. Stereo is rejected by native validation.
        selected = native.WavChannelMode.MONO
    else:
        try:
            selected = modes[mode]
        except KeyError as exc:
            raise ValueError(f"unsupported WAV channel interpretation: {mode}") from exc
    native_source = native.WavSource(str(path), selected)
    info = native_source.info
    semantic_type = "complex_iq" if info.complex_samples else "mono_real"
    return RecordingSource(
        native_source,
        SourceDescription(
            path=Path(path),
            source_format="WAV",
            sample_count=int(info.sample_count),
            channel_count=int(info.channel_count),
            scalar_format=info.scalar_format.name,
            semantic_type=semantic_type,
            sample_rate_hz=float(info.sample_rate_hz),
            sample_rate_source="wav_header",
            sample_rate_status="KNOWN",
            center_frequency_hz=None,
            center_frequency_source="missing",
            center_frequency_status="MISSING",
            amplitude_normalized=bool(info.normalized),
            conversion=info.conversion,
        ),
    )


def open_sigmf_source(meta_path: str | Path) -> RecordingSource:
    meta_path = Path(meta_path)
    if not str(meta_path).endswith(".sigmf-meta"):
        raise ValueError("SigMF metadata path must end with .sigmf-meta")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    global_meta = metadata.get("global", {})
    datatype = global_meta.get("core:datatype")
    mappings = {
        "cf32_le": ("float32", "little"),
        "cf32_be": ("float32", "big"),
        "ci16_le": ("int16", "little"),
        "ci16_be": ("int16", "big"),
        "ci8": ("int8", "little"),
        "ci8_le": ("int8", "little"),
        "cu8": ("uint8", "little"),
        "cu8_le": ("uint8", "little"),
    }
    if datatype not in mappings:
        raise ValueError(f"unsupported SigMF datatype: {datatype}")
    dtype, endian = mappings[datatype]
    sample_rate = global_meta.get("core:sample_rate")
    captures = metadata.get("captures", [])
    center_frequency = captures[0].get("core:frequency") if captures else None
    data_path = Path(str(meta_path).replace(".sigmf-meta", ".sigmf-data"))
    source = open_raw_iq_source(
        data_path,
        RawIQConfig(
            dtype,
            "IQ",
            endian,
            float(sample_rate) if sample_rate is not None else None,
            float(center_frequency) if center_frequency is not None else None,
        ),
    )
    return RecordingSource(
        source._native,
        replace(
            source.description,
            path=meta_path,
            source_format="SIGMF",
            sample_rate_source="sigmf-meta" if sample_rate is not None else "missing",
            center_frequency_source=(
                "sigmf-meta" if center_frequency is not None else "missing"
            ),
        ),
    )


def analyze_recording_source(
    source: RecordingSource,
    *,
    mode: AnalysisMode = AnalysisMode.PREVIEW,
    fft_size: int = 1024,
    hop_size: int | None = None,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
    preview_samples: int = DEFAULT_PREVIEW_SAMPLES,
    start_sample: int = 0,
    sample_count: int | None = None,
    max_stft_frames: int = 512,
    preprocessing: Any | None = None,
    real_passband: tuple[float, float] | None = None,
    cancellation: Any | None = None,
    progress: Any | None = None,
) -> SourceSpectralResult:
    """Analyze a bounded preview or stream the complete source through native sessions."""
    native = require_native()
    if start_sample < 0 or start_sample > source.description.sample_count:
        raise ValueError("start_sample is outside the recording")
    available = source.description.sample_count - start_sample
    if sample_count is not None:
        if sample_count < 0:
            raise ValueError("sample_count must be nonnegative")
        available = min(available, sample_count)
    if mode is AnalysisMode.PREVIEW:
        requested = min(available, preview_samples)
    else:
        requested = available
    if requested < 0 or preview_samples <= 0:
        raise ValueError("preview sample budget must be positive")
    if preprocessing is not None and source.description.semantic_type != "complex_iq":
        if preprocessing.mix_frequency_cycles_per_sample != 0.0:
            if real_passband is None or len(preprocessing.fir_taps) < 2:
                raise ValueError(
                    "mixing a real passband recording requires an explicit selected band "
                    "and a nontrivial band-selection FIR"
                )
    spectral_config = native.SpectralConfig()
    spectral_config.fft_size = fft_size
    spectral_config.hop_size = hop_size or fft_size // 2
    spectral_config.complex_input = source.description.semantic_type == "complex_iq"
    effective_sample_rate = source.description.sample_rate_hz
    output_rate_ratio = 1.0
    if effective_sample_rate is not None and preprocessing is not None:
        output_rate_ratio = preprocessing.resample_up / preprocessing.resample_down
        effective_sample_rate *= output_rate_ratio
    elif preprocessing is not None:
        output_rate_ratio = preprocessing.resample_up / preprocessing.resample_down
    spectral_config.sample_rate_hz = effective_sample_rate
    spectral_config.max_stft_frames = max_stft_frames
    analyzer = native.SpectralAnalyzer(spectral_config)
    scalar_format_name = source.description.scalar_format
    if scalar_format_name in {"F32", "COMPLEX_F32"}:
        clipping_level = float("inf")
    elif scalar_format_name in {"S8", "U8"}:
        clipping_level = 127 / 128
    else:
        clipping_level = 0.999
    statistics = native.StatisticsAccumulator(clipping_level)
    preprocessor = native.PreprocessorSession(preprocessing) if preprocessing is not None else None
    consumed = 0
    for chunk in source.iter_chunks(
        start=start_sample, count=requested, chunk_samples=chunk_samples
    ):
        if cancellation is not None and cancellation.is_cancelled:
            break
        statistics.update(chunk)
        processed = preprocessor.process(chunk).samples if preprocessor is not None else chunk
        analyzer.update(processed)
        consumed += int(chunk.size)
        if progress is not None:
            progress.update(consumed, requested)
    if preprocessor is not None and (cancellation is None or not cancellation.is_cancelled):
        tail = preprocessor.finish().samples
        if tail.size:
            analyzer.update(tail)
    analyzer.finish(include_partial=True)
    result = analyzer.result()
    time_offset = (
        (start_sample * output_rate_ratio) / effective_sample_rate
        if effective_sample_rate is not None
        else float(start_sample) * output_rate_ratio
    )
    absolute_times = result.times + time_offset
    complete = consumed == requested
    return SourceSpectralResult(
        execution_status=(
            SourceAnalysisStatus.COMPLETED
            if complete
            else SourceAnalysisStatus.CANCELLED
        ),
        frequencies=result.frequencies,
        psd=result.psd,
        times=absolute_times,
        stft_power=result.stft_power,
        frequency_unit=result.frequency_unit,
        time_unit=result.time_unit,
        power_unit=result.power_unit,
        welch_segments=int(result.welch_segments),
        dropped_stft_frames=int(result.dropped_stft_frames),
        coverage=ProcessedCoverage(
            mode=mode,
            source_samples=source.description.sample_count,
            processed_input_samples=consumed,
            first_sample=start_sample,
            end_sample_exclusive=start_sample + consumed,
            complete=(
                complete
                and start_sample == 0
                and requested == source.description.sample_count
            ),
        ),
        statistics=statistics.result(),
    )


__all__ = [
    "AnalysisMode",
    "ProcessedCoverage",
    "RecordingSource",
    "SourceDescription",
    "SourceChunk",
    "SourceAnalysisStatus",
    "SourceSpectralResult",
    "analyze_recording_source",
    "open_raw_iq_source",
    "open_sigmf_source",
    "open_wav_source",
]
