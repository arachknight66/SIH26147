"""GNU Radio sidecar used by every ordinary production import.

GNU Radio is deliberately invoked in its own Python runtime.  Distribution
packages often build its bindings for a different CPython version than this
application's virtual environment, so importing it directly here would make
the desktop application brittle.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import numpy as np

from .models import Diagnostic, MetadataStatus, MetadataValue, Severity, SignalRecording, SourceFormat
from .loaders import RawIQConfig


@dataclass(frozen=True)
class GNUradioRuntime:
    available: bool
    python_executable: str | None
    version: str | None
    reason: str


@dataclass(frozen=True)
class GNUradioPreprocessConfig:
    """Explicit File Source -> optional DSP -> File Sink flowgraph settings."""

    input_sample_rate_hz: float
    output_sample_rate_hz: float = 10_000.0
    frequency_shift_hz: float = 0.0
    lowpass_cutoff_hz: float | None = None

    def __post_init__(self) -> None:
        if self.input_sample_rate_hz <= 0 or self.output_sample_rate_hz <= 0:
            raise ValueError("GNU Radio input and output sample rates must be positive")
        if self.lowpass_cutoff_hz is not None and not 0 < self.lowpass_cutoff_hz < self.input_sample_rate_hz / 2:
            raise ValueError("GNU Radio low-pass cutoff must be between 0 and input Nyquist")


@lru_cache(maxsize=4)
def _discover_gnuradio_runtime(runtime_override: str | None) -> GNUradioRuntime:
    """Probe an interpreter once for a given explicit runtime override."""
    candidates = [runtime_override, "/usr/bin/python3", sys.executable]
    probe = "from gnuradio import gr; print(gr.version())"
    checked: set[str] = set()
    errors: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        try:
            result = subprocess.run([candidate, "-c", probe], capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{candidate}: {type(exc).__name__}")
            continue
        if result.returncode == 0:
            return GNUradioRuntime(True, candidate, result.stdout.strip() or None, "GNU Radio Python runtime available")
        errors.append(f"{candidate}: {result.stderr.strip() or 'import failed'}")
    return GNUradioRuntime(False, None, None, "; ".join(errors) or "GNU Radio runtime not found")


def gnuradio_runtime() -> GNUradioRuntime:
    """Find a GNU Radio runtime, caching the probe for this process.

    This probe starts an external Python interpreter. Caching it avoids that
    fixed cost for every file in a CLI directory batch or a GUI session while
    retaining support for an explicit ``SIH_GNURADIO_PYTHON`` override.
    """
    return _discover_gnuradio_runtime(os.environ.get("SIH_GNURADIO_PYTHON"))


def _preprocess_iq_file(path: Path, config: GNUradioPreprocessConfig, input_format: str) -> Path:
    """Run a supported raw GNU Radio source and return a temporary CF32 file.

    ``input_format`` is deliberately explicit.  GNU Radio does not infer byte
    layout, and all paths produce interleaved little-endian complex64 output.
    """
    runtime = gnuradio_runtime()
    if not runtime.available or runtime.python_executable is None:
        raise RuntimeError(f"GNU Radio preprocessing is required but unavailable: {runtime.reason}")
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    descriptor, output_name = tempfile.mkstemp(prefix="sih26147-gnuradio-", suffix=".cf32")
    os.close(descriptor)
    output = Path(output_name)
    runner = Path(__file__).with_name("gnuradio_runner.py")
    command = [
        runtime.python_executable, str(runner), "--input", str(source), "--output", str(output),
        "--input-rate", str(config.input_sample_rate_hz), "--output-rate", str(config.output_sample_rate_hz),
        "--frequency-shift", str(config.frequency_shift_hz), "--input-format", input_format,
    ]
    if config.lowpass_cutoff_hz is not None:
        command.extend(["--lowpass-cutoff", str(config.lowpass_cutoff_hz)])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"GNU Radio flowgraph failed: {result.stderr.strip() or result.stdout.strip()}")
    if output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("GNU Radio flowgraph produced an empty output stream")
    return output


def preprocess_complex64(path: Path, config: GNUradioPreprocessConfig) -> Path:
    """Run a little-endian complex64 GNU Radio File Source flowgraph."""
    return _preprocess_iq_file(path, config, "cf32_le")


def _effective_config(sample_rate: MetadataValue, config: GNUradioPreprocessConfig | None) -> tuple[GNUradioPreprocessConfig, bool]:
    """Resolve a flowgraph configuration without fabricating sample metadata."""
    rate_is_default = sample_rate.value is None
    effective_input_rate = 10_000.0 if rate_is_default else float(sample_rate.value)
    effective = config or GNUradioPreprocessConfig(effective_input_rate, effective_input_rate)
    if config is not None and abs(config.input_sample_rate_hz - effective_input_rate) > 1e-9:
        raise ValueError(
            "GNU Radio input sample rate must match the recording metadata or the 10 kS/s assumed default"
        )
    return effective, rate_is_default


def _output_sample_rate(
    original: MetadataValue, effective: GNUradioPreprocessConfig, rate_is_default: bool
) -> MetadataValue:
    if rate_is_default or effective.output_sample_rate_hz != effective.input_sample_rate_hz:
        return MetadataValue(
            effective.output_sample_rate_hz,
            "gnuradio_config" if not rate_is_default else "gnuradio_default_10ksps",
            MetadataStatus.ASSUMED,
            evidence="GNU Radio flowgraph configuration; not inferred from samples.",
        )
    return original


def _preprocess_diagnostic(effective: GNUradioPreprocessConfig) -> Diagnostic:
    return Diagnostic(
        Severity.INFO,
        "GNU_RADIO_PREPROCESSING",
        "Recording was processed by the configured GNU Radio File Source/preprocessing/File Sink graph.",
        f"input_rate={effective.input_sample_rate_hz}; output_rate={effective.output_sample_rate_hz}; "
        f"frequency_shift={effective.frequency_shift_hz}; lowpass_cutoff={effective.lowpass_cutoff_hz}",
    )


def preprocess_direct_raw_iq(
    path: Path, raw_config: RawIQConfig, config: GNUradioPreprocessConfig | None = None
) -> SignalRecording:
    """Run a GNU-Radio-compatible raw-IQ capture without an input staging copy.

    The direct path applies only to an already GNU-Radio-compatible layout:
    little-endian complex64 with I/Q order. Other raw representations retain
    the explicit loader conversion path, so byte layout is never guessed.
    """
    if not (
        raw_config.dtype in {"complex64", "int16"}
        and raw_config.iq_order.upper() == "IQ"
        and raw_config.endian == "little"
    ):
        raise ValueError("direct GNU Radio raw path requires little-endian complex64 or int16 IQ")
    source = Path(path).resolve()
    bytes_per_sample = np.dtype("<c8").itemsize if raw_config.dtype == "complex64" else 4
    if source.stat().st_size % bytes_per_sample:
        raise ValueError(f"raw {raw_config.dtype} source size is not a multiple of {bytes_per_sample} bytes")
    source_rate = MetadataValue(
        raw_config.sample_rate_hz,
        raw_config.sample_rate_source if raw_config.sample_rate_hz is not None else "user_input",
        raw_config.sample_rate_status if raw_config.sample_rate_hz is not None else MetadataStatus.MISSING,
    )
    effective, rate_is_default = _effective_config(source_rate, config)
    temporary_output = _preprocess_iq_file(
        source,
        effective,
        "cf32_le" if raw_config.dtype == "complex64" else "ci16_le",
    )
    try:
        output_samples = np.fromfile(temporary_output, dtype=np.complex64)
    finally:
        temporary_output.unlink(missing_ok=True)
    center_frequency = MetadataValue(
        raw_config.center_frequency_hz,
        "user_input",
        MetadataStatus.KNOWN if raw_config.center_frequency_hz is not None else MetadataStatus.MISSING,
    )
    return SignalRecording(
        samples=output_samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype=raw_config.dtype,
        semantic_type="complex_iq",
        sample_rate_hz=_output_sample_rate(source_rate, effective, rate_is_default),
        center_frequency_hz=center_frequency,
        provenance={
            "source_path": str(source),
            "file_size_bytes": source.stat().st_size,
            "loader": "GNUradioDirectRaw -> RawIQReader",
            "conversion_description": (
                f"little-endian {raw_config.dtype} IQ streamed directly into GNU Radio; "
                "no input staging copy"
            ),
            "gnuradio": {
                "input_sample_rate_hz": effective.input_sample_rate_hz,
                "output_sample_rate_hz": effective.output_sample_rate_hz,
                "frequency_shift_hz": effective.frequency_shift_hz,
                "lowpass_cutoff_hz": effective.lowpass_cutoff_hz,
            },
        },
        diagnostics=[_preprocess_diagnostic(effective)],
    )


def preprocess_recording(recording: SignalRecording, config: GNUradioPreprocessConfig | None = None) -> SignalRecording:
    """Route a recording through GNU Radio while preserving application semantics.

    GNU Radio receives a normalized complex64 stream.  A real or multichannel
    recording follows the same first-channel analysis convention already used
    by the receiver; its original semantic type remains unchanged so a
    passband/audio file is not relabelled as complex baseband.
    """
    effective, rate_is_default = _effective_config(recording.sample_rate_hz, config)
    input_samples = recording.samples if recording.samples.ndim == 1 else recording.samples[:, 0]
    input_samples = np.ascontiguousarray(input_samples, dtype=np.complex64)
    descriptor, input_name = tempfile.mkstemp(prefix="sih26147-gnuradio-input-", suffix=".cf32")
    os.close(descriptor)
    temporary_input = Path(input_name)
    try:
        input_samples.tofile(temporary_input)
        temporary_output = preprocess_complex64(temporary_input, effective)
        try:
            output_samples = np.fromfile(temporary_output, dtype=np.complex64)
        finally:
            temporary_output.unlink(missing_ok=True)
    finally:
        temporary_input.unlink(missing_ok=True)
    sample_rate = _output_sample_rate(recording.sample_rate_hz, effective, rate_is_default)
    return SignalRecording(
        samples=output_samples,
        source_format=recording.source_format,
        original_dtype=recording.original_dtype,
        semantic_type=recording.semantic_type,
        sample_rate_hz=sample_rate,
        center_frequency_hz=recording.center_frequency_hz,
        provenance={
            **recording.provenance,
            "loader": "GNUradioPreprocess -> " + str(recording.provenance.get("loader", "recording import")),
            "gnuradio": {
                "input_sample_rate_hz": effective.input_sample_rate_hz,
                "output_sample_rate_hz": effective.output_sample_rate_hz,
                "frequency_shift_hz": effective.frequency_shift_hz,
                "lowpass_cutoff_hz": effective.lowpass_cutoff_hz,
            },
        },
        diagnostics=[*recording.diagnostics, _preprocess_diagnostic(effective)],
    )
