from __future__ import annotations

import struct
import wave

import numpy as np
import pytest

from signal_analysis.jobs import AnalysisJob, JobState
from signal_analysis.loaders import RawIQConfig
from signal_analysis.native import new_cancellation_token, new_progress_state, require_native
from signal_analysis.sources import (
    AnalysisMode,
    analyze_recording_source,
    open_raw_iq_source,
    open_sigmf_source,
    open_wav_source,
)


def _pack_s24(values: list[int], endian: str = "little") -> bytes:
    packed = bytearray()
    for value in values:
        unsigned = value & 0xFFFFFF
        raw = unsigned.to_bytes(3, byteorder=endian, signed=False)
        packed.extend(raw)
    return bytes(packed)


@pytest.mark.parametrize(
    ("dtype", "endian", "payload", "expected"),
    [
        ("int8", "little", bytes([64, 192, 128, 0]), [0.5 - 0.5j, -1 + 0j]),
        ("uint8", "little", bytes([128, 255, 0, 128]), [127j / 128, -1 + 0j]),
        ("int16", "little", struct.pack("<hhhh", 16384, -16384, -32768, 0), [0.5 - 0.5j, -1 + 0j]),
        ("int16", "big", struct.pack(">hhhh", 16384, -16384, -32768, 0), [0.5 - 0.5j, -1 + 0j]),
        ("int24", "little", _pack_s24([4194304, -4194304, -8388608, 0]), [0.5 - 0.5j, -1 + 0j]),
        ("int32", "big", struct.pack(">iiii", 1073741824, -1073741824, -2147483648, 0), [0.5 - 0.5j, -1 + 0j]),
        ("float32", "little", struct.pack("<ffff", 0.5, -0.5, -1.0, 0.0), [0.5 - 0.5j, -1 + 0j]),
        ("complex64", "little", struct.pack("<ffff", 0.5, -0.5, -1.0, 0.0), [0.5 - 0.5j, -1 + 0j]),
    ],
)
def test_raw_source_normalizes_formats(tmp_path, dtype, endian, payload, expected):
    path = tmp_path / "capture.iq"
    path.write_bytes(payload)
    source = open_raw_iq_source(path, RawIQConfig(dtype, "IQ", endian))
    np.testing.assert_allclose(source.read_chunk(0, 20), np.asarray(expected, np.complex64))
    assert source.description.sample_rate_hz is None
    assert source.description.sample_rate_source == "missing"
    assert source.description.sample_rate_status == "MISSING"
    assert source.description.center_frequency_status == "MISSING"
    assert source.description.amplitude_normalized is (dtype not in {"float32", "complex64"})


def test_raw_qi_order_and_strict_size(tmp_path):
    path = tmp_path / "capture.iq"
    path.write_bytes(struct.pack("<hhhh", 16384, -16384, 0, -32768))
    source = open_raw_iq_source(path, RawIQConfig("int16", "QI", "little", 2e6))
    np.testing.assert_allclose(source.read_chunk(0, 2), [-0.5 + 0.5j, -1 + 0j])
    assert source.description.sample_rate_hz == 2e6

    malformed = tmp_path / "bad.iq"
    malformed.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="multiple"):
        open_raw_iq_source(malformed, RawIQConfig("int16", "IQ", "little"))


def test_source_result_owns_storage_after_source_lifetime(tmp_path):
    path = tmp_path / "capture.iq"
    path.write_bytes(bytes([128, 255, 0, 128]))
    source = open_raw_iq_source(path, RawIQConfig("uint8", "IQ", "little"))
    chunk = source.read_chunk_with_metadata(0, 2)
    assert chunk.start_sample == 0
    assert chunk.end_sample_exclusive == 2
    assert chunk.source_sample_count == 2
    samples = chunk.samples
    del source
    np.testing.assert_allclose(samples, [127j / 128, -1 + 0j])
    assert not samples.flags.owndata


def _write_pcm_wav(path, width: int, values: bytes, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(48_000)
        output.writeframes(values)


@pytest.mark.parametrize(
    ("width", "payload", "expected"),
    [
        (1, bytes([0, 128, 255]), [-1.0, 0.0, 127 / 128]),
        (2, struct.pack("<hhh", -32768, 0, 32767), [-1.0, 0.0, 32767 / 32768]),
        (3, _pack_s24([-8388608, 0, 8388607]), [-1.0, 0.0, 8388607 / 8388608]),
        (4, struct.pack("<iii", -2147483648, 0, 2147483647), [-1.0, 0.0, 2147483647 / 2147483648]),
    ],
)
def test_wav_pcm_formats_are_streamed_and_normalized(tmp_path, width, payload, expected):
    path = tmp_path / f"pcm{width}.wav"
    _write_pcm_wav(path, width, payload)
    source = open_wav_source(path)
    np.testing.assert_allclose(source.read_chunk(0, 20).real, expected, atol=1e-7)
    assert source.description.sample_rate_hz == 48_000
    assert source.description.semantic_type == "mono_real"


def test_float_wav_and_explicit_stereo_interpretation(tmp_path):
    float_data = struct.pack("<fff", -0.75, 0.0, 0.5)
    fmt = struct.pack("<HHIIHH", 3, 1, 48_000, 192_000, 4, 32)
    riff_size = 4 + 8 + len(fmt) + 8 + len(float_data)
    float_path = tmp_path / "float.wav"
    float_path.write_bytes(
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(float_data)) + float_data
    )
    np.testing.assert_allclose(open_wav_source(float_path).read_chunk(0, 3).real, [-0.75, 0, 0.5])

    stereo_path = tmp_path / "stereo.wav"
    _write_pcm_wav(stereo_path, 2, struct.pack("<hhhh", 16384, -16384, 0, 32767), channels=2)
    with pytest.raises(ValueError, match="explicit"):
        open_wav_source(stereo_path)
    iq = open_wav_source(stereo_path, "stereo_iq")
    np.testing.assert_allclose(iq.read_chunk(0, 2), [0.5 - 0.5j, 32767j / 32768])


def test_malformed_wav_is_rejected_before_chunk_processing(tmp_path):
    path = tmp_path / "truncated.wav"
    fmt = struct.pack("<HHIIHH", 1, 1, 48_000, 96_000, 2, 16)
    path.write_bytes(
        b"RIFF" + struct.pack("<I", 100) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", 64) + b"\x00\x00"
    )
    with pytest.raises(ValueError, match="extends beyond"):
        open_wav_source(path)


def test_sigmf_preserves_missing_rate_and_uses_native_data_source(tmp_path):
    meta = tmp_path / "recording.sigmf-meta"
    data = tmp_path / "recording.sigmf-data"
    meta.write_text('{"global":{"core:datatype":"ci16_le"},"captures":[]}', encoding="utf-8")
    data.write_bytes(struct.pack("<hhhh", 16384, 0, 0, -16384))
    source = open_sigmf_source(meta)
    assert source.description.source_format == "SIGMF"
    assert source.description.sample_rate_hz is None
    np.testing.assert_allclose(source.read_chunk(0, 2), [0.5 + 0j, -0.5j])


def _native_spectrum(samples: np.ndarray, partitions: list[int]):
    native = require_native()
    config = native.SpectralConfig()
    config.fft_size = 256
    config.hop_size = 96
    config.complex_input = True
    config.sample_rate_hz = None
    config.max_stft_frames = 100
    analyzer = native.SpectralAnalyzer(config)
    offset = 0
    for length in partitions:
        analyzer.update(samples[offset : offset + length])
        offset += length
    assert offset == len(samples)
    analyzer.finish()
    return analyzer.result()


def test_spectral_chunk_equivalence_axes_energy_and_band_detection():
    native = require_native()
    index = np.arange(4096)
    samples = np.exp(2j * np.pi * 0.125 * index).astype(np.complex64)
    whole = _native_spectrum(samples, [len(samples)])
    chunked = _native_spectrum(samples, [1, 17, 503, 1024, 2551])
    np.testing.assert_array_equal(chunked.frequencies, whole.frequencies)
    np.testing.assert_allclose(chunked.psd, whole.psd, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(chunked.stft_power, whole.stft_power, rtol=1e-6, atol=1e-8)
    assert chunked.frequency_unit == "cycles/sample"
    assert chunked.time_unit == "samples"
    assert chunked.frequencies[np.argmax(chunked.psd)] == pytest.approx(0.125)
    assert np.sum(chunked.psd) / 256 == pytest.approx(1.0, abs=0.02)
    bands = native.detect_spectral_bands(chunked.frequencies, chunked.psd, 8.0, 2)
    assert any(abs(band.center_frequency - 0.125) < 0.01 for band in bands)

    invalid = native.SpectralConfig()
    invalid.schema_version = 999
    with pytest.raises(ValueError, match="schema version"):
        native.SpectralAnalyzer(invalid)

    strict = native.SpectralAnalyzer(native.SpectralConfig())
    with pytest.raises(TypeError):
        strict.update(samples.astype(np.complex128))
    with pytest.raises(ValueError, match="C-contiguous"):
        strict.update(samples[::2])


def test_preprocessing_state_is_chunk_invariant():
    native = require_native()
    config = native.PreprocessingConfig()
    config.remove_dc = True
    config.enable_agc = True
    config.mix_frequency_cycles_per_sample = 0.125
    config.fir_taps = [0.25, 0.5, 0.25]
    config.resample_up = 3
    config.resample_down = 2
    samples = (
        np.exp(2j * np.pi * 0.125 * np.arange(2000)) + (0.2 + 0.1j)
    ).astype(np.complex64)

    whole_session = native.PreprocessorSession(config)
    whole = np.concatenate([whole_session.process(samples).samples, whole_session.finish().samples])
    chunked_session = native.PreprocessorSession(config)
    pieces = [chunked_session.process(part).samples for part in np.split(samples, [1, 9, 333, 901])]
    pieces.append(chunked_session.finish().samples)
    np.testing.assert_allclose(np.concatenate(pieces), whole, rtol=1e-6, atol=2e-6)


def test_streaming_statistics_and_float_clipping_semantics(tmp_path):
    path = tmp_path / "float.iq"
    values = np.array([0.5 + 2j, -0.5 - 2j], np.complex64)
    path.write_bytes(values.tobytes())
    source = open_raw_iq_source(path, RawIQConfig("complex64", "IQ", "little"))
    result = analyze_recording_source(source, mode=AnalysisMode.FULL, fft_size=16)
    assert result.statistics.mean_i == pytest.approx(0.0)
    assert result.statistics.mean_q == pytest.approx(0.0)
    assert result.statistics.rms_amplitude == pytest.approx(np.sqrt(4.25))
    assert not result.statistics.clipping_available

    native = require_native()
    accumulator = native.StatisticsAccumulator()
    accumulator.update(np.array([1 + 0j, 0 + 0j, -1j, 0 + 0j], np.complex64))
    statistics = accumulator.result()
    assert statistics.clipping_available
    assert statistics.clipping_fraction == pytest.approx(0.5)


def test_resampler_preserves_in_band_tone_and_rejects_alias_band():
    native = require_native()
    config = native.PreprocessingConfig()
    config.resample_up = 1
    config.resample_down = 2

    def resample(frequency):
        samples = np.exp(2j * np.pi * frequency * np.arange(4096)).astype(np.complex64)
        session = native.PreprocessorSession(config)
        return np.concatenate([session.process(samples).samples, session.finish().samples])

    in_band = resample(0.1)
    rejected = resample(0.4)
    stable = in_band[100:-100]
    phase_step = np.angle(np.mean(stable[1:] * np.conj(stable[:-1]))) / (2 * np.pi)
    assert phase_step == pytest.approx(0.2, abs=2e-3)
    assert np.sqrt(np.mean(np.abs(stable) ** 2)) == pytest.approx(1.0, abs=0.03)
    assert np.sqrt(np.mean(np.abs(rejected[100:-100]) ** 2)) < 0.03


def test_preview_of_sparse_one_gib_source_has_bounded_coverage(tmp_path):
    path = tmp_path / "large.iq"
    with path.open("wb") as output:
        output.seek((1 << 30) - 1)
        output.write(b"\x00")
    source = open_raw_iq_source(path, RawIQConfig("uint8", "IQ", "little"))
    progress = new_progress_state()
    result = analyze_recording_source(
        source,
        mode=AnalysisMode.PREVIEW,
        preview_samples=65_536,
        chunk_samples=4096,
        fft_size=256,
        max_stft_frames=8,
        progress=progress,
    )
    assert source.description.sample_count == (1 << 29)
    assert result.coverage.processed_input_samples == 65_536
    assert not result.coverage.complete
    assert result.stft_power.shape == (8, 256)
    assert result.dropped_stft_frames > 0
    assert progress.snapshot == (65_536, 65_536)

    job = AnalysisJob(
        lambda cancellation, job_progress: analyze_recording_source(
            source,
            mode=AnalysisMode.FULL,
            chunk_samples=4096,
            fft_size=256,
            max_stft_frames=1,
            cancellation=cancellation,
            progress=job_progress,
        )
    ).start()
    job.cancel()
    cancelled = job.result(timeout=5)
    assert cancelled.execution_status.name == "CANCELLED"
    assert job.snapshot().state is JobState.CANCELLED
    assert cancelled.coverage.processed_input_samples < source.description.sample_count


def test_full_source_mode_and_cancellation_report_coverage(tmp_path):
    path = tmp_path / "capture.iq"
    pairs = np.tile(np.array([64, -64], np.int8), 10_000)
    path.write_bytes(pairs.tobytes())
    source = open_raw_iq_source(path, RawIQConfig("int8", "IQ", "little"))
    full = analyze_recording_source(
        source, mode=AnalysisMode.FULL, chunk_samples=333, fft_size=256, max_stft_frames=2
    )
    assert full.coverage.complete
    assert full.execution_status.name == "COMPLETED"
    assert full.coverage.processed_input_samples == 10_000

    region = analyze_recording_source(
        source,
        mode=AnalysisMode.FULL,
        start_sample=100,
        sample_count=500,
        chunk_samples=73,
        fft_size=64,
    )
    assert region.coverage.first_sample == 100
    assert region.coverage.end_sample_exclusive == 600
    assert not region.coverage.complete
    assert region.times[0] >= 100

    cancellation = new_cancellation_token()
    cancellation.cancel()
    cancelled = analyze_recording_source(
        source, mode=AnalysisMode.FULL, chunk_samples=333, fft_size=256, cancellation=cancellation
    )
    assert not cancelled.coverage.complete
    assert cancelled.coverage.processed_input_samples == 0
    assert cancelled.execution_status.name == "CANCELLED"


def test_real_mixing_requires_explicit_band_and_filter(tmp_path):
    path = tmp_path / "real.wav"
    _write_pcm_wav(path, 2, struct.pack("<" + "h" * 100, *range(100)))
    source = open_wav_source(path)
    native = require_native()
    config = native.PreprocessingConfig()
    config.mix_frequency_cycles_per_sample = 0.1
    with pytest.raises(ValueError, match="explicit selected band"):
        analyze_recording_source(source, fft_size=64, preprocessing=config)
    config.fir_taps = native.design_lowpass_fir(0.1, 33)
    selected = analyze_recording_source(
        source,
        fft_size=64,
        preprocessing=config,
        real_passband=(0.05, 0.15),
    )
    assert selected.coverage.processed_input_samples == 100
