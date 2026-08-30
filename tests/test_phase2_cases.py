from __future__ import annotations
import json
import wave
from pathlib import Path
import numpy as np
import pytest
import scipy.signal as signal
from app.analysis.analyzer import AnalysisConfig, analyze_signal
from app.io.loader import load_signal
from app.io.raw_iq import RawIQConfig, RawIQReader
from app.io.wav import WavReader
from app.models.metadata import DiagnosticSeverity, MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import Endian, IQOrder, SignalRecording, SourceFormat

# -------------------------------------------------------------
# TEST A — Pure Complex Sinusoid
# -------------------------------------------------------------
def test_a_pure_complex_sinusoid():
    fs = 2_000_000.0  # 2 MHz
    f0 = 250_000.0   # +250 kHz (+0.125 normalized)
    amp = 3.5
    phase = np.pi / 4
    n_samples = 4096
    t = np.arange(n_samples, dtype=np.float32) / fs

    samples = (amp * np.exp(1j * (2 * np.pi * f0 * t + phase))).astype(np.complex64)
    rec = SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(fs, MetadataSource.USER_INPUT, MetadataStatus.KNOWN, 1.0, "Explicit"),
    )

    analysis = analyze_signal(rec, AnalysisConfig(fft_size=4096, window="rectangular"))
    assert analysis.spectrum is not None
    
    # Peak frequency in Hz
    peak_idx = int(np.argmax(analysis.spectrum.magnitude_spectrum))
    peak_f_hz = analysis.spectrum.frequencies[peak_idx]
    assert np.isclose(peak_f_hz, f0, atol=1000.0)
    
    # Peak amplitude in coherent normalization
    assert np.isclose(analysis.spectrum.magnitude_spectrum[peak_idx], amp, atol=0.05)
    
    # Time domain RMS and mean amplitude
    assert np.isclose(analysis.time_statistics.rms_amplitude, amp, atol=0.05)
    assert np.isclose(analysis.time_statistics.peak_amplitude, amp, atol=0.05)

# -------------------------------------------------------------
# TEST B — Two-Tone Signal
# -------------------------------------------------------------
def test_b_two_tone_signal():
    fs = 1_000_000.0
    f1 = -200_000.0  # -0.2 normalized
    f2 = +300_000.0  # +0.3 normalized
    a1 = 2.0
    a2 = 1.0
    n = 8192
    t = np.arange(n) / fs

    np.random.seed(42)
    noise = (np.random.normal(0, 0.02, n) + 1j * np.random.normal(0, 0.02, n)).astype(np.complex64)
    samples = (a1 * np.exp(2j * np.pi * f1 * t) + a2 * np.exp(2j * np.pi * f2 * t)).astype(np.complex64) + noise
    rec = SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(fs, MetadataSource.USER_INPUT, MetadataStatus.KNOWN, 1.0, "Explicit"),
    )

    analysis = analyze_signal(rec, AnalysisConfig(detection_threshold_db=10.0))
    # Should detect 2 candidate spectral regions
    spectral_regs = [r for r in analysis.detected_regions if r.method == "spectral_energy_threshold"]
    assert len(spectral_regs) == 2
    regs = sorted(spectral_regs, key=lambda r: r.center_freq_normalized)
    assert np.isclose(regs[0].center_freq_normalized, -0.2, atol=0.01)
    assert np.isclose(regs[1].center_freq_normalized, +0.3, atol=0.01)
    # Peak power of tone 1 should be approx 6 dB higher than tone 2 (2^2 / 1^2 = 4 = 6.02 dB)
    p_diff = regs[0].peak_power_db - regs[1].peak_power_db
    assert np.isclose(p_diff, 6.02, atol=1.0)

# -------------------------------------------------------------
# TEST C — Known-Bandwidth Signal
# -------------------------------------------------------------
def test_c_known_bandwidth_signal():
    np.random.seed(42)
    n = 16384
    noise = (np.random.normal(0, 1.0, n) + 1j * np.random.normal(0, 1.0, n)).astype(np.complex64)
    # Bandpass filter from -0.1 to +0.1 => total width = 0.20 (cutoff = 0.10 / 0.5 = 0.20 of Nyquist)
    b = signal.firwin(127, 0.20)
    filt = signal.lfilter(b, [1.0], noise).astype(np.complex64)

    rec = SignalRecording(
        samples=filt,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )

    analysis = analyze_signal(rec)
    bw_99 = next(e for e in analysis.bandwidth_candidates if e.method == "power_containment_99pct")
    assert bw_99.occupied_bandwidth_normalized is not None
    assert 0.18 <= bw_99.occupied_bandwidth_normalized <= 0.25

# -------------------------------------------------------------
# TEST D — AWGN + SNR Estimator
# -------------------------------------------------------------
@pytest.mark.parametrize("snr_target", [0.0, 5.0, 10.0, 20.0])
def test_d_awgn_snr(snr_target):
    np.random.seed(100 + int(snr_target))
    n = 32768
    t = np.arange(n)
    sig = np.exp(2j * np.pi * 0.1 * t).astype(np.complex64)
    n_power = 1.0 / (10.0 ** (snr_target / 10.0))
    noise = (np.random.normal(0, np.sqrt(n_power / 2), n) + 1j * np.random.normal(0, np.sqrt(n_power / 2), n)).astype(np.complex64)
    x = sig + noise

    rec = SignalRecording(
        samples=x,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )

    analysis = analyze_signal(rec)
    snr_est = next(s for s in analysis.snr_candidates if s.method == "spectral_noise_floor")
    assert snr_est.snr_db is not None
    assert abs(snr_est.snr_db - snr_target) < 1.0

# -------------------------------------------------------------
# TEST E — Burst Signal
# -------------------------------------------------------------
def test_e_burst_signal():
    np.random.seed(42)
    n = 4096
    noise = (np.random.normal(0, 0.05, n) + 1j * np.random.normal(0, 0.05, n)).astype(np.complex64)
    burst = 2.0 * np.exp(2j * np.pi * 0.1 * np.arange(2048)).astype(np.complex64)
    noise[1024:3072] += burst

    rec = SignalRecording(
        samples=noise,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )

    analysis = analyze_signal(rec)
    burst_regs = [r for r in analysis.detected_regions if r.start_sample is not None]
    assert len(burst_regs) >= 1
    assert abs(burst_regs[0].start_sample - 1024) < 100
    assert abs(burst_regs[0].end_sample - 3072) < 100
    assert analysis.activity_metrics is not None
    assert analysis.activity_metrics.burst_count >= 1
    assert 0.40 < analysis.activity_metrics.duty_cycle < 0.60

# -------------------------------------------------------------
# TEST F — Frequency-Shifted Signal
# -------------------------------------------------------------
def test_f_frequency_shifted_signal():
    n = 8192
    t = np.arange(n)
    for target_fn in [-0.35, -0.1, 0.05, 0.25]:
        x = np.exp(2j * np.pi * target_fn * t).astype(np.complex64)
        rec = SignalRecording(
            samples=x,
            source_format=SourceFormat.RAW_IQ,
            original_dtype="complex64",
            channels=2,
            semantic_type="complex_iq",
        )
        analysis = analyze_signal(rec)
        assert analysis.frequency_candidates[0].normalized_frequency is not None
        assert abs(analysis.frequency_candidates[0].normalized_frequency - target_fn) < 1e-4

# -------------------------------------------------------------
# TEST G — Unknown Sample Rate (Mandatory)
# -------------------------------------------------------------
def test_g_unknown_sample_rate():
    t = np.arange(4096)
    x = np.exp(2j * np.pi * 0.15 * t).astype(np.complex64)
    rec = SignalRecording(
        samples=x,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
        # Sample rate is unknown (default)
    )

    analysis = analyze_signal(rec)
    # Absolute frequency in Hz must NOT be manufactured
    assert analysis.sample_rate_hz.value is None
    assert analysis.duration_seconds is None
    assert analysis.spectrum is not None
    assert analysis.spectrum.frequency_unit == "cycles/sample"
    assert analysis.frequency_candidates[0].frequency_hz is None
    assert analysis.frequency_candidates[0].normalized_frequency is not None
    assert np.isclose(analysis.frequency_candidates[0].normalized_frequency, 0.15, atol=1e-4)

# -------------------------------------------------------------
# TEST H — Different Sample Rates Scaling
# -------------------------------------------------------------
def test_h_different_sample_rates_scaling():
    n = 4096
    t = np.arange(n)
    fn = 0.10  # normalized frequency
    x = np.exp(2j * np.pi * fn * t).astype(np.complex64)

    # Rate 1: 1 MHz => f = 100 kHz
    rec1 = SignalRecording(
        samples=x,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(1_000_000.0, MetadataSource.USER_INPUT, MetadataStatus.KNOWN, 1.0, "Explicit"),
    )
    # Rate 2: 2.4 MHz => f = 240 kHz
    rec2 = SignalRecording(
        samples=x,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(2_400_000.0, MetadataSource.USER_INPUT, MetadataStatus.KNOWN, 1.0, "Explicit"),
    )

    a1 = analyze_signal(rec1)
    a2 = analyze_signal(rec2)

    # Normalized measurements must be invariant
    assert np.isclose(a1.frequency_candidates[0].normalized_frequency, a2.frequency_candidates[0].normalized_frequency)
    # Physical Hz must scale appropriately
    assert np.isclose(a1.frequency_candidates[0].frequency_hz, 100_000.0, atol=100.0)
    assert np.isclose(a2.frequency_candidates[0].frequency_hz, 240_000.0, atol=100.0)

# -------------------------------------------------------------
# TEST I — Real WAV
# -------------------------------------------------------------
def test_i_real_wav(tmp_path: Path):
    path = tmp_path / "real.wav"
    t = np.arange(4000, dtype=np.float32)
    fs = 8000
    tone = np.rint(np.sin(2 * np.pi * 1000 * t / fs) * 20000).astype("<i2")
    
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(fs)
        wav.writeframes(tone.tobytes())

    recording = WavReader(path).read()
    assert recording.semantic_type == "mono_real"
    
    analysis = analyze_signal(recording)
    assert analysis.spectrum is not None
    assert analysis.spectrum.is_complex is False
    assert analysis.spectrum.frequencies_normalized[0] == 0.0
    assert analysis.spectrum.frequencies_normalized[-1] == 0.5
    # Peak should be at 1000 Hz
    peak_idx = int(np.argmax(analysis.spectrum.magnitude_spectrum))
    assert np.isclose(analysis.spectrum.frequencies[peak_idx], 1000.0, atol=50.0)

# -------------------------------------------------------------
# TEST J — Stereo IQ WAV
# -------------------------------------------------------------
def test_j_stereo_iq_wav(tmp_path: Path):
    path = tmp_path / "stereo_iq.wav"
    t = np.arange(4000, dtype=np.float32)
    fs = 8000
    f0 = 1000.0  # +1000 Hz
    i = np.rint(np.cos(2 * np.pi * f0 * t / fs) * 20000).astype("<i2")
    q = np.rint(np.sin(2 * np.pi * f0 * t / fs) * 20000).astype("<i2")
    pairs = np.column_stack((i, q))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(fs)
        wav.writeframes(pairs.tobytes())

    recording = WavReader(path, mode="stereo_iq").read()
    assert recording.semantic_type == "complex_iq"
    
    analysis = analyze_signal(recording)
    assert analysis.spectrum is not None
    assert analysis.spectrum.is_complex is True
    # Two-sided spectrum with peak at +1000 Hz (not -1000 Hz)
    peak_idx = int(np.argmax(analysis.spectrum.magnitude_spectrum))
    assert np.isclose(analysis.spectrum.frequencies[peak_idx], 1000.0, atol=50.0)

# -------------------------------------------------------------
# TEST K — DC Offset Detected (Not Removed)
# -------------------------------------------------------------
def test_k_dc_offset_detected():
    t = np.arange(2048)
    ac = np.exp(2j * np.pi * 0.1 * t)
    dc_i = 0.4
    dc_q = -0.2
    x = (ac + (dc_i + 1j * dc_q)).astype(np.complex64)

    rec = SignalRecording(
        samples=x,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )

    analysis = analyze_signal(rec)
    assert analysis.dc_offset.status == MetadataStatus.MEASURED
    assert np.isclose(analysis.dc_offset.i_offset, dc_i, atol=0.01)
    assert np.isclose(analysis.dc_offset.q_offset, dc_q, atol=0.01)
    # Samples in recording must NOT be modified
    np.testing.assert_array_equal(rec.samples, x)

# -------------------------------------------------------------
# TEST L — Clipping Detection
# -------------------------------------------------------------
def test_l_clipping_detection():
    # Int16 saturated recording
    n = 1000
    clipped = np.full(n, 32767 + 32767j, dtype=np.complex64)
    rec = SignalRecording(
        samples=clipped,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="int16",
        channels=2,
        semantic_type="complex_iq",
    )

    analysis = analyze_signal(rec)
    assert analysis.clipping_diagnostics.is_clipped is True
    codes = [d.code for d in analysis.diagnostics]
    assert "CLIPPING_DETECTED" in codes

# -------------------------------------------------------------
# TEST M — Short Recording Diagnostic
# -------------------------------------------------------------
def test_m_short_recording_diagnostic():
    # Only 16 samples
    x = np.ones(16, dtype=np.complex64)
    rec = SignalRecording(
        samples=x,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )

    analysis = analyze_signal(rec)
    assert analysis.sample_count == 16
    codes = [d.code for d in analysis.diagnostics]
    assert "SHORT_RECORDING" in codes

# -------------------------------------------------------------
# QUALITY GATE CASES 1–10
# -------------------------------------------------------------
def test_quality_gate_case1_high_snr_tone():
    t = np.arange(4096)
    x = np.exp(2j * np.pi * 0.125 * t).astype(np.complex64)
    rec = SignalRecording(samples=x, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    freq_est = analysis.frequency_candidates[0]
    assert freq_est.quality_score > 0.8

def test_quality_gate_case2_low_snr_tone():
    np.random.seed(42)
    n = 4096
    t = np.arange(n)
    sig = 0.5 * np.exp(2j * np.pi * 0.125 * t).astype(np.complex64)
    noise = (np.random.normal(0, 1.0, n) + 1j * np.random.normal(0, 1.0, n)).astype(np.complex64)
    rec = SignalRecording(samples=sig + noise, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    codes = [d.code for d in analysis.diagnostics]
    assert "LOW_SIGNAL_TO_NOISE" in codes

def test_quality_gate_case3_multiple_tones():
    t = np.arange(8192)
    x = (np.exp(2j * np.pi * -0.2 * t) + np.exp(2j * np.pi * 0.25 * t)).astype(np.complex64)
    rec = SignalRecording(samples=x, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    assert len(analysis.detected_regions) >= 2

def test_quality_gate_case4_no_signal():
    np.random.seed(42)
    noise = (np.random.normal(0, 1.0, 4096) + 1j * np.random.normal(0, 1.0, 4096)).astype(np.complex64)
    rec = SignalRecording(samples=noise, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    codes = [d.code for d in analysis.diagnostics]
    assert "NO_SIGNAL_DETECTED" in codes

def test_quality_gate_case5_metadata_free_iq():
    t = np.arange(2048)
    x = np.exp(2j * np.pi * 0.1 * t).astype(np.complex64)
    rec = SignalRecording(samples=x, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    assert analysis.sample_rate_hz.status == MetadataStatus.MISSING
    assert analysis.spectrum.frequency_unit == "cycles/sample"

def test_quality_gate_case6_short_capture():
    rec = SignalRecording(samples=np.ones(20, dtype=np.complex64), source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    codes = [d.code for d in analysis.diagnostics]
    assert "SHORT_RECORDING" in codes

def test_quality_gate_case7_clipped_capture():
    clipped = np.full(500, 32767 + 0j, dtype=np.complex64)
    rec = SignalRecording(samples=clipped, source_format=SourceFormat.RAW_IQ, original_dtype="int16", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    codes = [d.code for d in analysis.diagnostics]
    assert "CLIPPING_DETECTED" in codes

def test_quality_gate_case8_burst_signal():
    noise = np.zeros(2048, dtype=np.complex64)
    noise[512:1536] = 2.0 * np.exp(2j * np.pi * 0.1 * np.arange(1024))
    rec = SignalRecording(samples=noise, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    assert len(analysis.detected_regions) >= 1

def test_quality_gate_case9_real_valued_wav(tmp_path: Path):
    path = tmp_path / "real.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000)
        wav.writeframes(np.zeros(1000, dtype="<i2").tobytes())
    rec = WavReader(path).read()
    analysis = analyze_signal(rec)
    assert analysis.semantic_type == "mono_real"
    assert analysis.spectrum.is_complex is False

def test_quality_gate_case10_complex_iq():
    rec = SignalRecording(samples=np.ones(1024, dtype=np.complex64), source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    analysis = analyze_signal(rec)
    assert analysis.semantic_type == "complex_iq"
    assert analysis.spectrum.is_complex is True
