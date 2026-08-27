import numpy as np
import pytest
from app.models.metadata import MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import SignalRecording, SourceFormat
from app.modulation.analyzer import analyze_modulation
from app.modulation.models import (
    FeatureValidity,
    HypothesisStatus,
    ModulationAnalysisConfig,
    ModulationFamily,
)
from scripts.generate_modulated_dataset import generate_modulated_signal

def _make_rec(samples: np.ndarray, sample_rate: float | None = 100000.0) -> SignalRecording:
    meta_sr = (
        MetadataValue(
            value=sample_rate,
            source=MetadataSource.USER_INPUT,
            status=MetadataStatus.KNOWN,
            confidence=1.0,
            evidence=["test_case"],
        )
        if sample_rate is not None
        else MetadataValue(
            value=None,
            source=MetadataSource.UNKNOWN,
            status=MetadataStatus.UNAVAILABLE,
            confidence=0.0,
            evidence=["missing"],
        )
    )
    return SignalRecording(
        samples=samples.astype(np.complex64),
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
        sample_rate_hz=meta_sr,
    )

# Case 1: Clean BPSK
def test_case_1_clean_bpsk():
    samples, _ = generate_modulated_signal("BPSK", snr_db=25.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.PSK
    assert res.selected_hypothesis.order == 2

# Case 2: Clean QPSK
def test_case_2_clean_qpsk():
    samples, _ = generate_modulated_signal("QPSK", snr_db=25.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.PSK
    assert res.selected_hypothesis.order == 4

# Case 3: Clean 8PSK
def test_case_3_clean_8psk():
    samples, _ = generate_modulated_signal("8PSK", snr_db=25.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.PSK
    assert res.selected_hypothesis.order == 8

# Case 4: Clean BFSK
def test_case_4_clean_bfsk():
    samples, _ = generate_modulated_signal("BFSK", snr_db=25.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.FSK
    assert res.selected_hypothesis.order == 2

# Case 5: Clean 16QAM
def test_case_5_clean_16qam():
    samples, _ = generate_modulated_signal("16QAM", snr_db=25.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.QAM
    assert res.selected_hypothesis.order == 16

# Case 6: Low-SNR QPSK
def test_case_6_low_snr_qpsk():
    samples, _ = generate_modulated_signal("QPSK", snr_db=2.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert any(d.code in ("LOW_SIGNAL_TO_NOISE", "LOW_ANALYSIS_QUALITY", "UNKNOWN_OR_OOD_MODULATION", "AMBIGUOUS_MODULATION") for d in res.diagnostics) or (res.selected_hypothesis and res.selected_hypothesis.quality in ("MODERATE", "LOW"))

# Case 7: QPSK with CFO
def test_case_7_qpsk_with_cfo():
    samples, _ = generate_modulated_signal("QPSK", cfo_normalized=0.01, snr_db=20.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.hypotheses[0].family == ModulationFamily.PSK
    assert res.hypotheses[0].order in (4, 8)

# Case 8: QPSK with Fractional Timing Offset
def test_case_8_qpsk_with_timing():
    samples, _ = generate_modulated_signal("QPSK", timing_offset=0.35, snr_db=20.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.PSK
    assert res.selected_hypothesis.order == 4

# Case 9: QAM with Fading
def test_case_9_qam_with_fading():
    samples, _ = generate_modulated_signal("16QAM", fading="rician", snr_db=20.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.QAM

# Case 10: FSK with Amplitude Distortion
def test_case_10_fsk_with_amp_distortion():
    samples, _ = generate_modulated_signal("BFSK", iq_imbalance_db=3.0, clipping_ratio=0.7, snr_db=20.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.FSK

# Case 11: AM Out-of-Distribution
def test_case_11_am_ood():
    samples, _ = generate_modulated_signal("AM", seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.is_unknown or res.is_ambiguous or (res.selected_hypothesis and res.selected_hypothesis.quality == "LOW")

# Case 12: FM Out-of-Distribution
def test_case_12_fm_ood():
    samples, _ = generate_modulated_signal("FM", seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.is_unknown or res.is_ambiguous or (res.selected_hypothesis and res.selected_hypothesis.quality == "LOW")

# Case 13: GMSK Out-of-Distribution
def test_case_13_gmsk_ood():
    samples, _ = generate_modulated_signal("GMSK", seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.is_unknown or res.is_ambiguous or (res.selected_hypothesis and res.selected_hypothesis.quality in ("MODERATE", "LOW"))

# Case 14: OFDM Out-of-Distribution
def test_case_14_ofdm_ood():
    samples, _ = generate_modulated_signal("OFDM", seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.is_unknown or res.is_ambiguous or (res.selected_hypothesis and res.selected_hypothesis.quality == "LOW")

# Case 15: Noise Only Input
def test_case_15_noise_only():
    samples, _ = generate_modulated_signal("NOISE", seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.is_unknown or res.selected_hypothesis is None or (res.selected_hypothesis.quality == "LOW")

# Case 16: Multi-Signal Region
def test_case_16_multi_signal_region():
    s1, _ = generate_modulated_signal("BFSK", cfo_normalized=-0.15, snr_db=20.0, seed=42)
    s2, _ = generate_modulated_signal("QPSK", cfo_normalized=+0.15, snr_db=20.0, seed=43)
    mix = s1 + s2
    res = analyze_modulation(_make_rec(mix))
    assert len(res.hypotheses) > 0

# Case 17: Short Recording
def test_case_17_short_recording():
    samples, _ = generate_modulated_signal("QPSK", n_symbols=8, samples_per_symbol=4, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert any(d.code == "SHORT_RECORDING" for d in res.diagnostics)

# Case 18: Phase Invalid Recording (Near Zero)
def test_case_18_phase_invalid():
    zeros = np.zeros(512, dtype=np.complex64)
    res = analyze_modulation(_make_rec(zeros))
    assert res.feature_vector.phase.validity in (FeatureValidity.UNRELIABLE, FeatureValidity.UNAVAILABLE)
    assert res.is_unknown

# Case 19: Missing Sample Rate Metadata
def test_case_19_missing_sample_rate():
    samples, _ = generate_modulated_signal("QPSK", snr_db=20.0, seed=42)
    rec = _make_rec(samples, sample_rate=None)
    res = analyze_modulation(rec)
    assert res.selected_hypothesis is not None
    assert res.selected_hypothesis.family == ModulationFamily.PSK
    assert res.selected_hypothesis.candidate_parameters.get("candidate_symbol_rate_hz") is None

# Case 20: Repeated-Window Consistency Test
def test_case_20_window_consistency():
    samples, _ = generate_modulated_signal("QPSK", n_symbols=1024, samples_per_symbol=8, snr_db=20.0, seed=42)
    res = analyze_modulation(_make_rec(samples))
    assert res.window_consistency >= 0.75
