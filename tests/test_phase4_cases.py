import numpy as np
import pytest
from app.analysis.analyzer import analyze_signal
from app.models.metadata import MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import SignalRecording, SourceFormat
from app.modulation.analyzer import analyze_modulation
from app.modulation.models import (
    ModulationAnalysis,
    ModulationEvidence,
    ModulationFamily,
    ModulationHypothesis,
)
from app.recovery.analyzer import recover_all_regions, recover_signal
from app.recovery.models import ModulationFamily as RecModFamily, RecoveryQualityLevel, RecoveryStatus
from scripts.generate_modulated_dataset import generate_modulated_signal

def _make_rec(samples: np.ndarray, sample_rate: float | None = 100000.0) -> SignalRecording:
    meta_sr = (
        MetadataValue(
            value=sample_rate,
            source=MetadataSource.USER_INPUT,
            status=MetadataStatus.KNOWN,
            confidence=1.0,
            evidence=["test"],
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
def test_case_1_clean_bpsk_recovery():
    samples, _ = generate_modulated_signal("BPSK", n_symbols=512, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.family == RecModFamily.PSK
    assert res.selected_candidate.order in (2, 4)
    assert res.selected_candidate.constellation.evm_percent < 20.0

# Case 2: Clean QPSK
def test_case_2_clean_qpsk_recovery():
    samples, _ = generate_modulated_signal("QPSK", n_symbols=512, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.family == RecModFamily.PSK
    assert res.selected_candidate.order == 4
    assert res.selected_candidate.constellation.evm_percent < 20.0

# Case 3: Clean 8PSK
def test_case_3_clean_8psk_recovery():
    samples, _ = generate_modulated_signal("8PSK", n_symbols=512, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.family == RecModFamily.PSK

# Case 4: Clean BFSK
def test_case_4_clean_bfsk_recovery():
    samples, _ = generate_modulated_signal("BFSK", n_symbols=512, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.family == RecModFamily.FSK
    assert res.selected_candidate.order == 2

# Case 5: Clean 16QAM
def test_case_5_clean_16qam_recovery():
    samples, _ = generate_modulated_signal("16QAM", n_symbols=512, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.family == RecModFamily.QAM
    assert res.selected_candidate.order == 16

# Case 6: QPSK with CFO
def test_case_6_qpsk_cfo():
    samples, _ = generate_modulated_signal("QPSK", cfo_normalized=0.005, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.family == RecModFamily.PSK

# Case 7: QPSK with Fractional Timing Offset
def test_case_7_qpsk_timing_offset():
    samples, _ = generate_modulated_signal("QPSK", timing_offset=0.35, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.family == RecModFamily.PSK

# Case 8: QPSK with Low SNR
def test_case_8_qpsk_low_snr():
    samples, _ = generate_modulated_signal("QPSK", snr_db=2.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    # At 2 dB, should be inconclusive or low quality
    assert res.is_inconclusive or (res.selected_candidate and res.selected_candidate.quality.quality_level in (RecoveryQualityLevel.LOW, RecoveryQualityLevel.MODERATE))

# Case 9: 16QAM with Fading
def test_case_9_16qam_fading():
    samples, _ = generate_modulated_signal("16QAM", fading="rician", snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True

# Case 10: BFSK with IQ Imbalance & Clipping
def test_case_10_bfsk_impairments():
    samples, _ = generate_modulated_signal("BFSK", iq_imbalance_db=3.0, clipping_ratio=0.7, snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.family == RecModFamily.FSK

# Case 11: AM Out-of-Distribution
def test_case_11_am_ood():
    samples, _ = generate_modulated_signal("AM", seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_inconclusive or (res.selected_candidate and res.selected_candidate.quality.quality_level in (RecoveryQualityLevel.REJECTED, RecoveryQualityLevel.LOW))

# Case 12: FM Out-of-Distribution
def test_case_12_fm_ood():
    samples, _ = generate_modulated_signal("FM", seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_inconclusive or (res.selected_candidate and res.selected_candidate.quality.quality_level in (RecoveryQualityLevel.REJECTED, RecoveryQualityLevel.LOW))

# Case 13: GMSK Out-of-Distribution
def test_case_13_gmsk_ood():
    samples, _ = generate_modulated_signal("GMSK", seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_inconclusive or (res.selected_candidate and res.selected_candidate.quality.quality_level in (RecoveryQualityLevel.LOW, RecoveryQualityLevel.REJECTED))

# Case 14: OFDM Out-of-Distribution
def test_case_14_ofdm_ood():
    samples, _ = generate_modulated_signal("OFDM", seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_inconclusive or (res.selected_candidate and res.selected_candidate.quality.quality_level in (RecoveryQualityLevel.REJECTED, RecoveryQualityLevel.LOW))

# Case 15: Noise Only
def test_case_15_noise_only():
    samples, _ = generate_modulated_signal("NOISE", seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_inconclusive or (res.selected_candidate and res.selected_candidate.quality.quality_level in (RecoveryQualityLevel.REJECTED, RecoveryQualityLevel.LOW))

# Case 16: Wrong Phase 3 Hypothesis Promotion
def test_case_16_wrong_hypothesis_promotion():
    # Signal is 16QAM, but simulated Phase 3 favored QPSK incorrectly
    samples, _ = generate_modulated_signal("16QAM", snr_db=25.0, seed=42)
    rec = _make_rec(samples)
    an = analyze_signal(rec)
    
    # Fake Phase 3 with QPSK as #1, 16QAM as #2
    fake_mod_an = ModulationAnalysis(
        recording_reference="test",
        signal_region=None,
        hypotheses=[
            ModulationHypothesis(
                family=ModulationFamily.PSK,
                order=4,
                score=0.85,
                family_score=0.85,
                order_score=0.85,
                quality="HIGH",
                evidence=ModulationEvidence(),
                candidate_parameters={"candidate_samples_per_symbol": 8.0},
            ),
            ModulationHypothesis(
                family=ModulationFamily.QAM,
                order=16,
                score=0.40,
                family_score=0.40,
                order_score=0.40,
                quality="MODERATE",
                evidence=ModulationEvidence(),
                candidate_parameters={"candidate_samples_per_symbol": 8.0},
            ),
        ],
        selected_hypothesis=None,
        feature_vector=None,
        raw_distribution=None,
        window_consistency=1.0,
        is_ambiguous=False,
        is_unknown=False,
    )
    res = recover_signal(rec, analysis=an, modulation_analysis=fake_mod_an)
    # The receiver should promote 16-QAM because 16QAM locks and has lower EVM
    assert res.selected_candidate is not None
    assert res.selected_candidate.family == RecModFamily.QAM
    assert res.selected_candidate.order == 16
    assert res.wrong_hypothesis_detected is True

# Case 17: Short Recording
def test_case_17_short_recording():
    samples, _ = generate_modulated_signal("QPSK", n_symbols=4, samples_per_symbol=4, seed=42)
    rec = _make_rec(samples)
    res = recover_signal(rec)
    assert res.is_inconclusive is True
    assert any(d.code == "INSUFFICIENT_SAMPLES" for d in res.diagnostics)

# Case 18: All-Zero Signal
def test_case_18_all_zeros():
    zeros = np.zeros(512, dtype=np.complex64)
    rec = _make_rec(zeros)
    res = recover_signal(rec)
    assert res.is_inconclusive is True

# Case 19: Missing Sample Rate Metadata
def test_case_19_missing_sample_rate():
    samples, _ = generate_modulated_signal("QPSK", snr_db=25.0, seed=42)
    rec = _make_rec(samples, sample_rate=None)
    an = analyze_signal(rec)
    mod_an = analyze_modulation(rec, analysis=an)
    res = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
    assert res.is_recovered is True
    assert res.selected_candidate.symbol_rate_normalized == 0.125

# Case 20: Multi-Region Independent Recovery
def test_case_20_multi_region():
    s1, _ = generate_modulated_signal("BFSK", cfo_normalized=-0.15, snr_db=20.0, seed=42)
    s2, _ = generate_modulated_signal("QPSK", cfo_normalized=+0.15, snr_db=20.0, seed=43)
    mix = s1 + s2
    rec = _make_rec(mix)
    an = analyze_signal(rec)
    results = recover_all_regions(rec, analysis=an)
    assert len(results) >= 1
