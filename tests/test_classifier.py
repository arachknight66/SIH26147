import numpy as np
import pytest
from signal_analysis.models import SignalRecording, SourceFormat, MetadataValue, MetadataStatus, HypothesisStatus
from signal_analysis.features import extract_all_features
from signal_analysis.classifier import compute_classical_scores
from signal_analysis.hypotheses import evaluate_and_rank_hypotheses, check_temporal_consistency
from tests.test_synthesis import generate_synthetic_signal

def _create_rec(sig):
    return SignalRecording(
        samples=sig,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(None, "t", MetadataStatus.MISSING),
        center_frequency_hz=MetadataValue(None, "t", MetadataStatus.MISSING),
        provenance={}, diagnostics=[]
    )

def test_clean_signal_bpsk():
    sig = generate_synthetic_signal("BPSK", n_symbols=2000, sps=4, snr_db=30)
    rec = _create_rec(sig)
    fv = extract_all_features(rec)
    c_scores = compute_classical_scores(fv)
    
    hyps, sel, amb, unk = evaluate_and_rank_hypotheses(fv, c_scores, snr_estimate=30, config={}, recording=rec)
    assert not unk
    assert not amb
    assert sel is not None
    assert sel.label == "BPSK"
    assert sel.score > 0.70
    
def test_clean_signal_qpsk():
    sig = generate_synthetic_signal("QPSK", n_symbols=2000, sps=4, snr_db=30)
    rec = _create_rec(sig)
    fv = extract_all_features(rec)
    c_scores = compute_classical_scores(fv)
    hyps, sel, amb, unk = evaluate_and_rank_hypotheses(fv, c_scores, snr_estimate=30, config={}, recording=rec)
    assert sel.label == "QPSK"

def test_clean_signal_8psk():
    sig = generate_synthetic_signal("8PSK", n_symbols=4000, sps=4, snr_db=30)
    rec = _create_rec(sig)
    fv = extract_all_features(rec)
    c_scores = compute_classical_scores(fv)
    hyps, sel, amb, unk = evaluate_and_rank_hypotheses(fv, c_scores, snr_estimate=30, config={}, recording=rec)
    assert sel.label == "8PSK"

def test_clean_signal_16qam():
    sig = generate_synthetic_signal("16-QAM", n_symbols=4000, sps=4, snr_db=30)
    rec = _create_rec(sig)
    fv = extract_all_features(rec)
    c_scores = compute_classical_scores(fv)
    hyps, sel, amb, unk = evaluate_and_rank_hypotheses(fv, c_scores, snr_estimate=30, config={}, recording=rec)
    assert sel.label == "16-QAM"

def test_low_snr_degradation():
    sig = generate_synthetic_signal("QPSK", n_symbols=2000, sps=4, snr_db=2)
    rec = _create_rec(sig)
    fv = extract_all_features(rec)
    c_scores = compute_classical_scores(fv)
    
    hyps, sel, amb, unk = evaluate_and_rank_hypotheses(fv, c_scores, snr_estimate=2, config={}, recording=rec)
    # At 2dB SNR, it should be UNKNOWN or AMBIGUOUS, or at least not HIGH quality tier
    if sel is not None:
        assert sel.quality_tier == "LOW"
        assert sel.score < 0.9
        
def test_ood_rejection():
    # Pure noise
    sig = np.random.randn(8000) + 1j * np.random.randn(8000)
    sig = sig.astype(np.complex64)
    rec = _create_rec(sig)
    fv = extract_all_features(rec)
    c_scores = compute_classical_scores(fv)
    
    hyps, sel, amb, unk = evaluate_and_rank_hypotheses(fv, c_scores, snr_estimate=0, config={}, recording=rec)
    
    assert unk or amb or (sel is not None and sel.score < 0.6)

def test_multi_window_consistency():
    sig = generate_synthetic_signal("QPSK", n_symbols=4000, sps=4, snr_db=20)
    rec = _create_rec(sig)
    cons, diag = check_temporal_consistency(rec, {})
    assert cons >= 0.75
    assert diag is None
