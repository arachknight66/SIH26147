from __future__ import annotations
import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.models.analysis import SignalAnalysis, SNREstimate
from app.models.signal import SignalRecording, SourceFormat
from app.modulation.models import ModulationFamily
from app.recovery.models import RecoveredSignal
from app.verification.analyzer import verify_result
from app.verification.models import VerificationStatus
from scripts.generate_digital_dataset import generate_digital_stream

def _make_rec_sig(
    hard_bits: np.ndarray,
    soft_bits: np.ndarray | None = None,
    snr_db: float = 20.0,
    cfo: float = 0.0,
) -> RecoveredSignal:
    n_bits = len(hard_bits)
    n_syms = max(16, n_bits // 2)
    # Map bits to QPSK symbols
    padded_bits = np.pad(hard_bits, (0, max(0, n_syms * 2 - n_bits))) if len(hard_bits) < n_syms * 2 else hard_bits[: n_syms * 2]
    bit_pairs = padded_bits.reshape(-1, 2)
    # Gray mapping: 00 -> 1+1j, 01 -> -1+1j, 11 -> -1-1j, 10 -> 1-1j
    i_val = np.where(bit_pairs[:, 0] == 0, 1.0, -1.0)
    q_val = np.where(bit_pairs[:, 1] == 0, 1.0, -1.0)
    clean_syms = (i_val + 1j * q_val).astype(np.complex64) / np.sqrt(2.0)
    
    sigma = 10.0 ** (-snr_db / 20.0) if snr_db < 40.0 else 0.01
    noise = (np.random.normal(0, sigma, n_syms) + 1j * np.random.normal(0, sigma, n_syms)).astype(np.complex64)
    syms = clean_syms + noise

    dummy_indices = np.arange(len(syms), dtype=np.int32)
    sample_indices = np.arange(len(syms), dtype=np.float64) * 8.0

    return RecoveredSignal(
        symbols=syms,
        hard_bits=hard_bits.astype(np.uint8),
        soft_bits=soft_bits.astype(np.float32) if soft_bits is not None else np.where(hard_bits == 1, 1.0, -1.0).astype(np.float32),
        symbol_indices=dummy_indices,
        sample_indices=sample_indices,
        modulation_family=ModulationFamily.PSK,
        modulation_order=4,
        symbol_rate_normalized=0.125,
        samples_per_symbol=8.0,
        cfo_normalized=cfo,
        carrier_phase_rad=0.0,
        evm_percent=float(sigma * 100.0),
        decision_margin=0.95,
        rotational_ambiguities_deg=(0.0, 90.0, 180.0, 270.0),
        bit_polarity_status="unresolved",
    )

# 1. Clean Protocols A through E
def test_case_1_clean_protocol_a_verified():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True
    assert p6.status == VerificationStatus.INDEPENDENTLY_VERIFIED

def test_case_2_clean_protocol_b_verified():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_B", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True
    assert p6.status == VerificationStatus.INDEPENDENTLY_VERIFIED

def test_case_3_clean_protocol_c_verified():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True
    assert p6.status == VerificationStatus.INDEPENDENTLY_VERIFIED

def test_case_4_clean_protocol_d_verified():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_D", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True
    assert p6.status == VerificationStatus.INDEPENDENTLY_VERIFIED

def test_case_5_clean_protocol_e_verified():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_E", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True
    assert p6.status == VerificationStatus.INDEPENDENTLY_VERIFIED

# 2. Bit Offsets & Polarity Inversions
def test_case_6_bit_offset_verified():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, bit_offset=3, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True

def test_case_7_polarity_inverted_verified():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, invert_polarity=True, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True

# 3. Channel Noise & FEC Verification
def test_case_8_fec_under_channel_noise():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, ber=0.005, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True
    assert p6.fec_audit.is_beneficial is True
    assert p6.fec_audit.held_out_validation_passed is True

def test_case_9_burst_error_robustness():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, burst_error_len=6, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is True

# 4. Out-of-Distribution & Adversarial Rejection
def test_case_10_ood_random_rejection():
    rx, soft, _ = generate_digital_stream(protocol="OOD_RANDOM", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is False

def test_case_11_pure_noise_rejection():
    noise = np.random.randint(0, 2, 1024, dtype=np.uint8)
    rec = _make_rec_sig(noise)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is False

def test_case_12_all_zeros_rejection():
    zeros = np.zeros(512, dtype=np.uint8)
    rec = _make_rec_sig(zeros)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is False

def test_case_13_all_ones_rejection():
    ones = np.ones(512, dtype=np.uint8)
    rec = _make_rec_sig(ones)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is False

# 5. Boundary Perturbation Falsification
def test_case_14_boundary_perturbation_check():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.frame_audit.boundary_perturbation_passed is True

# 6. Leave-One-Frame-Out Stability
def test_case_15_leave_one_out_stability():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.robustness_audit.leave_one_out_stable is True
    assert p6.robustness_audit.high_leverage_frame_detected is False

# 7. Reproducibility Determinism
def test_case_16_reproducibility_determinism():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6_1 = verify_result(p5, rec)
    p6_2 = verify_result(p5, rec)
    assert p6_1.handoff.reproducibility_hash == p6_2.handoff.reproducibility_hash

# 8. Physical Consistency Audit
def test_case_17_physical_consistency():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    from app.analysis.analyzer import analyze_signal
    rec_sig = _make_rec_sig(rx, soft)
    p5 = recover_data(rec_sig)
    
    samples = np.exp(1j * np.linspace(0, 100, 1000)).astype(np.complex64)
    raw_rec = SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=1,
        semantic_type="iq",
    )
    p2 = analyze_signal(raw_rec)
    p6 = verify_result(p5, rec_sig, phase2_result=p2, phase1_result=raw_rec)
    assert p6.physical_audit.is_finite is True
    assert p6.physical_audit.measurement_consistent is True

# 9. Multiple-Testing Corrected P-Value
def test_case_18_multiple_testing_significance():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.integrity_audit.multiple_testing_corrected_p_value < 0.01

# 10. Temporal Window Stability
def test_case_19_temporal_window_stability():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.sync_audit.window_consistency_fraction >= 0.80

# 11. Error Budget Verification
def test_case_20_error_budget_computation():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.error_budget is not None
    assert p6.error_budget.total_composite_uncertainty > 0.0

# 12. Sequence Continuity Check
def test_case_21_sequence_continuity():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_B", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.frame_audit.sequence_is_continuous is True

# 13. Short Stream Safety
def test_case_22_short_stream_handling():
    rx = np.array([1, 0, 1], dtype=np.uint8)
    rec = _make_rec_sig(rx)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.is_verified is False

# 14. Scrambler Reproducibility Check
def test_case_23_scrambler_reproducibility():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_D", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.scrambler_audit.is_reproducible is True
    assert p6.scrambler_audit.is_verified is True

# 15. Cross-Validation on Held-Out Frames
def test_case_24_fec_cross_validation_held_out():
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=6, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    assert p6.fec_audit.held_out_validation_passed is True

# 16. Scientific Verification Report Formatting
def test_case_25_verification_report_generation():
    from app.verification.report import format_verification_report
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx, soft)
    p5 = recover_data(rec)
    p6 = verify_result(p5, rec)
    rep = format_verification_report(p6, "test_recording.iq")
    assert "SIH26147 PHASE 6 SCIENTIFIC VERIFICATION" in rep
    assert "CLAIM 1 — MODULATION" in rep
    assert "CLAIM 2 — FRAME STRUCTURE" in rep
    assert "FINAL ASSESSMENT" in rep
