import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.data_recovery.framing import detect_frame_boundaries, detect_sequence_continuity, slice_frames
from app.data_recovery.models import (
    DataQualityLevel,
    DataRecoveryStatus,
    EpistemicStatus,
    FrameCandidate,
)
from app.recovery.models import ModulationFamily, RecoveredSignal
from scripts.generate_digital_dataset import generate_digital_stream

def _make_rec_sig(
    hard_bits: np.ndarray,
    soft_bits: np.ndarray | None = None,
) -> RecoveredSignal:
    n_bits = len(hard_bits)
    dummy_syms = np.ones(max(1, n_bits // 2), dtype=np.complex64)
    dummy_indices = np.arange(len(dummy_syms), dtype=np.int32)
    sample_indices = np.arange(len(dummy_syms), dtype=np.float64) * 8.0

    return RecoveredSignal(
        symbols=dummy_syms,
        hard_bits=hard_bits.astype(np.uint8),
        soft_bits=soft_bits.astype(np.float32) if soft_bits is not None else None,
        symbol_indices=dummy_indices,
        sample_indices=sample_indices,
        modulation_family=ModulationFamily.PSK,
        modulation_order=4,
        symbol_rate_normalized=0.125,
        samples_per_symbol=8.0,
        cfo_normalized=0.0,
        carrier_phase_rad=0.0,
        evm_percent=5.0,
        decision_margin=0.95,
        rotational_ambiguities_deg=(0.0, 90.0, 180.0, 270.0),
        bit_polarity_status="unresolved",
    )

# Case 1: Clean Protocol A
def test_case_1_clean_protocol_a():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True
    assert res.selected_candidate is not None
    assert res.selected_candidate.integrity.valid_frame_count >= 1

# Case 2: Clean Protocol B
def test_case_2_clean_protocol_b():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_B", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True
    assert res.selected_candidate is not None
    assert res.selected_candidate.integrity.valid_frame_count >= 1

# Case 3: Clean Protocol C
def test_case_3_clean_protocol_c():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True
    assert res.selected_candidate is not None

# Case 4: Clean Protocol D
def test_case_4_clean_protocol_d():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_D", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True

# Case 5: Clean Protocol E
def test_case_5_clean_protocol_e():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_E", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True

# Case 6: Bit Offset 3 bits
def test_case_6_bit_offset_3():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, bit_offset=3, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True
    assert res.selected_candidate.integrity.valid_frame_count >= 1

# Case 7: Bit Offset 7 bits
def test_case_7_bit_offset_7():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, bit_offset=7, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True
    assert res.selected_candidate.integrity.valid_frame_count >= 1

# Case 8: Inverted Polarity
def test_case_8_inverted_polarity():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, invert_polarity=True, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True
    assert res.selected_candidate.integrity.valid_frame_count >= 1

# Case 9: Low BER on Protocol A
def test_case_9_low_ber():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, ber=0.001, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True

# Case 10: BER with FEC Error Correction
def test_case_10_fec_error_correction():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, ber=0.01, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True

# Case 11: Burst Error Injection
def test_case_11_burst_error():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, burst_error_len=6, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True

# Case 12: Multi-Frame Sequence Tracking
def test_case_12_multi_frame_sequence():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_B", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_recovered is True
    assert len(res.selected_candidate.frames) >= 4

# Case 13: Missing Sequence Number Detection
def test_case_13_missing_sequence():
    frames = [
        FrameCandidate(1, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 0, 100, True, True, False, sequence_number=1),
        FrameCandidate(2, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 100, 200, True, True, False, sequence_number=2),
        FrameCandidate(3, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 200, 300, True, True, False, sequence_number=4),
    ]
    is_cont, seqs, missing = detect_sequence_continuity(frames)
    assert is_cont is False
    assert missing == [3]

# Case 14: Out-of-Distribution Pure Random Bits Rejection
def test_case_14_ood_random_rejection():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="OOD_RANDOM", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.is_inconclusive is True
    assert res.status in (DataRecoveryStatus.INSUFFICIENT_DATA, DataRecoveryStatus.AMBIGUOUS)

# Case 15: Out-of-Distribution High Entropy Noise Rejection
def test_case_15_high_entropy_noise():
    noise_bits = np.random.randint(0, 2, 1024, dtype=np.uint8)
    rec = _make_rec_sig(noise_bits)
    res = recover_data(rec)
    assert res.is_inconclusive is True

# Case 16: Zero-length / Short Input Stream
def test_case_16_short_stream():
    short_bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    rec = _make_rec_sig(short_bits)
    res = recover_data(rec)
    assert res.is_inconclusive is True
    assert res.status == DataRecoveryStatus.INSUFFICIENT_DATA

# Case 17: All-Zero Bit Stream
def test_case_17_all_zeros():
    zeros = np.zeros(512, dtype=np.uint8)
    rec = _make_rec_sig(zeros)
    res = recover_data(rec)
    assert res.is_inconclusive is True

# Case 18: All-Ones Bit Stream
def test_case_18_all_ones():
    ones = np.ones(512, dtype=np.uint8)
    rec = _make_rec_sig(ones)
    res = recover_data(rec)
    assert res.is_inconclusive is True

# Case 19: Phase 6 Handoff Verification Contract
def test_case_19_phase6_handoff_contract():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec)
    assert res.phase6_handoff is not None
    assert len(res.phase6_handoff.raw_bits) > 0
    assert "Synchronous framing" in res.phase6_handoff.assumptions

# Case 20: Missing Soft Decisions Hard-Decision Fallback
def test_case_20_hard_decision_fallback():
    rx_bits, _, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_bits, soft_bits=None)
    res = recover_data(rec)
    assert res.is_recovered is True
    assert any(d.code == "NO_SOFT_INFORMATION" for d in res.diagnostics)
