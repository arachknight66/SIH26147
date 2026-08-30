import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.data_recovery.models import DataRecoveryStatus
from app.recovery.models import ModulationFamily, RecoveredSignal
from scripts.generate_digital_dataset import generate_digital_stream

def _make_rec_sig(hard_bits: np.ndarray, soft_bits: np.ndarray | None = None) -> RecoveredSignal:
    n_bits = len(hard_bits)
    dummy_syms = np.ones(max(1, n_bits // 2), dtype=np.complex64)
    dummy_indices = np.arange(len(dummy_syms), dtype=np.int32)
    sample_indices = np.arange(len(dummy_syms), dtype=np.float64) * 8.0

    return RecoveredSignal(
        symbols=dummy_syms,
        hard_bits=hard_bits.astype(np.uint8),
        soft_bits=soft_bits.astype(np.float32) if soft_bits is not None else np.where(hard_bits == 1, 1.0, -1.0).astype(np.float32),
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

def test_recover_data_end_to_end_protocol_a():
    rx_bits, rx_soft, manifest = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec_sig = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec_sig)

    assert res.is_recovered is True
    assert res.selected_candidate is not None
    assert res.status in (DataRecoveryStatus.INTEGRITY_SUPPORTED, DataRecoveryStatus.STRUCTURALLY_SUPPORTED)
    assert res.phase6_handoff is not None
    assert len(res.phase6_handoff.payload_bytes) > 0
    assert len(res.phase6_handoff.frame_boundaries) == len(res.selected_candidate.frames)

def test_recover_data_includes_bounded_bit_alignment_hypotheses():
    rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=3, seed=42)
    rec_sig = _make_rec_sig(rx_bits, rx_soft)
    res = recover_data(rec_sig)
    offsets = {candidate.bit_offset for candidate in res.bitstream_candidates}
    assert 0 in offsets
    assert any(offset > 0 for offset in offsets)

def test_recover_data_insufficient_data():
    short_bits = np.array([1, 0, 1], dtype=np.uint8)
    rec_sig = _make_rec_sig(short_bits)
    res = recover_data(rec_sig)

    assert res.is_inconclusive is True
    assert res.status == DataRecoveryStatus.INSUFFICIENT_DATA
