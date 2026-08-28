from __future__ import annotations
import numpy as np
import pytest
from app.modulation.models import ModulationFamily
from app.recovery.models import RecoveredSignal
from app.verification.models import AuditResultStatus
from app.verification.modulation_checks import audit_modulation_and_constellation

def _create_rec_sig(symbols: np.ndarray) -> RecoveredSignal:
    n_syms = len(symbols)
    dummy_indices = np.arange(n_syms, dtype=np.int32)
    sample_indices = np.arange(n_syms, dtype=np.float64) * 8.0
    hard_bits = np.zeros(n_syms * 2, dtype=np.uint8)
    soft_bits = np.zeros(n_syms * 2, dtype=np.float32)

    return RecoveredSignal(
        symbols=symbols,
        hard_bits=hard_bits,
        soft_bits=soft_bits,
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

def test_audit_modulation_clean_qpsk():
    qpsk_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    symbols = np.tile(qpsk_pts, 50)
    rec = _create_rec_sig(symbols)
    res, tests = audit_modulation_and_constellation(rec)
    assert res.evm_percent < 5.0
    assert res.is_consistent is True
    assert all(t.status in (AuditResultStatus.PASS, AuditResultStatus.WEAK_PASS) for t in tests)

def test_audit_modulation_high_evm_noise():
    symbols = (np.random.normal(0, 1, 200) + 1j * np.random.normal(0, 1, 200)).astype(np.complex64)
    rec = _create_rec_sig(symbols)
    res, tests = audit_modulation_and_constellation(rec)
    assert res.evm_percent > 30.0
    assert res.is_consistent is False
