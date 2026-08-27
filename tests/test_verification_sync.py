from __future__ import annotations
import numpy as np
import pytest
from app.modulation.models import ModulationFamily
from app.recovery.models import RecoveredSignal
from app.verification.models import AuditResultStatus
from app.verification.synchronization_checks import audit_synchronization_and_stability

def _create_rec_sig(symbols: np.ndarray, cfo: float = 0.0) -> RecoveredSignal:
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
        cfo_normalized=cfo,
        carrier_phase_rad=0.0,
        evm_percent=5.0,
        decision_margin=0.95,
        rotational_ambiguities_deg=(0.0, 90.0, 180.0, 270.0),
        bit_polarity_status="unresolved",
    )

def test_audit_sync_stable():
    qpsk_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    symbols = np.tile(qpsk_pts, 100)
    rec = _create_rec_sig(symbols, cfo=0.001)
    res, tests = audit_synchronization_and_stability(rec)
    assert res.is_stable is True
    assert res.window_consistency_fraction >= 0.80

def test_audit_sync_unstable():
    symbols = (np.random.normal(0, 1, 300) + 1j * np.random.normal(0, 1, 300)).astype(np.complex64)
    rec = _create_rec_sig(symbols, cfo=0.1)
    res, tests = audit_synchronization_and_stability(rec)
    assert res.is_stable is False
    assert res.window_consistency_fraction < 0.50
