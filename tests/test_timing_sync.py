import numpy as np
import pytest
from app.recovery.models import LockStatus
from app.recovery.timing_sync import gardner_timing_recovery
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_gardner_timing_recovery_clean_qpsk():
    samples, _ = generate_modulated_signal("QPSK", n_symbols=512, samples_per_symbol=8, timing_offset=0.25, snr_db=30.0, seed=42)
    syms, strobes, res = gardner_timing_recovery(samples, sps=8.0)
    assert res.valid is True
    assert res.lock_status == LockStatus.LOCKED
    assert res.ted_variance < 0.15
    assert len(syms) > 300

def test_gardner_timing_recovery_short():
    short_samples = np.ones(16, dtype=np.complex64)
    syms, strobes, res = gardner_timing_recovery(short_samples, sps=8.0)
    assert res.valid is False
    assert res.lock_status == LockStatus.UNLOCKED
