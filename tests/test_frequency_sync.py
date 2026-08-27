import numpy as np
import pytest
from app.recovery.frequency_sync import (
    correct_frequency_offset,
    estimate_bfsk_frequencies,
    estimate_coarse_cfo_mth_power,
)
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_coarse_cfo_estimation_qpsk():
    true_cfo = 0.008
    samples, _ = generate_modulated_signal("QPSK", cfo_normalized=true_cfo, snr_db=30.0, seed=42)
    res = estimate_coarse_cfo_mth_power(samples, order=4)
    assert res.valid is True
    assert abs(res.coarse_cfo_normalized - true_cfo) < 0.002

def test_coarse_cfo_estimation_bpsk():
    true_cfo = -0.012
    samples, _ = generate_modulated_signal("BPSK", cfo_normalized=true_cfo, snr_db=30.0, seed=42)
    res = estimate_coarse_cfo_mth_power(samples, order=2)
    assert res.valid is True
    assert abs(res.coarse_cfo_normalized - true_cfo) < 0.002

def test_bfsk_frequency_estimation():
    samples, _ = generate_modulated_signal("BFSK", snr_db=25.0, seed=42)
    f0, f1, delta_f, res = estimate_bfsk_frequencies(samples)
    assert res.valid is True
    assert 0.15 < delta_f < 0.35
    assert np.isclose(0.5 * (f0 + f1), 0.0, atol=0.05)

def test_frequency_correction():
    samples = np.ones(128, dtype=np.complex64)
    corr = correct_frequency_offset(samples, cfo_normalized=0.1)
    assert len(corr) == 128
    assert corr.dtype == np.complex64
