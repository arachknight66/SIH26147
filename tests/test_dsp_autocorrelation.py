import numpy as np
import pytest
from app.dsp.autocorrelation import compute_autocorrelation

def test_autocorrelation_white_noise():
    np.random.seed(42)
    noise = (np.random.normal(0, 1, 8192) + 1j * np.random.normal(0, 1, 8192)).astype(np.complex64)
    res = compute_autocorrelation(noise, max_lag=100)

    assert len(res.lags) == 101
    assert res.lags[0] == 0
    # Lag 0 normalized magnitude must be 1.0
    assert np.isclose(res.normalized_magnitude[0], 1.0)
    # Lags > 0 for white noise should be small (< 0.1)
    assert np.all(res.normalized_magnitude[1:] < 0.1)

def test_autocorrelation_periodic_signal():
    t = np.arange(4096)
    period = 16  # Normalized rate = 1/16 = 0.0625
    x = np.exp(2j * np.pi * (1.0 / period) * t).astype(np.complex64)
    res = compute_autocorrelation(x, max_lag=64)

    # Periodic complex sinusoid has constant magnitude autocorrelation |R[k]| = 1.0 (before window decay)
    assert np.isclose(res.normalized_magnitude[period], 1.0 - (period / 4096.0), atol=0.01)
