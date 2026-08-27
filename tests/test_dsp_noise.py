import numpy as np
import pytest
from app.dsp.noise import estimate_noise_floor
from app.dsp.psd import compute_psd

def test_noise_estimation_awgn():
    np.random.seed(123)
    n_samples = 16384
    # Complex circular Gaussian noise with variance 2.0 (I=1, Q=1)
    noise = (np.random.normal(0, 1.0, n_samples) + 1j * np.random.normal(0, 1.0, n_samples)).astype(np.complex64)
    
    psd_res = compute_psd(noise, segment_length=2048, is_complex=True)
    noise_est = estimate_noise_floor(psd_res.psd, method="iterative_sigma_clip")
    
    assert noise_est.noise_floor_db is not None
    assert noise_est.noise_power_linear is not None
    assert noise_est.is_signal_dominated is False
    # Theoretical noise power is 2.0 (3.01 dB)
    assert np.isclose(noise_est.noise_power_linear, 2.0, rtol=0.15)
    assert np.isclose(noise_est.noise_floor_db, 3.01, atol=0.6)

def test_noise_estimation_robust_to_strong_tones():
    np.random.seed(456)
    n_samples = 16384
    t = np.arange(n_samples)
    noise = (np.random.normal(0, 0.1, n_samples) + 1j * np.random.normal(0, 0.1, n_samples)).astype(np.complex64)
    tone1 = 5.0 * np.exp(2j * np.pi * 0.1 * t)
    tone2 = 3.0 * np.exp(2j * np.pi * -0.25 * t)
    x = (noise + tone1 + tone2).astype(np.complex64)

    psd_res = compute_psd(x, segment_length=2048, is_complex=True)
    noise_est = estimate_noise_floor(psd_res.psd, method="iterative_sigma_clip")
    
    # Noise variance is 2 * 0.01 = 0.02 (-16.99 dB)
    assert noise_est.noise_power_linear is not None
    assert np.isclose(noise_est.noise_power_linear, 0.02, rtol=0.20)
    assert np.isclose(noise_est.noise_floor_db, -16.99, atol=1.0)
