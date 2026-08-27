import numpy as np
import pytest
from app.dsp.frequency import compute_all_frequency_estimates, estimate_frequency_phase_progression, estimate_frequency_spectral_peak
from app.dsp.psd import compute_psd

def test_frequency_estimation_parabolic_interpolation():
    t = np.arange(8192)
    # Off-bin frequency (e.g. 0.123456 cycles/sample)
    true_f = 0.123456
    x = np.exp(2j * np.pi * true_f * t).astype(np.complex64)

    psd_res = compute_psd(x, segment_length=2048, is_complex=True)
    peak_est = estimate_frequency_spectral_peak(psd_res)
    assert peak_est.normalized_frequency is not None
    # Sub-bin interpolation should achieve high accuracy (< 1e-4)
    assert abs(peak_est.normalized_frequency - true_f) < 1e-4

    phase_est = estimate_frequency_phase_progression(x)
    assert phase_est.normalized_frequency is not None
    assert abs(phase_est.normalized_frequency - true_f) < 1e-5

def test_frequency_estimation_with_sample_rate():
    t = np.arange(8192)
    fs = 1_000_000.0  # 1 MHz
    true_hz = -150_000.0  # -150 kHz (-0.15 normalized)
    x = np.exp(2j * np.pi * (true_hz / fs) * t).astype(np.complex64)

    psd_res = compute_psd(x, segment_length=2048, sample_rate_hz=fs)
    estimates = compute_all_frequency_estimates(x, psd_res, sample_rate_hz=fs)
    
    for est in estimates:
        assert est.frequency_hz is not None
        assert abs(est.frequency_hz - true_hz) < 500.0
