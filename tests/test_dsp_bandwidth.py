import numpy as np
import pytest
import scipy.signal as signal
from app.dsp.bandwidth import compute_all_bandwidth_estimates, estimate_occupied_bandwidth_power, estimate_occupied_bandwidth_threshold
from app.dsp.noise import estimate_noise_floor
from app.dsp.psd import compute_psd

def test_occupied_bandwidth_bandlimited_noise():
    np.random.seed(789)
    n_samples = 32768
    # Generate white noise and filter with FIR lowpass to bandwidth 0.2 (+-0.1 around baseband)
    noise = (np.random.normal(0, 1.0, n_samples) + 1j * np.random.normal(0, 1.0, n_samples)).astype(np.complex64)
    # Bandpass filter from -0.1 to +0.1 (total BW = 0.20, cutoff = 0.10 / 0.5 = 0.20 of Nyquist)
    cutoff = 0.20  # fraction of Nyquist (Nyquist = 0.5 cycles/sample => 0.10 cycles/sample)
    b = signal.firwin(127, cutoff, pass_zero="lowpass")
    filt = signal.lfilter(b, [1.0], noise).astype(np.complex64)
    # Add light noise floor for realistic threshold detection
    filtered = filt + (np.random.normal(0, 0.03, n_samples) + 1j * np.random.normal(0, 0.03, n_samples)).astype(np.complex64)

    psd_res = compute_psd(filtered, segment_length=4096, is_complex=True)
    noise_est = estimate_noise_floor(psd_res.psd)

    bw_99 = estimate_occupied_bandwidth_power(psd_res, power_fraction=0.99)
    assert bw_99.occupied_bandwidth_normalized is not None
    # Filter passband is 0.20 (+ transition band ~ 0.02)
    assert 0.18 <= bw_99.occupied_bandwidth_normalized <= 0.25

    bw_thresh = estimate_occupied_bandwidth_threshold(psd_res, noise_est, threshold_db_offset=6.0)
    assert bw_thresh.occupied_bandwidth_normalized is not None
    assert 0.18 <= bw_thresh.occupied_bandwidth_normalized <= 0.25

def test_bandwidth_with_sample_rate():
    np.random.seed(789)
    fs = 2_000_000.0  # 2 MHz
    n_samples = 16384
    # Signal with known normalized BW ~0.20 => physical BW ~ 400 kHz
    cutoff = 0.20  # 0.20 * (Fs/2) = 200 kHz half-width => 400 kHz full width
    b = signal.firwin(65, cutoff)
    noise = (np.random.normal(0, 1.0, n_samples) + 1j * np.random.normal(0, 1.0, n_samples)).astype(np.complex64)
    filtered = signal.lfilter(b, [1.0], noise).astype(np.complex64)

    psd_res = compute_psd(filtered, segment_length=2048, sample_rate_hz=fs)
    noise_est = estimate_noise_floor(psd_res.psd)

    estimates = compute_all_bandwidth_estimates(psd_res, noise_est, sample_rate_hz=fs)
    bw_99 = next(e for e in estimates if e.method == "power_containment_99pct")
    assert bw_99.occupied_bandwidth_hz is not None
    assert 350_000.0 <= bw_99.occupied_bandwidth_hz <= 500_000.0
