import numpy as np
import pytest
from app.dsp.detection import detect_burst_regions_time, detect_signal_regions_spectral
from app.dsp.noise import estimate_noise_floor
from app.dsp.psd import compute_psd

def test_detect_spectral_regions_single_tone():
    t = np.arange(8192)
    tone = np.exp(2j * np.pi * 0.15 * t).astype(np.complex64)
    noise = (np.random.normal(0, 0.05, 8192) + 1j * np.random.normal(0, 0.05, 8192)).astype(np.complex64)
    x = tone + noise

    psd_res = compute_psd(x, segment_length=2048, is_complex=True)
    noise_est = estimate_noise_floor(psd_res.psd)
    regions = detect_signal_regions_spectral(psd_res, noise_est, threshold_db_offset=10.0)

    assert len(regions) == 1
    reg = regions[0]
    assert np.isclose(reg.center_freq_normalized, 0.15, atol=0.01)
    assert reg.detection_score > 0.5
    assert reg.estimated_snr_db > 10.0

def test_detect_spectral_regions_multiple_tones():
    t = np.arange(8192)
    tone1 = np.exp(2j * np.pi * -0.2 * t).astype(np.complex64)
    tone2 = np.exp(2j * np.pi * +0.3 * t).astype(np.complex64)
    noise = (np.random.normal(0, 0.05, 8192) + 1j * np.random.normal(0, 0.05, 8192)).astype(np.complex64)
    x = tone1 + tone2 + noise

    psd_res = compute_psd(x, segment_length=2048, is_complex=True)
    noise_est = estimate_noise_floor(psd_res.psd)
    regions = detect_signal_regions_spectral(psd_res, noise_est, threshold_db_offset=10.0)

    assert len(regions) == 2
    centers = sorted([r.center_freq_normalized for r in regions])
    assert np.isclose(centers[0], -0.2, atol=0.01)
    assert np.isclose(centers[1], +0.3, atol=0.01)

def test_detect_burst_regions_time():
    # 2048 samples: 512 noise, 1024 signal, 512 noise
    np.random.seed(42)
    n = 2048
    noise = (np.random.normal(0, 0.05, n) + 1j * np.random.normal(0, 0.05, n)).astype(np.complex64)
    burst = np.exp(2j * np.pi * 0.1 * np.arange(1024)).astype(np.complex64)
    noise[512:1536] += burst

    bursts = detect_burst_regions_time(noise, threshold_db_offset=6.0, smooth_window=32, min_burst_samples=128)
    assert len(bursts) == 1
    b = bursts[0]
    assert b.start_sample is not None and abs(b.start_sample - 512) < 50
    assert b.end_sample is not None and abs(b.end_sample - 1536) < 50
