import numpy as np
import pytest
from app.dsp.spectrum import compute_spectrum

def test_spectrum_complex_sinusoid():
    # Pure complex sinusoid at normalized frequency +0.125
    t = np.arange(1024, dtype=np.float32)
    freq = 0.125
    x = (2.0 * np.exp(2j * np.pi * freq * t)).astype(np.complex64)

    res = compute_spectrum(x, fft_size=1024, window_name="rectangular", is_complex=True)
    
    assert res.is_complex is True
    assert len(res.frequencies_normalized) == 1024
    assert np.isclose(res.frequencies_normalized[0], -0.5)
    assert res.frequencies_normalized[-1] < 0.5

    # Peak should be at +0.125
    peak_idx = int(np.argmax(res.magnitude_spectrum))
    peak_freq = res.frequencies_normalized[peak_idx]
    assert np.isclose(peak_freq, freq, atol=1e-3)
    # Peak amplitude should match 2.0 with rectangular window and coherent normalization
    assert np.isclose(res.magnitude_spectrum[peak_idx], 2.0, atol=1e-2)

def test_spectrum_real_signal():
    t = np.arange(1024, dtype=np.float32)
    freq = 0.2
    x = (np.cos(2 * np.pi * freq * t)).astype(np.complex64)

    res = compute_spectrum(x, fft_size=1024, window_name="hann", is_complex=False)
    assert res.is_complex is False
    assert len(res.frequencies_normalized) == 513
    assert np.isclose(res.frequencies_normalized[0], 0.0)
    assert np.isclose(res.frequencies_normalized[-1], 0.5)

def test_spectrum_with_sample_rate():
    t = np.arange(1024, dtype=np.float32)
    fs = 1_000_000.0  # 1 MHz
    freq_hz = 250_000.0  # +250 kHz (+0.25 normalized)
    x = np.exp(2j * np.pi * (freq_hz / fs) * t).astype(np.complex64)

    res = compute_spectrum(x, fft_size=1024, sample_rate_hz=fs, is_complex=True)
    assert res.frequency_unit == "Hz"
    peak_idx = int(np.argmax(res.magnitude_spectrum))
    assert np.isclose(res.frequencies[peak_idx], freq_hz, atol=1000.0)
