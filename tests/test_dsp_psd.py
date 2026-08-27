import numpy as np
import pytest
from app.dsp.psd import compute_psd

def test_psd_complex_frequency_axes():
    # 4096 samples of complex sinusoid
    t = np.arange(4096, dtype=np.float32)
    freq = -0.2
    x = np.exp(2j * np.pi * freq * t).astype(np.complex64)

    res = compute_psd(x, segment_length=1024, is_complex=True)
    assert res.is_two_sided is True
    assert len(res.frequencies_normalized) == 1024
    assert np.isclose(res.frequencies_normalized[0], -0.5)
    assert res.frequencies_normalized[-1] < 0.5
    
    # Peak should be at -0.2
    peak_idx = int(np.argmax(res.psd_db))
    assert np.isclose(res.frequencies_normalized[peak_idx], freq, atol=1e-2)

def test_psd_real_signal():
    t = np.arange(4096, dtype=np.float32)
    freq = 0.3
    x = np.sin(2 * np.pi * freq * t).astype(np.complex64)

    res = compute_psd(x, segment_length=1024, is_complex=False)
    assert res.is_two_sided is False
    assert len(res.frequencies_normalized) == 513
    assert np.isclose(res.frequencies_normalized[0], 0.0)
    assert np.isclose(res.frequencies_normalized[-1], 0.5)

def test_psd_with_sample_rate():
    t = np.arange(4096, dtype=np.float32)
    fs = 2_400_000.0
    freq_hz = 500_000.0
    x = np.exp(2j * np.pi * (freq_hz / fs) * t).astype(np.complex64)

    res = compute_psd(x, sample_rate_hz=fs, segment_length=1024)
    assert res.frequency_unit == "Hz"
    peak_idx = int(np.argmax(res.psd))
    assert np.isclose(res.frequencies[peak_idx], freq_hz, atol=3000.0)
