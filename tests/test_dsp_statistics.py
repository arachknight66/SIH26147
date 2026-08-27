import numpy as np
import pytest
from app.dsp.statistics import compute_dc_offset, compute_time_statistics, detect_clipping

def test_time_statistics_known_signal():
    # Signal with known I, Q distribution
    i = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    q = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    x = (i + 1j * q).astype(np.complex64)

    stats = compute_time_statistics(x)
    assert np.isclose(stats.mean_i, 2.5)
    assert np.isclose(stats.mean_q, 6.5)
    assert np.isclose(stats.variance_i, 1.25)
    assert np.isclose(stats.variance_q, 1.25)
    assert np.isclose(stats.iq_covariance, 1.25)
    assert np.isclose(stats.iq_correlation, 1.0)
    assert stats.phase_valid_fraction == 1.0

def test_dc_offset_detection():
    t = np.arange(1000)
    # AC sinusoid + DC offset
    ac = np.exp(2j * np.pi * 0.1 * t)
    dc = 0.5 - 0.3j
    x = (ac + dc).astype(np.complex64)

    dc_res = compute_dc_offset(x)
    assert np.isclose(dc_res.i_offset, 0.5, atol=0.01)
    assert np.isclose(dc_res.q_offset, -0.3, atol=0.01)
    assert np.isclose(dc_res.magnitude, np.sqrt(0.5**2 + 0.3**2), atol=0.01)

def test_clipping_detection():
    # Int16 saturated values
    x_clean = np.array([1000 + 1000j, 2000 - 2000j], dtype=np.complex64)
    clip_clean = detect_clipping(x_clean, original_dtype="int16")
    assert clip_clean.is_clipped is False

    x_clipped = np.array([32767 + 0j, 32767 + 100j, 32766 - 500j, 0 + 32767j], dtype=np.complex64)
    clip_sat = detect_clipping(x_clipped, original_dtype="int16")
    assert clip_sat.is_clipped is True
    assert clip_sat.fraction_near_extrema > 0.5
