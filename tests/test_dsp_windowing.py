import numpy as np
import pytest
from app.dsp.windowing import get_window

def test_hann_window():
    win, s1, s2 = get_window("hann", 1024)
    assert len(win) == 1024
    assert np.isclose(s1, 0.5, atol=1e-3)
    assert np.isclose(s2, 0.375, atol=1e-3)

def test_rectangular_window():
    win, s1, s2 = get_window("rectangular", 512)
    assert len(win) == 512
    assert np.all(win == 1.0)
    assert s1 == 1.0
    assert s2 == 1.0

def test_invalid_window_raises():
    with pytest.raises(ValueError, match="Unsupported window"):
        get_window("nonexistent_window_type", 128)

def test_invalid_length_raises():
    with pytest.raises(ValueError, match="positive integer"):
        get_window("hann", 0)
