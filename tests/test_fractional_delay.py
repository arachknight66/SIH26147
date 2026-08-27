import numpy as np
import pytest
from app.recovery.fractional_delay import (
    interpolate_sample_cubic,
    interpolate_sample_linear,
    interpolate_vector,
)

def test_fractional_delay_exact_points():
    samples = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j, 5.0 + 0j], dtype=np.complex64)
    # At integer points, cubic interpolation should return exact sample
    assert np.isclose(interpolate_sample_cubic(samples, 1.0), 2.0 + 0j)
    assert np.isclose(interpolate_sample_cubic(samples, 2.0), 3.0 + 0j)
    assert np.isclose(interpolate_sample_linear(samples, 2.0), 3.0 + 0j)

def test_fractional_delay_midpoint():
    samples = np.array([0.0 + 0j, 1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=np.complex64)
    # Midpoint between 1.0 and 2.0 is 1.5
    assert np.isclose(interpolate_sample_linear(samples, 1.5), 1.5 + 0j)
    assert np.isclose(interpolate_sample_cubic(samples, 1.5), 1.5 + 0j)

def test_fractional_delay_vector():
    samples = np.sin(np.linspace(0, 2 * np.pi, 64)).astype(np.complex64)
    indices = np.array([0.5, 1.5, 2.5, 3.5])
    vec = interpolate_vector(samples, indices, method="cubic")
    assert len(vec) == 4
    assert vec.dtype == np.complex64
