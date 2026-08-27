from __future__ import annotations
import numpy as np

def interpolate_sample_linear(
    samples: np.ndarray,
    index: float,
) -> complex:
    """
    Evaluate complex sample value at fractional index using linear interpolation.

    Parameters
    ----------
    samples : np.ndarray
        1D array of complex samples.
    index : float
        Fractional sample index (0-indexed).

    Returns
    -------
    val : complex
    """
    n = len(samples)
    if n == 0:
        return 0.0j
    if index <= 0.0:
        return complex(samples[0])
    if index >= n - 1:
        return complex(samples[-1])

    k = int(np.floor(index))
    mu = float(index - k)
    return (1.0 - mu) * samples[k] + mu * samples[k + 1]

def interpolate_sample_cubic(
    samples: np.ndarray,
    index: float,
) -> complex:
    """
    Evaluate complex sample value at fractional index using 4-point Cubic Hermite (Catmull-Rom) interpolation.

    Parameters
    ----------
    samples : np.ndarray
        1D array of complex samples.
    index : float
        Fractional sample index (0-indexed).

    Returns
    -------
    val : complex
    """
    n = len(samples)
    if n == 0:
        return 0.0j
    if index <= 0.0:
        return complex(samples[0])
    if index >= n - 1:
        return complex(samples[-1])

    k = int(np.floor(index))
    mu = float(index - k)

    # 4-point neighborhood with boundary extension
    y0 = samples[max(0, k - 1)]
    y1 = samples[k]
    y2 = samples[min(n - 1, k + 1)]
    y3 = samples[min(n - 1, k + 2)]

    c0 = y1
    c1 = 0.5 * (y2 - y0)
    c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)

    return ((c3 * mu + c2) * mu + c1) * mu + c0

def interpolate_vector(
    samples: np.ndarray,
    indices: np.ndarray,
    method: str = "cubic",
) -> np.ndarray:
    """
    Interpolate an array of fractional indices.

    Parameters
    ----------
    samples : np.ndarray
        Complex samples.
    indices : np.ndarray
        Array of fractional sample indices.
    method : str
        'cubic' or 'linear'.

    Returns
    -------
    out : np.ndarray
        Interpolated complex64 array.
    """
    fn = interpolate_sample_cubic if method == "cubic" else interpolate_sample_linear
    return np.array([fn(samples, idx) for idx in indices], dtype=np.complex64)
