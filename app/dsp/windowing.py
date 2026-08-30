from __future__ import annotations
import functools
import numpy as np
import scipy.signal.windows as windows

_SUPPORTED_WINDOWS = {
    "rectangular": windows.boxcar,
    "rect": windows.boxcar,
    "boxcar": windows.boxcar,
    "hann": windows.hann,
    "hanning": windows.hann,
    "hamming": windows.hamming,
    "blackman": windows.blackman,
    "flattop": windows.flattop,
    "blackmanharris": windows.blackmanharris,
}

@functools.lru_cache(maxsize=256)
def _cached_window_tuple(canonical_name: str, length: int) -> tuple[np.ndarray, float, float]:
    win_func = _SUPPORTED_WINDOWS[canonical_name]
    win = win_func(length).astype(np.float64)
    win.flags.writeable = False  # Protect cached array from mutation
    coherent_gain = float(np.mean(win))
    noise_power_gain = float(np.mean(win ** 2))
    if coherent_gain == 0.0:
        coherent_gain = 1.0
    return win, coherent_gain, noise_power_gain

def get_window(name: str, length: int) -> tuple[np.ndarray, float, float]:
    """
    Generate a window and compute its coherent gain (S1) and noise power gain (S2) with LRU caching.

    Parameters
    ----------
    name : str
        Window type ('rectangular', 'hann', 'blackman', 'hamming', 'flattop', etc.)
    length : int
        Window length in samples.

    Returns
    -------
    win : np.ndarray
        Window coefficients of shape (length,) and dtype float64.
    coherent_gain : float
        Mean of window coefficients: S1 = mean(w) = sum(w) / N.
    noise_power_gain : float
        Mean squared window coefficients: S2 = mean(w^2) = sum(w^2) / N.
    """
    if length <= 0:
        raise ValueError(f"Window length must be a positive integer, got {length}.")
    canonical_name = name.lower().strip()
    if canonical_name not in _SUPPORTED_WINDOWS:
        supported = ", ".join(sorted(set(_SUPPORTED_WINDOWS.keys())))
        raise ValueError(f"Unsupported window '{name}'. Supported windows: {supported}")

    return _cached_window_tuple(canonical_name, length)
