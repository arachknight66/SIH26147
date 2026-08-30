from __future__ import annotations
import numpy as np
import scipy.fft as sp_fft
from app.models.analysis import AutocorrelationResult

def compute_autocorrelation(
    samples: np.ndarray,
    *,
    max_lag: int = 2048,
) -> AutocorrelationResult:
    """
    Compute normalized autocorrelation R_xx[k] using optimized FFT-based correlation with fast lengths.
    """
    n_samples = len(samples)
    if n_samples == 0:
        raise ValueError("Cannot compute autocorrelation of empty array.")

    actual_max_lag = min(max_lag, n_samples - 1)
    if actual_max_lag <= 0:
        return AutocorrelationResult(
            lags=np.array([0]),
            complex_autocorrelation=np.array([1.0 + 0j], dtype=np.complex64),
            normalized_magnitude=np.array([1.0], dtype=np.float64),
            max_lag=0,
        )

    x = samples.astype(np.complex64, copy=False)
    total_energy = float(np.sum(x.real ** 2 + x.imag ** 2))
    if total_energy <= 1e-15:
        lags = np.arange(actual_max_lag + 1)
        return AutocorrelationResult(
            lags=lags,
            complex_autocorrelation=np.zeros(len(lags), dtype=np.complex64),
            normalized_magnitude=np.zeros(len(lags), dtype=np.float64),
            max_lag=actual_max_lag,
        )

    # Use fast 2,3,5-smooth composite length for FFT
    n_fft = sp_fft.next_fast_len(2 * n_samples - 1)
    fft_x = sp_fft.fft(x, n=n_fft)
    psd_x = fft_x.real ** 2 + fft_x.imag ** 2
    r_full = sp_fft.ifft(psd_x, n=n_fft)

    # Non-negative lags 0 .. actual_max_lag
    r_raw = r_full[: actual_max_lag + 1]

    # Scale by total zero-lag energy and sample count weighting
    r_norm = (r_raw / total_energy).astype(np.complex64)
    mag_norm = np.abs(r_norm).astype(np.float64)

    lags = np.arange(actual_max_lag + 1)

    return AutocorrelationResult(
        lags=lags,
        complex_autocorrelation=r_norm,
        normalized_magnitude=mag_norm,
        max_lag=actual_max_lag,
    )
