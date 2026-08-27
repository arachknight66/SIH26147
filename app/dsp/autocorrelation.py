from __future__ import annotations
import numpy as np
from app.models.analysis import AutocorrelationResult

def compute_autocorrelation(
    samples: np.ndarray,
    *,
    max_lag: int = 2048,
) -> AutocorrelationResult:
    """
    Compute normalized autocorrelation R_xx[k] using FFT-based correlation.

    R_xx[k] = (sum_{n} x[n+k] * conj(x[n])) / sum_{n} |x[n]|^2

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    AutocorrelationResult
    """
    n_samples = len(samples)
    if n_samples == 0:
        raise ValueError("Cannot compute autocorrelation of empty array.")

    actual_max_lag = min(max_lag, n_samples - 1)
    if actual_max_lag <= 0:
        return AutocorrelationResult(
            lags=np.array([0]),
            complex_autocorrelation=np.array([1.0 + 0j]),
            normalized_magnitude=np.array([1.0]),
            max_lag=0,
        )

    # Center samples by removing mean for correlation
    x = samples.astype(np.complex64)
    total_energy = float(np.sum(np.abs(x) ** 2))
    if total_energy <= 1e-15:
        lags = np.arange(actual_max_lag + 1)
        return AutocorrelationResult(
            lags=lags,
            complex_autocorrelation=np.zeros(len(lags), dtype=np.complex64),
            normalized_magnitude=np.zeros(len(lags), dtype=np.float64),
            max_lag=actual_max_lag,
        )

    # Next power of 2 for fast zero-padded FFT
    n_fft = 1 << int(np.ceil(np.log2(2 * n_samples - 1)))
    fft_x = np.fft.fft(x, n=n_fft)
    psd_x = np.abs(fft_x) ** 2
    r_full = np.fft.ifft(psd_x, n=n_fft)

    # Non-negative lags 0 .. actual_max_lag
    r_raw = r_full[: actual_max_lag + 1]
    
    # Scale by total zero-lag energy and sample count weighting
    # Biased estimator ensures positive semi-definiteness: R[k] = r_raw[k] / total_energy
    r_norm = (r_raw / total_energy).astype(np.complex64)
    mag_norm = np.abs(r_norm).astype(np.float64)

    lags = np.arange(actual_max_lag + 1)

    return AutocorrelationResult(
        lags=lags,
        complex_autocorrelation=r_norm,
        normalized_magnitude=mag_norm,
        max_lag=actual_max_lag,
    )
