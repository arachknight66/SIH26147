from __future__ import annotations
import numpy as np
import scipy.signal as signal
from app.models.analysis import PSDResult

def compute_psd(
    samples: np.ndarray,
    *,
    sample_rate_hz: float | None = None,
    segment_length: int = 4096,
    overlap: float = 0.5,
    window_name: str = "hann",
    fft_size: int | None = None,
    detrend: str = "constant",
    scaling: str = "density",
    is_complex: bool = True,
) -> PSDResult:
    """
    Compute Welch Power Spectral Density (PSD) estimate.

    Welch's method divides the signal into overlapping segments, windows each segment,
    computes periodograms, and averages them to reduce variance of the spectral estimate.
    The variance reduction factor is approximately 1/K for K independent segments,
    at the expense of frequency resolution (Delta f ~ 1/segment_length).

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.
    sample_rate_hz : float | None
        Sampling rate in Hz (if known from metadata).
    segment_length : int
        Number of samples per Welch segment (nperseg).
    overlap : float
        Fractional overlap between segments (e.g. 0.5 for 50%).
    window_name : str
        Window function ('hann', 'blackman', etc.).
    fft_size : int | None
        FFT size for each segment (if None, equal to segment_length).
    detrend : str
        Detrending mode ('constant', 'linear', or False).
    scaling : str
        'density' (power spectral density) or 'spectrum' (power spectrum).
    is_complex : bool
        True if complex IQ; False if real-valued.

    Returns
    -------
    PSDResult
    """
    if len(samples) == 0:
        raise ValueError("Cannot compute PSD of empty sample array.")
    
    n_samples = len(samples)
    nperseg = min(n_samples, max(16, segment_length))
    noverlap = int(np.clip(nperseg * overlap, 0, nperseg - 1))
    nfft = fft_size if (fft_size is not None and fft_size >= nperseg) else nperseg

    # Use normalized sample rate fs=1.0 for normalized computation
    fs_norm = 1.0

    detrend_arg = detrend if detrend in ("constant", "linear") else False

    if is_complex:
        freqs_raw, psd_raw = signal.welch(
            samples,
            fs=fs_norm,
            window=window_name,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend_arg,
            return_onesided=False,
            scaling=scaling,
        )
        # Shift to centered frequency ordering [-0.5, 0.5)
        freq_norm = np.fft.fftshift(freqs_raw)
        psd = np.fft.fftshift(psd_raw)
        is_two_sided = True
    else:
        freq_norm, psd = signal.welch(
            samples.real,
            fs=fs_norm,
            window=window_name,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend_arg,
            return_onesided=True,
            scaling=scaling,
        )
        is_two_sided = False

    # Prevent log10(0)
    eps = 1e-15
    psd_safe = np.maximum(psd, eps)
    psd_db = 10.0 * np.log10(psd_safe)

    if sample_rate_hz is not None and sample_rate_hz > 0:
        freq_hz = freq_norm * sample_rate_hz
        freq_unit = "Hz"
        bin_res = sample_rate_hz / nfft
    else:
        freq_hz = freq_norm.copy()
        freq_unit = "cycles/sample"
        bin_res = 1.0 / nfft

    return PSDResult(
        frequencies=freq_hz,
        frequencies_normalized=freq_norm,
        psd=psd,
        psd_db=psd_db,
        segment_length=nperseg,
        overlap=overlap,
        window=window_name,
        fft_size=nfft,
        detrend=str(detrend),
        scaling=scaling,
        frequency_unit=freq_unit,
        is_two_sided=is_two_sided,
        bin_resolution=bin_res,
    )
