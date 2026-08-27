from __future__ import annotations
import numpy as np
import scipy.signal as signal
from app.models.analysis import SpectrogramResult

def compute_spectrogram(
    samples: np.ndarray,
    *,
    sample_rate_hz: float | None = None,
    window_length: int = 2048,
    fft_size: int | None = None,
    overlap: float = 0.5,
    window_name: str = "hann",
    is_complex: bool = True,
) -> SpectrogramResult:
    """
    Compute STFT-based time-frequency spectrogram.

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.
    sample_rate_hz : float | None
        Physical sampling rate if known from metadata.
    window_length : int
        Window length in samples per time slice (nperseg).
    fft_size : int | None
        FFT size per slice (if None, equal to window_length).
    overlap : float
        Fractional overlap between slices (e.g. 0.5 for 50%).
    window_name : str
        Window function name.
    is_complex : bool
        True for complex IQ (two-sided frequency); False for real.

    Returns
    -------
    SpectrogramResult
    """
    if len(samples) == 0:
        raise ValueError("Cannot compute spectrogram of empty sample array.")

    n_samples = len(samples)
    nperseg = min(n_samples, max(16, window_length))
    noverlap = int(np.clip(nperseg * overlap, 0, nperseg - 1))
    nfft = fft_size if (fft_size is not None and fft_size >= nperseg) else nperseg
    hop_size = nperseg - noverlap

    fs_norm = 1.0

    if is_complex:
        freqs_raw, times_raw, sxx_raw = signal.spectrogram(
            samples,
            fs=fs_norm,
            window=window_name,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=False,
            mode="psd",
        )
        freq_norm = np.fft.fftshift(freqs_raw)
        sxx = np.fft.fftshift(sxx_raw, axes=0)
    else:
        freq_norm, times_raw, sxx = signal.spectrogram(
            samples.real,
            fs=fs_norm,
            window=window_name,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=True,
            mode="psd",
        )

    # dB matrix with floor
    eps = 1e-15
    power_matrix_db = 10.0 * np.log10(np.maximum(sxx, eps))

    if sample_rate_hz is not None and sample_rate_hz > 0:
        freq_axis = freq_norm * sample_rate_hz
        time_axis = times_raw / sample_rate_hz
        time_unit = "seconds"
        freq_unit = "Hz"
    else:
        freq_axis = freq_norm.copy()
        time_axis = times_raw.copy()  # in normalized sample steps
        time_unit = "samples"
        freq_unit = "cycles/sample"

    return SpectrogramResult(
        time_axis=time_axis,
        time_unit=time_unit,
        frequency_axis=freq_axis,
        frequency_axis_normalized=freq_norm,
        power_matrix_db=power_matrix_db,
        window_length=nperseg,
        fft_size=nfft,
        hop_size=hop_size,
        window=window_name,
        frequency_unit=freq_unit,
    )
