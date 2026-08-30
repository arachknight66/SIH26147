from __future__ import annotations
import numpy as np
import scipy.fft as sp_fft
from app.models.analysis import SpectrumResult
from .windowing import get_window

def compute_spectrum(
    samples: np.ndarray,
    *,
    fft_size: int = 4096,
    window_name: str = "hann",
    sample_rate_hz: float | None = None,
    center_frequency_hz: float | None = None,
    is_complex: bool = True,
    db_floor: float = -120.0,
    normalization: str = "coherent",
) -> SpectrumResult:
    """
    Compute FFT spectrum for complex IQ or real signals with optimized SciPy PocketFFT,
    rigorous frequency axes, and calibrated normalization.
    """
    if len(samples) == 0:
        raise ValueError("Cannot compute spectrum of empty sample array.")
    if fft_size <= 0:
        raise ValueError(f"FFT size must be positive, got {fft_size}.")

    n_samples = len(samples)
    seg_len = min(n_samples, fft_size)
    x = samples[:seg_len]

    win, s1, s2 = get_window(window_name, seg_len)
    x_win = x * win

    if seg_len < fft_size:
        pad_width = fft_size - seg_len
        x_win = np.pad(x_win, (0, pad_width), mode="constant")

    if is_complex:
        fft_raw = sp_fft.fft(x_win, n=fft_size)
        fft_shifted = sp_fft.fftshift(fft_raw)
        freq_norm = sp_fft.fftshift(sp_fft.fftfreq(fft_size, d=1.0))

        if normalization == "coherent":
            scale = 1.0 / (seg_len * s1)
            complex_spectrum = fft_shifted * scale
            power_linear = complex_spectrum.real ** 2 + complex_spectrum.imag ** 2
            magnitude = np.sqrt(power_linear)
        else:  # power
            scale = 1.0 / np.sqrt(seg_len * s2)
            complex_spectrum = fft_shifted * scale
            power_linear = (fft_shifted.real ** 2 + fft_shifted.imag ** 2) / (seg_len * s2)
            magnitude = np.sqrt(power_linear)

        is_two_sided = True
    else:
        # Real signal: compute rfft (one-sided [0, 0.5])
        fft_raw = sp_fft.rfft(x_win.real, n=fft_size)
        complex_spectrum = None
        freq_norm = sp_fft.rfftfreq(fft_size, d=1.0)

        raw_sq = fft_raw.real ** 2 + fft_raw.imag ** 2
        if normalization == "coherent":
            scale = 1.0 / (seg_len * s1)
            magnitude = np.sqrt(raw_sq) * scale
            if len(magnitude) > 2:
                magnitude[1:-1] *= np.sqrt(2.0)
            power_linear = magnitude ** 2
        else:
            power_linear = raw_sq / (seg_len * s2)
            if len(power_linear) > 2:
                power_linear[1:-1] *= 2.0
            magnitude = np.sqrt(power_linear)

        is_two_sided = False

    # Compute dB spectrum with scientific floor
    peak_p = float(np.max(power_linear)) if len(power_linear) > 0 else 0.0
    eps = 1e-15
    if peak_p > 0.0:
        min_p = peak_p * (10.0 ** (db_floor / 10.0))
        clamped_p = np.maximum(power_linear, max(min_p, eps))
        power_db = 10.0 * np.log10(clamped_p)
    else:
        power_db = np.full_like(power_linear, db_floor)

    # Frequency axes
    if sample_rate_hz is not None and sample_rate_hz > 0:
        freq_hz = freq_norm * sample_rate_hz
        freq_unit = "Hz"
        bin_res = sample_rate_hz / fft_size
        bin_res_unit = "Hz"
    else:
        freq_hz = freq_norm.copy()
        freq_unit = "cycles/sample"
        bin_res = 1.0 / fft_size
        bin_res_unit = "cycles/sample"

    ref = "recording_center_frequency" if (center_frequency_hz is not None and sample_rate_hz is not None) else "baseband_normalized"

    return SpectrumResult(
        frequencies=freq_hz,
        frequencies_normalized=freq_norm,
        magnitude_spectrum=magnitude,
        power_spectrum_db=power_db,
        complex_spectrum=complex_spectrum if is_complex else None,
        fft_size=fft_size,
        window=window_name,
        coherent_gain=s1,
        noise_power_gain=s2,
        db_floor=db_floor,
        frequency_unit=freq_unit,
        frequency_reference=ref,
        is_complex=is_complex,
        bin_resolution=bin_res,
        bin_resolution_unit=bin_res_unit,
    )
