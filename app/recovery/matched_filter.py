from __future__ import annotations
import numpy as np
import scipy.signal as signal

def design_rrc_filter(
    sps: float,
    alpha: float = 0.35,
    span_symbols: int = 8,
) -> np.ndarray:
    """
    Design a mathematically rigorous Root Raised Cosine (RRC) FIR filter.

    Parameters
    ----------
    sps : float
        Samples per symbol (oversampling factor).
    alpha : float
        Roll-off factor (0.1 to 0.9).
    span_symbols : int
        Total filter length in symbol intervals.

    Returns
    -------
    h : np.ndarray
        1D float64 impulse response normalized to unit energy (sum(h^2) == 1.0).
    """
    sps_int = max(2, int(round(sps)))
    n_taps = span_symbols * sps_int + 1
    t = np.arange(-(n_taps // 2), (n_taps // 2) + 1, dtype=np.float64) / sps
    h = np.zeros(len(t), dtype=np.float64)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-6:
            term1 = (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
            term2 = (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            h[i] = (alpha / np.sqrt(2.0)) * (term1 + term2)
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + (4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha)))
            denom = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / denom

    # Normalize to unit energy
    energy = np.sum(h ** 2)
    if energy > 0:
        h = h / np.sqrt(energy)

    return h

def apply_matched_filter(
    samples: np.ndarray,
    sps: float,
    alpha: float = 0.35,
    span_symbols: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Root Raised Cosine matched filtering to complex baseband samples.

    Parameters
    ----------
    samples : np.ndarray
        Input complex IQ samples.
    sps : float
        Samples per symbol.
    alpha : float
        RRC roll-off factor.
    span_symbols : int
        Filter span in symbols.

    Returns
    -------
    filtered : np.ndarray
        Filtered complex64 samples (same length as input).
    filter_taps : np.ndarray
        Applied RRC filter coefficients.
    """
    if len(samples) < 8:
        return samples.copy().astype(np.complex64), np.array([1.0], dtype=np.float64)

    h = design_rrc_filter(sps=sps, alpha=alpha, span_symbols=span_symbols)
    filtered = signal.convolve(samples, h, mode="same")
    return filtered.astype(np.complex64), h

def validate_rrc_properties(h: np.ndarray, sps: int) -> dict[str, Any]:
    """
    Verify mathematical properties of the RRC filter: symmetry, energy, group delay, and ISI.

    Parameters
    ----------
    h : np.ndarray
        Filter taps.
    sps : int
        Samples per symbol.

    Returns
    -------
    metrics : dict[str, Any]
    """
    # 1. Symmetry check: h[n] == h[N-1-n]
    symmetry_error = float(np.max(np.abs(h - h[::-1])))
    is_symmetric = symmetry_error < 1e-12

    # 2. Energy normalization check
    energy = float(np.sum(h ** 2))
    energy_error = abs(energy - 1.0)
    is_normalized = energy_error < 1e-10

    # 3. Group delay
    group_delay_samples = (len(h) - 1) // 2

    # 4. Cascaded Raised Cosine (RC) ISI Check: conv(h, h) should have Nyquist zero-crossings at k*sps (k != 0)
    rc = signal.convolve(h, h, mode="full")
    center_idx = len(rc) // 2
    rc_peak = rc[center_idx]
    
    isi_errors = []
    for k in range(1, (len(rc) // 2) // sps):
        val_pos = abs(rc[center_idx + k * sps]) / (abs(rc_peak) + 1e-12)
        val_neg = abs(rc[center_idx - k * sps]) / (abs(rc_peak) + 1e-12)
        isi_errors.extend([val_pos, val_neg])

    max_isi = float(np.max(isi_errors)) if isi_errors else 0.0

    return {
        "is_symmetric": is_symmetric,
        "symmetry_error": symmetry_error,
        "is_normalized": is_normalized,
        "energy": energy,
        "group_delay_samples": group_delay_samples,
        "max_nyquist_isi": max_isi,
    }
