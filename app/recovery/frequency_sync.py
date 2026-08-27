from __future__ import annotations
import numpy as np
import scipy.signal as signal
from .models import FrequencySyncResult, ModulationFamily

def estimate_coarse_cfo_mth_power(
    samples: np.ndarray,
    order: int = 4,
    family: ModulationFamily = ModulationFamily.PSK,
) -> FrequencySyncResult:
    """
    Estimate Carrier Frequency Offset (CFO) using non-linear M-th power phase progression and spectral line refinement.

    Parameters
    ----------
    samples : np.ndarray
        Input complex IQ samples.
    order : int
        Modulation order (2 for BPSK, 4 for QPSK/16QAM, 8 for 8PSK).
    family : ModulationFamily
        Modulation family.

    Returns
    -------
    FrequencySyncResult
    """
    n_samples = len(samples)
    if n_samples < 16:
        return FrequencySyncResult(
            coarse_cfo_normalized=0.0,
            residual_cfo_normalized=0.0,
            cfo_variance=0.0,
            capture_bandwidth=0.5,
            method="insufficient_samples",
            ambiguity_set=(0.0,),
            is_ambiguous=False,
            valid=False,
        )

    # Determine M power for non-linear collapse
    if family == ModulationFamily.PSK:
        m_power = order if order in (2, 4, 8) else 4
    elif family == ModulationFamily.QAM:
        m_power = 4  # Square 16-QAM collapses on 4th power
    else:
        m_power = 2

    # 1. Non-linear M-th power transformation: y[n] = (x[n] / |x[n]|)^M
    amp = np.abs(samples)
    mask = amp > (0.10 * np.mean(amp))
    if np.sum(mask) < 16:
        valid_x = samples
    else:
        valid_x = samples[mask]

    # Normalize magnitude before M-th power to avoid numerical overflow
    norm_x = valid_x / (np.abs(valid_x) + 1e-12)
    y = norm_x ** m_power

    # 2. Method A: Differential Phase Progression
    diff_prod = y[1:] * np.conj(y[:-1])
    diff_angles = np.angle(diff_prod)
    median_angle = float(np.median(diff_angles))
    cfo_diff = median_angle / (2.0 * np.pi * m_power)
    var_angle = float(np.var(diff_angles)) / ((2.0 * np.pi * m_power) ** 2)

    # 3. Method B: FFT Spectral Peak on y[n]
    n_fft = min(8192, 1 << int(np.floor(np.log2(len(y)))))
    if n_fft >= 64:
        win = np.hanning(n_fft)
        y_seg = y[:n_fft] * win
        spec = np.abs(np.fft.fft(y_seg, n=n_fft))
        peak_idx = int(np.argmax(spec))
        
        # Parabolic interpolation around peak
        alpha_val = float(spec[(peak_idx - 1) % n_fft])
        beta_val = float(spec[peak_idx])
        gamma_val = float(spec[(peak_idx + 1) % n_fft])
        denom = alpha_val - 2.0 * beta_val + gamma_val
        delta_p = 0.5 * (alpha_val - gamma_val) / denom if abs(denom) > 1e-9 else 0.0
        
        freq_bin = (peak_idx + delta_p)
        if freq_bin > n_fft / 2:
            freq_bin -= n_fft
        
        cfo_fft = float(freq_bin / (n_fft * m_power))
    else:
        cfo_fft = cfo_diff

    # Select more reliable estimator (FFT peak if line prominence is strong, otherwise diff median)
    cfo_est = cfo_fft if abs(cfo_fft - cfo_diff) < (0.25 / m_power) else cfo_diff
    
    # Capture bandwidth before aliasing is +/- 1 / (2*M)
    capture_bw = 1.0 / (2.0 * m_power)

    # Construct M-th power ambiguity set
    ambiguity_set = tuple(sorted([round(float(cfo_est + k / m_power), 6) for k in range(-m_power // 2 + 1, m_power // 2 + 1)]))
    is_ambiguous = abs(cfo_est) > (0.80 * capture_bw)

    return FrequencySyncResult(
        coarse_cfo_normalized=round(float(cfo_est), 6),
        residual_cfo_normalized=0.0,
        cfo_variance=round(float(var_angle), 8),
        capture_bandwidth=round(float(capture_bw), 4),
        method=f"{m_power}th_power_fft",
        ambiguity_set=ambiguity_set,
        is_ambiguous=is_ambiguous,
        valid=True,
    )

def estimate_bfsk_frequencies(
    samples: np.ndarray,
    sps: float = 8.0,
) -> tuple[float, float, float, FrequencySyncResult]:
    """
    Estimate FSK space (f0) and mark (f1) frequencies and separation from complex samples.

    Parameters
    ----------
    samples : np.ndarray
        Input samples.
    sps : float
        Estimated samples per symbol.

    Returns
    -------
    f0 : float
        Space frequency in cycles/sample.
    f1 : float
        Mark frequency in cycles/sample.
    delta_f : float
        Frequency deviation separation in cycles/sample.
    result : FrequencySyncResult
    """
    if len(samples) < 32:
        return -0.125, +0.125, 0.25, FrequencySyncResult(
            coarse_cfo_normalized=0.0,
            residual_cfo_normalized=0.0,
            cfo_variance=0.0,
            capture_bandwidth=0.5,
            method="fsk_default",
            valid=False,
        )

    # 1. Instantaneous frequency derivative
    diff_prod = samples[1:] * np.conj(samples[:-1])
    inst_freq = np.angle(diff_prod) / (2.0 * np.pi)

    # 2. Histogram cluster peaks
    hist, bin_edges = np.histogram(inst_freq, bins=64, range=(-0.5, 0.5))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Find two highest distinct peaks
    peaks, props = signal.find_peaks(hist, distance=8, prominence=np.max(hist) * 0.15)
    if len(peaks) >= 2:
        top_two = sorted(peaks, key=lambda p: -hist[p])[:2]
        f_peaks = sorted([float(centers[p]) for p in top_two])
        f0, f1 = f_peaks[0], f_peaks[1]
    else:
        # Fallback: estimate from positive and negative medians
        neg_vals = inst_freq[inst_freq < 0]
        pos_vals = inst_freq[inst_freq > 0]
        f0 = float(np.median(neg_vals)) if len(neg_vals) > 0 else -0.125
        f1 = float(np.median(pos_vals)) if len(pos_vals) > 0 else +0.125

    delta_f = abs(f1 - f0)
    cfo = 0.5 * (f0 + f1)

    result = FrequencySyncResult(
        coarse_cfo_normalized=round(float(cfo), 6),
        residual_cfo_normalized=0.0,
        cfo_variance=round(float(np.var(inst_freq)), 6),
        capture_bandwidth=round(float(delta_f), 4),
        method="fsk_bimodal_clustering",
        ambiguity_set=(round(float(cfo), 6),),
        is_ambiguous=False,
        valid=True,
    )
    return float(f0), float(f1), float(delta_f), result

def correct_frequency_offset(
    samples: np.ndarray,
    cfo_normalized: float,
) -> np.ndarray:
    """
    Apply carrier frequency offset correction: x_corr[n] = x[n] * exp(-j * 2 * pi * cfo * n).

    Parameters
    ----------
    samples : np.ndarray
        Complex samples.
    cfo_normalized : float
        Frequency offset to subtract in cycles/sample.

    Returns
    -------
    corrected : np.ndarray
    """
    if abs(cfo_normalized) < 1e-9 or len(samples) == 0:
        return samples.copy().astype(np.complex64)

    t = np.arange(len(samples), dtype=np.float64)
    rot = np.exp(-2j * np.pi * cfo_normalized * t).astype(np.complex64)
    return (samples * rot).astype(np.complex64)
