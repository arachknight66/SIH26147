from __future__ import annotations
import numpy as np
import scipy.signal as signal
from .models import FeatureValidity, FrequencyFeatures

def extract_frequency_features(samples: np.ndarray) -> FrequencyFeatures:
    """
    Extract instantaneous frequency moments and FSK bimodal state clustering features.

    Parameters
    ----------
    samples : np.ndarray
        Complex signal samples.

    Returns
    -------
    FrequencyFeatures
    """
    n_samples = len(samples)
    if n_samples < 16:
        return FrequencyFeatures(
            inst_freq_mean=0.0,
            inst_freq_var=0.0,
            inst_freq_median=0.0,
            inst_freq_mad=0.0,
            bimodal_separation=None,
            bimodal_prominence=0.0,
            state_occupancy_ratio=0.0,
            validity=FeatureValidity.UNAVAILABLE,
        )

    amp = np.abs(samples)
    rms_val = float(np.sqrt(np.mean(amp ** 2)))
    mask = (amp[1:] >= 0.05 * rms_val) & (amp[:-1] >= 0.05 * rms_val)

    if np.sum(mask) < 16:
        return FrequencyFeatures(
            inst_freq_mean=0.0,
            inst_freq_var=0.0,
            inst_freq_median=0.0,
            inst_freq_mad=0.0,
            bimodal_separation=None,
            bimodal_prominence=0.0,
            state_occupancy_ratio=0.0,
            validity=FeatureValidity.UNRELIABLE,
        )

    # Instantaneous frequency: f_inst = angle(x[n] * conj(x[n-1])) / (2 * pi)
    diff_prod = samples[1:] * np.conj(samples[:-1])
    valid_diff = diff_prod[mask]
    f_inst = (np.angle(valid_diff) / (2.0 * np.pi)).astype(np.float64)

    mean_f = float(np.mean(f_inst))
    var_f = float(np.var(f_inst))
    med_f = float(np.median(f_inst))
    mad_f = float(np.median(np.abs(f_inst - med_f)))

    # Histogram clustering for bimodal state separation (BFSK indicator)
    counts, bin_edges = np.histogram(f_inst, bins=64, range=(-0.5, 0.5), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    peaks, props = signal.find_peaks(counts, prominence=0.5, distance=3)
    prominences = props.get("prominences", np.zeros(len(peaks)))

    bimodal_sep: float | None = None
    bimodal_prom = 0.0
    state_ratio = 0.0

    if len(peaks) >= 2:
        # Sort top 2 peaks by prominence
        top_idx = np.argsort(prominences)[::-1][:2]
        p1, p2 = peaks[top_idx[0]], peaks[top_idx[1]]
        f1, f2 = bin_centers[p1], bin_centers[p2]
        bimodal_sep = float(abs(f2 - f1))
        
        # State occupancy around the two peaks
        c1, c2 = counts[p1], counts[p2]
        state_ratio = float(min(c1, c2) / max(c1, c2)) if max(c1, c2) > 0 else 0.0
        bimodal_prom = float(min(prominences[top_idx[0]], prominences[top_idx[1]]) / (np.max(counts) + 1e-12))
    elif len(peaks) == 1:
        bimodal_prom = 0.0
        state_ratio = 0.0

    validity = FeatureValidity.VALID if (n_samples >= 64 and len(f_inst) >= 32) else FeatureValidity.PARTIALLY_VALID

    return FrequencyFeatures(
        inst_freq_mean=round(mean_f, 6),
        inst_freq_var=round(var_f, 6),
        inst_freq_median=round(med_f, 6),
        inst_freq_mad=round(mad_f, 6),
        bimodal_separation=round(bimodal_sep, 6) if bimodal_sep is not None else None,
        bimodal_prominence=round(bimodal_prom, 4),
        state_occupancy_ratio=round(state_ratio, 4),
        validity=validity,
    )
