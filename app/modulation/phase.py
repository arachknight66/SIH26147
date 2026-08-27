from __future__ import annotations
import numpy as np
from .models import FeatureValidity, PhaseFeatures

def extract_phase_features(samples: np.ndarray) -> PhaseFeatures:
    """
    Extract masked phase increments and M-th power non-linear phase collapse features.

    Parameters
    ----------
    samples : np.ndarray
        Complex signal samples.

    Returns
    -------
    PhaseFeatures
    """
    n_samples = len(samples)
    if n_samples < 16:
        return PhaseFeatures(
            phase_inc_mean=0.0,
            phase_inc_var=0.0,
            phase_inc_kurtosis=0.0,
            var_phase_sq=1.0,
            var_phase_4th=1.0,
            var_phase_8th=1.0,
            valid_fraction=0.0,
            validity=FeatureValidity.UNAVAILABLE,
        )

    amp = np.abs(samples)
    rms_val = float(np.sqrt(np.mean(amp ** 2)))

    if rms_val <= 1e-12:
        return PhaseFeatures(
            phase_inc_mean=0.0,
            phase_inc_var=0.0,
            phase_inc_kurtosis=0.0,
            var_phase_sq=1.0,
            var_phase_4th=1.0,
            var_phase_8th=1.0,
            valid_fraction=0.0,
            validity=FeatureValidity.UNAVAILABLE,
        )

    # Magnitude masking threshold: 5% of RMS
    mask = amp >= (0.05 * rms_val)
    valid_fraction = float(np.mean(mask))

    if valid_fraction < 0.20 or np.sum(mask) < 16:
        return PhaseFeatures(
            phase_inc_mean=0.0,
            phase_inc_var=0.0,
            phase_inc_kurtosis=0.0,
            var_phase_sq=1.0,
            var_phase_4th=1.0,
            var_phase_8th=1.0,
            valid_fraction=round(valid_fraction, 4),
            validity=FeatureValidity.UNRELIABLE,
        )

    # 1. Phase increments Delta phi[n] = angle(x[n] * conj(x[n-1]))
    diff_prod = samples[1:] * np.conj(samples[:-1])
    diff_mask = mask[1:] & mask[:-1]

    if np.sum(diff_mask) >= 8:
        valid_diff = diff_prod[diff_mask]
        phase_increments = np.angle(valid_diff)
        
        # Circular mean and variance of increments
        mean_vec = np.mean(np.exp(1j * phase_increments))
        inc_mean = float(np.angle(mean_vec))
        inc_var = float(1.0 - np.abs(mean_vec))

        # Kurtosis of increments
        std_inc = float(np.std(phase_increments))
        if std_inc > 1e-6:
            z_inc = (phase_increments - np.mean(phase_increments)) / std_inc
            inc_kurt = float(np.mean(z_inc ** 4) - 3.0)
        else:
            inc_kurt = 0.0
    else:
        inc_mean = 0.0
        inc_var = 1.0
        inc_kurt = 0.0

    # 2. Static M-th power phase collapse on valid samples
    valid_x = samples[mask]
    
    # Square (BPSK collapse)
    x2 = valid_x ** 2
    r2_s = float(np.abs(np.mean(x2 / (np.abs(x2) + 1e-12))))
    var_sq = float(np.clip(1.0 - r2_s, 0.0, 1.0))

    # 4th power (QPSK collapse)
    x4 = valid_x ** 4
    r4_s = float(np.abs(np.mean(x4 / (np.abs(x4) + 1e-12))))
    var_4th = float(np.clip(1.0 - r4_s, 0.0, 1.0))

    # 8th power (8PSK collapse)
    x8 = valid_x ** 8
    r8_s = float(np.abs(np.mean(x8 / (np.abs(x8) + 1e-12))))
    var_8th = float(np.clip(1.0 - r8_s, 0.0, 1.0))

    validity = FeatureValidity.VALID if (n_samples >= 64 and valid_fraction >= 0.70) else FeatureValidity.PARTIALLY_VALID

    return PhaseFeatures(
        phase_inc_mean=round(inc_mean, 6),
        phase_inc_var=round(inc_var, 6),
        phase_inc_kurtosis=round(inc_kurt, 4),
        var_phase_sq=round(var_sq, 6),
        var_phase_4th=round(var_4th, 6),
        var_phase_8th=round(var_8th, 6),
        valid_fraction=round(valid_fraction, 4),
        validity=validity,
    )
