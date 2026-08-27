from __future__ import annotations
import numpy as np
from .models import AmplitudeFeatures, FeatureValidity

def extract_amplitude_features(samples: np.ndarray) -> AmplitudeFeatures:
    """
    Extract statistical moments and envelope variation features from complex samples.

    Parameters
    ----------
    samples : np.ndarray
        Complex signal samples (typically RMS-normalized).

    Returns
    -------
    AmplitudeFeatures
    """
    n_samples = len(samples)
    if n_samples < 16:
        return AmplitudeFeatures(
            mean=0.0,
            rms=0.0,
            variance=0.0,
            coeff_var=0.0,
            kurtosis=0.0,
            skewness=0.0,
            peak_to_rms=0.0,
            norm_variance=0.0,
            validity=FeatureValidity.UNAVAILABLE,
        )

    amp = np.abs(samples).astype(np.float64)
    rms_val = float(np.sqrt(np.mean(amp ** 2)))

    if rms_val <= 1e-12:
        return AmplitudeFeatures(
            mean=0.0,
            rms=0.0,
            variance=0.0,
            coeff_var=0.0,
            kurtosis=0.0,
            skewness=0.0,
            peak_to_rms=0.0,
            norm_variance=0.0,
            validity=FeatureValidity.UNRELIABLE,
        )

    mean_val = float(np.mean(amp))
    var_val = float(np.var(amp))
    std_val = float(np.std(amp))

    coeff_var = float(std_val / mean_val) if mean_val > 1e-12 else 0.0

    # Skewness and excess kurtosis
    if std_val > 1e-12:
        z = (amp - mean_val) / std_val
        skew_val = float(np.mean(z ** 3))
        kurt_val = float(np.mean(z ** 4) - 3.0)
    else:
        skew_val = 0.0
        kurt_val = 0.0

    peak_to_rms = float(np.max(amp) / rms_val)
    norm_var = float(var_val / (rms_val ** 2))

    validity = FeatureValidity.VALID if n_samples >= 64 else FeatureValidity.PARTIALLY_VALID

    return AmplitudeFeatures(
        mean=round(mean_val, 6),
        rms=round(rms_val, 6),
        variance=round(var_val, 6),
        coeff_var=round(coeff_var, 6),
        kurtosis=round(kurt_val, 4),
        skewness=round(skew_val, 4),
        peak_to_rms=round(peak_to_rms, 4),
        norm_variance=round(norm_var, 6),
        validity=validity,
    )
