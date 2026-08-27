from __future__ import annotations
import numpy as np
from app.models.analysis import FrequencyEstimate, PSDResult, SpectrumResult
from app.models.metadata import MetadataStatus

def estimate_frequency_spectral_peak(
    psd_result: PSDResult,
    *,
    sample_rate_hz: float | None = None,
) -> FrequencyEstimate:
    """
    Estimate dominant frequency via spectral peak with parabolic sub-bin interpolation.

    Parameters
    ----------
    psd_result : PSDResult
        Welch PSD result.
    sample_rate_hz : float | None
        Sample rate in Hz if known.

    Returns
    -------
    FrequencyEstimate
    """
    psd_db = psd_result.psd_db
    freqs_norm = psd_result.frequencies_normalized
    n_bins = len(psd_db)

    if n_bins < 3:
        return FrequencyEstimate(
            normalized_frequency=None,
            frequency_hz=None,
            method="spectral_peak_quadratic_interp",
            status=MetadataStatus.UNAVAILABLE,
            quality_score=0.0,
            evidence="Insufficient spectral bins.",
        )

    k_max = int(np.argmax(psd_db))
    bin_res_norm = 1.0 / psd_result.fft_size

    # Parabolic sub-bin interpolation
    if 0 < k_max < n_bins - 1:
        alpha = psd_db[k_max - 1]
        beta = psd_db[k_max]
        gamma = psd_db[k_max + 1]
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
        f_norm = float(freqs_norm[k_max] + delta * bin_res_norm)
    else:
        delta = 0.0
        f_norm = float(freqs_norm[k_max])

    # Ensure range [-0.5, 0.5]
    if f_norm >= 0.5:
        f_norm -= 1.0
    elif f_norm < -0.5:
        f_norm += 1.0

    f_hz = float(f_norm * sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None

    # Prominence as quality indicator
    local_median = float(np.median(psd_db))
    prominence = psd_db[k_max] - local_median
    quality = float(np.clip(prominence / 25.0, 0.1, 0.98))
    uncertainty_norm = bin_res_norm / (1.0 + prominence / 5.0)

    return FrequencyEstimate(
        normalized_frequency=round(f_norm, 7),
        frequency_hz=round(f_hz, 2) if f_hz is not None else None,
        method="spectral_peak_quadratic_interp",
        status=MetadataStatus.ESTIMATED,
        quality_score=round(quality, 3),
        uncertainty=round(uncertainty_norm * sample_rate_hz, 3) if (sample_rate_hz and sample_rate_hz > 0) else round(uncertainty_norm, 7),
        evidence=f"Peak at bin {k_max} ({psd_db[k_max]:.1f} dB, prominence {prominence:.1f} dB, sub-bin delta {delta:+.3f}).",
    )


def estimate_frequency_phase_progression(
    samples: np.ndarray,
    *,
    sample_rate_hz: float | None = None,
) -> FrequencyEstimate:
    """
    Estimate carrier frequency using average phase progression of narrowband signal.

    omega = angle(sum(x[n] * conj(x[n-1])))
    f_norm = omega / (2 * pi)

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.
    sample_rate_hz : float | None
        Sample rate in Hz if known.

    Returns
    -------
    FrequencyEstimate
    """
    if len(samples) < 16:
        return FrequencyEstimate(
            normalized_frequency=None,
            frequency_hz=None,
            method="phase_progression_narrowband",
            status=MetadataStatus.UNAVAILABLE,
            quality_score=0.0,
            evidence="Insufficient samples (<16).",
        )

    # Differential lag-1 product
    diff_prod = samples[1:] * np.conj(samples[:-1])
    sum_prod = np.sum(diff_prod)

    if np.abs(sum_prod) <= 1e-12:
        return FrequencyEstimate(
            normalized_frequency=None,
            frequency_hz=None,
            method="phase_progression_narrowband",
            status=MetadataStatus.AMBIGUOUS,
            quality_score=0.1,
            evidence="Differential phase sum has zero magnitude.",
        )

    mean_angle = float(np.angle(sum_prod))
    f_norm = float(mean_angle / (2.0 * np.pi))

    # Evaluate phase variance to check if signal is truly narrowband/coherent
    angles = np.angle(diff_prod)
    phase_var = float(np.var(np.exp(1j * (angles - mean_angle)).imag))
    
    # Low phase variance => high confidence in single tone
    quality = float(np.clip(1.0 - phase_var * 2.0, 0.05, 0.95))
    status = MetadataStatus.ESTIMATED if quality > 0.4 else MetadataStatus.AMBIGUOUS

    f_hz = float(f_norm * sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None

    return FrequencyEstimate(
        normalized_frequency=round(f_norm, 7),
        frequency_hz=round(f_hz, 2) if f_hz is not None else None,
        method="phase_progression_narrowband",
        status=status,
        quality_score=round(quality, 3),
        uncertainty=round(np.sqrt(phase_var) / (2.0 * np.pi * np.sqrt(len(samples))), 7),
        evidence=f"Mean phase progression: {mean_angle:+.4f} rad/sample, circular phase variance: {phase_var:.4f}.",
    )


def compute_all_frequency_estimates(
    samples: np.ndarray,
    psd_result: PSDResult,
    *,
    sample_rate_hz: float | None = None,
) -> list[FrequencyEstimate]:
    """Compute multi-method frequency estimates."""
    estimates: list[FrequencyEstimate] = []
    estimates.append(estimate_frequency_spectral_peak(psd_result, sample_rate_hz=sample_rate_hz))
    estimates.append(estimate_frequency_phase_progression(samples, sample_rate_hz=sample_rate_hz))
    return estimates
