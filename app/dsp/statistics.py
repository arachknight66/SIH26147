from __future__ import annotations
import numpy as np
from app.models.analysis import ClippingDiagnostics, DCOffsetEstimate, TimeStatistics
from app.models.metadata import DiagnosticSeverity, MetadataStatus

def compute_time_statistics(samples: np.ndarray) -> TimeStatistics:
    """
    Compute quantitative time-domain statistics, moments, power metrics, and masked phase statistics.

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.

    Returns
    -------
    TimeStatistics
    """
    if len(samples) == 0:
        raise ValueError("Cannot compute time statistics on empty array.")

    i = samples.real.astype(np.float64)
    q = samples.imag.astype(np.float64)
    
    mean_i = float(np.mean(i))
    mean_q = float(np.mean(q))
    mean_complex = complex(mean_i, mean_q)
    
    var_i = float(np.var(i))
    var_q = float(np.var(q))
    
    cov_iq = float(np.mean((i - mean_i) * (q - mean_q)))
    denom = np.sqrt(var_i * var_q)
    corr_iq = float(cov_iq / denom) if denom > 1e-15 else 0.0

    amp = np.abs(samples).astype(np.float64)
    power = (amp ** 2).astype(np.float64)

    mean_amp = float(np.mean(amp))
    med_amp = float(np.median(amp))
    std_amp = float(np.std(amp))
    rms_amp = float(np.sqrt(np.mean(power)))
    peak_amp = float(np.max(amp))

    mean_p = float(np.mean(power))
    med_p = float(np.median(power))
    var_p = float(np.var(power))
    peak_p = float(np.max(power))

    peak_to_rms = float(peak_amp / rms_amp) if rms_amp > 1e-15 else 1.0
    crest_factor = peak_to_rms

    pos_p = power[power > 1e-15]
    if len(pos_p) > 0:
        min_pos_p = float(np.min(pos_p))
        dynamic_range_db = float(10.0 * np.log10(max(peak_p, 1e-15) / min_pos_p))
    else:
        dynamic_range_db = 0.0

    # Phase statistics with magnitude-based stability masking
    # Discard samples with amplitude < 5% of RMS to avoid atan2 noise singularities
    phase_threshold = 0.05 * rms_amp
    valid_mask = amp >= phase_threshold
    valid_fraction = float(np.mean(valid_mask))

    if np.any(valid_mask):
        valid_samples = samples[valid_mask]
        phases = np.angle(valid_samples)
        # Circular mean and variance
        mean_vector = np.mean(np.exp(1j * phases))
        circ_mean = float(np.angle(mean_vector))
        # Circular variance = 1 - |R| in [0, 1]
        circ_var = float(1.0 - np.abs(mean_vector))
    else:
        circ_mean = 0.0
        circ_var = 1.0

    return TimeStatistics(
        mean_i=round(mean_i, 6),
        mean_q=round(mean_q, 6),
        mean_complex=complex(round(mean_i, 6), round(mean_q, 6)),
        variance_i=round(var_i, 6),
        variance_q=round(var_q, 6),
        iq_covariance=round(cov_iq, 6),
        iq_correlation=round(corr_iq, 6),
        mean_amplitude=round(mean_amp, 6),
        median_amplitude=round(med_amp, 6),
        std_amplitude=round(std_amp, 6),
        rms_amplitude=round(rms_amp, 6),
        peak_amplitude=round(peak_amp, 6),
        mean_power=round(mean_p, 6),
        median_power=round(med_p, 6),
        variance_power=round(var_p, 6),
        peak_power=round(peak_p, 6),
        peak_to_rms_ratio=round(peak_to_rms, 4),
        crest_factor=round(crest_factor, 4),
        dynamic_range_db=round(dynamic_range_db, 2),
        phase_mean=round(circ_mean, 4),
        phase_variance=round(circ_var, 4),
        phase_valid_fraction=round(valid_fraction, 4),
    )


def compute_dc_offset(samples: np.ndarray) -> DCOffsetEstimate:
    """
    Measure DC offset (I and Q means) explicitly without automatic removal.

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.

    Returns
    -------
    DCOffsetEstimate
    """
    if len(samples) == 0:
        return DCOffsetEstimate(
            i_offset=0.0,
            q_offset=0.0,
            magnitude=0.0,
            phase_rad=0.0,
            status=MetadataStatus.UNAVAILABLE,
            quality_score=0.0,
            evidence="Empty sample array.",
        )

    i_mean = float(np.mean(samples.real))
    q_mean = float(np.mean(samples.imag))
    mag = float(np.sqrt(i_mean ** 2 + q_mean ** 2))
    phase = float(np.arctan2(q_mean, i_mean))

    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    dc_to_rms = mag / rms if rms > 1e-15 else 0.0

    status = MetadataStatus.MEASURED
    quality = float(np.clip(1.0 - (1.0 / np.sqrt(len(samples))), 0.5, 0.99))
    evidence = f"I offset: {i_mean:+.4g}, Q offset: {q_mean:+.4g}, DC/RMS ratio: {dc_to_rms:.3f}."

    return DCOffsetEstimate(
        i_offset=round(i_mean, 6),
        q_offset=round(q_mean, 6),
        magnitude=round(mag, 6),
        phase_rad=round(phase, 4),
        status=status,
        quality_score=round(quality, 3),
        evidence=evidence,
    )


def detect_clipping(
    samples: np.ndarray,
    original_dtype: str = "float32",
) -> ClippingDiagnostics:
    """
    Detect possible clipping or ADC saturation based on original datatype extrema.

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.
    original_dtype : str
        Source datatype name (e.g. 'int16', 'int8', 'uint8', 'float32', 'pcm_s16le').

    Returns
    -------
    ClippingDiagnostics
    """
    if len(samples) == 0:
        return ClippingDiagnostics(
            is_clipped=False,
            fraction_near_extrema=0.0,
            sample_range_min=0.0,
            sample_range_max=0.0,
            clipping_threshold=0.0,
            evidence="Empty sample array.",
            severity=DiagnosticSeverity.INFO,
        )

    i_vals = samples.real
    q_vals = samples.imag
    s_min = float(min(np.min(i_vals), np.min(q_vals)))
    s_max = float(max(np.max(i_vals), np.max(q_vals)))

    dt = original_dtype.lower()
    if "int8" in dt or "ci8" in dt:
        threshold = 126.0
        clipped_mask = (np.abs(i_vals) >= threshold) | (np.abs(q_vals) >= threshold)
    elif "uint8" in dt or "cu8" in dt:
        threshold = 126.0  # relative to zero-centered representation
        clipped_mask = (np.abs(i_vals) >= threshold) | (np.abs(q_vals) >= threshold)
    elif "int16" in dt or "ci16" in dt or "s16" in dt:
        threshold = 32760.0
        clipped_mask = (np.abs(i_vals) >= threshold) | (np.abs(q_vals) >= threshold)
    elif "int32" in dt or "s32" in dt:
        threshold = 2147483600.0
        clipped_mask = (np.abs(i_vals) >= threshold) | (np.abs(q_vals) >= threshold)
    else:
        # Float: check if max amplitude > 0.999 if normalized or extreme outliers
        threshold = 0.999 * max(abs(s_min), abs(s_max)) if max(abs(s_min), abs(s_max)) > 0 else 1.0
        clipped_mask = (np.abs(i_vals) >= threshold) | (np.abs(q_vals) >= threshold)

    frac_clipped = float(np.mean(clipped_mask)) if len(samples) > 0 else 0.0
    is_clipped = frac_clipped > 0.001  # > 0.1% near extrema

    if is_clipped:
        severity = DiagnosticSeverity.WARNING if frac_clipped < 0.05 else DiagnosticSeverity.ERROR
        evidence = f"{frac_clipped * 100:.2f}% of samples reach within 0.1% of maximum dynamic range (threshold: {threshold:.1f})."
    else:
        severity = DiagnosticSeverity.INFO
        evidence = f"Peak value ({max(abs(s_min), abs(s_max)):.4g}) is within dynamic range limits (threshold: {threshold:.1f})."

    return ClippingDiagnostics(
        is_clipped=is_clipped,
        fraction_near_extrema=round(frac_clipped, 5),
        sample_range_min=round(s_min, 4),
        sample_range_max=round(s_max, 4),
        clipping_threshold=round(threshold, 2),
        evidence=evidence,
        severity=severity,
    )
