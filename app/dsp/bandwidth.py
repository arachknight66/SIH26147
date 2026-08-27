from __future__ import annotations
import numpy as np
from app.models.analysis import BandwidthEstimate, NoiseEstimate, PSDResult
from app.models.metadata import MetadataStatus

def estimate_occupied_bandwidth_power(
    psd_result: PSDResult,
    *,
    power_fraction: float = 0.99,
    sample_rate_hz: float | None = None,
) -> BandwidthEstimate:
    """
    Estimate Occupied Bandwidth (OBW) using fractional power containment.

    Finds frequency interval [f_low, f_high] containing the specified fraction (e.g., 99%)
    of total spectral power.

    Parameters
    ----------
    psd_result : PSDResult
        Welch PSD result.
    power_fraction : float
        Fraction of total power to contain (default: 0.99 for 99% OBW).
    sample_rate_hz : float | None
        Sample rate in Hz if known.

    Returns
    -------
    BandwidthEstimate
    """
    psd = psd_result.psd
    freqs_norm = psd_result.frequencies_normalized
    n_bins = len(psd)

    if n_bins < 4 or np.sum(psd) <= 0:
        return BandwidthEstimate(
            occupied_bandwidth_normalized=None,
            occupied_bandwidth_hz=None,
            method=f"power_containment_{int(power_fraction*100)}pct",
            fraction=power_fraction,
            status=MetadataStatus.UNAVAILABLE,
            quality_score=0.0,
            evidence="Insufficient or zero PSD power.",
        )

    cum_power = np.cumsum(psd)
    total_power = cum_power[-1]
    
    alpha = (1.0 - power_fraction) / 2.0
    p_low = alpha * total_power
    p_high = (1.0 - alpha) * total_power

    # Sub-bin interpolation for low frequency boundary
    idx_low = int(np.searchsorted(cum_power, p_low))
    if idx_low == 0:
        f_low = freqs_norm[0]
    else:
        prev_p = cum_power[idx_low - 1]
        cur_p = cum_power[idx_low]
        frac = (p_low - prev_p) / max(cur_p - prev_p, 1e-18)
        f_low = freqs_norm[idx_low - 1] + frac * (freqs_norm[idx_low] - freqs_norm[idx_low - 1])

    # Sub-bin interpolation for high frequency boundary
    idx_high = int(np.searchsorted(cum_power, p_high))
    if idx_high >= n_bins:
        f_high = freqs_norm[-1]
    elif idx_high == 0:
        f_high = freqs_norm[0]
    else:
        prev_p = cum_power[idx_high - 1]
        cur_p = cum_power[idx_high]
        frac = (p_high - prev_p) / max(cur_p - prev_p, 1e-18)
        f_high = freqs_norm[idx_high - 1] + frac * (freqs_norm[idx_high] - freqs_norm[idx_high - 1])

    bw_norm = float(np.abs(f_high - f_low))
    bw_hz = float(bw_norm * sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None

    # Uncertainty based on bin resolution
    bin_res_norm = 1.0 / psd_result.fft_size
    uncertainty_norm = bin_res_norm * 0.5

    return BandwidthEstimate(
        occupied_bandwidth_normalized=round(bw_norm, 6),
        occupied_bandwidth_hz=round(bw_hz, 2) if bw_hz is not None else None,
        method=f"power_containment_{int(power_fraction*100)}pct",
        fraction=power_fraction,
        status=MetadataStatus.ESTIMATED,
        quality_score=0.92,
        uncertainty=round(uncertainty_norm * sample_rate_hz, 2) if (sample_rate_hz and sample_rate_hz > 0) else round(uncertainty_norm, 6),
        evidence=f"Frequency span enclosing {power_fraction*100:.1f}% total integrated PSD power (indices {idx_low}..{idx_high}).",
    )


def estimate_occupied_bandwidth_threshold(
    psd_result: PSDResult,
    noise_estimate: NoiseEstimate,
    *,
    threshold_db_offset: float = 6.0,
    sample_rate_hz: float | None = None,
) -> BandwidthEstimate:
    """
    Estimate Occupied Bandwidth using noise-relative threshold (X-dB above noise floor).

    Parameters
    ----------
    psd_result : PSDResult
        Welch PSD result.
    noise_estimate : NoiseEstimate
        Noise floor estimate.
    threshold_db_offset : float
        Threshold margin in dB above estimated noise floor.
    sample_rate_hz : float | None
        Sample rate in Hz if known.

    Returns
    -------
    BandwidthEstimate
    """
    if noise_estimate.noise_floor_db is None or len(psd_result.psd_db) == 0:
        return BandwidthEstimate(
            occupied_bandwidth_normalized=None,
            occupied_bandwidth_hz=None,
            method="noise_threshold",
            threshold_db_above_noise=threshold_db_offset,
            status=MetadataStatus.UNAVAILABLE,
            quality_score=0.0,
            evidence="Noise floor is unavailable.",
        )

    thresh = noise_estimate.noise_floor_db + threshold_db_offset
    occupied_mask = psd_result.psd_db >= thresh
    
    if not np.any(occupied_mask):
        return BandwidthEstimate(
            occupied_bandwidth_normalized=0.0,
            occupied_bandwidth_hz=0.0 if sample_rate_hz else None,
            method="noise_threshold",
            threshold_db_above_noise=threshold_db_offset,
            status=MetadataStatus.ESTIMATED,
            quality_score=0.3,
            evidence="No spectral bins exceeded the threshold.",
        )

    active_indices = np.where(occupied_mask)[0]
    freqs_norm = psd_result.frequencies_normalized
    bin_res_norm = 1.0 / psd_result.fft_size

    min_idx, max_idx = int(active_indices[0]), int(active_indices[-1])
    f_low = freqs_norm[min_idx] - 0.5 * bin_res_norm
    f_high = freqs_norm[max_idx] + 0.5 * bin_res_norm
    bw_norm = float(f_high - f_low)
    bw_hz = float(bw_norm * sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None

    quality = float(np.clip(noise_estimate.quality_score * 0.9, 0.2, 0.90))

    return BandwidthEstimate(
        occupied_bandwidth_normalized=round(bw_norm, 6),
        occupied_bandwidth_hz=round(bw_hz, 2) if bw_hz is not None else None,
        method="noise_threshold",
        threshold_db_above_noise=threshold_db_offset,
        status=MetadataStatus.ESTIMATED,
        quality_score=round(quality, 3),
        uncertainty=round(bin_res_norm * sample_rate_hz, 2) if (sample_rate_hz and sample_rate_hz > 0) else round(bin_res_norm, 6),
        evidence=f"Span of {len(active_indices)} bins >= {thresh:.1f} dB (noise floor {noise_estimate.noise_floor_db:.1f} dB + {threshold_db_offset:.1f} dB).",
    )


def compute_all_bandwidth_estimates(
    psd_result: PSDResult,
    noise_estimate: NoiseEstimate,
    *,
    power_fractions: tuple[float, ...] = (0.99, 0.95),
    threshold_db_offset: float = 6.0,
    sample_rate_hz: float | None = None,
) -> list[BandwidthEstimate]:
    """Compute multi-method bandwidth estimates and cross-checks."""
    estimates: list[BandwidthEstimate] = []
    for frac in power_fractions:
        estimates.append(estimate_occupied_bandwidth_power(psd_result, power_fraction=frac, sample_rate_hz=sample_rate_hz))
    estimates.append(estimate_occupied_bandwidth_threshold(psd_result, noise_estimate, threshold_db_offset=threshold_db_offset, sample_rate_hz=sample_rate_hz))
    return estimates
