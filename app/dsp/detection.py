from __future__ import annotations
from typing import Sequence
import numpy as np
from app.models.analysis import DetectedRegion, NoiseEstimate, PSDResult

def detect_signal_regions_spectral(
    psd_result: PSDResult,
    noise_estimate: NoiseEstimate,
    *,
    threshold_db_offset: float = 10.0,
    min_bins: int = 3,
    merge_gap_bins: int = 2,
    sample_rate_hz: float | None = None,
) -> list[DetectedRegion]:
    """
    Detect candidate signal regions in the frequency domain using PSD energy thresholding.

    Parameters
    ----------
    psd_result : PSDResult
        Welch PSD result.
    noise_estimate : NoiseEstimate
        Estimated noise floor.
    threshold_db_offset : float
        Detection threshold offset in dB above estimated noise floor.
    min_bins : int
        Minimum number of contiguous frequency bins to constitute a valid region.
    merge_gap_bins : int
        Maximum gap in bins between active regions to merge into a single region.
    sample_rate_hz : float | None
        Sampling rate in Hz if known.

    Returns
    -------
    list[DetectedRegion]
    """
    if noise_estimate.noise_floor_db is None or len(psd_result.psd_db) == 0:
        return []

    thresh_db = noise_estimate.noise_floor_db + threshold_db_offset
    occupied_mask = psd_result.psd_db >= thresh_db
    n_bins = len(occupied_mask)

    if not np.any(occupied_mask):
        return []

    # Find contiguous run segments
    runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i in range(n_bins):
        if occupied_mask[i] and not in_run:
            in_run = True
            run_start = i
        elif not occupied_mask[i] and in_run:
            in_run = False
            runs.append((run_start, i - 1))
    if in_run:
        runs.append((run_start, n_bins - 1))

    if not runs:
        return []

    # Merge nearby runs separated by <= merge_gap_bins
    merged_runs: list[tuple[int, int]] = []
    current_start, current_end = runs[0]
    for next_start, next_end in runs[1:]:
        if next_start - current_end - 1 <= merge_gap_bins:
            current_end = next_end
        else:
            merged_runs.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_runs.append((current_start, current_end))

    # Filter runs with fewer than min_bins
    filtered_runs = [r for r in merged_runs if (r[1] - r[0] + 1) >= min_bins]

    regions: list[DetectedRegion] = []
    freq_norm = psd_result.frequencies_normalized
    psd_lin = psd_result.psd
    psd_db = psd_result.psd_db
    bin_res_norm = 1.0 / psd_result.fft_size

    for reg_id, (s_idx, e_idx) in enumerate(filtered_runs, start=1):
        reg_lin = psd_lin[s_idx : e_idx + 1]
        reg_db = psd_db[s_idx : e_idx + 1]
        reg_freqs_norm = freq_norm[s_idx : e_idx + 1]

        start_fn = float(reg_freqs_norm[0] - 0.5 * bin_res_norm)
        end_fn = float(reg_freqs_norm[-1] + 0.5 * bin_res_norm)
        bw_norm = float(end_fn - start_fn)

        # Power-weighted frequency centroid
        total_p = float(np.sum(reg_lin))
        if total_p > 0:
            center_fn = float(np.sum(reg_freqs_norm * reg_lin) / total_p)
        else:
            center_fn = float(0.5 * (start_fn + end_fn))

        peak_db = float(np.max(reg_db))
        excess_snr_db = float(peak_db - noise_estimate.noise_floor_db)
        
        # Detection score & confidence
        score = float(np.clip(excess_snr_db / 30.0, 0.05, 0.99))
        confidence = float(np.clip(score * noise_estimate.quality_score, 0.05, 0.98))

        # Physical Hz if Fs known
        if sample_rate_hz is not None and sample_rate_hz > 0:
            center_hz = float(center_fn * sample_rate_hz)
            bw_hz = float(bw_norm * sample_rate_hz)
        else:
            center_hz = None
            bw_hz = None

        regions.append(
            DetectedRegion(
                region_id=reg_id,
                start_freq_normalized=round(start_fn, 6),
                end_freq_normalized=round(end_fn, 6),
                center_freq_normalized=round(center_fn, 6),
                bandwidth_normalized=round(bw_norm, 6),
                center_freq_hz=round(center_hz, 2) if center_hz is not None else None,
                bandwidth_hz=round(bw_hz, 2) if bw_hz is not None else None,
                peak_power_db=round(peak_db, 2),
                estimated_snr_db=round(excess_snr_db, 2),
                detection_score=round(score, 3),
                method="spectral_energy_threshold",
                confidence=round(confidence, 3),
                assumptions=[
                    "Assumes stationary or persistent emission over PSD integration window.",
                    f"Threshold set to {threshold_db_offset:.1f} dB above estimated noise floor.",
                ],
            )
        )

    return regions


def detect_burst_regions_time(
    samples: np.ndarray,
    *,
    sample_rate_hz: float | None = None,
    threshold_db_offset: float = 6.0,
    smooth_window: int = 64,
    min_burst_samples: int = 128,
    merge_gap_samples: int = 64,
) -> list[DetectedRegion]:
    """
    Detect time-domain burst candidate regions via smoothed envelope energy.

    Parameters
    ----------
    samples : np.ndarray
        Signal samples.
    sample_rate_hz : float | None
        Sampling rate in Hz if known.
    threshold_db_offset : float
        Threshold in dB above estimated time-domain noise power floor.
    smooth_window : int
        Moving average smoothing window size.
    min_burst_samples : int
        Minimum burst duration in samples.
    merge_gap_samples : int
        Maximum gap in samples between burst segments to merge.

    Returns
    -------
    list[DetectedRegion]
    """
    n_samples = len(samples)
    if n_samples < min_burst_samples:
        return []

    inst_power = (np.abs(samples) ** 2).astype(np.float64)
    
    # Smooth power using moving average
    win_len = min(smooth_window, n_samples // 4)
    if win_len > 1:
        kernel = np.ones(win_len, dtype=np.float64) / win_len
        smoothed_power = np.convolve(inst_power, kernel, mode="same")
    else:
        smoothed_power = inst_power

    # Robust noise floor in time domain (20th percentile)
    noise_p = float(np.percentile(smoothed_power, 20.0))
    if noise_p <= 0:
        noise_p = 1e-12
    thresh_p = noise_p * (10.0 ** (threshold_db_offset / 10.0))

    active_mask = smoothed_power >= thresh_p
    if not np.any(active_mask):
        return []

    runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i in range(n_samples):
        if active_mask[i] and not in_run:
            in_run = True
            run_start = i
        elif not active_mask[i] and in_run:
            in_run = False
            runs.append((run_start, i - 1))
    if in_run:
        runs.append((run_start, n_samples - 1))

    if not runs:
        return []

    # Merge nearby runs
    merged_runs: list[tuple[int, int]] = []
    cur_s, cur_e = runs[0]
    for n_s, n_e in runs[1:]:
        if n_s - cur_e - 1 <= merge_gap_samples:
            cur_e = n_e
        else:
            merged_runs.append((cur_s, cur_e))
            cur_s, cur_e = n_s, n_e
    merged_runs.append((cur_s, cur_e))

    # Filter runs < min_burst_samples
    filtered_runs = [r for r in merged_runs if (r[1] - r[0] + 1) >= min_burst_samples]

    burst_regions: list[DetectedRegion] = []
    for b_id, (s_s, e_s) in enumerate(filtered_runs, start=1):
        seg_p = smoothed_power[s_s : e_s + 1]
        peak_p = float(np.max(seg_p))
        peak_db = float(10.0 * np.log10(max(peak_p, 1e-15)))
        snr_est = float(10.0 * np.log10(max(peak_p / noise_p, 1.0)))

        start_t = (s_s / sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None
        end_t = (e_s / sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None

        score = float(np.clip(snr_est / 25.0, 0.1, 0.99))

        burst_regions.append(
            DetectedRegion(
                region_id=b_id,
                start_sample=s_s,
                end_sample=e_s,
                start_time_s=round(start_t, 6) if start_t is not None else None,
                end_time_s=round(end_t, 6) if end_t is not None else None,
                peak_power_db=round(peak_db, 2),
                estimated_snr_db=round(snr_est, 2),
                detection_score=round(score, 3),
                method="time_domain_power_envelope",
                confidence=round(score * 0.85, 3),
                assumptions=["Assumes energy bursts exceed quiescent noise floor."],
            )
        )

    return burst_regions
