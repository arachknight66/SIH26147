from __future__ import annotations
import numpy as np
from app.models.analysis import NoiseEstimate

def estimate_noise_floor(
    psd_linear: np.ndarray,
    *,
    method: str = "iterative_sigma_clip",
    percentile: float = 30.0,
    max_clip_iterations: int = 5,
    sigma_clip_thresh: float = 3.0,
) -> NoiseEstimate:
    """
    Robust noise-floor estimator for power spectral densities.

    Supports:
    - 'iterative_sigma_clip': Iterative rejection of spectral peaks using Median Absolute Deviation (MAD),
      computing the linear mean of unrejected noise bins.
    - 'trimmed_percentile': Trimmed mean of bins below specified percentile.
    - 'median': Median of PSD bins.

    Parameters
    ----------
    psd_linear : np.ndarray
        Linear PSD array (1D).
    method : str
        Estimation method ('iterative_sigma_clip', 'trimmed_percentile', 'median').
    percentile : float
        Percentile threshold for trimmed percentile estimator (default: 30.0).
    max_clip_iterations : int
        Maximum iterations for sigma-clipping.
    sigma_clip_thresh : float
        Sigma threshold for peak rejection in sigma-clipping.

    Returns
    -------
    NoiseEstimate
    """
    if len(psd_linear) == 0:
        return NoiseEstimate(
            noise_floor_db=None,
            noise_power_linear=None,
            method=method,
            unit="relative_db",
            quality_score=0.0,
            uncertainty_db=None,
            is_signal_dominated=True,
            evidence="Empty PSD array.",
        )

    psd_clean = psd_linear[np.isfinite(psd_linear) & (psd_linear > 0)]
    if len(psd_clean) < 4:
        return NoiseEstimate(
            noise_floor_db=None,
            noise_power_linear=None,
            method=method,
            unit="relative_db",
            quality_score=0.0,
            uncertainty_db=None,
            is_signal_dominated=True,
            evidence="Insufficient valid PSD bins.",
        )

    n_bins = len(psd_clean)

    if method == "trimmed_percentile":
        p = np.clip(percentile, 5.0, 75.0)
        q_val = float(np.percentile(psd_clean, p))
        noise_subset = psd_clean[psd_clean <= q_val]
        
        # Mean of lower percentile bins with light adjustment for lower tail truncation
        raw_mean = float(np.mean(noise_subset)) if len(noise_subset) > 0 else q_val
        # For near-Gaussian Welch distribution, lower 30% mean is approx 0.85 of full mean
        correction = 1.0 / np.clip(p / 50.0, 0.5, 1.2) if p < 50.0 else 1.0
        noise_power_linear = raw_mean * correction
        noise_floor_db = float(10.0 * np.log10(max(noise_power_linear, 1e-15)))

        uncertainty_db = float(10.0 * np.log10(1.0 + (np.std(noise_subset) / max(noise_power_linear, 1e-15)) / np.sqrt(len(noise_subset))))
        ratio = np.mean(psd_clean) / max(noise_power_linear, 1e-15)
        is_sig_dom = ratio > 20.0
        quality = float(np.clip(0.85 - (0.3 if is_sig_dom else 0.0) - min(uncertainty_db / 4.0, 0.3), 0.1, 0.95))
        evidence = f"Trimmed {p:.0f}th-percentile mean estimator ({len(noise_subset)} bins)."

    elif method == "median":
        med_lin = float(np.median(psd_clean))
        noise_power_linear = med_lin
        noise_floor_db = float(10.0 * np.log10(max(noise_power_linear, 1e-15)))
        
        mad = float(np.median(np.abs(psd_clean - med_lin)))
        uncertainty_db = float(10.0 * np.log10(1.0 + (mad / max(med_lin, 1e-15)) / np.sqrt(n_bins)))
        is_sig_dom = (np.mean(psd_clean) / max(med_lin, 1e-15)) > 10.0
        quality = 0.7 if not is_sig_dom else 0.3
        evidence = f"Median PSD estimate."

    else:  # iterative_sigma_clip (default)
        psd_db = 10.0 * np.log10(psd_clean)
        mask = np.ones(n_bins, dtype=bool)
        
        for _ in range(max_clip_iterations):
            current_vals = psd_db[mask]
            if len(current_vals) < 4:
                break
            med = np.median(current_vals)
            mad = np.median(np.abs(current_vals - med))
            sigma = 1.4826 * mad
            if sigma < 1e-6:
                break
            new_mask = mask & (psd_db <= med + sigma_clip_thresh * sigma)
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

        noise_bins_lin = psd_clean[mask]
        retained_fraction = len(noise_bins_lin) / n_bins

        if len(noise_bins_lin) >= 4 and retained_fraction > 0.05:
            noise_power_linear = float(np.mean(noise_bins_lin))
            noise_floor_db = float(10.0 * np.log10(max(noise_power_linear, 1e-15)))
            noise_db_std = float(np.std(psd_db[mask]))
            uncertainty_db = float(noise_db_std / np.sqrt(len(noise_bins_lin)))
            is_sig_dom = retained_fraction < 0.20
            quality = float(np.clip(retained_fraction * (1.0 - min(uncertainty_db / 5.0, 0.8)), 0.1, 0.95))
            evidence = f"Iterative sigma-clip converged with {len(noise_bins_lin)}/{n_bins} ({retained_fraction*100:.1f}%) noise bins."
        else:
            q10 = float(np.percentile(psd_clean, 10.0))
            noise_power_linear = q10
            noise_floor_db = float(10.0 * np.log10(max(noise_power_linear, 1e-15)))
            uncertainty_db = 3.0
            is_sig_dom = True
            quality = 0.2
            evidence = "Sigma-clipping rejected almost all bins; signal-dominated spectrum."

    return NoiseEstimate(
        noise_floor_db=round(noise_floor_db, 2) if noise_floor_db is not None else None,
        noise_power_linear=noise_power_linear,
        method=method,
        unit="relative_db",
        quality_score=round(quality, 3),
        uncertainty_db=round(uncertainty_db, 2) if uncertainty_db is not None else None,
        is_signal_dominated=is_sig_dom,
        evidence=evidence,
    )
