from __future__ import annotations
import numpy as np
from app.models.analysis import DetectedRegion, NoiseEstimate, PSDResult, SNREstimate
from app.models.metadata import MetadataStatus

def estimate_snr_spectral(
    psd_result: PSDResult,
    noise_estimate: NoiseEstimate,
    detected_regions: list[DetectedRegion] | None = None,
) -> SNREstimate:
    """
    Estimate SNR from PSD power integration against estimated noise floor.

    Total signal power is estimated as max(0, total_power - noise_power_total),
    yielding full-band and in-band SNR metrics.

    Parameters
    ----------
    psd_result : PSDResult
        Welch PSD result.
    noise_estimate : NoiseEstimate
        Estimated noise floor.
    detected_regions : list[DetectedRegion] | None
        Detected spectral candidate regions (optional, for in-band SNR refinement).

    Returns
    -------
    SNREstimate
    """
    if noise_estimate.noise_power_linear is None or noise_estimate.noise_floor_db is None:
        return SNREstimate(
            snr_db=None,
            method="spectral_noise_floor",
            status=MetadataStatus.UNAVAILABLE,
            quality_score=0.0,
            evidence="Noise floor estimate is unavailable.",
            assumptions=["Requires valid noise floor estimate."],
        )

    psd = psd_result.psd
    n_bins = len(psd)
    n_floor = noise_estimate.noise_power_linear

    if psd_result.scaling == "density":
        # Total power is integral over normalized frequency span [ -0.5, 0.5 ) which has width 1.0
        # Integral = sum(psd * delta_f) = mean(psd)
        total_power = float(np.mean(psd))
        noise_power_total = float(n_floor)  # integrated noise density over width 1.0
    else:
        # Spectrum scaling (power per bin)
        total_power = float(np.sum(psd))
        noise_power_total = float(n_floor * n_bins)

    signal_power_total = max(0.0, total_power - noise_power_total)

    if signal_power_total <= 1e-18:
        # Near or below noise floor
        return SNREstimate(
            snr_db=0.0,
            method="spectral_noise_floor",
            status=MetadataStatus.ESTIMATED,
            quality_score=0.3,
            uncertainty_db=2.0,
            evidence=f"Total spectral power ({total_power:.4g}) is consistent with pure noise ({noise_power_total:.4g}).",
            assumptions=["Assumes noise is uniformly distributed across frequency band."],
        )

    snr_lin = signal_power_total / max(noise_power_total, 1e-18)
    snr_db = float(10.0 * np.log10(snr_lin))

    # Calculate uncertainty based on noise estimate uncertainty
    unc_db = (noise_estimate.uncertainty_db or 1.0) + 0.5
    quality = float(np.clip(noise_estimate.quality_score * (1.0 - min(unc_db / 10.0, 0.5)), 0.1, 0.95))

    return SNREstimate(
        snr_db=round(snr_db, 2),
        method="spectral_noise_floor",
        status=MetadataStatus.ESTIMATED,
        quality_score=round(quality, 3),
        uncertainty_db=round(unc_db, 2),
        evidence=f"Full-band SNR: {snr_db:.2f} dB (Signal power: {signal_power_total:.4g}, Noise power: {noise_power_total:.4g}).",
        assumptions=["Assumes additive noise with flat spectral density across measured bins."],
    )


def estimate_snr_m2m4(samples: np.ndarray) -> SNREstimate:
    """
    Estimate SNR using decision-independent 2nd and 4th order moments (M2M4 estimator).

    Assumes zero-mean circular complex Gaussian noise and constant modulus signal (k_s = 1).
    M2 = E[|x|^2] = S + N
    M4 = E[|x|^4] = S^2 + 4SN + 2N^2 = 2*M2^2 - S^2
    => S = sqrt(max(0, 2*M2^2 - M4)), N = M2 - S

    Parameters
    ----------
    samples : np.ndarray
        Signal samples (complex IQ).

    Returns
    -------
    SNREstimate
    """
    if len(samples) < 32:
        return SNREstimate(
            snr_db=None,
            method="m2m4_moments",
            status=MetadataStatus.UNAVAILABLE,
            quality_score=0.0,
            evidence="Insufficient samples (<32) for statistical moment estimation.",
            assumptions=["Requires sufficient sample count for 4th-order moment convergence."],
        )

    # Remove complex DC offset for moment estimation
    x = samples - np.mean(samples)
    p = np.abs(x) ** 2
    m2 = float(np.mean(p))
    m4 = float(np.mean(p ** 2))

    if m2 <= 0:
        return SNREstimate(
            snr_db=None,
            method="m2m4_moments",
            status=MetadataStatus.UNAVAILABLE,
            quality_score=0.0,
            evidence="Zero sample power.",
        )

    disc = 2.0 * (m2 ** 2) - m4
    if disc <= 0:
        # High noise or non-constant-modulus modulation causing discriminant <= 0
        return SNREstimate(
            snr_db=None,
            method="m2m4_moments",
            status=MetadataStatus.AMBIGUOUS,
            quality_score=0.15,
            evidence=f"Discriminant 2*M2^2 - M4 = {disc:.4g} <= 0 (indicates high noise or non-constant-modulus signal).",
            assumptions=["Assumes constant-envelope signal (M-PSK/FSK/tone) in circular complex AWGN."],
        )

    s = np.sqrt(disc)
    n = m2 - s

    if n <= 0 or s <= 0:
        return SNREstimate(
            snr_db=30.0,
            method="m2m4_moments",
            status=MetadataStatus.ESTIMATED,
            quality_score=0.5,
            uncertainty_db=3.0,
            evidence="Noise power estimate near zero; high SNR regime.",
            assumptions=["Constant envelope in AWGN."],
        )

    snr_lin = s / n
    snr_db = float(10.0 * np.log10(snr_lin))
    
    # Quality based on sample size and SNR range
    n_samp = len(samples)
    quality = float(np.clip((1.0 - np.exp(-n_samp / 1000.0)) * (0.85 if 0 <= snr_db <= 30 else 0.4), 0.1, 0.90))

    return SNREstimate(
        snr_db=round(snr_db, 2),
        method="m2m4_moments",
        status=MetadataStatus.ESTIMATED,
        quality_score=round(quality, 3),
        uncertainty_db=round(max(1.0, 15.0 / np.sqrt(n_samp)), 2),
        evidence=f"M2M4 moment ratio: S={s:.4g}, N={n:.4g}, SNR={snr_db:.2f} dB.",
        assumptions=[
            "Assumes constant-envelope signal in additive white circular complex Gaussian noise.",
            "Decision-independent estimator.",
        ],
    )


def compute_all_snr_estimates(
    samples: np.ndarray,
    psd_result: PSDResult,
    noise_estimate: NoiseEstimate,
    detected_regions: list[DetectedRegion] | None = None,
) -> list[SNREstimate]:
    """Compute multi-method SNR estimates."""
    estimates: list[SNREstimate] = []
    estimates.append(estimate_snr_spectral(psd_result, noise_estimate, detected_regions))
    estimates.append(estimate_snr_m2m4(samples))
    return estimates
