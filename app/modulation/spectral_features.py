from __future__ import annotations
import numpy as np
import scipy.signal as signal
from app.dsp.psd import compute_psd
from app.models.analysis import SignalAnalysis
from .models import FeatureValidity, SpectralFeatures

def extract_spectral_features(
    samples: np.ndarray,
    analysis: SignalAnalysis | None = None,
) -> SpectralFeatures:
    """
    Extract spectral distribution, flatness, asymmetry, and peak features.

    Parameters
    ----------
    samples : np.ndarray
        Complex signal samples.
    analysis : SignalAnalysis | None
        Precomputed Phase 2 analysis (optional).

    Returns
    -------
    SpectralFeatures
    """
    n_samples = len(samples)
    if n_samples < 32:
        return SpectralFeatures(
            spectral_centroid=0.0,
            spectral_spread=0.0,
            spectral_kurtosis=0.0,
            spectral_flatness=0.0,
            spectral_asymmetry=0.0,
            peak_count=0,
            occupied_bandwidth=None,
            validity=FeatureValidity.UNAVAILABLE,
        )

    if analysis is not None and analysis.psd is not None:
        psd_lin = analysis.psd.psd
        freqs_norm = analysis.psd.frequencies_normalized
        obw = analysis.bandwidth_candidates[0].occupied_bandwidth_normalized if analysis.bandwidth_candidates else None
    else:
        psd_res = compute_psd(samples, segment_length=min(1024, n_samples), is_complex=True)
        psd_lin = psd_res.psd
        freqs_norm = psd_res.frequencies_normalized
        obw = None

    total_p = float(np.sum(psd_lin))
    if total_p <= 1e-15:
        return SpectralFeatures(
            spectral_centroid=0.0,
            spectral_spread=0.0,
            spectral_kurtosis=0.0,
            spectral_flatness=0.0,
            spectral_asymmetry=0.0,
            peak_count=0,
            occupied_bandwidth=obw,
            validity=FeatureValidity.UNRELIABLE,
        )

    # 1. Spectral Centroid
    centroid = float(np.sum(freqs_norm * psd_lin) / total_p)

    # 2. Spectral Spread
    var_f = float(np.sum(((freqs_norm - centroid) ** 2) * psd_lin) / total_p)
    spread = float(np.sqrt(max(var_f, 1e-15)))

    # 3. Spectral Skewness / Asymmetry & Kurtosis
    if spread > 1e-6:
        dev = (freqs_norm - centroid) / spread
        skew_f = float(np.sum((dev ** 3) * psd_lin) / total_p)
        kurt_f = float(np.sum((dev ** 4) * psd_lin) / total_p - 3.0)
    else:
        skew_f = 0.0
        kurt_f = 0.0

    # 4. Spectral Flatness (Geometric Mean / Arithmetic Mean)
    psd_safe = np.maximum(psd_lin, 1e-15)
    geom_mean = float(np.exp(np.mean(np.log(psd_safe))))
    arith_mean = float(np.mean(psd_safe))
    flatness = float(np.clip(geom_mean / max(arith_mean, 1e-15), 0.0, 1.0))

    # 5. Prominent Peak Count in PSD (dB)
    psd_db = 10.0 * np.log10(psd_safe)
    med_db = float(np.median(psd_db))
    mad_db = float(np.median(np.abs(psd_db - med_db)))
    peaks, _ = signal.find_peaks(psd_db, prominence=max(3.0, 2.5 * (1.4826 * mad_db)), distance=8)
    peak_count = len(peaks)

    validity = FeatureValidity.VALID if n_samples >= 64 else FeatureValidity.PARTIALLY_VALID

    return SpectralFeatures(
        spectral_centroid=round(centroid, 6),
        spectral_spread=round(spread, 6),
        spectral_kurtosis=round(kurt_f, 4),
        spectral_flatness=round(flatness, 6),
        spectral_asymmetry=round(skew_f, 4),
        peak_count=peak_count,
        occupied_bandwidth=round(obw, 6) if obw is not None else None,
        validity=validity,
    )
