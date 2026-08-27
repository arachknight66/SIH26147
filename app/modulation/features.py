from __future__ import annotations
import numpy as np
from app.models.analysis import SignalAnalysis
from .amplitude import extract_amplitude_features
from .cumulants import extract_cumulant_features
from .cyclostationary import extract_cyclostationary_features
from .frequency import extract_frequency_features
from .models import FeatureValidity, ModulationFeatureVector
from .phase import extract_phase_features
from .spectral_features import extract_spectral_features

FEATURE_NAMES: list[str] = [
    "amp_mean",
    "amp_variance",
    "amp_coeff_var",
    "amp_kurtosis",
    "amp_skewness",
    "amp_peak_to_rms",
    "amp_norm_var",
    "phase_inc_var",
    "phase_inc_kurtosis",
    "var_phase_sq",
    "var_phase_4th",
    "var_phase_8th",
    "inst_freq_var",
    "inst_freq_mad",
    "bimodal_prominence",
    "bimodal_separation",
    "cumulant_f20",
    "cumulant_f40",
    "cumulant_f41",
    "cumulant_f42",
    "spectral_spread",
    "spectral_kurtosis",
    "spectral_flatness",
    "spectral_peaks",
    "periodicity_score",
]

def extract_modulation_features(
    samples: np.ndarray,
    analysis: SignalAnalysis | None = None,
) -> ModulationFeatureVector:
    """
    Extract full multi-domain feature vector for modulation analysis.

    Parameters
    ----------
    samples : np.ndarray
        Complex signal samples.
    analysis : SignalAnalysis | None
        Precomputed Phase 2 analysis (optional).

    Returns
    -------
    ModulationFeatureVector
    """
    if len(samples) == 0:
        empty_amp = extract_amplitude_features(samples)
        empty_ph = extract_phase_features(samples)
        empty_fr = extract_frequency_features(samples)
        empty_cu = extract_cumulant_features(samples)
        empty_sp = extract_spectral_features(samples)
        empty_cy = extract_cyclostationary_features(samples)
        return ModulationFeatureVector(
            amplitude=empty_amp,
            phase=empty_ph,
            frequency=empty_fr,
            cumulants=empty_cu,
            spectral=empty_sp,
            cyclostationary=empty_cy,
            overall_validity=FeatureValidity.UNAVAILABLE,
        )

    # RMS-normalized copy for scale-invariant feature extraction
    rms_val = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    if rms_val > 1e-12:
        x_norm = (samples / rms_val).astype(np.complex64)
    else:
        x_norm = samples.copy()

    amp_feat = extract_amplitude_features(x_norm)
    phase_feat = extract_phase_features(x_norm)
    freq_feat = extract_frequency_features(x_norm)
    cum_feat = extract_cumulant_features(x_norm)
    spec_feat = extract_spectral_features(x_norm, analysis=analysis)
    cyclo_feat = extract_cyclostationary_features(x_norm, analysis=analysis)

    validities = [
        amp_feat.validity,
        phase_feat.validity,
        freq_feat.validity,
        cum_feat.validity,
        spec_feat.validity,
    ]

    if FeatureValidity.UNAVAILABLE in validities:
        overall = FeatureValidity.UNAVAILABLE
    elif FeatureValidity.UNRELIABLE in validities:
        overall = FeatureValidity.UNRELIABLE
    elif FeatureValidity.PARTIALLY_VALID in validities:
        overall = FeatureValidity.PARTIALLY_VALID
    else:
        overall = FeatureValidity.VALID

    return ModulationFeatureVector(
        amplitude=amp_feat,
        phase=phase_feat,
        frequency=freq_feat,
        cumulants=cum_feat,
        spectral=spec_feat,
        cyclostationary=cyclo_feat,
        overall_validity=overall,
    )

def feature_vector_to_array(fv: ModulationFeatureVector) -> np.ndarray:
    """
    Convert ModulationFeatureVector to a 1D float32 numpy array matching FEATURE_NAMES.

    Parameters
    ----------
    fv : ModulationFeatureVector

    Returns
    -------
    np.ndarray of shape (len(FEATURE_NAMES),)
    """
    vals = [
        fv.amplitude.mean,
        fv.amplitude.variance,
        fv.amplitude.coeff_var,
        fv.amplitude.kurtosis,
        fv.amplitude.skewness,
        fv.amplitude.peak_to_rms,
        fv.amplitude.norm_variance,
        fv.phase.phase_inc_var,
        fv.phase.phase_inc_kurtosis,
        fv.phase.var_phase_sq,
        fv.phase.var_phase_4th,
        fv.phase.var_phase_8th,
        fv.frequency.inst_freq_var,
        fv.frequency.inst_freq_mad,
        fv.frequency.bimodal_prominence,
        fv.frequency.bimodal_separation or 0.0,
        fv.cumulants.f20,
        fv.cumulants.f40,
        fv.cumulants.f41,
        fv.cumulants.f42,
        fv.spectral.spectral_spread,
        fv.spectral.spectral_kurtosis,
        fv.spectral.spectral_flatness,
        float(fv.spectral.peak_count),
        fv.cyclostationary.periodicity_score,
    ]
    return np.asarray(vals, dtype=np.float32)
