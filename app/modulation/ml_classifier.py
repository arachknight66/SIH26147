from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .features import FEATURE_NAMES, feature_vector_to_array
from .models import FeatureValidity, ModulationFamily, ModulationFeatureVector

FEATURE_SCHEMA_VERSION = "1.0"
ML_MODEL_VERSION = "0.3.0-lightweight"

# Pre-calibrated discriminant reference profiles
_MODULATION_PROFILES: dict[tuple[ModulationFamily, int], dict[str, float]] = {
    (ModulationFamily.FSK, 2): {
        "amp_kurtosis": 0.0,
        "amp_norm_var": 0.01,
        "inst_freq_var": 0.04,
        "bimodal_prominence": 0.70,
        "cumulant_f20": 0.02,
        "cumulant_f40": 0.05,
        "cumulant_f42": 1.00,
        "var_phase_sq": 0.90,
        "var_phase_4th": 0.90,
    },
    (ModulationFamily.PSK, 2): {
        "amp_kurtosis": -0.20,
        "amp_norm_var": 0.10,
        "inst_freq_var": 0.01,
        "bimodal_prominence": 0.0,
        "cumulant_f20": 0.95,
        "cumulant_f40": 1.60,
        "cumulant_f42": 1.60,
        "var_phase_sq": 0.05,
        "var_phase_4th": 0.05,
    },
    (ModulationFamily.PSK, 4): {
        "amp_kurtosis": 1.20,
        "amp_norm_var": 0.08,
        "inst_freq_var": 0.01,
        "bimodal_prominence": 0.0,
        "cumulant_f20": 0.02,
        "cumulant_f40": 0.85,
        "cumulant_f42": 0.80,
        "var_phase_sq": 0.95,
        "var_phase_4th": 0.45,
    },
    (ModulationFamily.PSK, 8): {
        "amp_kurtosis": 0.80,
        "amp_norm_var": 0.07,
        "inst_freq_var": 0.01,
        "bimodal_prominence": 0.0,
        "cumulant_f20": 0.02,
        "cumulant_f40": 0.05,
        "cumulant_f42": 0.80,
        "var_phase_sq": 0.95,
        "var_phase_4th": 0.95,
    },
    (ModulationFamily.QAM, 16): {
        "amp_kurtosis": -0.70,
        "amp_norm_var": 0.14,
        "inst_freq_var": 0.01,
        "bimodal_prominence": 0.0,
        "cumulant_f20": 0.02,
        "cumulant_f40": 0.65,
        "cumulant_f42": 0.55,
        "var_phase_sq": 0.95,
        "var_phase_4th": 0.85,
    },
}

_FEATURE_SCALES: dict[str, float] = {
    "amp_kurtosis": 0.40,
    "amp_norm_var": 0.05,
    "inst_freq_var": 0.03,
    "bimodal_prominence": 0.25,
    "cumulant_f20": 0.20,
    "cumulant_f40": 0.25,
    "cumulant_f42": 0.25,
    "var_phase_sq": 0.20,
    "var_phase_4th": 0.20,
}

@dataclass(frozen=True)
class MLClassificationResult:
    scores: dict[tuple[ModulationFamily, int], float]
    model_version: str
    feature_schema_version: str
    uncertainty: float

def predict_ml_scores(fv: ModulationFeatureVector) -> MLClassificationResult:
    """
    Lightweight calibrated machine learning classifier computing model affinities.

    Parameters
    ----------
    fv : ModulationFeatureVector

    Returns
    -------
    MLClassificationResult
    """
    if fv.overall_validity in (FeatureValidity.UNAVAILABLE, FeatureValidity.UNRELIABLE):
        return MLClassificationResult(
            scores={cls_key: 0.10 for cls_key in _MODULATION_PROFILES},
            model_version=ML_MODEL_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            uncertainty=1.0,
        )

    # Feature lookup from vector
    curr_feats: dict[str, float] = {
        "amp_kurtosis": fv.amplitude.kurtosis,
        "amp_norm_var": fv.amplitude.norm_variance,
        "inst_freq_var": fv.frequency.inst_freq_var,
        "bimodal_prominence": fv.frequency.bimodal_prominence,
        "cumulant_f20": fv.cumulants.f20,
        "cumulant_f40": fv.cumulants.f40,
        "cumulant_f42": fv.cumulants.f42,
        "var_phase_sq": fv.phase.var_phase_sq,
        "var_phase_4th": fv.phase.var_phase_4th,
    }

    raw_distances: dict[tuple[ModulationFamily, int], float] = {}
    for cls_key, profile in _MODULATION_PROFILES.items():
        dist_sq = 0.0
        for feat_name, ref_val in profile.items():
            curr_val = curr_feats.get(feat_name, ref_val)
            scale = _FEATURE_SCALES.get(feat_name, 1.0)
            diff = (curr_val - ref_val) / scale
            dist_sq += diff ** 2
        raw_distances[cls_key] = float(dist_sq)

    # Softmin with temperature to obtain calibrated score distribution
    temp = 3.0
    logits = {k: -dist / temp for k, dist in raw_distances.items()}
    max_logit = max(logits.values())
    exp_logits = {k: np.exp(v - max_logit) for k, v in logits.items()}
    sum_exp = sum(exp_logits.values())
    
    calibrated_scores = {k: round(float(v / sum_exp), 4) for k, v in exp_logits.items()}
    
    # Entropy-based model uncertainty
    probs = np.array(list(calibrated_scores.values()))
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    max_entropy = np.log(len(probs))
    norm_uncertainty = float(np.clip(entropy / max_entropy, 0.0, 1.0))

    return MLClassificationResult(
        scores=calibrated_scores,
        model_version=ML_MODEL_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        uncertainty=round(norm_uncertainty, 3),
    )

def get_ml_model_metadata() -> dict[str, Any]:
    """Return model provenance metadata and feature schema."""
    return {
        "model_type": "Lightweight Calibrated Manifold Classifier",
        "model_version": ML_MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "supported_classes": [f"{fam.value}_{ord_}" for (fam, ord_) in _MODULATION_PROFILES],
        "feature_names": FEATURE_NAMES,
    }
