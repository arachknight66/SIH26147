from __future__ import annotations
import numpy as np
from app.dsp.autocorrelation import compute_autocorrelation
from app.dsp.rate_estimation import estimate_symbol_rate_candidates
from app.models.analysis import SignalAnalysis
from .models import CyclostationaryFeatures, FeatureValidity

def extract_cyclostationary_features(
    samples: np.ndarray,
    analysis: SignalAnalysis | None = None,
) -> CyclostationaryFeatures:
    """
    Extract symbol-rate periodicity and cyclostationary transition evidence.

    Parameters
    ----------
    samples : np.ndarray
        Complex signal samples.
    analysis : SignalAnalysis | None
        Precomputed Phase 2 analysis (optional).

    Returns
    -------
    CyclostationaryFeatures
    """
    n_samples = len(samples)
    if n_samples < 64:
        return CyclostationaryFeatures(
            periodicity_score=0.0,
            top_candidate_sps=None,
            top_candidate_rate=None,
            validity=FeatureValidity.UNAVAILABLE,
        )

    candidates = []
    if analysis is not None and analysis.symbol_rate_candidates:
        candidates = analysis.symbol_rate_candidates
    else:
        autocorr = compute_autocorrelation(samples, max_lag=min(128, n_samples - 1))
        candidates = estimate_symbol_rate_candidates(samples, autocorr_result=autocorr)

    if candidates:
        top_cand = candidates[0]
        periodicity_score = float(top_cand.score)
        top_sps = top_cand.estimated_samples_per_symbol
        top_rate = top_cand.normalized_rate
        validity = FeatureValidity.VALID
    else:
        periodicity_score = 0.0
        top_sps = None
        top_rate = None
        validity = FeatureValidity.PARTIALLY_VALID

    return CyclostationaryFeatures(
        periodicity_score=round(periodicity_score, 4),
        top_candidate_sps=round(top_sps, 3) if top_sps is not None else None,
        top_candidate_rate=round(top_rate, 6) if top_rate is not None else None,
        validity=validity,
    )
