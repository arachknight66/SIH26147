from __future__ import annotations
from typing import Any
from app.models.analysis import DetectedRegion, SignalAnalysis
from .models import ModulationHypothesis

def attach_candidate_parameters(
    hypotheses: list[ModulationHypothesis],
    analysis: SignalAnalysis,
    region: DetectedRegion | None,
) -> list[ModulationHypothesis]:
    """
    Attach candidate synchronization and physical parameters to each hypothesis for Phase 4 handoff.

    Parameters
    ----------
    hypotheses : list[ModulationHypothesis]
    analysis : SignalAnalysis
    region : DetectedRegion | None

    Returns
    -------
    list[ModulationHypothesis]
    """
    # Extract candidate parameters from Phase 2 analysis
    symbol_rate_hz = None
    symbol_rate_norm = None
    sps = None
    if analysis.symbol_rate_candidates:
        top_rate = analysis.symbol_rate_candidates[0]
        symbol_rate_hz = top_rate.rate_hz
        symbol_rate_norm = top_rate.normalized_rate
        sps = top_rate.estimated_samples_per_symbol

    center_freq_hz = None
    center_freq_norm = None
    bandwidth_hz = None
    bandwidth_norm = None

    if region is not None:
        center_freq_hz = region.center_freq_hz
        center_freq_norm = region.center_freq_normalized
        bandwidth_hz = region.bandwidth_hz
        bandwidth_norm = region.bandwidth_normalized
    elif analysis.detected_regions:
        r = analysis.detected_regions[0]
        center_freq_hz = r.center_freq_hz
        center_freq_norm = r.center_freq_normalized
        bandwidth_hz = r.bandwidth_hz
        bandwidth_norm = r.bandwidth_normalized
    elif analysis.frequency_candidates:
        f = analysis.frequency_candidates[0]
        center_freq_hz = f.frequency_hz
        center_freq_norm = f.normalized_frequency

    bw_est = next((b for b in analysis.bandwidth_candidates if b.method == "power_containment_99pct"), None)
    if bw_est and bandwidth_hz is None:
        bandwidth_hz = bw_est.occupied_bandwidth_hz
        bandwidth_norm = bw_est.occupied_bandwidth_normalized

    common_params: dict[str, Any] = {
        "candidate_symbol_rate_hz": symbol_rate_hz,
        "candidate_symbol_rate_normalized": symbol_rate_norm,
        "candidate_samples_per_symbol": sps,
        "candidate_center_frequency_hz": center_freq_hz,
        "candidate_center_frequency_normalized": center_freq_norm,
        "candidate_bandwidth_hz": bandwidth_hz,
        "candidate_bandwidth_normalized": bandwidth_norm,
    }

    updated: list[ModulationHypothesis] = []
    for h in hypotheses:
        params = dict(common_params)
        params["modulation_family"] = h.family.value
        params["modulation_order"] = h.order
        updated.append(
            ModulationHypothesis(
                family=h.family,
                order=h.order,
                score=h.score,
                family_score=h.family_score,
                order_score=h.order_score,
                quality=h.quality,
                evidence=h.evidence,
                status=h.status,
                candidate_parameters=params,
                assumptions=h.assumptions,
            )
        )

    return updated
