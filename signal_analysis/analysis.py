"""Python orchestration adapter for native Phase 3 estimation and inference."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import DEFAULT_MAX_ANALYSIS_SAMPLES

from .models import (
    CandidateParameters,
    HypothesisStatus,
    ModulationHypothesis,
    SignalRecording,
)
from .native import analyze_window as analyze_window_native
from .native import require_native


def analysis_summary(analysis: Any) -> Dict[str, Any]:
    """Return a JSON/deepcopy-safe summary for PipelineResult and reports."""
    def estimate(value: Any) -> Dict[str, Any]:
        return {
            "value": value.value,
            "uncertainty": value.uncertainty,
            "unit": value.unit,
            "validity": value.validity.name,
            "evidence": value.evidence,
        }

    return {
        "status": analysis.status.name,
        "processed_samples": analysis.processed_samples,
        "temporal_consistency": analysis.temporal_consistency,
        "snr": estimate(analysis.snr),
        "occupied_bandwidth": estimate(analysis.occupied_bandwidth),
        "carrier_offset": estimate(analysis.carrier_offset),
        "symbol_rate": estimate(analysis.symbol_rate),
        "rate_candidates": [
            {
                "symbol_rate": candidate.symbol_rate,
                "samples_per_symbol": candidate.samples_per_symbol,
                "score": candidate.score,
            }
            for candidate in analysis.rate_candidates
        ],
        "modulation_candidates": [
            {
                "label": candidate.label,
                "family": candidate.family,
                "score": candidate.score,
                "status": candidate.status.name,
                "constellation_error": candidate.constellation_error,
                "temporal_consistency": candidate.temporal_consistency,
                "evidence": list(candidate.evidence),
                "contradictions": list(candidate.contradictions),
            }
            for candidate in analysis.modulation_candidates
        ],
        "diagnostics": list(analysis.diagnostics),
    }


def _quality_tier(snr_db: Optional[float], validity: str) -> str:
    if snr_db is not None and snr_db >= 15.0 and validity == "VALID":
        return "HIGH"
    if snr_db is not None and snr_db >= 5.0 and validity != "UNAVAILABLE":
        return "MODERATE"
    return "LOW"


def analyze_modulation(
    recording: SignalRecording,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ModulationHypothesis], Optional[ModulationHypothesis], bool, bool, Any]:
    """Estimate parameters once and adapt ranked native candidates to GUI models."""
    options = config or {}
    native = require_native()
    native_config = native.AnalysisConfig()
    native_config.sample_rate_hz = recording.sample_rate_hz.value
    native_config.complex_input = recording.semantic_type == "complex_iq"
    native_config.minimum_samples_per_symbol = float(options.get("minimum_samples_per_symbol", 2.0))
    native_config.maximum_samples_per_symbol = float(options.get("maximum_samples_per_symbol", 32.0))
    native_config.maximum_candidates = int(options.get("maximum_modulation_candidates", 5))
    native_config.temporal_windows = int(options.get("window_count", 4))
    native_config.unknown_threshold = float(options.get("unknown_threshold", 0.55))
    native_config.ambiguity_margin = float(options.get("ambiguity_margin", 0.06))

    samples = recording.samples
    if samples.ndim > 1:
        samples = samples[:, 0]
    sample_limit = int(options.get("analysis_sample_limit", DEFAULT_MAX_ANALYSIS_SAMPLES))
    if sample_limit <= 0:
        raise ValueError("analysis_sample_limit must be positive")
    # Parameter estimation is a bounded preview operation.  Receivers retain
    # their own declared acquisition coverage; this prevents an unfamiliar
    # long WAV/IQ capture from feeding an unbounded O(N) estimator.
    samples = samples[:sample_limit]
    if samples.dtype != np.complex64 or not samples.flags.c_contiguous:
        samples = np.ascontiguousarray(samples, dtype=np.complex64)
    analysis = analyze_window_native(samples, config=native_config)

    status_name = analysis.status.name
    is_unknown = status_name in {"UNKNOWN", "UNSUPPORTED"}
    is_ambiguous = status_name == "AMBIGUOUS"
    snr_value = analysis.snr.value
    rate_value = analysis.symbol_rate.value
    rate_unit = analysis.symbol_rate.unit
    sps = None
    if rate_value and recording.sample_rate_hz.value and rate_unit == "Hz":
        sps = recording.sample_rate_hz.value / rate_value
    elif rate_value and rate_unit == "symbols/sample":
        sps = 1.0 / rate_value

    bandwidth_value = analysis.occupied_bandwidth.value
    parameters = CandidateParameters(
        symbol_rate=rate_value,
        symbol_rate_unit=rate_unit,
        samples_per_symbol=sps,
        center_frequency_hz=recording.center_frequency_hz.value,
        bandwidth_hz=bandwidth_value if analysis.occupied_bandwidth.unit == "Hz" else None,
        bandwidth=bandwidth_value,
        bandwidth_unit=analysis.occupied_bandwidth.unit,
        bandwidth_uncertainty=analysis.occupied_bandwidth.uncertainty,
        carrier_offset=analysis.carrier_offset.value,
        carrier_offset_unit=analysis.carrier_offset.unit,
        carrier_offset_uncertainty=analysis.carrier_offset.uncertainty,
        snr_db=snr_value,
        snr_uncertainty_db=analysis.snr.uncertainty,
        parameter_validity={
            "snr": analysis.snr.validity.name,
            "bandwidth": analysis.occupied_bandwidth.validity.name,
            "carrier_offset": analysis.carrier_offset.validity.name,
            "symbol_rate": analysis.symbol_rate.validity.name,
        },
    )
    quality = _quality_tier(snr_value, analysis.symbol_rate.validity.name)
    hypotheses: List[ModulationHypothesis] = []
    for index, candidate in enumerate(analysis.modulation_candidates):
        if status_name in {"UNKNOWN", "UNSUPPORTED"}:
            hypothesis_status = HypothesisStatus.UNKNOWN
        elif status_name == "AMBIGUOUS" and index < 2:
            hypothesis_status = HypothesisStatus.AMBIGUOUS
        else:
            hypothesis_status = HypothesisStatus.HYPOTHESIS_UNVERIFIED
        hypotheses.append(ModulationHypothesis(
            label=candidate.label,
            status=hypothesis_status,
            score=float(candidate.score),
            quality_tier=quality,
            candidate_parameters=parameters,
            evidence={
                "constellation_error": float(candidate.constellation_error),
                "temporal_consistency": float(candidate.temporal_consistency),
            },
            contradictions=list(candidate.contradictions),
        ))
    selected = hypotheses[0] if hypotheses and not is_unknown and not is_ambiguous else None
    return hypotheses, selected, is_ambiguous, is_unknown, analysis
