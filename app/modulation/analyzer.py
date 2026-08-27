from __future__ import annotations
from typing import Any
import numpy as np
from app.analysis.analyzer import analyze_signal
from app.models.analysis import DetectedRegion, SignalAnalysis
from app.models.metadata import Diagnostic, DiagnosticSeverity, MetadataStatus
from app.models.signal import SignalRecording
from .candidates import attach_candidate_parameters
from .classical_classifier import compute_classical_scores
from .features import extract_modulation_features
from .ml_classifier import get_ml_model_metadata, predict_ml_scores
from .models import (
    FeatureValidity,
    ModulationAnalysis,
    ModulationAnalysisConfig,
    ModulationHypothesis,
    RawComplexPlaneDistribution,
)
from .scoring import evaluate_and_rank_hypotheses

__MODULATION_ANALYSIS_VERSION__ = "0.3.0"

def _prepare_analysis_samples(
    recording: SignalRecording,
    region: DetectedRegion | None,
    max_samples: int = 65536,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Extract, frequency-shift, and condition signal samples for modulation analysis.
    """
    provenance_ops: dict[str, Any] = {}
    samples = recording.samples

    # 1. Time slicing if burst region specified
    if region is not None and region.start_sample is not None and region.end_sample is not None:
        s_idx = max(0, region.start_sample)
        e_idx = min(len(samples), region.end_sample)
        if e_idx > s_idx:
            samples = samples[s_idx:e_idx]
            provenance_ops["time_slice"] = {"start_sample": s_idx, "end_sample": e_idx}

    # 2. Limit total sample count to max_samples
    if len(samples) > max_samples:
        samples = samples[:max_samples]
        provenance_ops["subsampled"] = {"max_samples": max_samples}

    # 3. Frequency translation to region center if carrier offset detected (> 5 bins from DC)
    f_shift = 0.0
    if region is not None and abs(region.center_freq_normalized) >= 0.005:
        f_shift = region.center_freq_normalized
    
    if abs(f_shift) >= 0.005 and len(samples) > 0:
        t = np.arange(len(samples), dtype=np.float32)
        mix = np.exp(-2j * np.pi * f_shift * t).astype(np.complex64)
        samples = samples * mix
        provenance_ops["frequency_translation"] = {"shift_normalized": f_shift}

    return samples, provenance_ops

def analyze_modulation(
    recording: SignalRecording,
    analysis: SignalAnalysis | None = None,
    region: DetectedRegion | None = None,
    config: ModulationAnalysisConfig | None = None,
) -> ModulationAnalysis:
    """
    Perform research-grade modulation feature extraction, hypothesis generation, and uncertainty evaluation.

    Parameters
    ----------
    recording : SignalRecording
        Input canonical recording contract.
    analysis : SignalAnalysis | None
        Precomputed Phase 2 analysis (optional; computed automatically if None).
    region : DetectedRegion | None
        Target candidate signal region to analyze (optional; defaults to top detected region).
    config : ModulationAnalysisConfig | None
        Modulation engine configuration parameters.

    Returns
    -------
    ModulationAnalysis
    """
    cfg = config or ModulationAnalysisConfig()
    diagnostics: list[Diagnostic] = []

    # Ensure Phase 2 analysis is available
    if analysis is None:
        analysis = analyze_signal(recording)

    # Select target region if not explicitly provided
    selected_region = region
    if selected_region is None and analysis is not None and analysis.detected_regions:
        selected_region = max(analysis.detected_regions, key=lambda r: (r.bandwidth_normalized or 0.0))

    # Prepare conditioned signal view
    conditioned_samples, prep_prov = _prepare_analysis_samples(
        recording,
        selected_region,
        max_samples=cfg.max_analysis_samples,
    )

    n_samples = len(conditioned_samples)
    if n_samples < cfg.min_samples:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.WARNING,
                "SHORT_RECORDING",
                f"Analysis sample count ({n_samples}) is below minimum threshold ({cfg.min_samples}).",
                "Feature variance and cumulants have elevated uncertainty.",
            )
        )

    # Check SNR
    top_snr = analysis.snr_candidates[0] if analysis.snr_candidates else None
    if top_snr and top_snr.snr_db is not None and top_snr.snr_db < cfg.min_snr_db:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.WARNING,
                "LOW_SIGNAL_TO_NOISE",
                f"SNR ({top_snr.snr_db:.1f} dB) is below threshold ({cfg.min_snr_db:.1f} dB).",
                "Modulation classification confidence is penalized.",
            )
        )

    # 1. Full-window Feature Extraction
    feature_vector = extract_modulation_features(conditioned_samples, analysis=analysis)

    # 2. Classical Scoring
    classical_res = compute_classical_scores(feature_vector)

    # 3. Lightweight ML Classification
    ml_res = predict_ml_scores(feature_vector)

    # 4. Multi-Window Consistency Evaluation
    window_consistency = 1.0
    if cfg.window_count > 1 and n_samples >= (cfg.window_count * 32):
        sub_len = n_samples // cfg.window_count
        window_winners: list[tuple[str, int | None]] = []
        
        for w_i in range(cfg.window_count):
            sub_samples = conditioned_samples[w_i * sub_len : (w_i + 1) * sub_len]
            sub_fv = extract_modulation_features(sub_samples)
            sub_classical = compute_classical_scores(sub_fv)
            sub_ml = predict_ml_scores(sub_fv)
            sub_hyps, sub_sel, _, _ = evaluate_and_rank_hypotheses(sub_fv, sub_classical, sub_ml, cfg, snr_estimates=analysis.snr_candidates)
            if sub_hyps:
                window_winners.append((sub_hyps[0].family.value, sub_hyps[0].order))

        if window_winners:
            # Measure consensus with global top classical candidate
            top_global_key = max(classical_res.scores, key=classical_res.scores.get)
            matches = sum(1 for w in window_winners if (w[0] == top_global_key[0].value and w[1] == top_global_key[1]))
            window_consistency = float(matches / len(window_winners))

            if window_consistency < 0.60:
                diagnostics.append(
                    Diagnostic(
                        DiagnosticSeverity.WARNING,
                        "TEMPORAL_NONSTATIONARITY",
                        f"Modulation consistency across temporal sub-windows is low ({window_consistency*100:.1f}%).",
                        "May indicate signal burst boundaries, fading variations, or time-varying modulation.",
                    )
                )

    # 5. Evidence Fusion & Candidate Ranking
    ranked_hypotheses, selected_hypothesis, is_ambiguous, is_unknown = evaluate_and_rank_hypotheses(
        feature_vector,
        classical_res,
        ml_res,
        cfg,
        snr_estimates=analysis.snr_candidates,
    )

    # Attach candidate synchronization and physical parameters
    final_hypotheses = attach_candidate_parameters(ranked_hypotheses, analysis, selected_region)
    if selected_hypothesis is not None:
        selected_hypothesis = next((h for h in final_hypotheses if h.family == selected_hypothesis.family and h.order == selected_hypothesis.order), final_hypotheses[0])

    # 6. Diagnostics synthesis
    if is_unknown:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.INFO,
                "UNKNOWN_OR_OOD_MODULATION",
                "Max candidate score is below unknown threshold; signal is classified as UNKNOWN / OUT_OF_DISTRIBUTION.",
                f"Top candidate score was {final_hypotheses[0].score:.3f} (threshold: {cfg.unknown_threshold:.2f}).",
            )
        )
    elif is_ambiguous:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.INFO,
                "AMBIGUOUS_MODULATION",
                "Top competing modulation candidates have overlapping evidence within ambiguity margin.",
                f"Candidate separation is < {cfg.ambiguity_margin:.2f}.",
            )
        )

    # 7. Raw Complex-Plane Distribution (Subsampled for Visualization)
    scatter_len = min(2048, len(conditioned_samples))
    if scatter_len > 0:
        scatter_sub = conditioned_samples[:scatter_len]
        raw_dist = RawComplexPlaneDistribution(
            sample_subset_i=scatter_sub.real.astype(np.float32),
            sample_subset_q=scatter_sub.imag.astype(np.float32),
            radii=np.abs(scatter_sub).astype(np.float32),
            phases=np.angle(scatter_sub).astype(np.float32),
        )
    else:
        raw_dist = None

    # 8. Provenance
    provenance: dict[str, Any] = {
        "analysis_version": __MODULATION_ANALYSIS_VERSION__,
        "input_recording": str(recording.provenance.get("source_path", "in_memory")),
        "preparation_operations": prep_prov,
        "sample_count_analyzed": n_samples,
        "feature_validity": feature_vector.overall_validity.value,
        "ml_metadata": get_ml_model_metadata() if cfg.enable_ml else None,
        "window_consistency": round(window_consistency, 3),
        "is_ambiguous": is_ambiguous,
        "is_unknown": is_unknown,
    }

    return ModulationAnalysis(
        recording_reference=str(recording.provenance.get("source_path", "in_memory")),
        signal_region=selected_region,
        hypotheses=final_hypotheses,
        selected_hypothesis=selected_hypothesis,
        feature_vector=feature_vector,
        raw_distribution=raw_dist,
        window_consistency=round(window_consistency, 3),
        is_ambiguous=is_ambiguous,
        is_unknown=is_unknown,
        diagnostics=list(analysis.diagnostics) + diagnostics,
        provenance=provenance,
    )

def analyze_all_regions(
    recording: SignalRecording,
    analysis: SignalAnalysis | None = None,
    config: ModulationAnalysisConfig | None = None,
) -> list[ModulationAnalysis]:
    """
    Perform independent modulation analysis on each candidate signal region.
    """
    if analysis is None:
        analysis = analyze_signal(recording)

    if not analysis.detected_regions:
        return [analyze_modulation(recording, analysis=analysis, region=None, config=config)]

    results: list[ModulationAnalysis] = []
    for reg in analysis.detected_regions:
        results.append(analyze_modulation(recording, analysis=analysis, region=reg, config=config))
    return results
