from __future__ import annotations
from dataclasses import replace
from typing import Sequence
import numpy as np
from app.models.analysis import DetectedRegion, SignalAnalysis
from app.models.metadata import Diagnostic, DiagnosticSeverity
from app.models.signal import SignalRecording
from app.modulation.models import ModulationAnalysis, ModulationHypothesis
from .candidate_search import build_candidate_configurations, execute_candidate_recovery
from .models import (
    RecoveryAnalysis,
    RecoveryCandidate,
    RecoveryConfig,
    RecoveryQuality,
    RecoveryQualityLevel,
    RecoveryStatus,
)
from .preprocessing import prepare_recovery_samples
from .quality import evaluate_windowed_stability
from .ranking import rank_and_select_candidates

def recover_signal(
    recording: SignalRecording,
    analysis: SignalAnalysis | None = None,
    modulation_analysis: ModulationAnalysis | None = None,
    region: DetectedRegion | None = None,
    config: RecoveryConfig | None = None,
) -> RecoveryAnalysis:
    """
    Execute empirical receiver synchronization, carrier/timing recovery, and demodulation for a signal region.

    Parameters
    ----------
    recording : SignalRecording
        Input canonical complex64 signal recording.
    analysis : SignalAnalysis | None
        Quantitative signal analysis from Phase 2.
    modulation_analysis : ModulationAnalysis | None
        Ranked modulation hypotheses from Phase 3.
    region : DetectedRegion | None
        Target signal region.
    config : RecoveryConfig | None
        Recovery configuration.

    Returns
    -------
    RecoveryAnalysis
    """
    cfg = config or RecoveryConfig()
    all_diagnostics: list[Diagnostic] = []

    # 1. Target Region Resolution: pick dominant bandwidth region
    target_region = region
    if target_region is None and analysis is not None and analysis.detected_regions:
        target_region = max(analysis.detected_regions, key=lambda r: (r.bandwidth_normalized or 0.0))

    # 2. Non-Destructive Preprocessing
    cond_samples, prep_prov = prepare_recovery_samples(recording, region=target_region, config=cfg)

    # 3. Guard against empty / zero / insufficient samples
    max_amp = float(np.max(np.abs(cond_samples))) if len(cond_samples) > 0 else 0.0
    if len(cond_samples) < 32 or max_amp < 1e-5:
        all_diagnostics.append(
            Diagnostic(
                code="INSUFFICIENT_SAMPLES",
                message=f"Recording region has {len(cond_samples)} samples (max amp={max_amp:.2e}), insufficient for receiver synchronization.",
                severity=DiagnosticSeverity.ERROR,
            )
        )
        return RecoveryAnalysis(
            recording_reference="in_memory",
            signal_region=target_region,
            candidates=[],
            selected_candidate=None,
            recovered_signal=None,
            is_recovered=False,
            is_inconclusive=True,
            failure_reason="Insufficient or zero-power samples for recovery",
            diagnostics=all_diagnostics,
            provenance={"preprocessing": prep_prov},
        )

    # 4. Extract Candidate Receiver Configurations
    candidate_configs = build_candidate_configurations(modulation_analysis, config=cfg)

    # 5. Execute Multi-Candidate Recovery Loops
    executed_candidates: list[RecoveryCandidate] = []
    for cand_idx, (fam, ord_val, nominal_sps, p3_score) in enumerate(candidate_configs, start=1):
        cand_res = execute_candidate_recovery(
            cond_samples,
            candidate_id=cand_idx,
            family=fam,
            order=ord_val,
            nominal_sps=nominal_sps,
            phase3_score=p3_score,
            config=cfg,
        )
        executed_candidates.append(cand_res)

    # 6. Candidate Ranking, Selection & Wrong-Hypothesis Detection
    ranked, selected, rec_sig, wrong_hyp, rank_diags = rank_and_select_candidates(executed_candidates)
    all_diagnostics.extend(rank_diags)

    # 7. Evaluate Windowed Temporal Stability on Selected Candidate
    window_stability: float | None = None
    if selected is not None and selected.quality.quality_level in (RecoveryQualityLevel.HIGH, RecoveryQualityLevel.MODERATE):
        def runner(s_slice: np.ndarray) -> RecoveryCandidate:
            return execute_candidate_recovery(
                s_slice,
                candidate_id=selected.candidate_id,
                family=selected.family,
                order=selected.order or 4,
                nominal_sps=selected.samples_per_symbol,
                phase3_score=selected.phase3_score,
                config=cfg,
            )
        
        w_score, w_diags = evaluate_windowed_stability(cond_samples, runner, num_windows=cfg.window_count)
        all_diagnostics.extend(w_diags)
        window_stability = w_score
        selected = replace(selected, quality=replace(selected.quality, window_consistency_score=w_score))
        ranked = [selected if candidate.candidate_id == selected.candidate_id else candidate for candidate in ranked]

    is_recovered = (selected is not None and rec_sig is not None)
    is_inconclusive = not is_recovered

    return RecoveryAnalysis(
        recording_reference="in_memory",
        signal_region=target_region,
        candidates=ranked,
        selected_candidate=selected,
        recovered_signal=rec_sig,
        is_recovered=is_recovered,
        is_inconclusive=is_inconclusive,
        wrong_hypothesis_detected=wrong_hyp,
        failure_reason=None if is_recovered else "Recovery inconclusive across all attempted receiver candidates",
        diagnostics=all_diagnostics,
        provenance={
            "preprocessing": prep_prov,
            "num_candidates_attempted": len(executed_candidates),
            "window_stability": window_stability,
            "receiver_assumptions": {"rrc_rolloffs": list(cfg.rrc_rolloffs), "min_recovery_symbols": cfg.min_recovery_symbols},
        },
    )

def recover_candidate(
    recording: SignalRecording,
    analysis: SignalAnalysis | None,
    hypothesis: ModulationHypothesis,
    region: DetectedRegion | None = None,
    config: RecoveryConfig | None = None,
) -> RecoveryCandidate:
    """
    Directly execute receiver synchronization and demodulation for a single specified hypothesis.

    Parameters
    ----------
    recording : SignalRecording
    analysis : SignalAnalysis | None
    hypothesis : ModulationHypothesis
    region : DetectedRegion | None
    config : RecoveryConfig | None

    Returns
    -------
    RecoveryCandidate
    """
    cfg = config or RecoveryConfig()
    cond_samples, _ = prepare_recovery_samples(recording, region=region, config=cfg)
    sps = float(hypothesis.candidate_parameters.get("candidate_samples_per_symbol") or 8.0)
    ord_val = hypothesis.order or (4 if hypothesis.family == ModulationFamily.PSK else (16 if hypothesis.family == ModulationFamily.QAM else 2))

    return execute_candidate_recovery(
        cond_samples,
        candidate_id=1,
        family=hypothesis.family,
        order=ord_val,
        nominal_sps=sps,
        phase3_score=hypothesis.score,
        config=cfg,
    )

def recover_all_regions(
    recording: SignalRecording,
    analysis: SignalAnalysis | None = None,
    modulation_analyses: list[ModulationAnalysis] | None = None,
    config: RecoveryConfig | None = None,
) -> list[RecoveryAnalysis]:
    """
    Execute independent receiver recovery across all detected signal regions.

    Parameters
    ----------
    recording : SignalRecording
    analysis : SignalAnalysis | None
    modulation_analyses : list[ModulationAnalysis] | None
    config : RecoveryConfig | None

    Returns
    -------
    results : list[RecoveryAnalysis]
    """
    cfg = config or RecoveryConfig()
    regions = analysis.detected_regions if (analysis and analysis.detected_regions) else [None]
    results: list[RecoveryAnalysis] = []

    for i, reg in enumerate(regions):
        mod_an = modulation_analyses[i] if (modulation_analyses and i < len(modulation_analyses)) else None
        res = recover_signal(recording, analysis=analysis, modulation_analysis=mod_an, region=reg, config=cfg)
        results.append(res)

    return results
