import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import dataclasses

from .models import SignalRecording, Diagnostic, Severity, HypothesisStatus, CandidateParameters, ModulationHypothesis, FeatureValidity
from .features import extract_all_features, ModulationFeatureVector
from .classifier import compute_classical_scores, ClassScore
from .rate_estimation import estimate_symbol_rate_consensus

def evaluate_and_rank_hypotheses(
    feature_vector: ModulationFeatureVector,
    classical_scores: Dict[str, ClassScore],
    snr_estimate: float,
    config: Dict[str, Any],
    recording: SignalRecording
) -> Tuple[List[ModulationHypothesis], Optional[ModulationHypothesis], bool, bool]:
    
    unknown_threshold = config.get("unknown_threshold", 0.45)
    ambiguity_margin = config.get("ambiguity_margin", 0.08)
    
    # Base quality tier on SNR and validity
    if snr_estimate > 15 and feature_vector.overall_validity == FeatureValidity.VALID:
        base_tier = "HIGH"
    elif snr_estimate > 5 and feature_vector.overall_validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        base_tier = "MODERATE"
    else:
        base_tier = "LOW"
        
    # Get rate estimation for CandidateParameters
    consensus = estimate_symbol_rate_consensus(recording)
    if consensus is not None:
        rate, unit, status, conf = consensus
        if unit == "Hz" and recording.sample_rate_hz.value:
            sps = recording.sample_rate_hz.value / rate
        else:
            sps = 1.0 / rate if rate > 0 else None
    else:
        rate, unit, sps = None, "symbols/sample", None
        
    cf = recording.center_frequency_hz.value
    bw = rate * 1.35 if rate and unit == "Hz" else None
    
    cand_params = CandidateParameters(
        symbol_rate=rate,
        symbol_rate_unit=unit,
        samples_per_symbol=sps,
        center_frequency_hz=cf,
        bandwidth_hz=bw
    )
    
    hypotheses = []
    for label, cscore in classical_scores.items():
        snr_mult = 1.0
        if snr_estimate < 10:
            snr_mult = 0.8
        elif snr_estimate < 5:
            snr_mult = 0.6
            
        final_score = cscore.score * snr_mult
        
        hyp = ModulationHypothesis(
            label=label,
            status=HypothesisStatus.HYPOTHESIS_UNVERIFIED,
            score=final_score,
            quality_tier=base_tier,
            candidate_parameters=cand_params,
            evidence=cscore.evidence,
            contradictions=cscore.contradictions
        )
        hypotheses.append(hyp)
        
    hypotheses.sort(key=lambda h: h.score, reverse=True)
    
    top_score = hypotheses[0].score if hypotheses else 0.0
    
    is_unknown = top_score < unknown_threshold
    is_ambiguous = False
    selected = None
    
    if is_unknown:
        for h in hypotheses:
            object.__setattr__(h, 'status', HypothesisStatus.UNKNOWN)
    else:
        if len(hypotheses) > 1 and (top_score - hypotheses[1].score) < ambiguity_margin:
            is_ambiguous = True
            for h in hypotheses[:2]:
                object.__setattr__(h, 'status', HypothesisStatus.AMBIGUOUS)
        else:
            selected = hypotheses[0]
            
    return hypotheses, selected, is_ambiguous, is_unknown

def check_temporal_consistency(recording: SignalRecording, config: Dict[str, Any]) -> Tuple[float, Optional[Diagnostic]]:
    window_count = config.get("window_count", 4)
    n_samples = len(recording.samples)
    if n_samples < 256:
        return 1.0, None # Too short to partition
        
    window_size = n_samples // window_count
    
    fv_global = extract_all_features(recording)
    scores_global = compute_classical_scores(fv_global)
    global_winner = max(scores_global.items(), key=lambda x: x[1].score)[0]
    
    matches = 0
    for i in range(window_count):
        start = i * window_size
        end = start + window_size
        sub_samples = recording.samples[start:end]
        
        sub_rec = dataclasses.replace(recording, samples=sub_samples)
        fv_sub = extract_all_features(sub_rec)
        scores_sub = compute_classical_scores(fv_sub)
        sub_winner = max(scores_sub.items(), key=lambda x: x[1].score)[0]
        
        if sub_winner == global_winner:
            matches += 1
            
    consistency = matches / window_count
    
    diag = None
    if consistency < 0.60:
        diag = Diagnostic(
            severity=Severity.INFO,
            code="TEMPORAL_NONSTATIONARITY",
            message=f"Signal characteristics vary significantly over time (consistency {consistency:.2f})",
            evidence=f"Matched global class '{global_winner}' in only {matches}/{window_count} sub-windows."
        )
        
    return consistency, diag
