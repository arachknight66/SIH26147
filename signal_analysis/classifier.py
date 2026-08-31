import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import FeatureValidity
from .features import ModulationFeatureVector

@dataclass(frozen=True)
class ClassScore:
    score: float
    evidence: Dict[str, float]
    contradictions: List[str]

def _evaluate_bpsk(fv: ModulationFeatureVector) -> ClassScore:
    evidence = {}
    contradictions = []
    
    # BPSK f20 ~ 1.0, f40 ~ 2.0
    if fv.cumulant.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        c_score = 0.0
        if 0.8 < fv.cumulant.f20 < 1.2: c_score += 0.5
        if 1.7 < fv.cumulant.f40 < 2.3: c_score += 0.5
        evidence['cumulant'] = c_score
        
    # Phase collapse M=2 low variance
    if fv.phase.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        p_score = max(0.0, 1.0 - fv.phase.collapse_var_2 * 2)
        evidence['phase'] = p_score
        
    # Frequency: Should NOT be bimodal
    if fv.frequency.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        if fv.frequency.fsk_states > 1:
            contradictions.append(f"Frequency is bimodal ({fv.frequency.fsk_states} states), contradicting single-carrier PSK.")
            
    # Amplitude: Constant envelope roughly
    if fv.amplitude.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        if fv.amplitude.normalized_variance > 0.5:
            evidence['amplitude'] = 0.0
        else:
            evidence['amplitude'] = 1.0
            
    weights = {'cumulant': 0.4, 'phase': 0.4, 'amplitude': 0.2}
    total_score = sum(evidence.get(k, 0) * w for k, w in weights.items())
    
    if contradictions:
        total_score *= 0.1
        
    return ClassScore(score=total_score, evidence=evidence, contradictions=contradictions)

def _evaluate_qpsk(fv: ModulationFeatureVector) -> ClassScore:
    evidence = {}
    contradictions = []
    
    # QPSK f20 ~ 0.0, f40 ~ 1.0 (approx 1.0)
    if fv.cumulant.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        c_score = 0.0
        if fv.cumulant.f20 < 0.2: c_score += 0.5
        if 0.8 < fv.cumulant.f40 < 1.2: c_score += 0.5
        evidence['cumulant'] = c_score
        
    # Phase collapse M=4 low variance, M=2 high variance
    if fv.phase.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        p_score = 0.0
        if fv.phase.collapse_var_2 > 0.5: p_score += 0.5
        p_score += max(0.0, 0.5 * (1.0 - fv.phase.collapse_var_4 * 2))
        evidence['phase'] = p_score
        
    if fv.frequency.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        if fv.frequency.fsk_states > 1:
            contradictions.append(f"Frequency is bimodal ({fv.frequency.fsk_states} states), contradicting single-carrier PSK.")
            
    if fv.amplitude.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        evidence['amplitude'] = max(0.0, 1.0 - fv.amplitude.normalized_variance)
            
    weights = {'cumulant': 0.4, 'phase': 0.4, 'amplitude': 0.2}
    total_score = sum(evidence.get(k, 0) * w for k, w in weights.items())
    
    if contradictions:
        total_score *= 0.1
        
    return ClassScore(score=total_score, evidence=evidence, contradictions=contradictions)

def _evaluate_8psk(fv: ModulationFeatureVector) -> ClassScore:
    evidence = {}
    contradictions = []
    
    # 8PSK f20 ~ 0.0, f40 ~ 0.0
    if fv.cumulant.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        c_score = 0.0
        if fv.cumulant.f20 < 0.2: c_score += 0.5
        if fv.cumulant.f40 < 0.2: c_score += 0.5
        evidence['cumulant'] = c_score
        
    # Phase collapse M=8 low variance
    if fv.phase.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        p_score = 0.0
        if fv.phase.collapse_var_4 > 0.5: p_score += 0.5
        p_score += max(0.0, 0.5 * (1.0 - fv.phase.collapse_var_8 * 2))
        evidence['phase'] = p_score
        
    if fv.frequency.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        if fv.frequency.fsk_states > 1:
            contradictions.append(f"Frequency is bimodal ({fv.frequency.fsk_states} states), contradicting single-carrier PSK.")
            
    if fv.amplitude.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        evidence['amplitude'] = max(0.0, 1.0 - fv.amplitude.normalized_variance)
            
    weights = {'cumulant': 0.4, 'phase': 0.4, 'amplitude': 0.2}
    total_score = sum(evidence.get(k, 0) * w for k, w in weights.items())
    
    if contradictions:
        total_score *= 0.1
        
    return ClassScore(score=total_score, evidence=evidence, contradictions=contradictions)

def _evaluate_16qam(fv: ModulationFeatureVector) -> ClassScore:
    evidence = {}
    contradictions = []
    
    # 16QAM f20 ~ 0.0, f40 ~ 0.68
    if fv.cumulant.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        c_score = 0.0
        if fv.cumulant.f20 < 0.2: c_score += 0.5
        if 0.55 < fv.cumulant.f40 < 0.8: c_score += 0.5
        evidence['cumulant'] = c_score
        
    if fv.amplitude.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        # QAM has multi-ring, so high normalized variance and specific kurtosis
        a_score = 0.0
        if fv.amplitude.normalized_variance > 0.1: a_score += 0.5
        # 16QAM kurtosis is negative (sub-Gaussian), ~ -0.68
        if fv.amplitude.excess_kurtosis < -0.2: a_score += 0.5
        evidence['amplitude'] = a_score
        
    if fv.frequency.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        if fv.frequency.fsk_states > 1:
            contradictions.append(f"Frequency is bimodal ({fv.frequency.fsk_states} states), contradicting single-carrier QAM.")
            
    weights = {'cumulant': 0.6, 'amplitude': 0.4}
    total_score = sum(evidence.get(k, 0) * w for k, w in weights.items())
    
    if contradictions:
        total_score *= 0.1
        
    return ClassScore(score=total_score, evidence=evidence, contradictions=contradictions)

def _evaluate_2fsk(fv: ModulationFeatureVector) -> ClassScore:
    evidence = {}
    contradictions = []
    
    if fv.frequency.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        if fv.frequency.fsk_states == 2:
            evidence['frequency'] = 1.0
        elif fv.frequency.fsk_states > 2:
            contradictions.append(f"Found {fv.frequency.fsk_states} FSK states, expected 2.")
            evidence['frequency'] = 0.0
        else:
            contradictions.append("No bimodal frequency distribution found.")
            evidence['frequency'] = 0.0
            
    if fv.amplitude.validity in (FeatureValidity.VALID, FeatureValidity.PARTIALLY_VALID):
        evidence['amplitude'] = max(0.0, 1.0 - fv.amplitude.normalized_variance * 2)
        
    weights = {'frequency': 0.8, 'amplitude': 0.2}
    total_score = sum(evidence.get(k, 0) * w for k, w in weights.items())
    
    if contradictions:
        total_score *= 0.1
        
    return ClassScore(score=total_score, evidence=evidence, contradictions=contradictions)

def compute_classical_scores(feature_vector: ModulationFeatureVector) -> Dict[str, ClassScore]:
    """
        Compute scores for classical modulations based on features.

        Parameters
        ----------
        feature_vector : ModulationFeatureVector
            Extracted features.

        Returns
        -------
        Dict[str, ClassScore]
            Dictionary mapping modulation labels to their classification scores.
        """
    return {
        "BPSK": _evaluate_bpsk(feature_vector),
        "QPSK": _evaluate_qpsk(feature_vector),
        "8PSK": _evaluate_8psk(feature_vector),
        "16-QAM": _evaluate_16qam(feature_vector),
        "2-FSK": _evaluate_2fsk(feature_vector)
    }
