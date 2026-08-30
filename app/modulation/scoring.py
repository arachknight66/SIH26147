from __future__ import annotations
from typing import Sequence
import numpy as np
from app.models.analysis import SNREstimate
from .classical_classifier import ClassicalScores
from .ml_classifier import MLClassificationResult
from .models import (
    FeatureValidity,
    HypothesisStatus,
    ModulationAnalysisConfig,
    ModulationEvidence,
    ModulationFamily,
    ModulationFeatureVector,
    ModulationHypothesis,
)

def evaluate_and_rank_hypotheses(
    fv: ModulationFeatureVector,
    classical_res: ClassicalScores,
    ml_res: MLClassificationResult,
    config: ModulationAnalysisConfig,
    snr_estimates: Sequence[SNREstimate] | None = None,
) -> tuple[list[ModulationHypothesis], ModulationHypothesis | None, bool, bool]:
    """
    Fuse independent classical and ML evidence into ranked candidate hypotheses.

    Applies contradiction penalties, ambiguity margin logic, and out-of-distribution / unknown rejection.

    Parameters
    ----------
    fv : ModulationFeatureVector
    classical_res : ClassicalScores
    ml_res : MLClassificationResult
    config : ModulationAnalysisConfig
    snr_estimates : Sequence[SNREstimate] | None

    Returns
    -------
    hypotheses : list[ModulationHypothesis]
    selected : ModulationHypothesis | None
    is_ambiguous : bool
    is_unknown : bool
    """
    # Determine best SNR from available multi-method estimators
    snr_val: float | None = None
    if snr_estimates:
        valid_snrs = [s.snr_db for s in snr_estimates if s.snr_db is not None]
        if valid_snrs:
            snr_val = float(max(valid_snrs))

    if snr_val is not None:
        snr_quality = float(np.clip((snr_val + 2.0) / 10.0, 0.40, 1.0))
    else:
        snr_quality = 0.75

    # Feature validity penalty
    if fv.overall_validity == FeatureValidity.VALID:
        val_penalty = 0.0
    elif fv.overall_validity == FeatureValidity.PARTIALLY_VALID:
        val_penalty = 0.05
    elif fv.overall_validity == FeatureValidity.UNRELIABLE:
        val_penalty = 0.25
    else:
        val_penalty = 0.50

    hypotheses: list[ModulationHypothesis] = []
    all_classes = set(classical_res.scores.keys()) | set(ml_res.scores.keys())

    for cls_key in all_classes:
        fam, ord_ = cls_key
        c_score = classical_res.scores.get(cls_key, 0.05)
        m_score = ml_res.scores.get(cls_key, 0.05) if config.enable_ml else 0.0
        
        c_breakdown = classical_res.evidence_breakdown.get(cls_key, {})
        c_notes = classical_res.supporting_notes.get(cls_key, [])
        c_contra = classical_res.contradictions.get(cls_key, [])

        # Contradiction penalty
        contra_penalty = 0.35 * len(c_contra)

        # Fused candidate score: w_c=0.55, w_ml=0.45
        if config.enable_ml:
            base_score = 0.55 * c_score + 0.45 * m_score
        else:
            base_score = c_score

        fused_score = float(np.clip((base_score - contra_penalty) * snr_quality - val_penalty, 0.0, 0.99))

        # Quality assessment
        if fused_score >= 0.70 and snr_quality >= 0.70 and fv.overall_validity == FeatureValidity.VALID and len(c_contra) == 0:
            quality = "HIGH"
        elif fused_score >= 0.45 and snr_quality >= 0.35 and len(c_contra) <= 1:
            quality = "MODERATE"
        else:
            quality = "LOW"

        evidence_obj = ModulationEvidence(
            amplitude_score=round(c_breakdown.get("amplitude", 0.5), 3),
            phase_score=round(c_breakdown.get("phase", 0.5), 3),
            frequency_score=round(c_breakdown.get("frequency", 0.5), 3),
            cumulant_score=round(c_breakdown.get("cumulants", 0.5), 3),
            spectral_score=round(c_breakdown.get("spectral", 0.5), 3),
            periodicity_score=round(c_breakdown.get("periodicity", 0.0), 3),
            classical_model_score=round(c_score, 3),
            ml_score=round(m_score, 3),
            snr_quality=round(snr_quality, 3),
            contradiction_penalty=round(contra_penalty, 3),
            supporting_evidence=tuple(c_notes),
            assumptions=(
                "Assumes single dominant emission in selected signal region.",
                "Subject to downstream carrier & timing synchronization verification in Phase 4.",
            ),
        )

        hypotheses.append(
            ModulationHypothesis(
                family=fam,
                order=ord_,
                score=round(fused_score, 3),
                family_score=round(base_score, 3),
                order_score=round(fused_score, 3),
                quality=quality,
                evidence=evidence_obj,
                status=HypothesisStatus.HYPOTHESIS_UNVERIFIED,
                assumptions=[
                    "Hypothesis constructed from statistical moments and spectral features.",
                    "Demodulation and symbol slicing have not yet been performed.",
                ],
            )
        )

    # Sort hypotheses by score descending
    sorted_hypotheses = sorted(hypotheses, key=lambda h: -h.score)[: config.max_hypotheses]

    if not sorted_hypotheses:
        return [], None, False, True

    top_h = sorted_hypotheses[0]
    is_unknown = top_h.score < config.unknown_threshold
    is_ambiguous = False

    if len(sorted_hypotheses) >= 2:
        second_h = sorted_hypotheses[1]
        score_diff = top_h.score - second_h.score
        if not is_unknown and score_diff < config.ambiguity_margin:
            is_ambiguous = True

    if is_unknown:
        selected = None
        updated = [
            ModulationHypothesis(
                family=h.family,
                order=h.order,
                score=h.score,
                family_score=h.family_score,
                order_score=h.order_score,
                quality="LOW",
                evidence=h.evidence,
                status=HypothesisStatus.UNKNOWN,
                assumptions=h.assumptions,
            )
            for h in sorted_hypotheses
        ]
        return updated, selected, is_ambiguous, is_unknown

    elif is_ambiguous:
        selected = None
        updated = [
            ModulationHypothesis(
                family=h.family,
                order=h.order,
                score=h.score,
                family_score=h.family_score,
                order_score=h.order_score,
                quality=h.quality,
                evidence=h.evidence,
                status=HypothesisStatus.AMBIGUOUS if (top_h.score - h.score < config.ambiguity_margin) else HypothesisStatus.HYPOTHESIS_UNVERIFIED,
                assumptions=h.assumptions,
            )
            for h in sorted_hypotheses
        ]
        return updated, selected, is_ambiguous, is_unknown

    else:
        selected = top_h
        return sorted_hypotheses, selected, is_ambiguous, is_unknown
