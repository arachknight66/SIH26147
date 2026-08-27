import numpy as np
import pytest
from app.modulation.classical_classifier import compute_classical_scores
from app.modulation.features import extract_modulation_features
from app.modulation.ml_classifier import predict_ml_scores
from app.modulation.models import HypothesisStatus, ModulationAnalysisConfig, ModulationFamily
from app.modulation.scoring import evaluate_and_rank_hypotheses
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_scoring_and_ranking_qpsk():
    samples, _ = generate_modulated_signal("QPSK", snr_db=25.0, seed=42)
    fv = extract_modulation_features(samples)
    c_res = compute_classical_scores(fv)
    m_res = predict_ml_scores(fv)
    cfg = ModulationAnalysisConfig()

    hyps, selected, is_amb, is_unk = evaluate_and_rank_hypotheses(fv, c_res, m_res, cfg)

    assert not is_unk
    assert not is_amb
    assert selected is not None
    assert selected.family == ModulationFamily.PSK
    assert selected.order == 4
    assert selected.score > 0.60
    assert selected.status == HypothesisStatus.HYPOTHESIS_UNVERIFIED

def test_scoring_unknown_rejection_noise():
    samples, _ = generate_modulated_signal("NOISE", snr_db=25.0, seed=42)
    fv = extract_modulation_features(samples)
    c_res = compute_classical_scores(fv)
    m_res = predict_ml_scores(fv)
    cfg = ModulationAnalysisConfig(unknown_threshold=0.45)

    hyps, selected, is_amb, is_unk = evaluate_and_rank_hypotheses(fv, c_res, m_res, cfg)

    # Pure noise should be rejected as UNKNOWN
    assert is_unk
    assert selected is None
