import numpy as np
import pytest
from app.modulation.classical_classifier import compute_classical_scores
from app.modulation.features import extract_modulation_features
from app.modulation.models import ModulationFamily
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_classical_scoring_bpsk():
    samples, _ = generate_modulated_signal("BPSK", snr_db=25.0, seed=42)
    fv = extract_modulation_features(samples)
    scores = compute_classical_scores(fv)

    # BPSK should be the highest scored candidate
    best_cls = max(scores.scores, key=scores.scores.get)
    assert best_cls == (ModulationFamily.PSK, 2)
    assert scores.scores[best_cls] > 0.70

def test_classical_scoring_qpsk():
    samples, _ = generate_modulated_signal("QPSK", snr_db=25.0, seed=42)
    fv = extract_modulation_features(samples)
    scores = compute_classical_scores(fv)

    best_cls = max(scores.scores, key=scores.scores.get)
    assert best_cls == (ModulationFamily.PSK, 4)
    assert scores.scores[best_cls] > 0.70

def test_classical_scoring_bfsk():
    samples, _ = generate_modulated_signal("BFSK", snr_db=25.0, seed=42)
    fv = extract_modulation_features(samples)
    scores = compute_classical_scores(fv)

    best_cls = max(scores.scores, key=scores.scores.get)
    assert best_cls == (ModulationFamily.FSK, 2)
    assert scores.scores[best_cls] > 0.70
