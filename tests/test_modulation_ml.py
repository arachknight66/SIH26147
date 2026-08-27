import numpy as np
import pytest
from app.modulation.features import extract_modulation_features
from app.modulation.ml_classifier import get_ml_model_metadata, predict_ml_scores
from app.modulation.models import ModulationFamily
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_ml_metadata():
    meta = get_ml_model_metadata()
    assert "model_version" in meta
    assert "feature_schema_version" in meta
    assert meta["feature_schema_version"] == "1.0"
    assert len(meta["feature_names"]) == 25

def test_ml_inference_qpsk():
    samples, _ = generate_modulated_signal("QPSK", snr_db=25.0, seed=42)
    fv = extract_modulation_features(samples)
    res = predict_ml_scores(fv)

    assert res.feature_schema_version == "1.0"
    assert res.uncertainty < 0.85
    # QPSK should be top scored in ML
    best_ml = max(res.scores, key=res.scores.get)
    assert best_ml == (ModulationFamily.PSK, 4)
