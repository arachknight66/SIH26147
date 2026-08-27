import numpy as np
import pytest
from app.modulation.cumulants import extract_cumulant_features
from app.modulation.models import FeatureValidity
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_cumulants_bpsk():
    # Theoretical BPSK: f20 ~ 1.0, f40 ~ 2.0, f42 ~ 2.0
    samples, _ = generate_modulated_signal("BPSK", pulse_shape="rect", snr_db=90.0, seed=42)
    feat = extract_cumulant_features(samples)

    assert feat.validity == FeatureValidity.VALID
    assert np.isclose(feat.f20, 1.0, atol=0.15)
    assert feat.f40 > 1.4
    assert feat.f42 > 1.4

def test_cumulants_qpsk():
    # Theoretical QPSK: f20 ~ 0.0, f40 ~ 1.0, f42 ~ 1.0
    samples, _ = generate_modulated_signal("QPSK", pulse_shape="rect", snr_db=90.0, seed=42)
    feat = extract_cumulant_features(samples)

    assert feat.validity == FeatureValidity.VALID
    assert np.isclose(feat.f20, 0.0, atol=0.15)
    assert np.isclose(feat.f40, 1.0, atol=0.25)
    assert np.isclose(feat.f42, 1.0, atol=0.25)

def test_cumulants_16qam():
    # Theoretical 16-QAM: f20 ~ 0.0, f40 ~ 0.68, f42 ~ 0.68
    samples, _ = generate_modulated_signal("16QAM", pulse_shape="rect", snr_db=90.0, seed=42)
    feat = extract_cumulant_features(samples)

    assert feat.validity == FeatureValidity.VALID
    assert np.isclose(feat.f20, 0.0, atol=0.15)
    assert np.isclose(feat.f40, 0.68, atol=0.25)
    assert np.isclose(feat.f42, 0.68, atol=0.25)
