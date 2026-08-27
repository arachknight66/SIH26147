import numpy as np
import pytest
from app.modulation.models import FeatureValidity
from app.modulation.phase import extract_phase_features
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_phase_features_bpsk_collapse():
    # BPSK has 180-deg phase transitions; x^2 collapses to a single carrier line
    samples, _ = generate_modulated_signal("BPSK", pulse_shape="rect", snr_db=90.0, seed=42)
    feat = extract_phase_features(samples)

    assert feat.validity == FeatureValidity.VALID
    # Squaring collapses BPSK phase -> circular variance near 0
    assert feat.var_phase_sq < 0.15

def test_phase_features_qpsk_collapse():
    # QPSK has 90-deg phase transitions; x^4 collapses to a single carrier line
    samples, _ = generate_modulated_signal("QPSK", pulse_shape="rect", snr_db=90.0, seed=42)
    feat = extract_phase_features(samples)

    assert feat.validity == FeatureValidity.VALID
    # 4th power collapses QPSK phase
    assert feat.var_phase_4th < 0.20
    # Squaring does NOT collapse QPSK phase
    assert feat.var_phase_sq > 0.50

def test_phase_features_8psk_collapse():
    # 8-PSK has 45-deg phase transitions; x^8 collapses
    samples, _ = generate_modulated_signal("8PSK", pulse_shape="rect", snr_db=90.0, seed=42)
    feat = extract_phase_features(samples)

    assert feat.validity == FeatureValidity.VALID
    # 8th power collapses 8PSK phase
    assert feat.var_phase_8th < 0.30
    assert feat.var_phase_4th > 0.50
