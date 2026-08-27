import numpy as np
import pytest
from app.modulation.amplitude import extract_amplitude_features
from app.modulation.models import FeatureValidity
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_amplitude_features_constant_envelope_psk():
    # Pure QPSK with constant envelope
    samples, _ = generate_modulated_signal("QPSK", pulse_shape="rect", snr_db=90.0, seed=42)
    feat = extract_amplitude_features(samples)

    assert feat.validity == FeatureValidity.VALID
    # Normalized amplitude variance should be small for constant envelope
    assert feat.norm_variance < 0.05
    assert feat.coeff_var < 0.05
    assert np.isclose(feat.peak_to_rms, 1.0, atol=0.08)

def test_amplitude_features_varying_envelope_qam():
    # 16-QAM with multi-ring varying envelope
    samples, _ = generate_modulated_signal("16QAM", pulse_shape="rect", snr_db=90.0, seed=42)
    feat = extract_amplitude_features(samples)

    assert feat.validity == FeatureValidity.VALID
    # 16-QAM has higher normalized variance
    assert feat.norm_variance > 0.04
    assert feat.coeff_var > 0.15
    assert feat.peak_to_rms > 1.2

def test_amplitude_features_short_signal():
    short_samples = np.ones(8, dtype=np.complex64)
    feat = extract_amplitude_features(short_samples)
    assert feat.validity == FeatureValidity.UNAVAILABLE
