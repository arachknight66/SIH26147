import numpy as np
import pytest
from app.modulation.frequency import extract_frequency_features
from app.modulation.models import FeatureValidity
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_frequency_features_bfsk_bimodal():
    # BFSK with 2 frequency states
    samples, _ = generate_modulated_signal("BFSK", snr_db=90.0, seed=42)
    feat = extract_frequency_features(samples)

    assert feat.validity == FeatureValidity.VALID
    assert feat.bimodal_separation is not None
    # Nominal delta_f in generator is 0.25 (between -0.125 and +0.125)
    assert np.isclose(feat.bimodal_separation, 0.25, atol=0.03)
    assert feat.bimodal_prominence > 0.30
    assert feat.state_occupancy_ratio > 0.50

def test_frequency_features_single_carrier_psk():
    # QPSK has single carrier
    samples, _ = generate_modulated_signal("QPSK", snr_db=90.0, seed=42)
    feat = extract_frequency_features(samples)

    assert feat.validity == FeatureValidity.VALID
    # QPSK should not have a prominent bimodal separation
    assert feat.bimodal_prominence < 0.25 or feat.bimodal_separation is None
