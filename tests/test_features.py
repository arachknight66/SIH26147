import numpy as np
import pytest
from signal_analysis.features import extract_cumulant_features
from tests.test_synthesis import generate_synthetic_signal

def test_cumulant_bpsk():
    sig = generate_synthetic_signal("BPSK", n_symbols=4000, sps=1, snr_db=30)
    cf = extract_cumulant_features(sig)
    assert abs(cf.f20 - 1.0) < 0.15
    assert abs(cf.f40 - 2.0) < 0.25

def test_cumulant_qpsk():
    sig = generate_synthetic_signal("QPSK", n_symbols=4000, sps=1, snr_db=30)
    cf = extract_cumulant_features(sig)
    assert abs(cf.f20 - 0.0) < 0.15
    assert abs(cf.f40 - 1.0) < 0.15

def test_cumulant_8psk():
    sig = generate_synthetic_signal("8PSK", n_symbols=4000, sps=1, snr_db=30)
    cf = extract_cumulant_features(sig)
    assert abs(cf.f20 - 0.0) < 0.15
    assert abs(cf.f40 - 0.0) < 0.15

def test_cumulant_16qam():
    sig = generate_synthetic_signal("16-QAM", n_symbols=4000, sps=1, snr_db=30)
    cf = extract_cumulant_features(sig)
    assert abs(cf.f20 - 0.0) < 0.15
    assert abs(cf.f40 - 0.68) < 0.1
