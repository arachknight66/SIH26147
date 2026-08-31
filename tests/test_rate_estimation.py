import numpy as np
from signal_analysis.models import SignalRecording, SourceFormat, MetadataValue, MetadataStatus
from signal_analysis.rate_estimation import estimate_symbol_rate_consensus
from tests.test_synthesis import generate_synthetic_signal

def _create_rec(sig, sr=None):
    return SignalRecording(
        samples=sig,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(sr, "t", MetadataStatus.KNOWN if sr else MetadataStatus.MISSING),
        center_frequency_hz=MetadataValue(None, "t", MetadataStatus.MISSING),
        provenance={}, diagnostics=[]
    )

def test_rate_estimation_bpsk():
    sig = generate_synthetic_signal("BPSK", n_symbols=2000, sps=4, snr_db=20, pulse_shape='rrc')
    rec = _create_rec(sig)
    res = estimate_symbol_rate_consensus(rec)
    assert res is not None
    rate, unit, status, conf = res
    assert unit == "symbols/sample"
    assert status == "ESTIMATED"
    assert abs(rate - 0.25) < 0.01

def test_rate_estimation_cfo_invariance():
    sig = generate_synthetic_signal("QPSK", n_symbols=2000, sps=8, snr_db=20, cfo_norm=0.15, pulse_shape='rrc')
    rec = _create_rec(sig)
    res = estimate_symbol_rate_consensus(rec)
    assert res is not None
    rate, unit, status, conf = res
    assert abs(rate - 0.125) < 0.01
    
def test_rate_estimation_known_fs():
    sig = generate_synthetic_signal("BPSK", n_symbols=2000, sps=4, snr_db=20, pulse_shape='rrc')
    rec = _create_rec(sig, sr=1e6)
    res = estimate_symbol_rate_consensus(rec)
    assert res is not None
    rate, unit, status, conf = res
    assert unit == "Hz"
    assert abs(rate - 250000.0) < 10000.0

def test_harmonic_aliasing():
    # If sps=4, fundamental rate=0.25
    # Let's create an artificial list of candidates inside the estimation function
    # Wait, the logic is inside estimate_symbol_rate_consensus.
    # It will automatically detect 0.25 and possibly 0.5 (harmonic).
    sig = generate_synthetic_signal("BPSK", n_symbols=2000, sps=4, snr_db=30, pulse_shape='rrc')
    rec = _create_rec(sig)
    res = estimate_symbol_rate_consensus(rec)
    assert res is not None
    rate, unit, status, conf = res
    assert abs(rate - 0.25) < 0.02

