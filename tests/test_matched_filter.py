import numpy as np
import pytest
from app.recovery.matched_filter import apply_matched_filter, design_rrc_filter, validate_rrc_properties

def test_rrc_design_properties():
    h = design_rrc_filter(sps=8.0, alpha=0.35, span_symbols=8)
    assert len(h) == 8 * 8 + 1
    # Check energy normalization
    assert np.isclose(np.sum(h ** 2), 1.0, atol=1e-10)
    # Check symmetry
    assert np.allclose(h, h[::-1], atol=1e-12)

def test_rrc_validation_helper():
    sps = 8
    h = design_rrc_filter(sps=float(sps), alpha=0.35, span_symbols=8)
    props = validate_rrc_properties(h, sps=sps)
    assert props["is_symmetric"] is True
    assert props["is_normalized"] is True
    assert props["group_delay_samples"] == (len(h) - 1) // 2
    assert props["max_nyquist_isi"] < 0.05

def test_apply_matched_filter():
    sig = np.random.randn(256) + 1j * np.random.randn(256)
    filtered, taps = apply_matched_filter(sig, sps=8.0, alpha=0.35)
    assert len(filtered) == len(sig)
    assert len(taps) == 8 * 8 + 1
    assert filtered.dtype == np.complex64
