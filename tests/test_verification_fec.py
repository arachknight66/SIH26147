from __future__ import annotations
import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.verification.fec_checks import audit_fec_and_cross_validation
from app.verification.models import TestResultStatus
from scripts.generate_digital_dataset import generate_digital_stream
from tests.test_phase6_cases import _make_rec_sig

def test_audit_fec_protocol_c():
    rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, ber=0.005, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    res, tests = audit_fec_and_cross_validation(p5)
    assert res.anti_overcorrection_passed is True
    assert res.held_out_validation_passed is True
    assert res.is_beneficial is True
    assert res.correction_fraction <= 0.10

def test_audit_fec_uncoded():
    rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    res, tests = audit_fec_and_cross_validation(p5)
    assert res.code_name == "UNCODED"
    assert res.is_beneficial is True
