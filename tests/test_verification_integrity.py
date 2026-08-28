from __future__ import annotations
import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.verification.integrity_checks import audit_integrity_and_null_model
from app.verification.models import TestResultStatus
from scripts.generate_digital_dataset import generate_digital_stream
from tests.test_phase6_cases import _make_rec_sig

def test_audit_integrity_protocol_a():
    rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    res, tests = audit_integrity_and_null_model(p5)
    assert res.crc_name.startswith("CRC")
    assert res.validation_valid_count >= 1
    assert res.multiple_testing_corrected_p_value < 0.01
    assert res.is_statistically_significant is True
    assert all(t.status == TestResultStatus.PASS for t in tests)
