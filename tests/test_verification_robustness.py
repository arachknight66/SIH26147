from __future__ import annotations
import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.verification.models import TestResultStatus
from app.verification.robustness import audit_robustness_and_leave_one_out
from scripts.generate_digital_dataset import generate_digital_stream
from tests.test_phase6_cases import _make_rec_sig

def test_audit_robustness_protocol_a():
    rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    res, tests = audit_robustness_and_leave_one_out(p5)
    assert res.leave_one_out_stable is True
    assert res.high_leverage_frame_detected is False
    assert res.bit_flip_tolerance_score >= 0.80
    assert all(t.status == TestResultStatus.PASS for t in tests)
