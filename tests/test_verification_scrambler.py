from __future__ import annotations
import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.verification.models import TestResultStatus
from app.verification.scrambler_checks import audit_scrambler
from scripts.generate_digital_dataset import generate_digital_stream
from tests.test_phase6_cases import _make_rec_sig

def test_audit_scrambler_protocol_d():
    rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_D", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    res, tests = audit_scrambler(p5)
    assert res.is_reproducible is True
    assert res.is_verified is True
    assert all(t.status == TestResultStatus.PASS for t in tests)
