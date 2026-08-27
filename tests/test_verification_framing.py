from __future__ import annotations
import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.data_recovery.models import DataQualityLevel, DataRecoveryAnalysis, DataRecoveryStatus
from app.verification.frame_checks import audit_framing_and_periodicity
from app.verification.models import AuditResultStatus
from scripts.generate_digital_dataset import generate_digital_stream
from tests.test_phase6_cases import _make_rec_sig

def test_audit_framing_protocol_a():
    rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    res, tests = audit_framing_and_periodicity(p5)
    assert res.total_frames >= 4
    assert res.interval_cv < 0.05
    assert res.is_structurally_sound is True
    assert res.boundary_perturbation_passed is True

def test_audit_framing_no_frames():
    analysis = DataRecoveryAnalysis(
        recording_reference="test",
        bitstream_candidates=[],
        reconstruction_candidates=[],
        selected_candidate=None,
        status=DataRecoveryStatus.INSUFFICIENT_DATA,
        quality_level=DataQualityLevel.VERY_LOW,
        is_recovered=False,
        is_inconclusive=True,
        is_ambiguous=False,
    )
    res, tests = audit_framing_and_periodicity(analysis)
    assert res.total_frames == 0
    assert res.is_structurally_sound is False
