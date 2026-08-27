from __future__ import annotations
import numpy as np
import pytest
from app.data_recovery.analyzer import recover_data
from app.verification.analyzer import verify_result
from app.verification.models import VerificationStatus
from scripts.generate_digital_dataset import generate_digital_stream
from tests.test_phase6_cases import _make_rec_sig

def test_verification_pipeline_protocol_a():
    rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    p6 = verify_result(phase5_result=p5, phase4_result=rec)
    assert p6.status == VerificationStatus.INDEPENDENTLY_VERIFIED
    assert p6.is_verified is True
    assert p6.handoff is not None
    assert len(p6.handoff.reproducibility_hash) == 64

def test_verification_pipeline_ood_random():
    rx_b, rx_s, _ = generate_digital_stream(protocol="OOD_RANDOM", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    p6 = verify_result(phase5_result=p5, phase4_result=rec)
    assert p6.is_verified is False
    assert p6.status in (VerificationStatus.INSUFFICIENT_EVIDENCE, VerificationStatus.AMBIGUOUS, VerificationStatus.FALSIFIED)
