from __future__ import annotations
import numpy as np
import pytest
from app.models.signal import SignalRecording, SourceFormat
from app.analysis.analyzer import analyze_signal
from app.verification.models import AuditResultStatus
from app.verification.signal_checks import audit_signal_and_physics

def test_audit_signal_finite_and_clean():
    samples = np.exp(1j * np.linspace(0, 100, 1000)).astype(np.complex64)
    rec = SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=1,
        semantic_type="iq",
    )
    analysis = analyze_signal(rec)
    res, tests = audit_signal_and_physics(rec, analysis)
    assert res.is_finite is True
    assert res.clipping_fraction < 0.05
    assert len(tests) >= 2
    assert all(t.status in (AuditResultStatus.PASS, AuditResultStatus.WEAK_PASS) for t in tests)

def test_audit_signal_non_finite():
    samples = np.array([1.0, np.nan, 2.0], dtype=np.complex64)
    rec = SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=1,
        semantic_type="iq",
    )
    res, tests = audit_signal_and_physics(rec)
    assert res.is_finite is False
    assert any(t.status == AuditResultStatus.FAIL for t in tests)
