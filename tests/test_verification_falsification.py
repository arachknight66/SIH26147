from __future__ import annotations
import pytest
from app.verification.falsification import audit_falsification
from app.verification.models import FalsificationOutcome, TestResultStatus, VerificationTest

def test_falsification_all_pass():
    tests = [
        VerificationTest("T1", "Finite", "phys", "Desc", TestResultStatus.PASS, 1.0, is_critical=True),
        VerificationTest("T2", "EVM", "mod", "Desc", TestResultStatus.PASS, 0.95, is_critical=True),
    ]
    res = audit_falsification(tests)
    assert res.outcome == FalsificationOutcome.NOT_FALSIFIED
    assert res.critical_failure_count == 0
    assert len(res.major_contradictions) == 0

def test_falsification_critical_fail():
    tests = [
        VerificationTest("T1", "Finite", "phys", "Desc", TestResultStatus.PASS, 1.0, is_critical=True),
        VerificationTest("T2", "EVM", "mod", "Desc", TestResultStatus.FAIL, 0.0, counter_evidence="EVM is 45%", is_critical=True),
    ]
    res = audit_falsification(tests)
    assert res.outcome == FalsificationOutcome.FALSIFIED
    assert res.critical_failure_count == 1
    assert len(res.major_contradictions) == 1
