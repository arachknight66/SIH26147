from __future__ import annotations
import pytest
from app.verification.models import (
    ClaimStatus,
    ErrorBudget,
    FalsificationAuditResult,
    FalsificationOutcome,
    FECAuditResult,
    FrameAuditResult,
    IndependenceLevel,
    IntegrityAuditResult,
    ModulationAuditResult,
    PhysicalAuditResult,
    RobustnessAuditResult,
    ScramblerAuditResult,
    SyncAuditResult,
    TestResultStatus,
    VerificationAnalysis,
    VerificationClaim,
    VerificationConfig,
    VerificationHandoff,
    VerificationQualityLevel,
    VerificationStatus,
    VerificationTest,
)

def test_verification_enums():
    assert VerificationStatus.INDEPENDENTLY_VERIFIED.value == "independently_verified"
    assert VerificationQualityLevel.HIGH.value == "HIGH"
    assert ClaimStatus.SUPPORTED.value == "supported"
    assert IndependenceLevel.INDEPENDENT.value == "independent"
    assert TestResultStatus.PASS.value == "PASS"
    assert FalsificationOutcome.NOT_FALSIFIED.value == "not_falsified"

def test_verification_dataclasses():
    test = VerificationTest(
        test_id="T01",
        name="Test 1",
        category="general",
        description="A test",
        status=TestResultStatus.PASS,
        score=0.95,
        p_value=0.001,
        is_critical=True,
    )
    assert test.test_id == "T01"
    assert test.is_critical is True

    claim = VerificationClaim(
        claim_id=1,
        claim_text="Modulation is QPSK",
        status=ClaimStatus.SUPPORTED,
        evidence_category="modulation",
        tests=(test,),
        confidence=0.95,
        independence_level=IndependenceLevel.INDEPENDENT,
    )
    assert claim.claim_id == 1
    assert len(claim.tests) == 1

def test_error_budget():
    budget = ErrorBudget(
        carrier_uncertainty=0.01,
        timing_uncertainty=0.005,
        bit_error_rate_proxy=0.002,
        fec_residual_uncertainty=0.0002,
        total_composite_uncertainty=0.0114,
        summary="Test error budget",
    )
    assert budget.total_composite_uncertainty == 0.0114
