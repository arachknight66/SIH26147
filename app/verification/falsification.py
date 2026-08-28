from __future__ import annotations
from typing import Sequence
from .models import FalsificationAuditResult, FalsificationOutcome, TestResultStatus, VerificationConfig, VerificationTest

def audit_falsification(
    tests: Sequence[VerificationTest],
    config: VerificationConfig | None = None,
) -> FalsificationAuditResult:
    """
    Execute falsification engine: search for critical contradictions and hypothesis failures.

    Parameters
    ----------
    tests : Sequence[VerificationTest]
    config : VerificationConfig | None

    Returns
    -------
    FalsificationAuditResult
    """
    total = len(tests)
    failed = [t for t in tests if t.status == TestResultStatus.FAIL]
    critical_fails = [t for t in failed if t.is_critical]

    contradictions: list[str] = []
    for t in failed:
        if t.counter_evidence:
            contradictions.append(f"[{t.test_id}] {t.counter_evidence}")
        else:
            contradictions.append(f"[{t.test_id}] {t.name} failed verification threshold.")

    if critical_fails:
        outcome = FalsificationOutcome.FALSIFIED
    elif failed:
        outcome = FalsificationOutcome.PARTIALLY_FALSIFIED
    elif total == 0:
        outcome = FalsificationOutcome.INCONCLUSIVE
    else:
        outcome = FalsificationOutcome.NOT_FALSIFIED

    return FalsificationAuditResult(
        total_falsification_tests=total,
        falsified_test_count=len(failed),
        critical_failure_count=len(critical_fails),
        major_contradictions=tuple(contradictions),
        outcome=outcome,
    )
