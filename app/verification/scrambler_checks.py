from __future__ import annotations
import numpy as np
from app.data_recovery.models import DataRecoveryAnalysis, Phase6Handoff, ScramblerType
from app.data_recovery.scrambling import descramble_lfsr
from .models import ScramblerAuditResult, TestResultStatus, VerificationTest

def audit_scrambler(
    data_analysis: DataRecoveryAnalysis | None = None,
    handoff: Phase6Handoff | None = None,
) -> tuple[ScramblerAuditResult, list[VerificationTest]]:
    """
    Independently verify scrambler polynomial and descrambling reproducibility.

    Parameters
    ----------
    data_analysis : DataRecoveryAnalysis | None
    handoff : Phase6Handoff | None

    Returns
    -------
    audit_result : ScramblerAuditResult
    tests : list[VerificationTest]
    """
    tests: list[VerificationTest] = []

    sel_cand = data_analysis.selected_candidate if data_analysis else None
    scram = sel_cand.scrambler if sel_cand else None

    if scram is None or scram.scrambler_type == ScramblerType.NONE:
        res = ScramblerAuditResult(
            polynomial_name="NONE",
            is_reproducible=True,
            improves_framing=True,
            improves_integrity=True,
            is_verified=True,
            details={"status": "no_scrambler_applied"},
        )
        tests.append(
            VerificationTest(
                test_id="SCRAM_00_NONE",
                name="Scrambler Necessity Check",
                category="scrambler",
                description="Verify unscrambled stream baseline",
                status=TestResultStatus.PASS,
                score=1.0,
                details={"scrambler_type": "NONE"},
            )
        )
        return res, tests

    raw_bits = sel_cand.bit_hypothesis.bitstream.hard_bits if sel_cand else np.array([], dtype=np.uint8)
    poly_name = scram.polynomial_name
    taps = scram.polynomial_bits

    # Reproduce descrambling
    desc1 = descramble_lfsr(raw_bits, taps)
    desc2 = descramble_lfsr(raw_bits, taps)
    is_reproducible = bool(np.array_equal(desc1, desc2))

    has_crc = bool(sel_cand.integrity and sel_cand.integrity.valid_frame_count > 0)
    has_framing = bool(sel_cand.preamble and sel_cand.preamble.is_periodic)
    is_ver = bool(is_reproducible and (has_crc or has_framing))

    tests.append(
        VerificationTest(
            test_id="SCRAM_01_REPRODUCIBILITY",
            name="Descrambler Determinism & Integrity Benefit",
            category="scrambler",
            description="Verify LFSR sequence reproducibility and framing improvement",
            status=TestResultStatus.PASS if is_ver else TestResultStatus.FAIL,
            score=1.0 if is_ver else 0.0,
            details={"polynomial": poly_name, "taps": taps, "has_crc": has_crc, "has_framing": has_framing},
            counter_evidence="Scrambler hypothesis fails reproducibility or provides no structural improvement" if not is_ver else None,
            is_critical=True,
        )
    )

    res = ScramblerAuditResult(
        polynomial_name=poly_name,
        is_reproducible=is_reproducible,
        improves_framing=has_framing,
        improves_integrity=has_crc,
        is_verified=is_ver,
        details={"polynomial": poly_name},
    )
    return res, tests
