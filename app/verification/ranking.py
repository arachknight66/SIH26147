from __future__ import annotations
from typing import Sequence
import numpy as np
from app.data_recovery.models import DataRecoveryStatus, ReconstructionCandidate
from app.models.metadata import Diagnostic, DiagnosticSeverity
from .models import (
    ErrorBudget,
    FalsificationAuditResult,
    FalsificationOutcome,
    VerificationClaim,
    VerificationQualityLevel,
    VerificationStatus,
)

def build_error_budget(
    candidate: ReconstructionCandidate | None,
    snr_db: float = 20.0,
) -> ErrorBudget:
    """Construct itemized scientific uncertainty budget."""
    carrier_unc = float(np.clip(1.0 / max(1.0, snr_db), 0.001, 0.05))
    timing_unc = float(np.clip(0.5 / max(1.0, snr_db), 0.001, 0.03))
    
    corr_frac = candidate.fec_decode.correction_fraction if (candidate and candidate.fec_decode) else 0.0
    ber_proxy = float(max(0.001, corr_frac))
    fec_residual = float(corr_frac * 0.10)

    total_unc = float(np.sqrt(carrier_unc**2 + timing_unc**2 + ber_proxy**2 + fec_residual**2))
    summary = f"Carrier: {carrier_unc*100:.2f}%, Timing: {timing_unc*100:.2f}%, BER Proxy: {ber_proxy*100:.2f}%, Total: {total_unc*100:.2f}%"

    return ErrorBudget(
        carrier_uncertainty=round(carrier_unc, 4),
        timing_uncertainty=round(timing_unc, 4),
        bit_error_rate_proxy=round(ber_proxy, 4),
        fec_residual_uncertainty=round(fec_residual, 4),
        total_composite_uncertainty=round(total_unc, 4),
        summary=summary,
    )

def determine_final_verification_status(
    falsification: FalsificationAuditResult,
    claims: Sequence[VerificationClaim],
    candidate: ReconstructionCandidate | None,
    snr_db: float = 20.0,
) -> tuple[VerificationStatus, VerificationQualityLevel, bool, bool, bool, list[Diagnostic]]:
    """
    Apply conservative epistemic decision rules to determine the final Phase 6 status.

    Returns
    -------
    status : VerificationStatus
    quality_level : VerificationQualityLevel
    is_verified : bool
    is_falsified : bool
    is_ambiguous : bool
    diagnostics : list[Diagnostic]
    """
    diags: list[Diagnostic] = []

    if falsification.outcome == FalsificationOutcome.FALSIFIED:
        for contra in falsification.major_contradictions:
            diags.append(Diagnostic(code="VERIFICATION_FALSIFIED", message=contra, severity=DiagnosticSeverity.ERROR))
        return VerificationStatus.FALSIFIED, VerificationQualityLevel.VERY_LOW, False, True, False, diags

    if candidate is None or not candidate.frames:
        diags.append(
            Diagnostic(
                code="INSUFFICIENT_RECOVERY_EVIDENCE",
                message="No valid candidate reconstruction survived Phase 5 for verification.",
                severity=DiagnosticSeverity.INFO,
            )
        )
        return VerificationStatus.INSUFFICIENT_EVIDENCE, VerificationQualityLevel.VERY_LOW, False, False, False, diags

    has_crc = bool(candidate.integrity and candidate.integrity.valid_frame_count > 0 and candidate.integrity.crc_valid_fraction >= 0.50)
    has_framing = bool(candidate.preamble and candidate.preamble.is_periodic and len(candidate.frames) >= 2)
    has_fec = bool(candidate.fec_decode and candidate.fec_decode.valid)
    no_crit = bool(falsification.critical_failure_count == 0)

    # 1. Independent Verification Rule
    if no_crit and has_crc and has_framing and falsification.falsified_test_count == 0:
        status = VerificationStatus.INDEPENDENTLY_VERIFIED
        q_level = VerificationQualityLevel.HIGH if snr_db >= 10.0 else VerificationQualityLevel.MEDIUM
        is_ver = True
        is_fal = False
        is_amb = False

    # 2. Strong Structural Support Rule
    elif no_crit and (has_crc or has_framing) and falsification.falsified_test_count <= 2:
        status = VerificationStatus.STRONGLY_SUPPORTED
        q_level = VerificationQualityLevel.MEDIUM
        is_ver = False
        is_fal = False
        is_amb = False

    # 3. Partial Verification Rule
    elif has_framing:
        status = VerificationStatus.PARTIALLY_VERIFIED
        q_level = VerificationQualityLevel.LOW
        is_ver = False
        is_fal = False
        is_amb = False

    # 4. Inconclusive / Insufficient Evidence
    else:
        status = VerificationStatus.INSUFFICIENT_EVIDENCE
        q_level = VerificationQualityLevel.VERY_LOW
        is_ver = False
        is_fal = False
        is_amb = False

    return status, q_level, is_ver, is_fal, is_amb, diags
