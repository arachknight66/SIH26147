from __future__ import annotations
import hashlib
from typing import Any
from app.data_recovery.models import DataRecoveryAnalysis
from app.models.analysis import SignalAnalysis
from app.models.metadata import Diagnostic
from app.models.signal import SignalRecording
from app.modulation.models import ModulationAnalysis
from app.recovery.models import RecoveryAnalysis
from .bit_checks import audit_bitstream
from .falsification import audit_falsification
from .fec_checks import audit_fec_and_cross_validation
from .frame_checks import audit_framing_and_periodicity
from .integrity_checks import audit_integrity_and_null_model
from .models import (
    ClaimStatus,
    IndependenceLevel,
    VerificationAnalysis,
    VerificationClaim,
    VerificationConfig,
    VerificationHandoff,
    VerificationTest,
)
from .modulation_checks import audit_modulation_and_constellation
from .ranking import build_error_budget, determine_final_verification_status
from .robustness import audit_robustness_and_leave_one_out
from .scrambler_checks import audit_scrambler
from .signal_checks import audit_signal_and_physics
from .synchronization_checks import audit_synchronization_and_stability

def verify_result(
    phase5_result: DataRecoveryAnalysis,
    phase4_result: RecoveryAnalysis | None = None,
    phase3_result: ModulationAnalysis | None = None,
    phase2_result: SignalAnalysis | None = None,
    phase1_result: SignalRecording | None = None,
    config: VerificationConfig | None = None,
) -> VerificationAnalysis:
    """
    Execute full independent scientific verification, falsification, and uncertainty quantification.

    Parameters
    ----------
    phase5_result : DataRecoveryAnalysis
    phase4_result : RecoveryAnalysis | None
    phase3_result : ModulationAnalysis | None
    phase2_result : SignalAnalysis | None
    phase1_result : SignalRecording | None
    config : VerificationConfig | None

    Returns
    -------
    VerificationAnalysis
    """
    cfg = config or VerificationConfig()
    all_tests: list[VerificationTest] = []
    claims: list[VerificationClaim] = []
    all_diagnostics: list[Diagnostic] = []

    handoff_p5 = phase5_result.phase6_handoff
    sel_cand = phase5_result.selected_candidate

    # 1. Physical & Signal Checks
    phys_res, phys_tests = audit_signal_and_physics(recording=phase1_result, analysis=phase2_result)
    all_tests.extend(phys_tests)
    claims.append(
        VerificationClaim(
            claim_id=1,
            claim_text="The physical signal representation is finite, non-clipping, and consistent.",
            status=ClaimStatus.SUPPORTED if phys_res.measurement_consistent else ClaimStatus.CONTRADICTED,
            evidence_category="physical",
            tests=tuple(phys_tests),
            confidence=0.95 if phys_res.measurement_consistent else 0.20,
            independence_level=IndependenceLevel.INDEPENDENT,
        )
    )

    # 2. Modulation Checks
    mod_res, mod_tests = audit_modulation_and_constellation(recovery=phase4_result, mod_analysis=phase3_result)
    all_tests.extend(mod_tests)
    claims.append(
        VerificationClaim(
            claim_id=2,
            claim_text=f"The recovered modulation is {mod_res.modulation_name}.",
            status=ClaimStatus.SUPPORTED if mod_res.is_consistent else ClaimStatus.AMBIGUOUS if not mod_res.is_consistent else ClaimStatus.CONTRADICTED,
            evidence_category="modulation",
            tests=tuple(mod_tests),
            confidence=0.90 if mod_res.is_consistent else 0.40,
            independence_level=IndependenceLevel.INDEPENDENT,
        )
    )

    # 3. Synchronization & Stability Checks
    sync_res, sync_tests = audit_synchronization_and_stability(recovery=phase4_result, config=cfg)
    all_tests.extend(sync_tests)
    claims.append(
        VerificationClaim(
            claim_id=3,
            claim_text="Carrier and symbol synchronization is temporally stable across signal windows.",
            status=ClaimStatus.SUPPORTED if sync_res.is_stable else ClaimStatus.CONTRADICTED,
            evidence_category="synchronization",
            tests=tuple(sync_tests),
            confidence=sync_res.window_consistency_fraction,
            independence_level=IndependenceLevel.INDEPENDENT,
        )
    )

    # 4. Bitstream Checks
    bit_res, bit_tests = audit_bitstream(data_analysis=phase5_result, handoff=handoff_p5)
    all_tests.extend(bit_tests)

    # 5. Framing & Boundary Perturbation Checks
    frame_res, frame_tests = audit_framing_and_periodicity(data_analysis=phase5_result, handoff=handoff_p5, config=cfg)
    all_tests.extend(frame_tests)
    claims.append(
        VerificationClaim(
            claim_id=4,
            claim_text=f"The detected frame boundaries (length {frame_res.frame_length_bits} bits) are genuine and sharp.",
            status=ClaimStatus.SUPPORTED if frame_res.is_structurally_sound else ClaimStatus.WEAKLY_SUPPORTED,
            evidence_category="framing",
            tests=tuple(frame_tests),
            confidence=0.90 if frame_res.is_structurally_sound else 0.40,
            independence_level=IndependenceLevel.INDEPENDENT,
        )
    )

    # 6. FEC Checks
    fec_res, fec_tests = audit_fec_and_cross_validation(data_analysis=phase5_result, handoff=handoff_p5, config=cfg)
    all_tests.extend(fec_tests)
    if fec_res.code_name != "UNCODED":
        claims.append(
            VerificationClaim(
                claim_id=5,
                claim_text=f"Forward error correction ({fec_res.code_name}) improves the reconstruction without over-correction.",
                status=ClaimStatus.SUPPORTED if fec_res.is_beneficial else ClaimStatus.CONTRADICTED,
                evidence_category="fec",
                tests=tuple(fec_tests),
                confidence=0.90 if fec_res.is_beneficial else 0.20,
                independence_level=IndependenceLevel.INDEPENDENT,
            )
        )

    # 7. Integrity & Null Model Checks
    integ_res, integ_tests = audit_integrity_and_null_model(data_analysis=phase5_result, handoff=handoff_p5, config=cfg)
    all_tests.extend(integ_tests)
    if integ_res.crc_name != "none":
        claims.append(
            VerificationClaim(
                claim_id=6,
                claim_text=f"The CRC integrity hypothesis ({integ_res.crc_name}) is statistically significant on held-out frames.",
                status=ClaimStatus.STRONGLY_SUPPORTED if (integ_res.is_statistically_significant and integ_res.validation_valid_count > 0) else ClaimStatus.WEAKLY_SUPPORTED,
                evidence_category="integrity",
                tests=tuple(integ_tests),
                confidence=0.95 if (integ_res.is_statistically_significant and integ_res.validation_valid_count > 0) else 0.40,
                independence_level=IndependenceLevel.INDEPENDENT,
            )
        )

    # 8. Scrambler Checks
    scram_res, scram_tests = audit_scrambler(data_analysis=phase5_result, handoff=handoff_p5)
    all_tests.extend(scram_tests)

    # 9. Robustness & Leave-One-Out Checks
    rob_res, rob_tests = audit_robustness_and_leave_one_out(data_analysis=phase5_result, candidate=sel_cand)
    all_tests.extend(rob_tests)

    # 10. Falsification Engine
    fals_res = audit_falsification(all_tests, config=cfg)

    # 11. Error Budget
    snr_est = phys_res.estimated_snr_db if phys_res else 20.0
    err_budget = build_error_budget(sel_cand, snr_db=snr_est)

    # 12. Final Status & Quality Level Determination
    status, q_level, is_ver, is_fal, is_amb, rank_diags = determine_final_verification_status(
        falsification=fals_res,
        claims=claims,
        candidate=sel_cand,
        snr_db=snr_est,
    )
    all_diagnostics.extend(rank_diags)

    # 13. Assemble Verification Handoff
    payload_b = sel_cand.recovered_payload_bytes if sel_cand else b""
    hash_obj = hashlib.sha256(payload_b + status.value.encode("utf-8")).hexdigest()

    handoff = VerificationHandoff(
        is_verified=is_ver,
        verified_payload=payload_b,
        status=status,
        quality_level=q_level,
        claims_summary={str(c.claim_id): c.status.value for c in claims},
        error_budget=err_budget,
        assumptions=("Linear AWGN/fading channel", "Standard framing and polynomial bounds"),
        limitations=("Semantic payload correctness unverified", "Cryptographic authenticity unverified"),
        reproducibility_hash=hash_obj,
    )

    return VerificationAnalysis(
        status=status,
        quality_level=q_level,
        is_verified=is_ver,
        is_falsified=is_fal,
        is_ambiguous=is_amb,
        claims=claims,
        physical_audit=phys_res,
        modulation_audit=mod_res,
        sync_audit=sync_res,
        bitstream_audit=bit_res,
        frame_audit=frame_res,
        fec_audit=fec_res,
        integrity_audit=integ_res,
        scrambler_audit=scram_res,
        robustness_audit=rob_res,
        falsification_audit=fals_res,
        error_budget=err_budget,
        diagnostics=all_diagnostics,
        provenance={"total_tests": len(all_tests), "falsified_tests": fals_res.falsified_test_count},
        handoff=handoff,
    )
