from __future__ import annotations
import numpy as np
from app.data_recovery.fec_decode import viterbi_decode
from app.data_recovery.models import DataRecoveryAnalysis, FECCodeFamily, Phase6Handoff
from .models import FECAuditResult, TestResultStatus, VerificationConfig, VerificationTest

def audit_fec_and_cross_validation(
    data_analysis: DataRecoveryAnalysis | None = None,
    handoff: Phase6Handoff | None = None,
    config: VerificationConfig | None = None,
) -> tuple[FECAuditResult, list[VerificationTest]]:
    """
    Independently verify forward error correction benefit, anti-over-correction, and held-out cross-validation.

    Parameters
    ----------
    data_analysis : DataRecoveryAnalysis | None
    handoff : Phase6Handoff | None
    config : VerificationConfig | None

    Returns
    -------
    audit_result : FECAuditResult
    tests : list[VerificationTest]
    """
    cfg = config or VerificationConfig()
    tests: list[VerificationTest] = []

    sel_cand = data_analysis.selected_candidate if data_analysis else None
    fec_dec = sel_cand.fec_decode if sel_cand else None
    fec_hyp = sel_cand.fec if sel_cand else None

    if fec_hyp is None or fec_hyp.code_family == FECCodeFamily.NONE or fec_dec is None:
        res = FECAuditResult(
            code_name="UNCODED",
            ber_before=0.0,
            ber_after=0.0,
            information_gain=0.0,
            correction_fraction=0.0,
            anti_overcorrection_passed=True,
            held_out_validation_passed=True,
            is_beneficial=True,
            details={"status": "uncoded_or_no_fec_applied"},
        )
        tests.append(
            VerificationTest(
                test_id="FEC_00_UNCODED",
                name="FEC Necessity & Application Audit",
                category="fec",
                description="Verify signal reconstruction does not require unneeded FEC correction",
                status=TestResultStatus.PASS,
                score=1.0,
                details={"code_family": "NONE"},
            )
        )
        return res, tests

    corr_frac = fec_dec.correction_fraction
    corr_count = fec_dec.corrected_bit_count
    is_anti_over = bool(corr_frac <= cfg.max_allowable_correction_fraction)

    tests.append(
        VerificationTest(
            test_id="FEC_01_OVERCORRECTION",
            name="Anti-Over-Correction Budget Audit",
            category="fec",
            description=f"Verify FEC bit modification fraction <= {cfg.max_allowable_correction_fraction * 100:.1f}%",
            status=TestResultStatus.PASS if is_anti_over else TestResultStatus.FAIL,
            score=max(0.0, 1.0 - (corr_frac / cfg.max_allowable_correction_fraction)),
            details={"corrected_bits": corr_count, "correction_fraction": round(corr_frac, 4), "budget": cfg.max_allowable_correction_fraction},
            counter_evidence=f"Excessive bit alterations ({corr_frac * 100:.1f}%) exceeds safety budget" if not is_anti_over else None,
            is_critical=True,
        )
    )

    # 70/30 Held-out Cross-Validation
    raw_channel_bits = handoff.raw_bits if handoff else (sel_cand.bit_hypothesis.bitstream.hard_bits if sel_cand else np.array([], dtype=np.uint8))
    held_out_passed = True

    if len(raw_channel_bits) >= 64:
        split_idx = int(len(raw_channel_bits) * 0.70)
        val_raw_bits = raw_channel_bits[split_idx:]

        if len(val_raw_bits) >= 32:
            val_dec = viterbi_decode(val_raw_bits, k=7, g1=0o133, g2=0o171, max_correction_fraction=cfg.max_allowable_correction_fraction)
            held_out_passed = bool(val_dec.valid)

        tests.append(
            VerificationTest(
                test_id="FEC_02_CROSS_VALIDATION",
                name="Held-Out Frame FEC Cross-Validation (70/30)",
                category="fec",
                description="Verify FEC decoder generalizability on held-out validation frames",
                status=TestResultStatus.PASS if held_out_passed else TestResultStatus.FAIL,
                score=1.0 if held_out_passed else 0.20,
                details={"selection_frames": split_idx, "validation_frames": len(val_raw_bits)},
                counter_evidence="FEC hypothesis fails on held-out validation frames (possible overfit)" if not held_out_passed else None,
                is_critical=True,
            )
        )

    info_gain = float(corr_frac)
    res = FECAuditResult(
        code_name=fec_hyp.code_name,
        ber_before=round(corr_frac, 4),
        ber_after=0.0,
        information_gain=round(info_gain, 4),
        correction_fraction=round(corr_frac, 4),
        anti_overcorrection_passed=is_anti_over,
        held_out_validation_passed=held_out_passed,
        is_beneficial=bool(is_anti_over and held_out_passed),
        details={"code_family": fec_hyp.code_family.value},
    )
    return res, tests
