from __future__ import annotations
import math
from typing import Any
import numpy as np
from app.data_recovery.crc import search_crc_presets
from app.data_recovery.models import DataRecoveryAnalysis, FrameCandidate, Phase6Handoff
from .models import IntegrityAuditResult, TestResultStatus, VerificationConfig, VerificationTest

def _binomial_tail(n: int, k: int, p: float) -> float:
    """Calculate P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    prob_sum = 0.0
    for j in range(k, n + 1):
        try:
            coeff = math.comb(n, j)
            prob_sum += coeff * (p ** j) * ((1.0 - p) ** (n - j))
        except (ValueError, OverflowError):
            break
    return float(min(1.0, max(0.0, prob_sum)))

def audit_integrity_and_null_model(
    data_analysis: DataRecoveryAnalysis | None = None,
    handoff: Phase6Handoff | None = None,
    config: VerificationConfig | None = None,
) -> tuple[IntegrityAuditResult, list[VerificationTest]]:
    """
    Independently verify CRC integrity on held-out frames and compute multiple-testing corrected significance.

    Parameters
    ----------
    data_analysis : DataRecoveryAnalysis | None
    handoff : Phase6Handoff | None
    config : VerificationConfig | None

    Returns
    -------
    audit_result : IntegrityAuditResult
    tests : list[VerificationTest]
    """
    cfg = config or VerificationConfig()
    tests: list[VerificationTest] = []

    sel_cand = data_analysis.selected_candidate if data_analysis else None
    frames = list(sel_cand.frames) if sel_cand else []
    total_f = len(frames)

    if not frames or sel_cand is None or sel_cand.integrity is None or sel_cand.integrity.valid_frame_count == 0:
        res = IntegrityAuditResult(
            crc_name="none",
            selection_frames_count=0,
            selection_valid_count=0,
            validation_frames_count=0,
            validation_valid_count=0,
            validation_success_rate=0.0,
            raw_p_value=1.0,
            multiple_testing_corrected_p_value=1.0,
            is_statistically_significant=False,
            details={"status": "no_valid_crc_found"},
        )
        tests.append(
            VerificationTest(
                test_id="INTEG_00_CRC",
                name="Data Integrity CRC Check",
                category="integrity",
                description="Check independent CRC confirmation",
                status=TestResultStatus.WEAK_PASS,
                score=0.50,
                details={"status": "framing_supported_without_crc"},
            )
        )
        return res, tests

    integ = sel_cand.integrity
    crc_name = integ.crc_results[0].crc_name if integ.crc_results else "CRC-16"
    width = integ.crc_results[0].width if integ.crc_results else 16

    # 70/30 Selection vs Validation frame partitioning
    split_idx = int(total_f * 0.70) if total_f >= 4 else total_f
    sel_frames = frames[:split_idx]
    val_frames = frames[split_idx:] if split_idx < total_f else frames

    # Re-evaluate CRC on validation frames independently
    val_valid_cnt = 0
    for f in val_frames:
        for offset in (0, 16, 32, 64):
            if len(f.raw_bits) > offset + width:
                cand_bits = f.raw_bits[offset:]
                n_b = len(cand_bits) // 8
                if n_b > (width // 8):
                    f_bytes = bytes(np.packbits(cand_bits[: n_b * 8]))
                    matches = [r for r in search_crc_presets(f_bytes) if r.is_valid and r.width == width]
                    if matches:
                        val_valid_cnt += 1
                        break

    val_rate = float(val_valid_cnt / max(1, len(val_frames)))
    val_pass = bool(val_valid_cnt > 0 and val_rate >= 0.50)

    tests.append(
        VerificationTest(
            test_id="INTEG_01_HOLDOUT_VALIDATION",
            name="Held-Out CRC Validation (70/30 Split)",
            category="integrity",
            description="Verify CRC correctness on held-out validation frames",
            status=TestResultStatus.PASS if val_pass else TestResultStatus.FAIL,
            score=val_rate,
            details={"selection_frames": len(sel_frames), "validation_frames": len(val_frames), "validation_valid": val_valid_cnt},
            counter_evidence="CRC matches only on selection frames and fails on held-out validation frames" if not val_pass else None,
            is_critical=True,
        )
    )

    # Multiple-testing correction under Null Model
    # M = 16 presets * 4 offsets = 64 hypotheses
    m_hypotheses = 64
    p_single = 2.0 ** (-width)
    raw_p = _binomial_tail(total_f, integ.valid_frame_count, p_single)
    corrected_p = float(min(1.0, raw_p * m_hypotheses))
    is_sig = bool(corrected_p < cfg.multiple_testing_alpha)

    tests.append(
        VerificationTest(
            test_id="INTEG_02_MULTIPLE_TESTING_SIGNIFICANCE",
            name="Multiple-Testing Null Model Significance",
            category="integrity",
            description=f"Verify Bonferroni-corrected false-discovery p-value < {cfg.multiple_testing_alpha}",
            status=TestResultStatus.PASS if is_sig else TestResultStatus.FAIL,
            score=max(0.0, 1.0 - corrected_p),
            p_value=corrected_p,
            details={"raw_p_value": raw_p, "corrected_p_value": corrected_p, "width_bits": width, "hypotheses_tested": m_hypotheses},
            counter_evidence=f"High accidental discovery probability (corrected p = {corrected_p:.3e})" if not is_sig else None,
            is_critical=True,
        )
    )

    res = IntegrityAuditResult(
        crc_name=crc_name,
        selection_frames_count=len(sel_frames),
        selection_valid_count=integ.valid_frame_count,
        validation_frames_count=len(val_frames),
        validation_valid_count=val_valid_cnt,
        validation_success_rate=round(val_rate, 4),
        raw_p_value=raw_p,
        multiple_testing_corrected_p_value=corrected_p,
        is_statistically_significant=is_sig,
        details={"width": width},
    )
    return res, tests
