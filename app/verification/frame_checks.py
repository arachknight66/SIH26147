from __future__ import annotations
import numpy as np
from app.data_recovery.framing import detect_sequence_continuity
from app.data_recovery.models import DataRecoveryAnalysis, Phase6Handoff
from .models import FrameAuditResult, TestResultStatus, VerificationConfig, VerificationTest
from .perturbation import evaluate_boundary_perturbations

def audit_framing_and_periodicity(
    data_analysis: DataRecoveryAnalysis | None = None,
    handoff: Phase6Handoff | None = None,
    config: VerificationConfig | None = None,
) -> tuple[FrameAuditResult, list[VerificationTest]]:
    """
    Independently verify frame boundary stability, interval variance, and sequence continuity.

    Parameters
    ----------
    data_analysis : DataRecoveryAnalysis | None
    handoff : Phase6Handoff | None
    config : VerificationConfig | None

    Returns
    -------
    audit_result : FrameAuditResult
    tests : list[VerificationTest]
    """
    cfg = config or VerificationConfig()
    tests: list[VerificationTest] = []

    sel_cand = data_analysis.selected_candidate if data_analysis else None
    if sel_cand is None or not sel_cand.frames:
        res = FrameAuditResult(
            preamble_name="none",
            frame_length_bits=0,
            total_frames=0,
            interval_mean=0.0,
            interval_std=0.0,
            interval_cv=1.0,
            sequence_is_continuous=False,
            missing_sequences=(),
            boundary_perturbation_passed=False,
            is_structurally_sound=False,
            details={"status": "no_frames_detected"},
        )
        tests.append(
            VerificationTest(
                test_id="FRAME_00_INPUT",
                name="Frame Structure Input Check",
                category="framing",
                description="Check availability of detected digital frames",
                status=TestResultStatus.FAIL,
                score=0.0,
                counter_evidence="No frame candidates available for independent structural verification",
                is_critical=True,
            )
        )
        return res, tests

    frames = list(sel_cand.frames)
    total_f = len(frames)
    preamble_hex = sel_cand.preamble.pattern_hex if sel_cand.preamble else "unknown"

    # Compute frame intervals
    starts = [f.start_bit for f in frames]
    if len(starts) >= 2:
        spacings = np.diff(starts)
        mean_sp = float(np.mean(spacings))
        std_sp = float(np.std(spacings))
        cv = float(std_sp / max(1e-6, mean_sp))
    else:
        mean_sp = float(frames[0].end_bit - frames[0].start_bit)
        std_sp = 0.0
        cv = 0.0

    # Interval CV verification test
    is_interval_pass = bool(cv <= cfg.max_allowable_interval_cv and total_f >= 2)
    tests.append(
        VerificationTest(
            test_id="FRAME_01_INTERVAL_STABILITY",
            name="Frame Interval Periodicity & Dispersion",
            category="framing",
            description=f"Verify interval coefficient of variation (CV <= {cfg.max_allowable_interval_cv * 100:.1f}%)",
            status=TestResultStatus.PASS if is_interval_pass else (TestResultStatus.WEAK_PASS if cv <= 0.10 else TestResultStatus.FAIL),
            score=float(np.clip(1.0 - (cv / 0.10), 0.0, 1.0)),
            details={"interval_mean": round(mean_sp, 2), "interval_std": round(std_sp, 2), "interval_cv": round(cv, 4), "num_frames": total_f},
            counter_evidence=f"Irregular frame spacing (CV = {cv:.3f}) suggests false or unlocked framing" if not is_interval_pass else None,
            is_critical=True,
        )
    )

    # Sequence Number Continuity
    is_seq_cont, seqs, missing = detect_sequence_continuity(frames)
    if len(seqs) >= 2:
        tests.append(
            VerificationTest(
                test_id="FRAME_02_SEQUENCE_CONTINUITY",
                name="Sequence Number Monotonicity",
                category="framing",
                description="Verify monotonic sequential counter progression",
                status=TestResultStatus.PASS if is_seq_cont else TestResultStatus.WEAK_PASS,
                score=1.0 if is_seq_cont else max(0.0, 1.0 - len(missing) * 0.20),
                details={"observed_sequences": seqs, "missing_sequences": missing},
                counter_evidence=f"Sequence numbers contain gaps: missing {missing}" if missing else None,
            )
        )

    # Boundary Perturbation Falsification Test
    perturb_pass, perturb_score, perturb_details = evaluate_boundary_perturbations(sel_cand)
    tests.append(
        VerificationTest(
            test_id="FRAME_03_BOUNDARY_PERTURBATION",
            name="Boundary Sharpness & Perturbation Falsification",
            category="framing",
            description="Verify that boundary shifts (+/-1, +/-2, +/-4 bits) collapse structural evidence",
            status=TestResultStatus.PASS if perturb_pass else TestResultStatus.FAIL,
            score=perturb_score,
            details=perturb_details,
            counter_evidence="Frame structure failed boundary perturbation test (alternative offsets yield comparable structure)" if not perturb_pass else None,
            is_critical=True,
        )
    )

    is_sound = bool(is_interval_pass and perturb_pass)

    res = FrameAuditResult(
        preamble_name=preamble_hex,
        frame_length_bits=int(mean_sp),
        total_frames=total_f,
        interval_mean=round(mean_sp, 2),
        interval_std=round(std_sp, 2),
        interval_cv=round(cv, 4),
        sequence_is_continuous=is_seq_cont,
        missing_sequences=tuple(missing),
        boundary_perturbation_passed=perturb_pass,
        is_structurally_sound=is_sound,
        details={"num_frames": total_f},
    )
    return res, tests
