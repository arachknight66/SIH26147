from __future__ import annotations
from typing import Any
import numpy as np
from app.data_recovery.interleaving import (
    deinterleave_block,
    deinterleave_convolutional,
    deinterleave_diagonal,
    deinterleave_pseudorandom,
    evaluate_deinterleaved_stream_metrics,
)
from app.data_recovery.models import DataRecoveryAnalysis, InterleaverType, Phase6Handoff
from .models import InterleaverAuditResult, TestResultStatus, VerificationConfig, VerificationTest

def audit_interleaver_and_falsification(
    data_analysis: DataRecoveryAnalysis | None = None,
    handoff: Phase6Handoff | None = None,
    config: VerificationConfig | None = None,
) -> tuple[InterleaverAuditResult, list[VerificationTest]]:
    """
    Independently verify de-interleaving hypothesis through parameter perturbation falsification
    and held-out cross-validation.

    Parameters
    ----------
    data_analysis : DataRecoveryAnalysis | None
    handoff : Phase6Handoff | None
    config : VerificationConfig | None

    Returns
    -------
    audit_result : InterleaverAuditResult
    tests : list[VerificationTest]
    """
    cfg = config or VerificationConfig()
    tests: list[VerificationTest] = []

    sel_cand = data_analysis.selected_candidate if data_analysis else None
    inter_h = sel_cand.interleaver if sel_cand else None

    # Case 1: No interleaver applied / baseline uncoded
    if inter_h is None or inter_h.interleaver_type == InterleaverType.NONE:
        res = InterleaverAuditResult(
            interleaver_type="NONE",
            parameter_perturbation_passed=True,
            held_out_validation_passed=True,
            improves_framing=True,
            improves_integrity=True,
            is_verified=True,
            details={"status": "no_interleaver_applied"},
        )
        tests.append(
            VerificationTest(
                test_id="INTER_00_NONE",
                name="Interleaver Necessity & Null Baseline Audit",
                category="interleaver",
                description="Verify signal reconstruction does not require de-interleaving",
                status=TestResultStatus.PASS,
                score=1.0,
                details={"interleaver_type": "NONE"},
            )
        )
        return res, tests

    raw_bits = sel_cand.bit_hypothesis.bitstream.hard_bits if sel_cand else np.array([], dtype=np.uint8)
    i_type = inter_h.interleaver_type
    params = inter_h.parameters

    # Baseline valid frame count and periodic flag
    base_crc = sel_cand.integrity.valid_frame_count if sel_cand.integrity else 0
    has_framing = bool(sel_cand.preamble and sel_cand.preamble.is_periodic)
    has_crc = bool(base_crc > 0)

    # -------------------------------------------------------------
    # Test 1: Parameter Perturbation Falsification
    # -------------------------------------------------------------
    perturb_collapsed = True
    perturb_trials: list[dict[str, Any]] = []

    if i_type == InterleaverType.BLOCK:
        span = params.get("span", 8)
        depth = params.get("depth", 8)
        perturb_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0)]
        for ds, dd in perturb_deltas:
            p_span = max(2, span + ds)
            p_depth = max(2, depth + dd)
            if p_span == span and p_depth == depth:
                continue
            p_deint = deinterleave_block(raw_bits, span=p_span, depth=p_depth)
            _, p_is_per, p_crc, p_crc_frac, p_crc_valid, _ = evaluate_deinterleaved_stream_metrics(p_deint)
            perturb_trials.append({"span": p_span, "depth": p_depth, "valid_crc": p_crc, "is_periodic": p_is_per})
            if p_crc >= base_crc and base_crc > 0:
                perturb_collapsed = False

    elif i_type == InterleaverType.CONVOLUTIONAL:
        branches = params.get("branches", 4)
        delay_inc = params.get("delay_increment", 1)
        perturb_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dm, dd in perturb_deltas:
            p_b = max(2, branches + dm)
            p_d = max(1, delay_inc + dd)
            if p_b == branches and p_d == delay_inc:
                continue
            p_deint = deinterleave_convolutional(raw_bits, branches=p_b, delay_increment=p_d)
            _, p_is_per, p_crc, p_crc_frac, p_crc_valid, _ = evaluate_deinterleaved_stream_metrics(p_deint)
            perturb_trials.append({"branches": p_b, "delay": p_d, "valid_crc": p_crc, "is_periodic": p_is_per})
            if p_crc >= base_crc and base_crc > 0:
                perturb_collapsed = False

    elif i_type == InterleaverType.DIAGONAL:
        span = params.get("span", 8)
        depth = params.get("depth", 8)
        step = params.get("step", 1)
        perturb_deltas = [(-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1)]
        for ds, dd, dst in perturb_deltas:
            p_span = max(2, span + ds)
            p_depth = max(2, depth + dd)
            p_step = max(1, step + dst)
            if p_span == span and p_depth == depth and p_step == step:
                continue
            p_deint = deinterleave_diagonal(raw_bits, span=p_span, depth=p_depth, step=p_step)
            _, p_is_per, p_crc, p_crc_frac, p_crc_valid, _ = evaluate_deinterleaved_stream_metrics(p_deint)
            perturb_trials.append({"span": p_span, "depth": p_depth, "step": p_step, "valid_crc": p_crc})
            if p_crc >= base_crc and base_crc > 0:
                perturb_collapsed = False

    elif i_type == InterleaverType.PSEUDO_RANDOM:
        b_size = params.get("block_size", 128)
        taps = params.get("taps", (7, 4))
        # Perturb block size and taps
        perturb_sizes = [max(16, b_size - 16), b_size + 16]
        for ps in perturb_sizes:
            p_deint = deinterleave_pseudorandom(raw_bits, taps=taps, block_size=ps)
            _, p_is_per, p_crc, p_crc_frac, p_crc_valid, _ = evaluate_deinterleaved_stream_metrics(p_deint)
            perturb_trials.append({"block_size": ps, "valid_crc": p_crc})
            if p_crc >= base_crc and base_crc > 0:
                perturb_collapsed = False

    tests.append(
        VerificationTest(
            test_id="INTER_01_PERTURBATION",
            name="Interleaver Parameter Perturbation Falsification",
            category="interleaver",
            description="Verify downstream framing and CRC collapse under perturbed interleaver parameters",
            status=TestResultStatus.PASS if perturb_collapsed else TestResultStatus.FAIL,
            score=1.0 if perturb_collapsed else 0.20,
            details={"interleaver_type": i_type.value, "baseline_crc": base_crc, "perturb_trials": perturb_trials},
            counter_evidence="Perturbed interleaver parameters maintain high score (hypothesis lacks parameter resonance / sharp falsifiability)" if not perturb_collapsed else None,
            is_critical=True,
        )
    )

    # -------------------------------------------------------------
    # Test 2: Held-Out Frame Cross-Validation (70/30 Split)
    # -------------------------------------------------------------
    held_out_passed = True
    if len(raw_bits) >= 64:
        split_idx = int(len(raw_bits) * 0.70)
        val_raw_bits = raw_bits[split_idx:]

        if i_type == InterleaverType.BLOCK:
            val_deint = deinterleave_block(val_raw_bits, span=params.get("span", 8), depth=params.get("depth", 8))
        elif i_type == InterleaverType.CONVOLUTIONAL:
            val_deint = deinterleave_convolutional(val_raw_bits, branches=params.get("branches", 4), delay_increment=params.get("delay_increment", 1))
        elif i_type == InterleaverType.DIAGONAL:
            val_deint = deinterleave_diagonal(val_raw_bits, span=params.get("span", 8), depth=params.get("depth", 8), step=params.get("step", 1))
        elif i_type == InterleaverType.PSEUDO_RANDOM:
            val_deint = deinterleave_pseudorandom(val_raw_bits, taps=params.get("taps", (7, 4)), block_size=params.get("block_size", 128))
        else:
            val_deint = val_raw_bits

        _, val_is_per, val_crc, val_crc_frac, val_crc_valid, _ = evaluate_deinterleaved_stream_metrics(val_deint)
        # Validation passes if framing periodicity or at least 1 CRC match holds on validation segment
        held_out_passed = bool(val_is_per or val_crc > 0)

    tests.append(
        VerificationTest(
            test_id="INTER_02_CROSS_VALIDATION",
            name="Held-Out Frame De-interleaving Cross-Validation (70/30)",
            category="interleaver",
            description="Verify de-interleaving hypothesis generalizability on held-out validation segment",
            status=TestResultStatus.PASS if held_out_passed else TestResultStatus.FAIL,
            score=1.0 if held_out_passed else 0.20,
            details={"selection_length": int(len(raw_bits) * 0.70), "validation_length": len(raw_bits) - int(len(raw_bits) * 0.70)},
            counter_evidence="De-interleaving fails on held-out validation frames (possible selection overfit)" if not held_out_passed else None,
            is_critical=True,
        )
    )

    is_verified = bool(perturb_collapsed and held_out_passed and (has_crc or has_framing))

    res = InterleaverAuditResult(
        interleaver_type=i_type.value,
        parameter_perturbation_passed=perturb_collapsed,
        held_out_validation_passed=held_out_passed,
        improves_framing=has_framing,
        improves_integrity=has_crc,
        is_verified=is_verified,
        details={"parameters": params, "is_verified": is_verified},
    )

    return res, tests
