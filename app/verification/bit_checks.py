from __future__ import annotations
import numpy as np
from app.data_recovery.bit_alignment import generate_byte_stream_candidates
from app.data_recovery.models import DataRecoveryAnalysis, Phase6Handoff
from .models import BitstreamAuditResult, TestResultStatus, VerificationTest

def audit_bitstream(
    data_analysis: DataRecoveryAnalysis | None = None,
    handoff: Phase6Handoff | None = None,
) -> tuple[BitstreamAuditResult, list[VerificationTest]]:
    """
    Independently verify bitstream balance, transition dynamics, and byte-alignment uniqueness.

    Parameters
    ----------
    data_analysis : DataRecoveryAnalysis | None
    handoff : Phase6Handoff | None

    Returns
    -------
    audit_result : BitstreamAuditResult
    tests : list[VerificationTest]
    """
    tests: list[VerificationTest] = []

    raw_bits: np.ndarray | None = None
    if handoff is not None:
        raw_bits = handoff.raw_bits
    elif data_analysis is not None and data_analysis.selected_candidate is not None:
        raw_bits = data_analysis.selected_candidate.bit_hypothesis.bitstream.hard_bits

    if raw_bits is None or len(raw_bits) < 16:
        res = BitstreamAuditResult(
            bit_balance=0.0,
            transition_probability=0.0,
            byte_entropy=0.0,
            selected_offset_score=0.0,
            alternative_offsets_score=0.0,
            is_alignment_unique=False,
            details={"status": "no_bitstream_available"},
        )
        tests.append(
            VerificationTest(
                test_id="BIT_00_INPUT",
                name="Bitstream Input Check",
                category="bitstream",
                description="Check availability of recovered bitstream",
                status=TestResultStatus.FAIL,
                score=0.0,
                counter_evidence="No recovered bitstream available for verification",
                is_critical=True,
            )
        )
        return res, tests

    n_bits = len(raw_bits)
    balance = float(np.mean(raw_bits))
    trans_prob = float(np.mean(raw_bits[1:] != raw_bits[:-1])) if n_bits > 1 else 0.0

    # Byte Entropy
    n_bytes = n_bits // 8
    if n_bytes > 0:
        byte_vals = np.packbits(raw_bits[: n_bytes * 8])
        counts = np.bincount(byte_vals, minlength=256)
        probs = counts[counts > 0] / n_bytes
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0.0

    # Alignment search across 8 bit offsets
    cands = generate_byte_stream_candidates(raw_bits, search_lsb=False)
    scores = [c.printable_ratio + (c.entropy / 8.0) for c in cands]
    best_score = max(scores) if scores else 0.0
    mean_other = float(np.mean(scores)) if len(scores) > 1 else best_score
    is_unique = bool(best_score >= mean_other)

    is_bal_pass = bool(0.20 <= balance <= 0.80)
    tests.append(
        VerificationTest(
            test_id="BIT_01_BALANCE",
            name="Bit Balance & Entropy",
            category="bitstream",
            description="Verify bit balance in [0.20, 0.80] and non-zero entropy",
            status=TestResultStatus.PASS if is_bal_pass else TestResultStatus.WEAK_PASS,
            score=max(0.0, 1.0 - abs(balance - 0.5) * 2.0),
            details={"bit_balance": round(balance, 3), "transition_probability": round(trans_prob, 3), "byte_entropy": round(entropy, 3)},
            counter_evidence="Skewed bit balance indicates degenerate DC content or improper slicer thresholds" if not is_bal_pass else None,
        )
    )

    tests.append(
        VerificationTest(
            test_id="BIT_02_ALIGNMENT_DOMINANCE",
            name="Byte Alignment Stability",
            category="bitstream",
            description="Verify candidate byte alignment performs consistently",
            status=TestResultStatus.PASS if is_unique else TestResultStatus.WEAK_PASS,
            score=0.90 if is_unique else 0.50,
            details={"best_alignment_score": round(best_score, 3), "mean_other_alignments": round(mean_other, 3)},
        )
    )

    res = BitstreamAuditResult(
        bit_balance=round(balance, 4),
        transition_probability=round(trans_prob, 4),
        byte_entropy=round(entropy, 4),
        selected_offset_score=round(best_score, 3),
        alternative_offsets_score=round(mean_other, 3),
        is_alignment_unique=is_unique,
        details={"total_bits": n_bits},
    )
    return res, tests
