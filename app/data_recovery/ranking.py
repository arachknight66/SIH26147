from __future__ import annotations
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity
from .models import (
    DataQualityLevel,
    DataRecoveryStatus,
    EpistemicStatus,
    Phase6Handoff,
    ReconstructionCandidate,
)

def rank_and_select_reconstructions(
    candidates: list[ReconstructionCandidate],
) -> tuple[
    list[ReconstructionCandidate],
    ReconstructionCandidate | None,
    DataRecoveryStatus,
    DataQualityLevel,
    Phase6Handoff | None,
    list[Diagnostic],
]:
    """
    Empirically rank candidate reconstructions, determine overall status, and assemble Phase 6 handoff.

    Parameters
    ----------
    candidates : list[ReconstructionCandidate]

    Returns
    -------
    ranked_candidates : list[ReconstructionCandidate]
    selected_candidate : ReconstructionCandidate | None
    status : DataRecoveryStatus
    quality_level : DataQualityLevel
    phase6_handoff : Phase6Handoff | None
    diagnostics : list[Diagnostic]
    """
    diagnostics: list[Diagnostic] = []
    if not candidates:
        return [], None, DataRecoveryStatus.INSUFFICIENT_DATA, DataQualityLevel.VERY_LOW, None, [
            Diagnostic(
                code="NO_RECONSTRUCTION_CANDIDATES",
                message="No reconstruction candidates were evaluated.",
                severity=DiagnosticSeverity.ERROR,
            )
        ]

    # Deterministic multi-tier sort key:
    # 1. Composite score
    # 2. CRC valid count
    # 3. Preamble periodic
    # 4. -Complexity penalty
    # 5. -Candidate ID (tie breaking)
    def rank_key(cand: ReconstructionCandidate) -> tuple[float, int, bool, float, int]:
        crc_count = cand.integrity.valid_frame_count if cand.integrity else 0
        preamble_per = cand.preamble.is_periodic if cand.preamble else False
        return (
            cand.composite_score,
            crc_count,
            preamble_per,
            -cand.complexity_penalty,
            -cand.candidate_id,
        )

    ranked = sorted(candidates, key=rank_key, reverse=True)
    winner = ranked[0]

    # Determine Overall Phase 5 Status
    has_crc = bool(winner.integrity and winner.integrity.valid_frame_count > 0 and winner.integrity.crc_valid_fraction >= 0.50)
    has_fec = bool(winner.fec_decode and winner.fec_decode.valid and winner.fec_decode.corrected_bit_count > 0)
    has_framing = bool(
        (winner.scrambler is None or has_crc)
        and winner.preamble
        and winner.preamble.is_periodic
        and winner.preamble.confidence >= 0.80
        and len(winner.frames) >= 3
        and (winner.preamble.length_bits >= 16 or winner.preamble.match_count >= 4)
        and winner.composite_score >= 0.50
    )

    if has_crc and has_fec:
        status = DataRecoveryStatus.CORRECTED
        q_level = winner.data_quality_level
    elif has_crc:
        status = DataRecoveryStatus.INTEGRITY_SUPPORTED
        q_level = winner.data_quality_level
    elif has_framing:
        status = DataRecoveryStatus.STRUCTURALLY_SUPPORTED
        q_level = winner.data_quality_level
    elif winner.composite_score < 0.35:
        status = DataRecoveryStatus.INSUFFICIENT_DATA
        q_level = DataQualityLevel.VERY_LOW
        diagnostics.append(
            Diagnostic(
                code="INSUFFICIENT_STRUCTURE",
                message="No repeatable digital framing or integrity structure detected above statistical noise.",
                severity=DiagnosticSeverity.INFO,
            )
        )
    else:
        status = DataRecoveryStatus.AMBIGUOUS
        q_level = DataQualityLevel.LOW
        diagnostics.append(
            Diagnostic(
                code="STRUCTURE_AMBIGUOUS",
                message="Multiple candidate digital reconstructions exist with comparable weak evidence.",
                severity=DiagnosticSeverity.WARNING,
            )
        )

    # Acceptance threshold for selected candidate
    selected = winner if status in (
        DataRecoveryStatus.CORRECTED,
        DataRecoveryStatus.INTEGRITY_SUPPORTED,
        DataRecoveryStatus.STRUCTURALLY_SUPPORTED,
    ) else None

    # Assemble Phase 6 Handoff
    handoff: Phase6Handoff | None = None
    if selected is not None:
        raw_b = selected.bit_hypothesis.bitstream.hard_bits
        corr_b = selected.fec_decode.decoded_bits if selected.fec_decode else raw_b
        
        # Frame boundaries tuple
        boundaries = tuple(
            f.start_bit for f in selected.frames
        )

        corr_masks = (selected.fec_decode.correction_mask,) if selected.fec_decode else ()

        handoff = Phase6Handoff(
            raw_bits=raw_b,
            corrected_bits=corr_b,
            payload_bytes=selected.recovered_payload_bytes,
            frame_boundaries=(),
            fec_parameters={"code_name": selected.fec.code_name if selected.fec else "none"},
            scrambler_parameters={"scrambler_name": selected.scrambler.polynomial_name if selected.scrambler else "none"},
            crc_parameters={"crc_name": selected.integrity.crc_results[0].crc_name if (selected.integrity and selected.integrity.crc_results) else "none"},
            correction_masks=corr_masks,
            structural_evidence={
                "composite_score": selected.composite_score,
                "complexity_penalty": selected.complexity_penalty,
                "num_frames": len(selected.frames),
                "crc_valid_fraction": selected.integrity.crc_valid_fraction if selected.integrity else 0.0,
            },
            candidate_ranking_provenance={"winner_id": selected.candidate_id, "score": selected.composite_score},
            assumptions=("Synchronous framing", "Standard polynomial assumptions"),
            uncertainties=("Cryptographic integrity not verified", "Protocol semantic validity unverified"),
        )

    return ranked, selected, status, q_level, handoff, diagnostics
