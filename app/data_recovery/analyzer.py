from __future__ import annotations
from typing import Sequence
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity
from app.recovery.models import RecoveredSignal, RecoveryAnalysis
from .ambiguity import generate_ambiguity_hypotheses
from .models import (
    BitHypothesis,
    BitStream,
    DataQualityLevel,
    DataRecoveryAnalysis,
    DataRecoveryConfig,
    DataRecoveryStatus,
    FrameCandidate,
    LineCodeType,
    ReconstructionCandidate,
)
from .preprocessing import extract_bitstream_from_recovery
from .ranking import rank_and_select_reconstructions
from .reconstruction import build_reconstruction_candidate

def recover_data(
    recovery: RecoveryAnalysis | RecoveredSignal,
    config: DataRecoveryConfig | None = None,
) -> DataRecoveryAnalysis:
    """
    Execute scientific data-stream reconstruction, framing, scrambler analysis, and FEC error correction.

    Parameters
    ----------
    recovery : RecoveryAnalysis | RecoveredSignal
        Phase 4 recovered signal output.
    config : DataRecoveryConfig | None
        Configuration and computational limits.

    Returns
    -------
    DataRecoveryAnalysis
    """
    cfg = config or DataRecoveryConfig()
    all_diagnostics: list[Diagnostic] = []

    # 1. Canonical BitStream Extraction
    bitstream, extract_diags = extract_bitstream_from_recovery(recovery, config=cfg)
    all_diagnostics.extend(extract_diags)

    if bitstream.length < 16 or np.all(bitstream.hard_bits == 0) or np.all(bitstream.hard_bits == 1):
        all_diagnostics.append(
            Diagnostic(
                code="INSUFFICIENT_OR_UNIFORM_BITSTREAM",
                message=f"Bitstream length ({bitstream.length} bits) or uniform DC content is insufficient for framing or error correction.",
                severity=DiagnosticSeverity.WARNING,
            )
        )
        return DataRecoveryAnalysis(
            recording_reference="in_memory",
            bitstream_candidates=[],
            reconstruction_candidates=[],
            selected_candidate=None,
            status=DataRecoveryStatus.INSUFFICIENT_DATA,
            quality_level=DataQualityLevel.VERY_LOW,
            is_recovered=False,
            is_inconclusive=True,
            is_ambiguous=False,
            failure_reason="Insufficient or uniform bitstream for data recovery",
            diagnostics=all_diagnostics,
            provenance={"bit_count": bitstream.length},
            phase6_handoff=None,
        )

    # 2. Generate Ambiguity Hypotheses (Polarity, Rotations)
    bit_hypotheses = generate_ambiguity_hypotheses(bitstream, config=cfg)

    # 3. Execute Reconstruction Pipeline on each Bit Hypothesis
    recon_candidates: list[ReconstructionCandidate] = []
    for cand_idx, bit_hyp in enumerate(bit_hypotheses, start=1):
        recon = build_reconstruction_candidate(cand_idx, bit_hyp, config=cfg)
        recon_candidates.append(recon)

    # 4. Multi-Evidence Ranking & Selection
    ranked, selected, status, q_level, handoff, rank_diags = rank_and_select_reconstructions(recon_candidates)
    all_diagnostics.extend(rank_diags)

    is_rec = (selected is not None and status in (
        DataRecoveryStatus.CORRECTED,
        DataRecoveryStatus.INTEGRITY_SUPPORTED,
        DataRecoveryStatus.STRUCTURALLY_SUPPORTED,
    ))
    is_inconc = not is_rec
    is_ambig = (status == DataRecoveryStatus.AMBIGUOUS)

    return DataRecoveryAnalysis(
        recording_reference="in_memory",
        bitstream_candidates=bit_hypotheses,
        reconstruction_candidates=ranked,
        selected_candidate=selected,
        status=status,
        quality_level=q_level,
        is_recovered=is_rec,
        is_inconclusive=is_inconc,
        is_ambiguous=is_ambig,
        failure_reason=None if is_rec else "Data recovery inconclusive across candidate reconstructions",
        diagnostics=all_diagnostics,
        provenance={
            "num_candidates_attempted": len(recon_candidates),
            "bit_hypothesis_count": len(bit_hypotheses),
            "search_controls": {
                "evaluate_all_bit_offsets": cfg.evaluate_all_bit_offsets,
                "evaluate_polarity_inversion": cfg.evaluate_polarity_inversion,
                "evaluate_rotational_ambiguities": cfg.evaluate_rotational_ambiguities,
                "enable_viterbi": cfg.enable_viterbi,
                "enable_descrambler": cfg.enable_descrambler,
                "enable_hamming": cfg.enable_hamming,
            },
        },
        phase6_handoff=handoff,
    )

def recover_frame_stream(
    bitstream: BitStream,
    config: DataRecoveryConfig | None = None,
) -> list[FrameCandidate]:
    """
    Convenience helper to extract sliced FrameCandidates directly from a BitStream.

    Parameters
    ----------
    bitstream : BitStream
    config : DataRecoveryConfig | None

    Returns
    -------
    list[FrameCandidate]
    """
    hyp = BitHypothesis(
        hypothesis_id=1,
        bitstream=bitstream,
        phase_rotation_deg=0.0,
        polarity=bitstream.bit_polarity,
        line_code=LineCodeType.NONE,
        bit_order=bitstream.bit_order,
        bit_offset=bitstream.bit_offset,
    )
    recon = build_reconstruction_candidate(1, hyp, config=config)
    return list(recon.frames)

def recover_candidate_stream(
    bit_hypothesis: BitHypothesis,
    config: DataRecoveryConfig | None = None,
) -> ReconstructionCandidate:
    """
    Directly execute reconstruction on a specific BitHypothesis.

    Parameters
    ----------
    bit_hypothesis : BitHypothesis
    config : DataRecoveryConfig | None

    Returns
    -------
    ReconstructionCandidate
    """
    return build_reconstruction_candidate(1, bit_hypothesis, config=config)
