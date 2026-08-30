import numpy as np
import pytest
from app.data_recovery.models import (
    BitHypothesis,
    BitOrder,
    BitPolarity,
    BitStream,
    DataQualityLevel,
    DataRecoveryStatus,
    EpistemicStatus,
    FrameCandidate,
    LineCodeType,
    PreambleCandidate,
    ReconstructionCandidate,
)
from app.data_recovery.ranking import rank_and_select_reconstructions

def test_rank_and_select_reconstructions_winner():
    bs = BitStream(
        hard_bits=np.ones(64, dtype=np.uint8),
        soft_bits=None,
        symbol_indices=None,
    )
    hyp = BitHypothesis(
        hypothesis_id=1,
        bitstream=bs,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
    )

    preamble = PreambleCandidate(
        pattern_bits=np.array([0, 1, 0, 1] * 4, dtype=np.uint8),
        pattern_hex="5555",
        length_bits=16,
        match_indices=(0, 32, 64),
        match_count=3,
        mean_spacing=32.0,
        spacing_variance=0.0,
        hamming_distance_dist=(0.0, 0.0, 0.0),
        is_periodic=True,
        confidence=0.90,
    )

    cand1 = ReconstructionCandidate(
        candidate_id=1,
        bit_hypothesis=hyp,
        preamble=preamble,
        frames=(
            FrameCandidate(1, np.zeros(32, dtype=np.uint8), np.array([]), np.array([]), np.array([]), np.array([]), 0, 32, True, False, False),
            FrameCandidate(2, np.zeros(32, dtype=np.uint8), np.array([]), np.array([]), np.array([]), np.array([]), 32, 64, True, False, False),
            FrameCandidate(3, np.zeros(32, dtype=np.uint8), np.array([]), np.array([]), np.array([]), np.array([]), 64, 96, True, False, False),
        ),
        line_code=None,
        interleaver=None,
        scrambler=None,
        fec=None,
        fec_decode=None,
        integrity=None,
        correction_quality=None,
        recovered_payload_bytes=b"Payload",
        composite_score=0.85,
        complexity_penalty=0.02,
        data_quality_level=DataQualityLevel.HIGH,
        epistemic_status=EpistemicStatus.INFERRED,
    )

    cand2 = ReconstructionCandidate(
        candidate_id=2,
        bit_hypothesis=hyp,
        preamble=None,
        frames=(),
        line_code=None,
        interleaver=None,
        scrambler=None,
        fec=None,
        fec_decode=None,
        integrity=None,
        correction_quality=None,
        recovered_payload_bytes=b"Payload",
        composite_score=0.40,
        complexity_penalty=0.10,
        data_quality_level=DataQualityLevel.LOW,
        epistemic_status=EpistemicStatus.UNKNOWN,
    )

    ranked, selected, status, q_level, handoff, diags = rank_and_select_reconstructions([cand1, cand2])
    assert len(ranked) == 2
    assert ranked[0].candidate_id == 1
    assert q_level == DataQualityLevel.HIGH
