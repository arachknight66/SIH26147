from __future__ import annotations
import numpy as np
import pytest
from app.data_recovery.models import (
    BitHypothesis,
    BitOrder,
    BitPolarity,
    BitStream,
    DataQualityLevel,
    DataRecoveryAnalysis,
    DataRecoveryStatus,
    EpistemicStatus,
    LineCodeType,
    ReconstructionCandidate,
)
from app.verification.bit_checks import audit_bitstream
from app.verification.models import AuditResultStatus

def test_audit_bitstream_clean():
    bits = np.random.randint(0, 2, 512, dtype=np.uint8)
    bs = BitStream(hard_bits=bits, soft_bits=None, symbol_indices=None)
    hyp = BitHypothesis(
        hypothesis_id=1,
        bitstream=bs,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
    )
    cand = ReconstructionCandidate(
        candidate_id=1,
        bit_hypothesis=hyp,
        preamble=None,
        frames=(),
        line_code=None,
        scrambler=None,
        fec=None,
        fec_decode=None,
        integrity=None,
        correction_quality=None,
        recovered_payload_bytes=b"",
        composite_score=0.8,
        complexity_penalty=0.0,
        data_quality_level=DataQualityLevel.HIGH,
        epistemic_status=EpistemicStatus.INFERRED,
    )
    analysis = DataRecoveryAnalysis(
        recording_reference="test",
        bitstream_candidates=[hyp],
        reconstruction_candidates=[cand],
        selected_candidate=cand,
        status=DataRecoveryStatus.INTEGRITY_SUPPORTED,
        quality_level=DataQualityLevel.HIGH,
        is_recovered=True,
        is_inconclusive=False,
        is_ambiguous=False,
    )
    res, tests = audit_bitstream(analysis)
    assert 0.30 <= res.bit_balance <= 0.70
    assert all(t.status in (AuditResultStatus.PASS, AuditResultStatus.WEAK_PASS) for t in tests)
