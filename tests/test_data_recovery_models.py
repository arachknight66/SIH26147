import numpy as np
import pytest
from app.data_recovery.models import (
    BitHypothesis,
    BitOrder,
    BitPolarity,
    BitStream,
    ByteStreamCandidate,
    CorrectionQuality,
    CRCResult,
    DataQualityLevel,
    DataRecoveryAnalysis,
    DataRecoveryConfig,
    DataRecoveryStatus,
    EpistemicStatus,
    FECCodeFamily,
    FECDecodeResult,
    FECHypothesis,
    FrameBoundary,
    FrameCandidate,
    IntegrityResult,
    LineCodeHypothesis,
    LineCodeType,
    Phase6Handoff,
    PreambleCandidate,
    ReconstructionCandidate,
    ScramblerHypothesis,
    ScramblerType,
)

def test_data_recovery_models_instantiation():
    cfg = DataRecoveryConfig(max_bit_hypotheses=4, max_correction_fraction=0.10)
    assert cfg.max_bit_hypotheses == 4
    assert cfg.max_correction_fraction == 0.10

    bs = BitStream(
        hard_bits=np.array([1, 0, 1, 1], dtype=np.uint8),
        soft_bits=np.array([1.2, -1.1, 0.9, 1.5], dtype=np.float32),
        symbol_indices=np.array([0, 1, 0, 1], dtype=np.int32),
        bit_order=BitOrder.MSB_FIRST,
        bit_polarity=BitPolarity.NORMAL,
        bit_offset=0,
        source_candidate="QPSK",
    )
    assert bs.length == 4
    assert bs.bit_order == BitOrder.MSB_FIRST

    hyp = BitHypothesis(
        hypothesis_id=1,
        bitstream=bs,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
        score=0.90,
        epistemic_status=EpistemicStatus.OBSERVED,
    )
    assert hyp.phase_rotation_deg == 0.0

    crc_res = CRCResult(
        crc_name="CRC-16-CCITT",
        width=16,
        polynomial=0x1021,
        init_value=0xFFFF,
        xor_out=0x0000,
        reflect_in=False,
        reflect_out=False,
        calculated_crc=0x1234,
        expected_crc=0x1234,
        is_valid=True,
    )
    assert crc_res.is_valid is True

    fec_res = FECDecodeResult(
        input_bits=np.array([1, 0, 1], dtype=np.uint8),
        decoded_bits=np.array([1, 0], dtype=np.uint8),
        correction_mask=np.array([False, False, True]),
        corrected_bit_count=1,
        correction_fraction=0.333,
        path_metric=1.0,
        normalized_path_metric=0.333,
        is_overcorrected=False,
        code_family=FECCodeFamily.CONVOLUTIONAL,
        valid=True,
    )
    assert fec_res.corrected_bit_count == 1
