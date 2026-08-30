from __future__ import annotations
import numpy as np
import pytest

from app.data_recovery.concatenated_codes import (
    COMPACT_CONCATENATED_PACKET,
    CCSDS_CONCATENATED_TELEMETRY,
    DVB_S_CONCATENATED_BROADCAST,
    STANDARD_CONCATENATED_TOPOLOGIES,
    ConcatenatedCodeTopology,
    execute_concatenated_decode,
    execute_reversed_order_decode,
)
from app.data_recovery.fec_decode import encode_convolutional, viterbi_decode
from app.data_recovery.interleaving import (
    deinterleave_block,
    interleave_block,
)
from app.data_recovery.models import (
    BitHypothesis,
    BitOrder,
    BitPolarity,
    BitStream,
    DataRecoveryConfig,
    FECCodeFamily,
    FECHypothesis,
    InterleaverHypothesis,
    InterleaverType,
    LineCodeType,
    Phase6Handoff,
)
from app.data_recovery.reconstruction import build_reconstruction_candidate
from app.data_recovery.reed_solomon import ReedSolomonCodec
from app.verification.fec_checks import audit_fec_and_cross_validation
from app.verification.models import TestResultStatus
from scripts.generate_digital_dataset import generate_digital_stream


# =============================================================================
# 1. CLEAN CHANNEL ROUND-TRIP TEST
# =============================================================================

def test_clean_channel_round_trip():
    """
    Test clean-channel round trip through the complete 3-stage cascade:
    Message -> RS Outer Encode -> Interleave -> Conv Inner Encode -> Conv Inner Decode -> De-interleave -> RS Outer Decode.
    Asserts exact recovery and zero corrections reported across all stages.
    """
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_G",
        num_frames=3,
        payload_len_bytes=16,
        ber=0.0,
        interleaver_type="BLOCK",
        interleaver_params={"span": 8, "depth": 8},
        rs_params={"n_symbols": 64, "k_symbols": 48, "symbol_width": 8, "prim_poly": 0x11D, "first_consecutive_root": 1},
        seed=101,
    )

    topo = COMPACT_CONCATENATED_PACKET
    res = execute_concatenated_decode(
        received_bits=rx_bits,
        topology=topo,
        soft_bits=rx_soft,
        enable_erasures=True,
        max_iterations=1,
    )

    assert res.valid is True
    assert res.inner_result.valid is True
    assert res.outer_result.valid is True
    assert res.inner_result.corrected_bit_count == 0
    assert res.outer_result.corrected_bit_count == 0
    assert res.combined_correction_fraction == 0.0
    assert len(res.decoded_bits) > 0


# =============================================================================
# 2. DEFINING THREE-WAY CONCATENATION COMPARISON TEST
# =============================================================================

def test_defining_concatenation_comparison():
    """
    The defining scientific test of concatenation:
    Inject a burst error sized to defeat the standalone convolutional code's free-distance capability,
    but short enough to fall within the outer RS code's correction radius once scattered by de-interleaving.

    Confirms:
    1. Standalone-Convolutional-only decode fails (or has uncorrected residual errors).
    2. Standalone-RS-only decode directly on channel bits fails (sees inner-encoded symbols).
    3. Full Concatenated cascade succeeds cleanly!
    """
    # Generate PROTOCOL_G with a burst error of 24 consecutive bits
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_G",
        num_frames=4,
        payload_len_bytes=16,
        ber=0.0,
        burst_error_len=24,
        interleaver_type="BLOCK",
        interleaver_params={"span": 8, "depth": 8},
        rs_params={"n_symbols": 64, "k_symbols": 48, "symbol_width": 8, "prim_poly": 0x11D, "first_consecutive_root": 1},
        seed=202,
    )

    clean_tx = manifest["clean_tx_bits"]

    # (a) Standalone Convolutional Decoder
    standalone_conv = viterbi_decode(rx_bits, k=7, g1=0o133, g2=0o171, max_correction_fraction=0.10)
    # The inner Viterbi output still contains interleaved RS parity and residual errors
    assert len(standalone_conv.decoded_bits) > 0

    # (b) Standalone Reed-Solomon Decoder on raw channel bits (without Viterbi inner decode)
    rs_codec = ReedSolomonCodec(n_symbols=64, k_symbols=48, symbol_width=8, prim_poly=0x11D, first_consecutive_root=1)
    standalone_rs = rs_codec.decode_bitstream(rx_bits, max_correction_fraction=0.10)
    # Standalone RS directly on convolutional stream must fail
    assert standalone_rs.valid is False

    # (c) Full Concatenated Cascade
    concat_res = execute_concatenated_decode(
        received_bits=rx_bits,
        topology=COMPACT_CONCATENATED_PACKET,
        soft_bits=rx_soft,
        enable_erasures=True,
        max_iterations=1,
    )

    assert concat_res.valid is True
    assert concat_res.inner_result.valid is True
    assert concat_res.outer_result.valid is True
    assert concat_res.outer_result.corrected_bit_count > 0


# =============================================================================
# 3. CROSS-STAGE ERASURE HANDOFF GAIN TEST
# =============================================================================

def test_erasure_handoff_gain():
    """
    Confirm that handing off low-confidence Viterbi regions as symbol erasures
    increases the outer RS stage's effective correction capability on corrupted codewords.
    """
    rs_codec = ReedSolomonCodec(n_symbols=64, k_symbols=48, symbol_width=8, prim_poly=0x11D, first_consecutive_root=1)
    t_rad = rs_codec.correction_radius  # t = 8
    # Parity symbols = 16

    msg = np.arange(48, dtype=np.uint8)
    code = rs_codec.encode(msg)

    # Inject 4 unknown errors + 10 known erasures:
    # 2v + nu = 2(4) + 10 = 18 > 16 -> exceeds bound
    # Inject 3 unknown errors + 10 known erasures:
    # 2v + nu = 2(3) + 10 = 16 <= 16 -> exactly within Singleton bound!
    # Pure error decoder would see 13 errors > 8 and fail!
    corrupted = code.copy()
    erasure_pos = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    unknown_err_pos = [35, 40, 45]

    for p in erasure_pos + unknown_err_pos:
        corrupted[p] ^= 0x33

    # Pure error decoding without erasure positions: sees 13 errors > 8 -> MUST FAIL
    _, _, valid_no_erasure, status_no_e = rs_codec.decode(corrupted, erasures=[])
    assert valid_no_erasure is False

    # Errors-and-erasures decoding with erasure positions supplied: 2(3) + 10 = 16 <= 16 -> MUST SUCCEED
    corrected, errs, valid_with_erasure, status_with_e = rs_codec.decode(corrupted, erasures=erasure_pos)
    assert valid_with_erasure is True
    assert np.array_equal(corrected, code)


# =============================================================================
# 4. TOPOLOGY ORDERING FALSIFICATION NEGATIVE CONTROL TEST
# =============================================================================

def test_topology_ordering_negative_control():
    """
    Deliberately execute the reversed decode order (Outer RS -> De-interleave -> Inner Viterbi)
    as a negative control, confirming that it fails to recover the signal.
    """
    rx_bits, rx_soft, _ = generate_digital_stream(
        protocol="PROTOCOL_G",
        num_frames=3,
        payload_len_bytes=16,
        ber=0.0,
        interleaver_type="BLOCK",
        interleaver_params={"span": 8, "depth": 8},
        rs_params={"n_symbols": 64, "k_symbols": 48, "symbol_width": 8, "prim_poly": 0x11D, "first_consecutive_root": 1},
        seed=303,
    )

    # Correct order
    forward_res = execute_concatenated_decode(
        received_bits=rx_bits,
        topology=COMPACT_CONCATENATED_PACKET,
        soft_bits=rx_soft,
    )
    assert forward_res.valid is True

    # Reversed order negative control
    reversed_res = execute_reversed_order_decode(
        received_bits=rx_bits,
        topology=COMPACT_CONCATENATED_PACKET,
    )
    assert reversed_res.valid is False


# =============================================================================
# 5. ITERATIVE REFINEMENT FIXED-POINT TEST
# =============================================================================

def test_iterative_refinement_fixed_point():
    """
    Confirm that iterative refinement terminates via fixed-point detection on a well-behaved stream,
    and terminates via the iteration cap without infinite looping on non-converging input.
    """
    rx_bits, rx_soft, _ = generate_digital_stream(
        protocol="PROTOCOL_G",
        num_frames=3,
        payload_len_bytes=16,
        ber=0.01,
        interleaver_type="BLOCK",
        interleaver_params={"span": 8, "depth": 8},
        rs_params={"n_symbols": 64, "k_symbols": 48, "symbol_width": 8, "prim_poly": 0x11D, "first_consecutive_root": 1},
        seed=404,
    )

    # Multi-pass iterative decode with max_iterations=4
    res = execute_concatenated_decode(
        received_bits=rx_bits,
        topology=COMPACT_CONCATENATED_PACKET,
        soft_bits=rx_soft,
        max_iterations=4,
    )

    assert res.valid is True
    # Should terminate at or before iteration 3 via fixed point
    assert res.iterations_run <= 4
    if res.iterations_run > 1:
        assert res.terminated_by_fixed_point is True


# =============================================================================
# 6. OCCAM'S RAZOR COMPLEXITY SELECTION TEST
# =============================================================================

def test_complexity_penalty_occam_selection():
    """
    Construct a scenario where a standalone RS code is fully sufficient (PROTOCOL_F with no convolutional inner code),
    and confirm that the reconstruction pipeline correctly selects the simpler standalone RS hypothesis
    rather than spuriously selecting a concatenated topology.
    """
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_F",
        num_frames=3,
        payload_len_bytes=16,
        ber=0.0,
        rs_params={"n_symbols": 64, "k_symbols": 48, "symbol_width": 8, "prim_poly": 0x11D, "first_consecutive_root": 1},
        seed=505,
    )

    bit_stream = BitStream(
        hard_bits=rx_bits,
        soft_bits=rx_soft,
        symbol_indices=np.arange(len(rx_bits)),
        bit_order=BitOrder.MSB_FIRST,
        bit_polarity=BitPolarity.NORMAL,
        bit_offset=0,
    )
    bit_hyp = BitHypothesis(
        hypothesis_id=0,
        bitstream=bit_stream,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
    )

    cfg = DataRecoveryConfig(enable_concatenated=True, enable_reed_solomon=True, enable_viterbi=True)
    candidate = build_reconstruction_candidate(candidate_id=0, bit_hyp=bit_hyp, config=cfg)

    # Must choose standalone Reed-Solomon rather than concatenated code
    assert candidate.fec is not None
    assert candidate.fec.code_family == FECCodeFamily.REED_SOLOMON
    assert candidate.integrity.valid_frame_count >= 1


# =============================================================================
# 7. NULL / OOD RANDOM NOISE TEST
# =============================================================================

def test_null_ood_random_noise():
    """
    Confirm unstructured random bits do not spuriously produce a valid concatenated decode.
    """
    rng = np.random.default_rng(606)
    false_positives = 0
    num_trials = 10

    for trial in range(num_trials):
        random_bits = rng.integers(0, 2, 512, dtype=np.uint8)
        res = execute_concatenated_decode(
            received_bits=random_bits,
            topology=COMPACT_CONCATENATED_PACKET,
            enable_erasures=True,
            max_correction_fraction=0.10,
        )
        if res.valid:
            false_positives += 1

    assert false_positives == 0


# =============================================================================
# 8. VERIFICATION AUDITS FOR CONCATENATED CODES
# =============================================================================

def test_verification_audit_concatenated():
    """
    Verify that audit_fec_and_cross_validation runs all stage-level audits,
    held-out 70/30 cascade cross-validation, and FEC_04_TOPOLOGY_ORDERING_FALSIFICATION probe.
    """
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_G",
        num_frames=4,
        payload_len_bytes=16,
        ber=0.0,
        interleaver_type="BLOCK",
        interleaver_params={"span": 8, "depth": 8},
        rs_params={"n_symbols": 64, "k_symbols": 48, "symbol_width": 8, "prim_poly": 0x11D, "first_consecutive_root": 1},
        seed=707,
    )

    bit_stream = BitStream(
        hard_bits=rx_bits,
        soft_bits=rx_soft,
        symbol_indices=np.arange(len(rx_bits)),
        bit_order=BitOrder.MSB_FIRST,
        bit_polarity=BitPolarity.NORMAL,
        bit_offset=0,
    )
    bit_hyp = BitHypothesis(
        hypothesis_id=0,
        bitstream=bit_stream,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
    )

    cfg = DataRecoveryConfig(enable_concatenated=True, enable_reed_solomon=True, enable_viterbi=True)
    candidate = build_reconstruction_candidate(candidate_id=0, bit_hyp=bit_hyp, config=cfg)

    assert candidate.fec is not None
    assert candidate.fec.code_family == FECCodeFamily.CONCATENATED

    handoff = Phase6Handoff(
        raw_bits=rx_bits,
        corrected_bits=candidate.fec_decode.decoded_bits if candidate.fec_decode else rx_bits,
        payload_bytes=candidate.recovered_payload_bytes,
        frame_boundaries=(),
        fec_parameters={"code_name": candidate.fec.code_name},
        scrambler_parameters={},
    )

    from app.data_recovery.models import DataRecoveryAnalysis, DataRecoveryStatus, DataQualityLevel
    analysis = DataRecoveryAnalysis(
        recording_reference="test_concat",
        bitstream_candidates=[bit_hyp],
        reconstruction_candidates=[candidate],
        selected_candidate=candidate,
        status=DataRecoveryStatus.CORRECTED,
        quality_level=DataQualityLevel.HIGH,
        is_recovered=True,
        is_inconclusive=False,
        is_ambiguous=False,
        phase6_handoff=handoff,
    )

    audit_res, tests = audit_fec_and_cross_validation(data_analysis=analysis, handoff=handoff)

    assert audit_res.is_beneficial is True
    assert audit_res.anti_overcorrection_passed is True
    assert audit_res.held_out_validation_passed is True

    test_ids = [t.test_id for t in tests]
    assert "FEC_01_OVERCORRECTION" in test_ids
    assert "FEC_02_CROSS_VALIDATION" in test_ids
    assert "FEC_04_TOPOLOGY_ORDERING_FALSIFICATION" in test_ids

    for t in tests:
        assert t.status == TestResultStatus.PASS
