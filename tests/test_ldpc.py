"""
Unit and regression test suite for LDPC coding, Tanner graphs, and belief-propagation decoding.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.data_recovery.fec_models import STANDARD_FEC_CONFIGURATIONS
from app.data_recovery.ldpc import (
    STANDARD_LDPC_SPECS,
    LDPCCodeSpec,
    LDPCDecodeMode,
    LDPCTerminationStatus,
    build_gallager_matrix,
    build_qc_ldpc_matrix,
    build_tanner_graph,
    compute_graph_girth,
    decode_ldpc,
    decode_ldpc_bitstream,
    encode_ldpc,
)
from app.data_recovery.models import (
    BitHypothesis,
    BitOrder,
    BitPolarity,
    BitStream,
    DataQualityLevel,
    DataRecoveryAnalysis,
    DataRecoveryConfig,
    DataRecoveryStatus,
    EpistemicStatus,
    FECCodeFamily,
    FECDecodeResult,
    FrameCandidate,
    IntegrityResult,
    LineCodeType,
    Phase6Handoff,
    PreambleCandidate,
    ReconstructionCandidate,
)
from app.data_recovery.reconstruction import build_reconstruction_candidate
from app.verification.fec_checks import audit_fec_and_cross_validation
from app.verification.models import TestResultStatus, VerificationConfig


# =============================================================================
# 1. Matrix Construction, Sparsity, Degrees & Girth
# =============================================================================

def test_matrix_construction_sparsity_degrees_and_girth():
    """
    Verify Gallager and Quasi-Cyclic matrix constructions, exact degree distributions,
    sparsity statistics, and assert girth >= 6 (no 4-cycles) for standard configs.
    """
    # 1. Gallager (3, 6) N=96, M=48
    h_gal = build_gallager_matrix(n_bits=96, d_v=3, d_c=6, rng_seed=42)
    assert h_gal.shape == (48, 96)
    col_weights = np.sum(h_gal, axis=0)
    row_weights = np.sum(h_gal, axis=1)
    assert np.all(col_weights == 3), f"Gallager bit degrees must all equal 3, got {col_weights}"
    assert np.all(row_weights == 6), f"Gallager check degrees must all equal 6, got {row_weights}"

    graph_gal = build_tanner_graph(h_gal)
    assert graph_gal.girth >= 6, f"Gallager graph girth must be >= 6, got {graph_gal.girth}"

    # 2. Quasi-Cyclic N=128, M=64 (Z=16)
    spec_qc128 = STANDARD_LDPC_SPECS["QC_LDPC_N128_R12"]
    assert spec_qc128.n_bits == 128
    assert spec_qc128.m_checks == 64
    assert spec_qc128.k_info_bits == 64
    assert spec_qc128.rate == 0.5
    assert spec_qc128.sparsity > 0.90, f"QC-LDPC must be sparse (>90% zeros), got {spec_qc128.sparsity:.3f}"
    assert spec_qc128.girth >= 6, f"Standard QC-LDPC girth must be >= 6, got {spec_qc128.girth}"

    # 3. Standard specifications registry validation
    for name, spec in STANDARD_LDPC_SPECS.items():
        assert spec.girth >= 6, f"Code {name} has pathological girth {spec.girth} < 6"
        assert spec.g_matrix is not None, f"Code {name} must have a valid systematic generator matrix"
        # Check parity check orthogonality: H * G^T = 0 (mod 2)
        ortho = np.dot(spec.h_matrix, spec.g_matrix.T) % 2
        assert np.all(ortho == 0), f"Parity check matrix H and generator G must satisfy H*G^T=0 for {name}"


# =============================================================================
# 2. Clean-Channel Round Trip
# =============================================================================

def test_clean_channel_round_trip():
    """
    Generate valid systematic codewords via generator matrix, decode with zero noise,
    and assert convergence in <= 1 iteration, 0 bit corrections, and exact recovery.
    """
    rng = np.random.default_rng(42)

    for name in ["QC_LDPC_N128_R12", "GALLAGER_N96_R12"]:
        spec = STANDARD_LDPC_SPECS[name]
        msg = rng.integers(0, 2, size=spec.k_info_bits, dtype=np.uint8)
        codeword = encode_ldpc(msg, spec)

        # Verify it is a true codeword
        syndrome = np.dot(spec.h_matrix, codeword) % 2
        assert np.all(syndrome == 0), "Generated vector is not in the code nullspace"

        # Decode clean hard bits
        res = decode_ldpc(codeword, code_spec=spec, max_iterations=20)
        assert res.valid is True
        assert res.iterations_used <= 1
        assert res.corrected_bit_count == 0
        assert np.array_equal(res.decoded_bits, codeword)
        assert res.termination_status == LDPCTerminationStatus.CONVERGED
        assert res.final_syndrome_weight == 0


# =============================================================================
# 3. Waterfall / Threshold Behavior Test
# =============================================================================

def test_waterfall_threshold_behavior():
    """
    Simulate soft LLR transmission across a sweep of channel noise / SNR levels.
    Confirm the characteristic LDPC waterfall behavior: high success at high SNR,
    dropping sharply at low SNR.
    """
    spec = STANDARD_LDPC_SPECS["QC_LDPC_N128_R12"]
    rng = np.random.default_rng(123)
    num_trials = 10

    msg = rng.integers(0, 2, size=spec.k_info_bits, dtype=np.uint8)
    codeword = encode_ldpc(msg, spec)
    # BPSK mapping: 0 -> +1.0, 1 -> -1.0
    bpsk_syms = np.where(codeword == 0, 1.0, -1.0)

    # Low noise (high SNR = 6.0 dB, sigma = 0.5)
    high_snr_success = 0
    sigma_low = 0.5
    for _ in range(num_trials):
        noise = rng.normal(0.0, sigma_low, size=spec.n_bits)
        rx_signal = bpsk_syms + noise
        llr = 2.0 * rx_signal / (sigma_low ** 2)
        rx_bits = (llr < 0).astype(np.uint8)

        res = decode_ldpc(rx_bits, code_spec=spec, soft_bits=llr, max_iterations=30)
        if res.valid and np.array_equal(res.decoded_bits, codeword):
            high_snr_success += 1

    assert high_snr_success >= 9, f"High SNR success rate should be >= 90%, got {high_snr_success}/{num_trials}"

    # Extreme noise (low SNR = -3.0 dB, sigma = 1.4)
    low_snr_success = 0
    sigma_high = 1.4
    for _ in range(num_trials):
        noise = rng.normal(0.0, sigma_high, size=spec.n_bits)
        rx_signal = bpsk_syms + noise
        llr = 2.0 * rx_signal / (sigma_high ** 2)
        rx_bits = (llr < 0).astype(np.uint8)

        res = decode_ldpc(rx_bits, code_spec=spec, soft_bits=llr, max_iterations=20)
        if res.valid:
            low_snr_success += 1

    assert low_snr_success <= 3, f"Low SNR success rate should drop sharply (<= 30%), got {low_snr_success}/{num_trials}"


# =============================================================================
# 4. Sum-Product vs. Min-Sum Comparison
# =============================================================================

def test_sum_product_vs_min_sum_comparison():
    """
    Confirm both Sum-Product and Min-Sum modes converge to the same correct codeword
    on low/moderate noise inputs, and verify that the mode is correctly reported.
    """
    spec = STANDARD_LDPC_SPECS["GALLAGER_N96_R12"]
    rng = np.random.default_rng(789)

    msg = rng.integers(0, 2, size=spec.k_info_bits, dtype=np.uint8)
    codeword = encode_ldpc(msg, spec)
    bpsk_syms = np.where(codeword == 0, 1.0, -1.0)

    # Moderate noise
    sigma = 0.65
    noise = rng.normal(0.0, sigma, size=spec.n_bits)
    rx_signal = bpsk_syms + noise
    llr = 2.0 * rx_signal / (sigma ** 2)
    rx_bits = (llr < 0).astype(np.uint8)

    res_sp = decode_ldpc(rx_bits, code_spec=spec, soft_bits=llr, mode=LDPCDecodeMode.SUM_PRODUCT, max_iterations=40)
    res_ms = decode_ldpc(rx_bits, code_spec=spec, soft_bits=llr, mode=LDPCDecodeMode.MIN_SUM, max_iterations=40)

    assert res_sp.decoding_mode == LDPCDecodeMode.SUM_PRODUCT
    assert res_ms.decoding_mode == LDPCDecodeMode.MIN_SUM

    # Both should converge on moderate noise
    assert res_sp.valid is True
    assert res_ms.valid is True
    assert np.array_equal(res_sp.decoded_bits, codeword)
    assert np.array_equal(res_ms.decoded_bits, codeword)


# =============================================================================
# 5. Extrinsic Exclusion Regression Test
# =============================================================================

def test_extrinsic_exclusion_regression():
    """
    Verify that the decoder strictly excludes extrinsic messages (q_{v->c} = L_total - r_{c->v}).
    """
    spec = STANDARD_LDPC_SPECS["QC_LDPC_N128_R12"]
    rng = np.random.default_rng(456)
    msg = rng.integers(0, 2, size=spec.k_info_bits, dtype=np.uint8)
    codeword = encode_ldpc(msg, spec)

    # Corrupt 3 bits
    corrupted = codeword.copy()
    corrupted[5] ^= 1
    corrupted[20] ^= 1
    corrupted[50] ^= 1

    res = decode_ldpc(corrupted, code_spec=spec, max_iterations=30)
    assert res.valid is True
    assert np.array_equal(res.decoded_bits, codeword)
    assert res.corrected_bit_count == 3
    # Check correction mask
    assert np.all(res.correction_mask == (corrupted != codeword))


# =============================================================================
# 6. Non-Convergence & Trapping-Set Diagnosis
# =============================================================================

def test_non_convergence_and_trapping_set_diagnosis():
    """
    Verify that an uncorrectable corrupted input halts within max_iterations,
    reports valid=False, and surfaces the syndrome weight trajectory and termination status.
    """
    spec = STANDARD_LDPC_SPECS["GALLAGER_N96_R12"]
    rng = np.random.default_rng(999)

    # Heavy random corruption (e.g. 35% bit flips)
    corrupted = rng.integers(0, 2, size=spec.n_bits, dtype=np.uint8)

    res = decode_ldpc(corrupted, code_spec=spec, max_iterations=15)
    assert res.valid is False
    assert res.iterations_used == 15
    assert res.final_syndrome_weight > 0
    assert res.termination_status in (
        LDPCTerminationStatus.ITERATION_CAP_REACHED,
        LDPCTerminationStatus.TRAPPING_SET_STALLED,
    )
    assert len(res.syndrome_weight_history) > 1
    # Hard bits must NOT be returned as successfully corrected
    assert res.corrected_bit_count == 0


# =============================================================================
# 7. Null / Out-of-Distribution Random Noise Test
# =============================================================================

def test_null_ood_random_noise():
    """
    Confirm random unstructured noise vectors essentially never satisfy an LDPC parity check.
    """
    spec = STANDARD_LDPC_SPECS["QC_LDPC_N128_R12"]
    rng = np.random.default_rng(777)
    num_trials = 50
    false_positives = 0

    for _ in range(num_trials):
        random_bits = rng.integers(0, 2, size=spec.n_bits, dtype=np.uint8)
        res = decode_ldpc(random_bits, code_spec=spec, max_iterations=10, max_correction_fraction=0.10)
        if res.valid:
            false_positives += 1

    assert false_positives == 0, f"Observed {false_positives} false positives on pure random noise"


# =============================================================================
# 8. Independent Verification Audit & Held-Out Cross-Validation
# =============================================================================

def test_verification_audit_ldpc():
    """
    Verify audit_fec_and_cross_validation properly performs independent syndrome verification,
    checks iteration limits, and cross-validates on held-out blocks for FECCodeFamily.LDPC.
    """
    spec = STANDARD_LDPC_SPECS["QC_LDPC_N128_R12"]
    rng = np.random.default_rng(333)

    # Construct 4 blocks of valid codewords with 2% channel error
    num_blocks = 4
    codewords = []
    corrupted_blocks = []

    for _ in range(num_blocks):
        msg = rng.integers(0, 2, size=spec.k_info_bits, dtype=np.uint8)
        cw = encode_ldpc(msg, spec)
        corr = cw.copy()
        # Flip 2 bits per block (~1.5% BER)
        corr[0] ^= 1
        corr[15] ^= 1
        codewords.append(cw)
        corrupted_blocks.append(corr)

    raw_channel_bits = np.concatenate(corrupted_blocks)
    full_cw_bits = np.concatenate(codewords)

    fec_dec = decode_ldpc_bitstream(raw_channel_bits, code_spec=spec, max_correction_fraction=0.10)
    assert fec_dec.valid is True

    ldpc_hyp = next(h for h in STANDARD_FEC_CONFIGURATIONS if h.code_family == FECCodeFamily.LDPC and spec.name in h.code_name)

    bit_stream = BitStream(hard_bits=raw_channel_bits, soft_bits=None, symbol_indices=None)
    bit_hyp = BitHypothesis(
        hypothesis_id=1,
        bitstream=bit_stream,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
        epistemic_status=EpistemicStatus.INFERRED,
    )

    reconstruction = ReconstructionCandidate(
        candidate_id=1,
        bit_hypothesis=bit_hyp,
        preamble=None,
        frames=(),
        line_code=None,
        interleaver=None,
        scrambler=None,
        fec=ldpc_hyp,
        fec_decode=fec_dec,
        integrity=None,
        correction_quality=None,
        recovered_payload_bytes=b"Payload",
        composite_score=0.80,
        complexity_penalty=0.06,
        data_quality_level=DataQualityLevel.HIGH,
        epistemic_status=EpistemicStatus.CORRECTED,
    )

    analysis = DataRecoveryAnalysis(
        recording_reference="test_ldpc_rec",
        bitstream_candidates=[bit_hyp],
        reconstruction_candidates=[reconstruction],
        selected_candidate=reconstruction,
        status=DataRecoveryStatus.INTEGRITY_SUPPORTED,
        quality_level=DataQualityLevel.HIGH,
        is_recovered=True,
        is_inconclusive=False,
        is_ambiguous=False,
    )

    handoff = Phase6Handoff(
        raw_bits=raw_channel_bits,
        corrected_bits=full_cw_bits,
        payload_bytes=b"Payload",
        frame_boundaries=(),
        fec_parameters={"code_name": spec.name},
        scrambler_parameters={},
    )

    audit_res, tests = audit_fec_and_cross_validation(analysis, handoff=handoff)
    assert audit_res.is_beneficial is True
    assert audit_res.anti_overcorrection_passed is True
    assert audit_res.held_out_validation_passed is True

    test_ids = [t.test_id for t in tests]
    assert "FEC_01_OVERCORRECTION" in test_ids
    assert "FEC_02_CROSS_VALIDATION" in test_ids
    assert "FEC_05_LDPC_FALSIFICATION" in test_ids

    for t in tests:
        assert t.status == TestResultStatus.PASS, f"Test {t.test_id} failed: {t.counter_evidence}"
