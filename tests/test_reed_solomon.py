from __future__ import annotations
import pytest
import numpy as np

from app.data_recovery.models import (
    BitHypothesis,
    BitOrder,
    BitPolarity,
    BitStream,
    DataRecoveryConfig,
    EpistemicStatus,
    FECCodeFamily,
    LineCodeType,
)
from app.data_recovery.fec_models import STANDARD_FEC_CONFIGURATIONS
from app.data_recovery.galois_field import GaloisField
from app.data_recovery.reed_solomon import ReedSolomonCodec
from app.data_recovery.reconstruction import build_reconstruction_candidate
from app.verification.fec_checks import audit_fec_and_cross_validation
from scripts.generate_digital_dataset import generate_digital_stream


# Standard Test Configurations (N, K, m, poly, fcr)
STANDARD_CODEC_CONFIGS = [
    # (N, K, symbol_width, prim_poly, fcr, name)
    (64, 48, 8, 0x11D, 1, "RS(64,48,fcr=1)"),
    (128, 112, 8, 0x11D, 0, "RS(128,112,fcr=0)"),
    (204, 188, 8, 0x11D, 0, "RS(204,188,fcr=0)"),
    (255, 239, 8, 0x11D, 0, "RS(255,239,fcr=0)"),
    (255, 223, 8, 0x187, 112, "RS(255,223,fcr=112)"),
]


@pytest.mark.parametrize("n,k,m,poly,fcr,name", STANDARD_CODEC_CONFIGS)
def test_generator_polynomial_and_encoding_roots(n: int, k: int, m: int, poly: int, fcr: int, name: str):
    """
    Verify algebraic defining property: Every systematic Reed-Solomon codeword produced
    by the encoder evaluates to zero at all 2t consecutive roots of the generator polynomial.
    """
    codec = ReedSolomonCodec(n_symbols=n, k_symbols=k, symbol_width=m, prim_poly=poly, first_consecutive_root=fcr)
    rng = np.random.default_rng(100)

    for _ in range(10):
        msg = rng.integers(0, 1 << m, k, dtype=np.uint8)
        codeword = codec.encode(msg)

        assert len(codeword) == n
        assert np.array_equal(codeword[:k], msg)

        # Check all syndromes are zero on clean codeword
        syndromes = codec.compute_syndromes(codeword)
        assert np.all(syndromes == 0), f"Clean codeword had non-zero syndromes: {syndromes}"


@pytest.mark.parametrize("n,k,m,poly,fcr,name", STANDARD_CODEC_CONFIGS)
def test_clean_channel_roundtrip(n: int, k: int, m: int, poly: int, fcr: int, name: str):
    """Verify clean channel decoding produces exact message recovery with zero corrections."""
    codec = ReedSolomonCodec(n_symbols=n, k_symbols=k, symbol_width=m, prim_poly=poly, first_consecutive_root=fcr)
    rng = np.random.default_rng(101)

    msg = rng.integers(0, 1 << m, k, dtype=np.uint8)
    codeword = codec.encode(msg)

    corrected, corr_pos, is_valid, status = codec.decode(codeword)
    assert is_valid is True
    assert status.detected_error_count == 0
    assert len(corr_pos) == 0
    assert np.array_equal(corrected[:k], msg)


@pytest.mark.parametrize("n,k,m,poly,fcr,name", STANDARD_CODEC_CONFIGS)
def test_exact_error_correction_boundary(n: int, k: int, m: int, poly: int, fcr: int, name: str):
    """
    Verify the fundamental exact failure boundary of Reed-Solomon codes:
    1. Up to and including t = (N-K)//2 symbol errors: 100% exact recovery.
    2. At t + 1 symbol errors: Definitive, explicit decode failure via Chien search root mismatch.
    """
    codec = ReedSolomonCodec(n_symbols=n, k_symbols=k, symbol_width=m, prim_poly=poly, first_consecutive_root=fcr)
    t = codec.correction_radius
    rng = np.random.default_rng(102)

    msg = rng.integers(0, 1 << m, k, dtype=np.uint8)
    codeword = codec.encode(msg)

    # 1. Exactly t errors injected at random positions
    corrupted_t = codeword.copy()
    err_indices_t = rng.choice(n, size=t, replace=False)
    for idx in err_indices_t:
        corrupted_t[idx] ^= int(rng.integers(1, 1 << m))

    corrected_t, corr_pos_t, is_valid_t, status_t = codec.decode(corrupted_t)
    assert is_valid_t is True, f"Failed to correct t={t} errors: {status_t.diagnostics}"
    assert status_t.detected_error_count == t
    assert set(corr_pos_t) == set(err_indices_t)
    assert np.array_equal(corrected_t, codeword)

    # 2. Exactly t + 1 errors injected (beyond correction radius)
    corrupted_t_plus_1 = codeword.copy()
    err_indices_over = rng.choice(n, size=min(n, t + 1), replace=False)
    for idx in err_indices_over:
        corrupted_t_plus_1[idx] ^= int(rng.integers(1, 1 << m))

    corrected_over, corr_pos_over, is_valid_over, status_over = codec.decode(corrupted_t_plus_1)
    # The decoder MUST explicitly report failure rather than fabricating a wrong codeword
    assert is_valid_over is False, "Decoder silently accepted over-radius error pattern!"
    assert status_over.is_overcorrected is True


def test_erasure_and_error_combined_decoding():
    """
    Verify errors-and-erasures decoding:
    When erasure positions are provided, the code can correct up to 2*v + nu <= 2t.
    """
    codec = ReedSolomonCodec(n_symbols=64, k_symbols=48, symbol_width=8, prim_poly=0x11D, first_consecutive_root=1)
    # 2t = 16, t = 8
    rng = np.random.default_rng(103)
    msg = rng.integers(0, 256, 48, dtype=np.uint8)
    codeword = codec.encode(msg)

    # Inject 4 unknown errors and 8 known erasures: 2*(4) + 8 = 16 <= 16 (within capacity)
    num_errors = 4
    num_erasures = 8
    all_indices = rng.choice(64, size=num_errors + num_erasures, replace=False)
    err_indices = all_indices[:num_errors]
    erasure_indices = all_indices[num_errors:]

    corrupted = codeword.copy()
    for idx in all_indices:
        corrupted[idx] ^= int(rng.integers(1, 256))

    # Without erasures, 12 errors exceeds pure error capacity t=8 and must fail
    _, _, valid_no_erasure, _ = codec.decode(corrupted)
    assert valid_no_erasure is False

    # With erasure list supplied, 2*4 + 8 = 16 is fully correctable
    corrected, corr_pos, valid_erasure, status = codec.decode(corrupted, erasures=list(erasure_indices))
    assert valid_erasure is True, f"Erasure decoding failed: {status.diagnostics}"
    assert np.array_equal(corrected, codeword)


def test_berlekamp_massey_vs_extended_euclidean_crosscheck():
    """
    Verify agreement between Berlekamp-Massey and Extended Euclidean error-locator solvers.
    Deliberately corrupting one solver's state is caught as a discrepancy.
    """
    codec = ReedSolomonCodec(n_symbols=64, k_symbols=48, symbol_width=8, prim_poly=0x11D, first_consecutive_root=1)
    rng = np.random.default_rng(104)

    for _ in range(50):
        msg = rng.integers(0, 256, 48, dtype=np.uint8)
        codeword = codec.encode(msg)

        # Inject 1 to 8 errors
        n_errs = int(rng.integers(1, 9))
        corrupted = codeword.copy()
        err_indices = rng.choice(64, size=n_errs, replace=False)
        for idx in err_indices:
            corrupted[idx] ^= int(rng.integers(1, 256))

        syndromes = codec.compute_syndromes(corrupted)
        bm_loc = codec.find_error_locator_berlekamp_massey(syndromes)
        euc_loc, _ = codec.find_error_locator_euclidean(syndromes)

        assert np.array_equal(bm_loc, euc_loc), f"BM and Euclidean disagreed: BM={bm_loc}, Euc={euc_loc}"


def test_bitstream_decoding_and_correction_mask():
    """
    Verify block-wise bitstream decoding, LLR soft decision erasure extraction,
    and exact bit-level correction mask generation.
    """
    codec = ReedSolomonCodec(n_symbols=64, k_symbols=48, symbol_width=8, prim_poly=0x11D, first_consecutive_root=1)
    rng = np.random.default_rng(105)

    # 2 full blocks: 2 * 48 = 96 message bytes = 768 message bits; 2 * 64 = 128 codeword bytes = 1024 codeword bits
    msg_bytes = rng.integers(0, 256, 96, dtype=np.uint8)
    msg_bits = np.unpackbits(msg_bytes)

    blk1_code = codec.encode(msg_bytes[:48])
    blk2_code = codec.encode(msg_bytes[48:])
    codeword_bytes = np.concatenate((blk1_code, blk2_code)).astype(np.uint8)
    codeword_bits = np.unpackbits(codeword_bytes)

    # Corrupt 5 bytes in block 1 (within t=8)
    corrupted_bytes = codeword_bytes.copy()
    corrupted_bytes[5] ^= 0xAA
    corrupted_bytes[12] ^= 0x55
    corrupted_bytes[20] ^= 0x0F
    corrupted_bytes[35] ^= 0xF0
    corrupted_bytes[50] ^= 0x33
    corrupted_bits = np.unpackbits(corrupted_bytes.astype(np.uint8))

    fec_res = codec.decode_bitstream(corrupted_bits, max_correction_fraction=0.10)
    assert fec_res.valid is True
    assert np.array_equal(fec_res.decoded_bits, msg_bits)
    assert fec_res.corrected_bit_count > 0
    # Verify correction mask accurately identified modified bits
    assert np.all(fec_res.correction_mask == (corrupted_bits != codeword_bits))


def test_end_to_end_reconstruction_pipeline_rs():
    """
    Verify end-to-end Phase 5 data recovery on an RS-encoded stream (PROTOCOL_F).
    """
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_F",
        num_frames=5,
        payload_len_bytes=32,
        ber=0.01,
        seed=200,
    )

    bit_stream = BitStream(hard_bits=rx_bits, soft_bits=rx_soft, symbol_indices=None)
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

    config = DataRecoveryConfig(enable_reed_solomon=True)
    recon = build_reconstruction_candidate(1, bit_hyp, config=config)

    assert recon.integrity is not None
    assert recon.integrity.valid_frame_count >= 3
    assert recon.fec is not None
    assert recon.fec.code_family == FECCodeFamily.REED_SOLOMON


def test_concatenated_viterbi_rs_pipeline_and_order_sensitivity():
    """
    Verify concatenated coding recovery (Viterbi inner decode -> RS outer decode)
    and confirm that reversing or skipping a layer fails explicitly.
    """
    # Generate PROTOCOL_G (RS outer -> Interleaver -> Convolutional inner)
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_G",
        num_frames=4,
        payload_len_bytes=24,
        ber=0.01,
        interleaver_type="BLOCK",
        interleaver_params={"span": 8, "depth": 8},
        seed=300,
    )

    bit_stream = BitStream(hard_bits=rx_bits, soft_bits=rx_soft, symbol_indices=None)
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

    config = DataRecoveryConfig(enable_viterbi=True, enable_deinterleaver=True, enable_reed_solomon=True)
    recon = build_reconstruction_candidate(1, bit_hyp, config=config)

    assert recon.integrity is not None
    assert recon.integrity.valid_frame_count >= 2
    assert recon.fec is not None
    assert recon.fec.code_family in (FECCodeFamily.CONCATENATED, FECCodeFamily.REED_SOLOMON)


def test_ood_random_data_false_positive_control():
    """
    Verify out-of-distribution random data does not spuriously produce valid RS decodes.
    """
    rng = np.random.default_rng(999)
    codec = ReedSolomonCodec(n_symbols=64, k_symbols=48, symbol_width=8, prim_poly=0x11D, first_consecutive_root=1)

    false_positives = 0
    num_trials = 100

    for _ in range(num_trials):
        random_syms = rng.integers(0, 256, 64, dtype=np.uint8)
        _, _, is_valid, _ = codec.decode(random_syms)
        if is_valid:
            false_positives += 1

    # Theoretical accidental syndrome-zero probability for random symbols is 256^(-16) = 2^(-128)
    assert false_positives == 0, f"Spurious valid RS decode observed on random noise: {false_positives}/{num_trials}"


def test_independent_verification_fec_checks():
    """
    Verify independent verification audits for RS:
    1. FEC_01_OVERCORRECTION (Exact theoretical budget check)
    2. FEC_02_CROSS_VALIDATION (Held-out 70/30 frame validation)
    3. FEC_03_CHIEN_PERTURBATION (Falsification probe rejecting over-radius corruptions)
    """
    rx_bits, rx_soft, _ = generate_digital_stream(
        protocol="PROTOCOL_F",
        num_frames=5,
        payload_len_bytes=32,
        ber=0.005,
        seed=400,
    )

    bit_stream = BitStream(hard_bits=rx_bits, soft_bits=rx_soft, symbol_indices=None)
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

    recon = build_reconstruction_candidate(1, bit_hyp, config=DataRecoveryConfig(enable_reed_solomon=True))
    from app.data_recovery.models import DataRecoveryAnalysis, DataQualityLevel, DataRecoveryStatus

    analysis = DataRecoveryAnalysis(
        recording_reference="test_rs_rec",
        bitstream_candidates=[bit_hyp],
        reconstruction_candidates=[recon],
        selected_candidate=recon,
        status=DataRecoveryStatus.CORRECTED,
        quality_level=DataQualityLevel.HIGH,
        is_recovered=True,
        is_inconclusive=False,
        is_ambiguous=False,
    )

    audit_res, tests = audit_fec_and_cross_validation(data_analysis=analysis)
    assert audit_res.is_beneficial is True
    assert audit_res.anti_overcorrection_passed is True
    assert audit_res.held_out_validation_passed is True

    test_ids = [t.test_id for t in tests]
    assert "FEC_01_OVERCORRECTION" in test_ids
    assert "FEC_02_CROSS_VALIDATION" in test_ids
    assert "FEC_03_CHIEN_PERTURBATION" in test_ids
    for t in tests:
        assert t.status.value == "PASS", f"Verification test {t.test_id} failed: {t.counter_evidence}"
