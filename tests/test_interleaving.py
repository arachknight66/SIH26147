import numpy as np
import pytest
from app.data_recovery.interleaving import (
    STANDARD_BLOCK_CONFIGS,
    STANDARD_CONVOLUTIONAL_CONFIGS,
    STANDARD_DIAGONAL_CONFIGS,
    deinterleave_block,
    deinterleave_convolutional,
    deinterleave_diagonal,
    deinterleave_pseudorandom,
    generate_interleaver_hypotheses,
    get_convolutional_latency,
    interleave_block,
    interleave_convolutional,
    interleave_diagonal,
    interleave_pseudorandom,
)
from app.data_recovery.models import (
    BitHypothesis,
    BitOrder,
    BitPolarity,
    BitStream,
    DataRecoveryAnalysis,
    DataRecoveryConfig,
    DataRecoveryStatus,
    EpistemicStatus,
    InterleaverHypothesis,
    InterleaverType,
    LineCodeType,
)
from app.data_recovery.preprocessing import compute_digital_statistics
from app.data_recovery.ranking import rank_and_select_reconstructions
from app.data_recovery.reconstruction import build_reconstruction_candidate
from app.data_recovery.scrambling import STANDARD_LFSR_POLYNOMIALS
from app.verification.interleaving_checks import audit_interleaver_and_falsification
from scripts.generate_digital_dataset import generate_digital_stream


# =============================================================================
# 1. EXACT ROUND-TRIP TESTS FOR ALL FOUR TRANSFORM FAMILIES
# =============================================================================

@pytest.mark.parametrize("span,depth", STANDARD_BLOCK_CONFIGS)
def test_block_interleaver_roundtrip(span: int, depth: int):
    """Test exact bit recovery for block interleaver across diverse parameters and lengths."""
    rng = np.random.default_rng(42)
    # Test lengths with integer multiple of blocks and with remainder bits
    for n in (span * depth * 3, span * depth * 3 + 7):
        bits = rng.integers(0, 2, n, dtype=np.uint8)
        interleaved = interleave_block(bits, span=span, depth=depth)
        deinterleaved = deinterleave_block(interleaved, span=span, depth=depth)
        assert np.array_equal(bits, deinterleaved), f"Failed for span={span}, depth={depth}, length={n}"


@pytest.mark.parametrize("branches,delay_inc", STANDARD_CONVOLUTIONAL_CONFIGS)
def test_convolutional_interleaver_roundtrip(branches: int, delay_inc: int):
    """Test exact bit recovery for Ramsey/Forney convolutional cross-interleaver."""
    rng = np.random.default_rng(42)
    lat = get_convolutional_latency(branches, delay_inc)
    n = lat * 4 + 100
    bits = rng.integers(0, 2, n, dtype=np.uint8)

    interleaved = interleave_convolutional(bits, branches=branches, delay_increment=delay_inc)
    deinterleaved = deinterleave_convolutional(interleaved, branches=branches, delay_increment=delay_inc)

    # After exact end-to-end latency Delta = M*(M-1)*D bits, bits match exactly
    assert np.array_equal(
        bits[: n - lat],
        deinterleaved[lat:n],
    ), f"Failed complementary latency alignment for branches={branches}, delay={delay_inc}"


@pytest.mark.parametrize("span,depth,step", STANDARD_DIAGONAL_CONFIGS)
def test_diagonal_interleaver_roundtrip(span: int, depth: int, step: int):
    """Test exact bit recovery for diagonal matrix interleaver."""
    rng = np.random.default_rng(42)
    for n in (span * depth * 2, span * depth * 2 + 5):
        bits = rng.integers(0, 2, n, dtype=np.uint8)
        interleaved = interleave_diagonal(bits, span=span, depth=depth, step=step)
        deinterleaved = deinterleave_diagonal(interleaved, span=span, depth=depth, step=step)
        assert np.array_equal(bits, deinterleaved), f"Failed for span={span}, depth={depth}, step={step}"


@pytest.mark.parametrize("poly_name,taps,deg", STANDARD_LFSR_POLYNOMIALS[:3])
@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_pseudorandom_interleaver_roundtrip(poly_name: str, taps: tuple[int, ...], deg: int, block_size: int):
    """Test exact bit recovery for deterministic LFSR-driven pseudo-random interleaver."""
    rng = np.random.default_rng(42)
    for n in (block_size * 2, block_size * 2 + 11):
        bits = rng.integers(0, 2, n, dtype=np.uint8)
        interleaved = interleave_pseudorandom(bits, taps=taps, block_size=block_size)
        deinterleaved = deinterleave_pseudorandom(interleaved, taps=taps, block_size=block_size)
        assert np.array_equal(bits, deinterleaved), f"Failed for {poly_name}, block_size={block_size}"


# =============================================================================
# 2. BURST-ERROR-CONVERSION MECHANISM TEST
# =============================================================================

def test_burst_error_dispersion_mechanism():
    """
    Verify the fundamental physical purpose of interleaving:
    A contiguous burst error injected into an interleaved stream is scattered
    into isolated single-bit errors in the de-interleaved domain.
    """
    rng = np.random.default_rng(100)
    span = 8
    depth = 8
    block_size = span * depth
    n = block_size * 4  # 256 bits

    clean_bits = rng.integers(0, 2, n, dtype=np.uint8)
    interleaved = interleave_block(clean_bits, span=span, depth=span)

    # Inject a contiguous 6-bit burst error into the interleaved stream
    burst_len = 6
    burst_start = block_size // 2
    corrupted = interleaved.copy()
    corrupted[burst_start : burst_start + burst_len] ^= 1

    # In the interleaved (channel) domain, the error pattern has a contiguous run of length 6
    channel_error_mask = (interleaved ^ corrupted)
    assert np.sum(channel_error_mask) == burst_len
    # Error bits are consecutive in channel
    error_indices_channel = np.where(channel_error_mask == 1)[0]
    assert np.all(np.diff(error_indices_channel) == 1)

    # De-interleave the corrupted stream
    deinterleaved = deinterleave_block(corrupted, span=span, depth=depth)
    deint_error_mask = (clean_bits ^ deinterleaved)
    assert np.sum(deint_error_mask) == burst_len

    # In the de-interleaved domain, errors MUST be scattered (separated by at least depth=8 bits)
    error_indices_deint = np.sort(np.where(deint_error_mask == 1)[0])
    spacings = np.diff(error_indices_deint)
    # Consecutive errors in a column become separated by depth when read row-wise
    assert np.all(spacings >= depth - 1), f"Errors were not dispersed: spacings = {spacings}"

    # Verify run length of error bits (1s) dropped to 1.0 (isolated single-bit errors)
    diffs_deint = np.diff(np.concatenate(([0], deint_error_mask, [0])))
    run_starts = np.where(diffs_deint == 1)[0]
    run_ends = np.where(diffs_deint == -1)[0]
    error_run_lengths = run_ends - run_starts
    assert np.all(error_run_lengths == 1), f"Errors were not isolated: {error_run_lengths}"


# =============================================================================
# 3. END-TO-END RECONSTRUCTION TESTS
# =============================================================================

def test_end_to_end_block_interleaved_recovery():
    """
    End-to-end reconstruction: Generate block-interleaved Protocol A stream.
    Reconstruction pipeline must identify the block interleaver, de-interleave,
    and recover valid frames and valid CRCs.
    """
    span = 8
    depth = 8
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_A",
        num_frames=4,
        payload_len_bytes=32,
        interleaver_type="BLOCK",
        interleaver_params={"span": span, "depth": depth},
        seed=123,
    )

    bs = BitStream(
        hard_bits=rx_bits,
        soft_bits=rx_soft,
        symbol_indices=None,
        bit_order=BitOrder.UNKNOWN,
        bit_polarity=BitPolarity.NORMAL,
    )
    bit_hyp = BitHypothesis(
        hypothesis_id=0,
        bitstream=bs,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
    )

    cand = build_reconstruction_candidate(0, bit_hyp, config=DataRecoveryConfig(enable_deinterleaver=True))

    assert cand.interleaver is not None
    assert cand.interleaver.interleaver_type == InterleaverType.BLOCK
    assert cand.interleaver.parameters["span"] == span
    assert cand.interleaver.parameters["depth"] == depth
    assert cand.interleaver.valid is True
    assert cand.interleaver.confidence >= 0.80

    # Downstream CRC and framing must be restored
    assert cand.integrity is not None
    assert cand.integrity.valid_frame_count >= 3
    assert cand.integrity.crc_valid_fraction >= 0.75
    assert cand.preamble is not None


def test_end_to_end_diagonal_interleaved_recovery():
    """End-to-end reconstruction: Generate diagonal-interleaved Protocol A stream and verify recovery."""
    span = 8
    depth = 8
    step = 1
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_A",
        num_frames=4,
        payload_len_bytes=32,
        interleaver_type="DIAGONAL",
        interleaver_params={"span": span, "depth": depth, "step": step},
        seed=456,
    )

    bs = BitStream(
        hard_bits=rx_bits,
        soft_bits=rx_soft,
        symbol_indices=None,
        bit_order=BitOrder.UNKNOWN,
        bit_polarity=BitPolarity.NORMAL,
    )
    bit_hyp = BitHypothesis(
        hypothesis_id=0,
        bitstream=bs,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
    )

    cand = build_reconstruction_candidate(0, bit_hyp, config=DataRecoveryConfig(enable_deinterleaver=True))

    assert cand.interleaver is not None
    assert cand.interleaver.interleaver_type == InterleaverType.DIAGONAL
    assert cand.interleaver.valid is True
    assert cand.integrity is not None
    assert cand.integrity.valid_frame_count >= 3


# =============================================================================
# 4. NULL-HYPOTHESIS DISCRIMINATION TEST (FALSE-POSITIVE CONTROL)
# =============================================================================

def test_null_hypothesis_discrimination():
    """
    False-positive control: On a standard non-interleaved stream, the search must
    retain the null hypothesis (InterleaverType.NONE) and reject spurious interleavers.
    """
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_A",
        num_frames=5,
        payload_len_bytes=32,
        interleaver_type=None,
        seed=789,
    )

    hyps = generate_interleaver_hypotheses(rx_bits)
    top_hyp = hyps[0]

    assert top_hyp.interleaver_type == InterleaverType.NONE
    assert top_hyp.valid is True
    assert top_hyp.confidence >= 0.80

    # No non-null interleaver should be promoted to valid=True when data is already clean
    non_null_valid = [h for h in hyps if h.interleaver_type != InterleaverType.NONE and h.valid]
    assert len(non_null_valid) == 0, f"Spurious interleaver promoted: {non_null_valid}"


# =============================================================================
# 5. EPISTEMIC AMBIGUITY & SINGLE-METRIC CAPPING TEST
# =============================================================================

def test_ambiguity_and_single_metric_capping():
    """
    Epistemic contract: An interleaver candidate that provides only a single weak
    signal (or accidental match) without dual corroboration must remain capped at AMBIGUOUS / valid=False.
    """
    rng = np.random.default_rng(999)
    # Random unmodulated/unstructured bits
    random_bits = rng.integers(0, 2, 512, dtype=np.uint8)

    hyps = generate_interleaver_hypotheses(random_bits)
    for h in hyps:
        if h.interleaver_type != InterleaverType.NONE:
            assert h.valid is False, f"Uncorroborated candidate on random bits was marked valid: {h}"
            assert h.confidence <= 0.55, f"Uncorroborated candidate had high confidence: {h.confidence}"


# =============================================================================
# 6. VERIFICATION FALSIFICATION & CROSS-VALIDATION AUDIT TESTS
# =============================================================================

def test_verification_perturbation_and_held_out_audit():
    """
    Verification layer: Verify that audit_interleaver_and_falsification passes for a genuine
    recovered interleaver (showing parameter collapse on perturbation and passing 70/30 held-out).
    """
    span = 8
    depth = 8
    rx_bits, rx_soft, manifest = generate_digital_stream(
        protocol="PROTOCOL_A",
        num_frames=5,
        payload_len_bytes=32,
        interleaver_type="BLOCK",
        interleaver_params={"span": span, "depth": depth},
        seed=321,
    )

    bs = BitStream(
        hard_bits=rx_bits,
        soft_bits=rx_soft,
        symbol_indices=None,
        bit_order=BitOrder.UNKNOWN,
        bit_polarity=BitPolarity.NORMAL,
    )
    bit_hyp = BitHypothesis(
        hypothesis_id=0,
        bitstream=bs,
        phase_rotation_deg=0.0,
        polarity=BitPolarity.NORMAL,
        line_code=LineCodeType.NONE,
        bit_order=BitOrder.MSB_FIRST,
        bit_offset=0,
    )

    cand = build_reconstruction_candidate(0, bit_hyp)
    ranked, sel, status, q_level, handoff, diag = rank_and_select_reconstructions([cand])

    assert sel is not None
    assert handoff is not None
    assert handoff.interleaver_parameters["interleaver_type"] == "block"

    data_analysis = DataRecoveryAnalysis(
        recording_reference="test_interleaved_rec",
        bitstream_candidates=[bit_hyp],
        reconstruction_candidates=[cand],
        selected_candidate=sel,
        status=status,
        quality_level=q_level,
        is_recovered=True,
        is_inconclusive=False,
        is_ambiguous=False,
        phase6_handoff=handoff,
    )

    audit_res, tests = audit_interleaver_and_falsification(data_analysis=data_analysis, handoff=handoff)

    assert audit_res.is_verified is True
    assert audit_res.parameter_perturbation_passed is True
    assert audit_res.held_out_validation_passed is True

    test_ids = {t.test_id: t for t in tests}
    assert "INTER_01_PERTURBATION" in test_ids
    assert test_ids["INTER_01_PERTURBATION"].status.value == "PASS"
    assert "INTER_02_CROSS_VALIDATION" in test_ids
    assert test_ids["INTER_02_CROSS_VALIDATION"].status.value == "PASS"
