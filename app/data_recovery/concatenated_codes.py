from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from .fec_decode import encode_convolutional, viterbi_decode
from .interleaving import (
    deinterleave_block,
    deinterleave_convolutional,
    deinterleave_diagonal,
    deinterleave_pseudorandom,
    interleave_block,
    interleave_convolutional,
    interleave_diagonal,
    interleave_pseudorandom,
)
from .models import (
    FECCodeFamily,
    FECDecodeResult,
    FECHypothesis,
    InterleaverHypothesis,
    InterleaverType,
)
from .reed_solomon import ReedSolomonCodec


@dataclass(frozen=True)
class ConcatenatedCodeTopology:
    """
    Specification of a named concatenated FEC topology composing outer RS code,
    inter-code interleaver, and inner convolutional code.

    Physical Invariants:
    1. Transmitter ordering: Message -> RS Outer Encode -> Inter-code Interleave -> Convolutional Inner Encode.
    2. Receiver ordering: Channel -> Convolutional Inner Decode (Viterbi) -> Inter-code De-Interleave -> RS Outer Decode.
    3. The inter-code interleaver converts inner Viterbi residual error bursts into scattered symbol errors for the outer RS code.
    4. Viterbi low-confidence divergence regions are mapped through the de-interleaver to supply symbol erasures to the RS decoder.
    """
    name: str
    outer_fec: FECHypothesis
    inner_fec: FECHypothesis
    interleaver: InterleaverHypothesis
    confidence_prior: float = 0.35
    description: str = ""
    erasure_threshold: float = 0.30
    max_iterations: int = 1


@dataclass(frozen=True)
class ConcatenatedDecodeResult:
    """
    Multi-stage decode result capturing stage-level evidence, cross-stage erasure handoffs,
    and composite validity across the concatenated cascade.
    """
    input_bits: np.ndarray
    decoded_bits: np.ndarray
    inner_result: FECDecodeResult
    outer_result: FECDecodeResult
    erasure_positions: tuple[int, ...]
    used_erasure_count: int
    combined_correction_fraction: float
    iterations_run: int
    terminated_by_fixed_point: bool
    valid: bool
    topology: ConcatenatedCodeTopology
    code_family: FECCodeFamily = FECCodeFamily.CONCATENATED


# =============================================================================
# STANDARD NAMED CONCATENATED TOPOLOGIES
# =============================================================================

# 1. CCSDS Telemetry Concatenated Code (CCSDS 131.0-B-3)
CCSDS_CONCATENATED_TELEMETRY = ConcatenatedCodeTopology(
    name="CCSDS_CONCATENATED_TELEMETRY",
    outer_fec=FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_255_223_CCSDS",
        rate=223.0 / 255.0,
        constraint_length=None,
        generator_polynomials=(0x187,),
        block_size=255,
        assumptions=("N=255, K=223, 2t=32 (t=16), fcr=112, poly=0x187", "CCSDS outer code"),
        confidence=0.50,
        valid=True,
    ),
    inner_fec=FECHypothesis(
        code_family=FECCodeFamily.CONVOLUTIONAL,
        code_name="CONV_K7_R12_NASA",
        rate=0.5,
        constraint_length=7,
        generator_polynomials=(0o133, 0o171),
        block_size=None,
        assumptions=("K=7, R=1/2", "NASA standard inner code"),
        confidence=0.50,
        valid=True,
    ),
    interleaver=InterleaverHypothesis(
        interleaver_type=InterleaverType.BLOCK,
        parameters={"span": 8, "depth": 8, "block_size": 64},
        permutation_map=None,
        confidence=0.85,
        entropy_improvement=0.1,
        structural_improvement=1.0,
        valid=True,
    ),
    confidence_prior=0.35,
    description="CCSDS 131.0-B-3 Blue Book standard for deep-space and near-Earth telemetry",
    erasure_threshold=0.30,
)

# 2. DVB-S Broadcast Concatenated Code (ETSI EN 300 421)
DVB_S_CONCATENATED_BROADCAST = ConcatenatedCodeTopology(
    name="DVB_S_CONCATENATED_BROADCAST",
    outer_fec=FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_204_188_DVB_SHORTENED",
        rate=188.0 / 204.0,
        constraint_length=None,
        generator_polynomials=(0x11D,),
        block_size=204,
        assumptions=("N=204, K=188, 2t=16 (t=8), fcr=0, poly=0x11D", "DVB MPEG-TS outer code"),
        confidence=0.50,
        valid=True,
    ),
    inner_fec=FECHypothesis(
        code_family=FECCodeFamily.CONVOLUTIONAL,
        code_name="CONV_K7_R12_NASA",
        rate=0.5,
        constraint_length=7,
        generator_polynomials=(0o133, 0o171),
        block_size=None,
        assumptions=("K=7, R=1/2", "DVB-S standard inner code"),
        confidence=0.50,
        valid=True,
    ),
    interleaver=InterleaverHypothesis(
        interleaver_type=InterleaverType.CONVOLUTIONAL,
        parameters={"branches": 4, "delay_increment": 2, "latency_bits": 24},
        permutation_map=None,
        confidence=0.85,
        entropy_improvement=0.1,
        structural_improvement=1.0,
        valid=True,
    ),
    confidence_prior=0.35,
    description="ETSI EN 300 421 DVB-S standard for digital satellite broadcast",
    erasure_threshold=0.30,
)

# 3. Voyager Interstellar Telemetry Concatenated Code
VOYAGER_CONCATENATED_CLASSIC = ConcatenatedCodeTopology(
    name="VOYAGER_CONCATENATED_CLASSIC",
    outer_fec=FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_255_223_CCSDS",
        rate=223.0 / 255.0,
        constraint_length=None,
        generator_polynomials=(0x187,),
        block_size=255,
        assumptions=("N=255, K=223, 2t=32 (t=16), fcr=112, poly=0x187", "Voyager RS outer code"),
        confidence=0.50,
        valid=True,
    ),
    inner_fec=FECHypothesis(
        code_family=FECCodeFamily.CONVOLUTIONAL,
        code_name="CONV_K7_R12_NASA",
        rate=0.5,
        constraint_length=7,
        generator_polynomials=(0o133, 0o171),
        block_size=None,
        assumptions=("K=7, R=1/2", "Voyager inner convolutional code"),
        confidence=0.50,
        valid=True,
    ),
    interleaver=InterleaverHypothesis(
        interleaver_type=InterleaverType.BLOCK,
        parameters={"span": 16, "depth": 8, "block_size": 128},
        permutation_map=None,
        confidence=0.80,
        entropy_improvement=0.1,
        structural_improvement=1.0,
        valid=True,
    ),
    confidence_prior=0.30,
    description="NASA Deep Space Network Voyager interstellar telemetry profile",
    erasure_threshold=0.30,
)

# 4. Compact Low-Latency Packet Concatenated Code
COMPACT_CONCATENATED_PACKET = ConcatenatedCodeTopology(
    name="COMPACT_CONCATENATED_PACKET",
    outer_fec=FECHypothesis(
        code_family=FECCodeFamily.REED_SOLOMON,
        code_name="RS_64_48_COMPACT",
        rate=48.0 / 64.0,
        constraint_length=None,
        generator_polynomials=(0x11D,),
        block_size=64,
        assumptions=("N=64, K=48, 2t=16 (t=8), fcr=1, poly=0x11D", "Low latency RS outer code"),
        confidence=0.45,
        valid=True,
    ),
    inner_fec=FECHypothesis(
        code_family=FECCodeFamily.CONVOLUTIONAL,
        code_name="CONV_K7_R12_NASA",
        rate=0.5,
        constraint_length=7,
        generator_polynomials=(0o133, 0o171),
        block_size=None,
        assumptions=("K=7, R=1/2", "Standard inner code"),
        confidence=0.50,
        valid=True,
    ),
    interleaver=InterleaverHypothesis(
        interleaver_type=InterleaverType.BLOCK,
        parameters={"span": 8, "depth": 8, "block_size": 64},
        permutation_map=None,
        confidence=0.85,
        entropy_improvement=0.1,
        structural_improvement=1.0,
        valid=True,
    ),
    confidence_prior=0.35,
    description="Low-latency tactical/micro-satellite packet telemetry with short frame overhead",
    erasure_threshold=0.30,
)

STANDARD_CONCATENATED_TOPOLOGIES: list[ConcatenatedCodeTopology] = [
    CCSDS_CONCATENATED_TELEMETRY,
    DVB_S_CONCATENATED_BROADCAST,
    VOYAGER_CONCATENATED_CLASSIC,
    COMPACT_CONCATENATED_PACKET,
]


# =============================================================================
# CASCADE EXECUTION & CROSS-STAGE CONFIDENCE HANDOFF
# =============================================================================

def _apply_deinterleaver(bits: np.ndarray, interleaver_h: InterleaverHypothesis) -> np.ndarray:
    """Apply the specified inverse de-interleaver transform."""
    itype = interleaver_h.interleaver_type
    p = interleaver_h.parameters

    if itype == InterleaverType.NONE:
        return bits.copy()
    elif itype == InterleaverType.BLOCK:
        return deinterleave_block(bits, span=p["span"], depth=p["depth"])
    elif itype == InterleaverType.CONVOLUTIONAL:
        deint_raw = deinterleave_convolutional(bits, branches=p["branches"], delay_increment=p["delay_increment"])
        lat = p.get("latency_bits", 0)
        return deint_raw[lat:] if len(deint_raw) > lat + 32 else deint_raw
    elif itype == InterleaverType.DIAGONAL:
        return deinterleave_diagonal(bits, span=p["span"], depth=p["depth"], step=p.get("step", 1))
    elif itype == InterleaverType.PSEUDO_RANDOM:
        return deinterleave_pseudorandom(bits, taps=p["taps"], block_size=p["block_size"])
    else:
        return bits.copy()


def _apply_interleaver(bits: np.ndarray, interleaver_h: InterleaverHypothesis) -> np.ndarray:
    """Apply the forward interleaver transform."""
    itype = interleaver_h.interleaver_type
    p = interleaver_h.parameters

    if itype == InterleaverType.NONE:
        return bits.copy()
    elif itype == InterleaverType.BLOCK:
        return interleave_block(bits, span=p["span"], depth=p["depth"])
    elif itype == InterleaverType.CONVOLUTIONAL:
        return interleave_convolutional(bits, branches=p["branches"], delay_increment=p["delay_increment"])
    elif itype == InterleaverType.DIAGONAL:
        return interleave_diagonal(bits, span=p["span"], depth=p["depth"], step=p.get("step", 1))
    elif itype == InterleaverType.PSEUDO_RANDOM:
        return interleave_pseudorandom(bits, taps=p["taps"], block_size=p["block_size"])
    else:
        return bits.copy()


def _build_rs_codec_from_hyp(fec_hyp: FECHypothesis) -> ReedSolomonCodec:
    """Construct ReedSolomonCodec from FECHypothesis assumptions and polynomials."""
    n_syms = fec_hyp.block_size or 64
    k_syms = int(round(n_syms * fec_hyp.rate))
    poly = fec_hyp.generator_polynomials[0] if fec_hyp.generator_polynomials else 0x11D
    assumptions_str = "".join(fec_hyp.assumptions)
    fcr = 112 if "fcr=112" in assumptions_str else (1 if "fcr=1" in assumptions_str else 0)
    return ReedSolomonCodec(
        n_symbols=n_syms,
        k_symbols=k_syms,
        symbol_width=8,
        prim_poly=poly,
        first_consecutive_root=fcr,
    )


def execute_concatenated_decode(
    received_bits: np.ndarray,
    topology: ConcatenatedCodeTopology,
    soft_bits: np.ndarray | None = None,
    enable_erasures: bool = True,
    max_iterations: int = 1,
    max_correction_fraction: float = 0.10,
) -> ConcatenatedDecodeResult:
    """
    Execute complete 3-stage concatenated decode cascade:
    Stage 1: Inner Viterbi Convolutional Decode with soft/hard inputs and confidence extraction.
    Stage 2: Inter-code De-interleaving with spatial mapping of erasure coordinates.
    Stage 3: Outer Reed-Solomon Errors-and-Erasures Decode over GF(2^m).

    Parameters
    ----------
    received_bits : np.ndarray
        1D uint8 channel received bitstream.
    topology : ConcatenatedCodeTopology
        Named cascade configuration.
    soft_bits : np.ndarray | None
        Optional 1D float32 LLR soft decision stream.
    enable_erasures : bool
        Whether to hand off low-confidence Viterbi regions as symbol erasures to RS decoder.
    max_iterations : int
        Maximum number of iterative feedback passes (default 1 = single-pass).
    max_correction_fraction : float
        Maximum allowable bit alteration fraction per stage.

    Returns
    -------
    ConcatenatedDecodeResult
    """
    if len(received_bits) < 32:
        empty_res = FECDecodeResult(
            input_bits=received_bits,
            decoded_bits=received_bits.copy(),
            correction_mask=np.zeros(len(received_bits), dtype=bool),
            corrected_bit_count=0,
            correction_fraction=0.0,
            path_metric=0.0,
            normalized_path_metric=0.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.CONCATENATED,
            valid=False,
        )
        return ConcatenatedDecodeResult(
            input_bits=received_bits,
            decoded_bits=received_bits.copy(),
            inner_result=empty_res,
            outer_result=empty_res,
            erasure_positions=(),
            used_erasure_count=0,
            combined_correction_fraction=0.0,
            iterations_run=0,
            terminated_by_fixed_point=False,
            valid=False,
            topology=topology,
        )

    inner_k = topology.inner_fec.constraint_length or 7
    inner_g1 = topology.inner_fec.generator_polynomials[0] if len(topology.inner_fec.generator_polynomials) > 0 else 0o133
    inner_g2 = topology.inner_fec.generator_polynomials[1] if len(topology.inner_fec.generator_polynomials) > 1 else 0o171

    rs_codec = _build_rs_codec_from_hyp(topology.outer_fec)
    m = rs_codec.symbol_width
    n_syms = rs_codec.n_symbols
    k_syms = rs_codec.k_symbols
    bits_per_rs_block = n_syms * m
    parity_syms = rs_codec.parity_symbols

    curr_rx_bits = received_bits.copy()
    curr_soft_bits = soft_bits.copy() if soft_bits is not None else None

    iterations_run = 0
    terminated_by_fixed_point = False
    prev_decoded_bits: np.ndarray | None = None

    last_inner_res: FECDecodeResult | None = None
    last_outer_res: FECDecodeResult | None = None
    flagged_erasures_tuple: tuple[int, ...] = ()
    used_erasure_count = 0

    max_iter_bound = max(1, min(max_iterations, 5))

    for iter_idx in range(1, max_iter_bound + 1):
        iterations_run = iter_idx

        # ---------------------------------------------------------------------
        # Stage 1: Inner Viterbi Decode
        # ---------------------------------------------------------------------
        inner_res = viterbi_decode(
            input_bits=curr_rx_bits,
            soft_bits=curr_soft_bits,
            k=inner_k,
            g1=inner_g1,
            g2=inner_g2,
            max_correction_fraction=max_correction_fraction,
        )
        last_inner_res = inner_res

        if not inner_res.valid or len(inner_res.decoded_bits) < bits_per_rs_block:
            break

        # Compute per-bit confidence proxies for inner decoded stream
        # Discrepancy between re-encoded convolutional stream and received stream indicates burst regions
        reencoded = encode_convolutional(inner_res.decoded_bits, k=inner_k, g1=inner_g1, g2=inner_g2)
        min_c_len = min(len(reencoded), len(curr_rx_bits))
        raw_bit_errors = np.zeros(len(inner_res.decoded_bits), dtype=np.float32)

        if min_c_len >= 2:
            # Map paired channel errors back to information bits
            pair_errs = (reencoded[:min_c_len:2] != curr_rx_bits[:min_c_len:2]).astype(np.float32) + \
                        (reencoded[1:min_c_len:2] != curr_rx_bits[1:min_c_len:2]).astype(np.float32)
            n_pairs = min(len(pair_errs), len(raw_bit_errors))
            raw_bit_errors[:n_pairs] = pair_errs[:n_pairs]

        # ---------------------------------------------------------------------
        # Stage 2: Inter-code De-interleaving
        # ---------------------------------------------------------------------
        deint_bits = _apply_deinterleaver(inner_res.decoded_bits, topology.interleaver)
        deint_err_proxy = _apply_deinterleaver(
            (raw_bit_errors * 255.0).astype(np.uint8), topology.interleaver
        ).astype(np.float32) / 255.0

        # ---------------------------------------------------------------------
        # Stage 3: Outer Reed-Solomon Decode with Erasure Handoff
        # ---------------------------------------------------------------------
        n_blocks = len(deint_bits) // bits_per_rs_block
        if n_blocks == 0:
            break

        all_blocks_valid = True
        total_outer_corrected_bits = 0
        outer_decoded_chunks: list[np.ndarray] = []
        outer_corr_mask = np.zeros(len(deint_bits), dtype=bool)
        collected_erasures: list[int] = []

        for b_idx in range(n_blocks):
            blk_bits = deint_bits[b_idx * bits_per_rs_block : (b_idx + 1) * bits_per_rs_block]
            symbols = np.packbits(blk_bits.reshape(n_syms, m), axis=1).squeeze(-1)

            # Erasure extraction for this RS block
            block_erasures: list[int] = []
            if enable_erasures:
                blk_err_scores = deint_err_proxy[b_idx * bits_per_rs_block : (b_idx + 1) * bits_per_rs_block]
                sym_err_rates = np.mean(blk_err_scores.reshape(n_syms, m), axis=1)

                # Identify symbol positions with elevated error rates
                cand_erasures = np.where(sym_err_rates > topology.erasure_threshold)[0].tolist()
                # Limit to parity budget to ensure Singleton bound feasibility
                if 0 < len(cand_erasures) <= parity_syms:
                    block_erasures = cand_erasures
                    collected_erasures.extend([b_idx * n_syms + pos for pos in block_erasures])

            corr_syms, err_pos, blk_valid, status = rs_codec.decode(symbols, erasures=block_erasures)
            if not blk_valid:
                all_blocks_valid = False

            used_erasure_count += len(block_erasures)
            msg_syms = corr_syms[:k_syms]
            msg_bits = np.unpackbits(msg_syms.astype(np.uint8)[:, None], axis=1)[:, 8 - m :].ravel()
            outer_decoded_chunks.append(msg_bits)

            # Re-encode to compute correction mask on de-interleaved bits
            re_syms = rs_codec.encode(msg_syms)
            re_bits = np.unpackbits(re_syms.astype(np.uint8)[:, None], axis=1)[:, 8 - m :].ravel()
            blk_mask = (re_bits != blk_bits)
            outer_corr_mask[b_idx * bits_per_rs_block : (b_idx + 1) * bits_per_rs_block] = blk_mask
            total_outer_corrected_bits += int(np.sum(blk_mask))

        curr_decoded_outer = np.concatenate(outer_decoded_chunks) if outer_decoded_chunks else np.array([], dtype=np.uint8)
        outer_corr_frac = float(total_outer_corrected_bits / (n_blocks * bits_per_rs_block)) if n_blocks > 0 else 0.0
        outer_is_over = bool(outer_corr_frac > max_correction_fraction or not all_blocks_valid)

        outer_res = FECDecodeResult(
            input_bits=deint_bits,
            decoded_bits=curr_decoded_outer,
            correction_mask=outer_corr_mask,
            corrected_bit_count=total_outer_corrected_bits,
            correction_fraction=round(outer_corr_frac, 4),
            path_metric=float(total_outer_corrected_bits),
            normalized_path_metric=round(outer_corr_frac, 4),
            is_overcorrected=outer_is_over,
            code_family=FECCodeFamily.REED_SOLOMON,
            valid=bool(all_blocks_valid and not outer_is_over),
        )
        last_outer_res = outer_res
        flagged_erasures_tuple = tuple(collected_erasures)

        # Check for Fixed-Point Termination in Iterative Mode
        if prev_decoded_bits is not None and np.array_equal(prev_decoded_bits, curr_decoded_outer):
            terminated_by_fixed_point = True
            break

        prev_decoded_bits = curr_decoded_outer

        # Iterative feedback refinement: re-encode outer message -> interleave -> update inner inputs
        if iter_idx < max_iter_bound and all_blocks_valid:
            # Reconstruct clean interleaved codeword bits from outer message
            full_code_syms_list: list[np.ndarray] = []
            for b_idx in range(n_blocks):
                m_chunk = curr_decoded_outer[b_idx * k_syms * m : (b_idx + 1) * k_syms * m]
                m_syms = np.packbits(m_chunk.reshape(k_syms, m), axis=1).squeeze(-1)
                full_code_syms_list.append(rs_codec.encode(m_syms))
            all_code_syms = np.concatenate(full_code_syms_list)
            all_code_bits = np.unpackbits(all_code_syms.astype(np.uint8)[:, None], axis=1)[:, 8 - m :].ravel()
            reinterleaved = _apply_interleaver(all_code_bits, topology.interleaver)

            # Convolutional re-encode to synthesize high-confidence soft guidance
            re_conv = encode_convolutional(reinterleaved, k=inner_k, g1=inner_g1, g2=inner_g2)
            c_len = min(len(re_conv), len(curr_rx_bits))
            # Blend feedback with original received channel bits
            curr_rx_bits[:c_len] = re_conv[:c_len]
            if curr_soft_bits is not None and len(curr_soft_bits) >= c_len:
                curr_soft_bits[:c_len] = np.where(re_conv[:c_len] == 1, 2.0, -2.0)

    # -------------------------------------------------------------------------
    # Composite Result Synthesis
    # -------------------------------------------------------------------------
    if last_inner_res is None or last_outer_res is None:
        empty_res = FECDecodeResult(
            input_bits=received_bits,
            decoded_bits=received_bits.copy(),
            correction_mask=np.zeros(len(received_bits), dtype=bool),
            corrected_bit_count=0,
            correction_fraction=0.0,
            path_metric=0.0,
            normalized_path_metric=0.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.CONCATENATED,
            valid=False,
        )
        return ConcatenatedDecodeResult(
            input_bits=received_bits,
            decoded_bits=received_bits.copy(),
            inner_result=empty_res,
            outer_result=empty_res,
            erasure_positions=(),
            used_erasure_count=0,
            combined_correction_fraction=0.0,
            iterations_run=iterations_run,
            terminated_by_fixed_point=terminated_by_fixed_point,
            valid=False,
            topology=topology,
        )

    # Honest combined correction fraction across both stages
    inner_rho = last_inner_res.correction_fraction
    outer_rho = last_outer_res.correction_fraction
    combined_rho = round(1.0 - (1.0 - inner_rho) * (1.0 - outer_rho), 4)

    is_overall_valid = bool(
        last_inner_res.valid
        and last_outer_res.valid
        and not last_inner_res.is_overcorrected
        and not last_outer_res.is_overcorrected
    )

    return ConcatenatedDecodeResult(
        input_bits=received_bits,
        decoded_bits=last_outer_res.decoded_bits,
        inner_result=last_inner_res,
        outer_result=last_outer_res,
        erasure_positions=flagged_erasures_tuple,
        used_erasure_count=used_erasure_count,
        combined_correction_fraction=combined_rho,
        iterations_run=iterations_run,
        terminated_by_fixed_point=terminated_by_fixed_point,
        valid=is_overall_valid,
        topology=topology,
    )


# =============================================================================
# TOPOLOGY ORDERING FALSIFICATION PROBE
# =============================================================================

def execute_reversed_order_decode(
    received_bits: np.ndarray,
    topology: ConcatenatedCodeTopology,
    max_correction_fraction: float = 0.10,
) -> ConcatenatedDecodeResult:
    """
    Deliberately execute the reversed decode sequence:
    Stage 1: Outer RS decode directly on raw channel stream.
    Stage 2: De-interleaving.
    Stage 3: Inner Viterbi decode.

    This function serves exclusively as a negative control / falsification probe
    to confirm that the recovered structure collapses under reverse order.
    """
    rs_codec = _build_rs_codec_from_hyp(topology.outer_fec)
    m = rs_codec.symbol_width
    n_syms = rs_codec.n_symbols
    bits_per_rs_block = n_syms * m

    if len(received_bits) < bits_per_rs_block:
        empty_res = FECDecodeResult(
            input_bits=received_bits,
            decoded_bits=received_bits.copy(),
            correction_mask=np.zeros(len(received_bits), dtype=bool),
            corrected_bit_count=0,
            correction_fraction=0.0,
            path_metric=0.0,
            normalized_path_metric=0.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.CONCATENATED,
            valid=False,
        )
        return ConcatenatedDecodeResult(
            input_bits=received_bits,
            decoded_bits=received_bits.copy(),
            inner_result=empty_res,
            outer_result=empty_res,
            erasure_positions=(),
            used_erasure_count=0,
            combined_correction_fraction=0.0,
            iterations_run=1,
            terminated_by_fixed_point=False,
            valid=False,
            topology=topology,
        )

    # 1. Reverse Stage 1: RS decode on channel bits
    rs_res = rs_codec.decode_bitstream(received_bits, max_correction_fraction=max_correction_fraction)

    # 2. Reverse Stage 2: De-interleave RS decoded stream
    deint_bits = _apply_deinterleaver(rs_res.decoded_bits, topology.interleaver)

    # 3. Reverse Stage 3: Inner Viterbi decode
    inner_k = topology.inner_fec.constraint_length or 7
    inner_g1 = topology.inner_fec.generator_polynomials[0] if len(topology.inner_fec.generator_polynomials) > 0 else 0o133
    inner_g2 = topology.inner_fec.generator_polynomials[1] if len(topology.inner_fec.generator_polynomials) > 1 else 0o171

    vit_res = viterbi_decode(
        input_bits=deint_bits,
        k=inner_k,
        g1=inner_g1,
        g2=inner_g2,
        max_correction_fraction=max_correction_fraction,
    )

    combined_rho = round(1.0 - (1.0 - rs_res.correction_fraction) * (1.0 - vit_res.correction_fraction), 4)
    is_valid = bool(rs_res.valid and vit_res.valid and not rs_res.is_overcorrected and not vit_res.is_overcorrected)

    return ConcatenatedDecodeResult(
        input_bits=received_bits,
        decoded_bits=vit_res.decoded_bits,
        inner_result=vit_res,
        outer_result=rs_res,
        erasure_positions=(),
        used_erasure_count=0,
        combined_correction_fraction=combined_rho,
        iterations_run=1,
        terminated_by_fixed_point=False,
        valid=is_valid,
        topology=topology,
    )
