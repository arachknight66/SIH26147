from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .concatenated_codes import (
    STANDARD_CONCATENATED_TOPOLOGIES,
    ConcatenatedCodeTopology,
    ConcatenatedDecodeResult,
    execute_concatenated_decode,
)
from .crc import search_crc_presets
from .fec_decode import decode_hamming_7_4, viterbi_decode
from .fec_models import STANDARD_FEC_CONFIGURATIONS
from .framing import detect_frame_boundaries, slice_frames
from .integrity import evaluate_multi_frame_integrity
from .interleaving import (
    deinterleave_block,
    deinterleave_convolutional,
    deinterleave_diagonal,
    deinterleave_pseudorandom,
    generate_interleaver_hypotheses,
)
from .ldpc import STANDARD_LDPC_SPECS, decode_ldpc_bitstream
from .line_coding import decode_line_code
from .models import (
    BitHypothesis,
    CorrectionQuality,
    DataQualityLevel,
    DataRecoveryConfig,
    EpistemicStatus,
    FECCodeFamily,
    FECDecodeResult,
    FECHypothesis,
    FrameCandidate,
    IntegrityResult,
    InterleaverHypothesis,
    InterleaverType,
    LineCodeType,
    PreambleCandidate,
    ReconstructionCandidate,
    ScramblerHypothesis,
    ScramblerType,
)
from .reed_solomon import ReedSolomonCodec
from .scrambling import STANDARD_LFSR_POLYNOMIALS, descramble_lfsr, evaluate_scrambler_hypotheses
from .synchronization import detect_preamble_candidates

@dataclass
class _PipelinePath:
    working_bits: np.ndarray
    preamble: PreambleCandidate | None
    frames: list[FrameCandidate]
    interleaver: InterleaverHypothesis | None
    scrambler: ScramblerHypothesis | None
    fec_hyp: FECHypothesis | None
    fec_decode: FECDecodeResult | None
    integrity: IntegrityResult
    is_periodic: bool
    score: float
    is_concatenated: bool = False
    inner_fec_decode: FECDecodeResult | None = None

def build_reconstruction_candidate(
    candidate_id: int,
    bit_hyp: BitHypothesis,
    config: DataRecoveryConfig | None = None,
) -> ReconstructionCandidate:
    """
    Execute full post-demodulation reconstruction pipeline for a single bit hypothesis.

    Parameters
    ----------
    candidate_id : int
    bit_hyp : BitHypothesis
    config : DataRecoveryConfig | None

    Returns
    -------
    ReconstructionCandidate
    """
    cfg = config or DataRecoveryConfig()
    raw_bits = bit_hyp.bitstream.hard_bits
    soft_bits = bit_hyp.bitstream.soft_bits

    if len(raw_bits) < 16 or np.all(raw_bits == 0) or np.all(raw_bits == 1):
        return ReconstructionCandidate(
            candidate_id=candidate_id,
            bit_hypothesis=bit_hyp,
            preamble=None,
            frames=(),
            line_code=None,
            interleaver=None,
            scrambler=None,
            fec=None,
            fec_decode=None,
            integrity=None,
            correction_quality=None,
            recovered_payload_bytes=b"",
            composite_score=0.0,
            complexity_penalty=0.0,
            data_quality_level=DataQualityLevel.VERY_LOW,
            epistemic_status=EpistemicStatus.UNKNOWN,
        )

    # 1. Line Code Application if specified
    if bit_hyp.line_code != LineCodeType.NONE:
        proc_bits, _ = decode_line_code(raw_bits, bit_hyp.line_code)
    else:
        proc_bits = raw_bits

    def _evaluate_bits(
        bits_to_eval: np.ndarray,
        interleaver_h: InterleaverHypothesis | None = None,
        scrambler_h: ScramblerHypothesis | None = None,
        fec_h: FECHypothesis | None = None,
        fec_dec: FECDecodeResult | None = None,
        inner_fec_dec: FECDecodeResult | None = None,
        is_concat: bool = False,
    ) -> _PipelinePath:
        preambles = detect_preamble_candidates(bits_to_eval)
        cand_preambles = preambles[:4] if preambles else [None]

        best_eval_path: _PipelinePath | None = None

        for p_cand in cand_preambles:
            boundaries, p_info = detect_frame_boundaries(bits_to_eval, preamble=p_cand)
            frames = slice_frames(bits_to_eval, boundaries)
            integ = evaluate_multi_frame_integrity(frames)
            is_per = bool(p_info.get("is_periodic", False))

            s_framing = 0.80 if (p_cand and is_per) else (0.40 if p_cand else 0.10)
            s_integrity = integ.crc_valid_fraction
            s_fec = 0.80 if (fec_dec and fec_dec.valid) else (0.50 if (fec_h and fec_h.code_family == FECCodeFamily.NONE) else 0.10)
            s_scram = scrambler_h.confidence if scrambler_h else 0.50
            s_inter = interleaver_h.confidence if (interleaver_h and interleaver_h.interleaver_type != InterleaverType.NONE) else 0.50

            path_score = 0.30 * s_framing + 0.30 * s_integrity + 0.15 * s_fec + 0.15 * s_scram + 0.10 * s_inter
            cand_p = _PipelinePath(
                working_bits=bits_to_eval,
                preamble=p_cand,
                frames=frames,
                interleaver=interleaver_h,
                scrambler=scrambler_h,
                fec_hyp=fec_h or STANDARD_FEC_CONFIGURATIONS[0],
                fec_decode=fec_dec,
                integrity=integ,
                is_periodic=is_per,
                score=path_score,
                is_concatenated=is_concat,
                inner_fec_decode=inner_fec_dec,
            )

            if best_eval_path is None or (integ.valid_frame_count, is_per, path_score) > (
                best_eval_path.integrity.valid_frame_count,
                best_eval_path.is_periodic,
                best_eval_path.score,
            ):
                best_eval_path = cand_p

        return best_eval_path

    paths: list[_PipelinePath] = []

    # 2. De-interleaving Hypotheses Search
    interleaver_hyps = generate_interleaver_hypotheses(proc_bits, config=cfg)
    # Take top candidate hypotheses (null plus any viable interleaver candidates)
    inter_candidates = [h for h in interleaver_hyps if h.valid or h.interleaver_type == InterleaverType.NONE][:4]
    if not any(h.interleaver_type == InterleaverType.NONE for h in inter_candidates):
        inter_candidates.insert(0, interleaver_hyps[0])

    rs_configs = [
        (64, 48, 8, 0x11D, 1, STANDARD_FEC_CONFIGURATIONS[10]),
        (128, 112, 8, 0x11D, 0, STANDARD_FEC_CONFIGURATIONS[9]),
        (204, 188, 8, 0x11D, 0, STANDARD_FEC_CONFIGURATIONS[8]),
        (255, 239, 8, 0x11D, 0, STANDARD_FEC_CONFIGURATIONS[7]),
        (255, 223, 8, 0x187, 112, STANDARD_FEC_CONFIGURATIONS[6]),
    ]

    for inter_h in inter_candidates:
        # Apply de-interleaving transform
        if inter_h.interleaver_type == InterleaverType.NONE:
            deint_bits = proc_bits
        elif inter_h.interleaver_type == InterleaverType.BLOCK:
            deint_bits = deinterleave_block(proc_bits, inter_h.parameters["span"], inter_h.parameters["depth"])
        elif inter_h.interleaver_type == InterleaverType.CONVOLUTIONAL:
            deint_raw = deinterleave_convolutional(proc_bits, inter_h.parameters["branches"], inter_h.parameters["delay_increment"])
            lat = inter_h.parameters.get("latency_bits", 0)
            deint_bits = deint_raw[lat:] if len(deint_raw) > lat + 32 else deint_raw
        elif inter_h.interleaver_type == InterleaverType.DIAGONAL:
            deint_bits = deinterleave_diagonal(proc_bits, inter_h.parameters["span"], inter_h.parameters["depth"], inter_h.parameters.get("step", 1))
        elif inter_h.interleaver_type == InterleaverType.PSEUDO_RANDOM:
            deint_bits = deinterleave_pseudorandom(proc_bits, inter_h.parameters["taps"], block_size=inter_h.parameters["block_size"])
        else:
            deint_bits = proc_bits

        # Path A: Direct Uncoded / Descrambled on de-interleaved bits
        paths.append(_evaluate_bits(deint_bits, interleaver_h=inter_h))

        # Path B: Candidate Descramblers
        for name, taps, deg in STANDARD_LFSR_POLYNOMIALS[: cfg.max_scrambler_candidates] if cfg.enable_descrambler else []:
            desc_bits = descramble_lfsr(deint_bits, taps)
            scram_h = ScramblerHypothesis(
                scrambler_type=ScramblerType.LFSR_SYNCHRONOUS,
                polynomial_name=name,
                polynomial_bits=taps,
                initial_state=tuple([1] * deg),
                period=(1 << deg) - 1,
                linear_complexity=deg,
                entropy_improvement=0.1,
                crc_improvement=1.0,
                confidence=0.85,
                valid=True,
            )
            paths.append(_evaluate_bits(desc_bits, interleaver_h=inter_h, scrambler_h=scram_h))

        # Path C: Convolutional FEC (Viterbi K=7, R=1/2)
        if cfg.enable_viterbi and len(deint_bits) >= 64:
            vit_res = viterbi_decode(
                deint_bits,
                soft_bits=soft_bits if inter_h.interleaver_type == InterleaverType.NONE else None,
                k=7,
                g1=0o133,
                g2=0o171,
                max_correction_fraction=cfg.max_correction_fraction,
            )
            if vit_res.valid:
                paths.append(
                    _evaluate_bits(
                        vit_res.decoded_bits,
                        interleaver_h=inter_h,
                        fec_h=STANDARD_FEC_CONFIGURATIONS[1],
                        fec_dec=vit_res,
                    )
                )
                # Path D: FEC Decoded + Descrambled (Protocol E)
                for name, taps, deg in STANDARD_LFSR_POLYNOMIALS[:1] if cfg.enable_descrambler else []:
                    desc_after_fec = descramble_lfsr(vit_res.decoded_bits, taps)
                    scram_h = ScramblerHypothesis(
                        scrambler_type=ScramblerType.LFSR_SYNCHRONOUS,
                        polynomial_name=name,
                        polynomial_bits=taps,
                        initial_state=tuple([1] * deg),
                        period=(1 << deg) - 1,
                        linear_complexity=deg,
                        entropy_improvement=0.1,
                        crc_improvement=1.0,
                        confidence=0.85,
                        valid=True,
                    )
                    paths.append(
                        _evaluate_bits(
                            desc_after_fec,
                            interleaver_h=inter_h,
                            scrambler_h=scram_h,
                            fec_h=STANDARD_FEC_CONFIGURATIONS[1],
                            fec_dec=vit_res,
                        )
                    )

        # Path E: Hamming(7,4) syndrome decoding
        if cfg.enable_hamming and len(deint_bits) >= 28:
            hamming_res = decode_hamming_7_4(deint_bits, max_correction_fraction=cfg.max_correction_fraction)
            if hamming_res.valid:
                paths.append(_evaluate_bits(hamming_res.decoded_bits, interleaver_h=inter_h, fec_h=STANDARD_FEC_CONFIGURATIONS[3], fec_dec=hamming_res))

        # Path F: Reed-Solomon Single-Code Decoding
        if cfg.enable_reed_solomon:
            for n_sym, k_sym, m_sym, poly, fcr, rs_hyp in rs_configs:
                if len(deint_bits) >= n_sym * m_sym:
                    rs_codec = ReedSolomonCodec(
                        n_symbols=n_sym,
                        k_symbols=k_sym,
                        symbol_width=m_sym,
                        prim_poly=poly,
                        first_consecutive_root=fcr,
                    )
                    rs_res = rs_codec.decode_bitstream(
                        deint_bits, max_correction_fraction=cfg.max_correction_fraction, soft_bits=soft_bits
                    )
                    if rs_res.valid:
                        paths.append(
                            _evaluate_bits(
                                rs_res.decoded_bits,
                                interleaver_h=inter_h,
                                fec_h=rs_hyp,
                                fec_dec=rs_res,
                            )
                        )

        # Path H: Standard Named LDPC Decoding
        if cfg.enable_ldpc:
            for ldpc_name, ldpc_spec in STANDARD_LDPC_SPECS.items():
                if len(deint_bits) >= ldpc_spec.n_bits:
                    ldpc_hyp = next(
                        (h for h in STANDARD_FEC_CONFIGURATIONS if h.code_family == FECCodeFamily.LDPC and ldpc_name in h.code_name),
                        None,
                    )
                    if ldpc_hyp is not None:
                        ldpc_res = decode_ldpc_bitstream(
                            deint_bits,
                            code_spec=ldpc_spec,
                            soft_bits=soft_bits if inter_h.interleaver_type == InterleaverType.NONE else None,
                            max_correction_fraction=cfg.max_correction_fraction,
                        )
                        if ldpc_res.valid:
                            paths.append(
                                _evaluate_bits(
                                    ldpc_res.decoded_bits,
                                    interleaver_h=inter_h,
                                    fec_h=ldpc_hyp,
                                    fec_dec=ldpc_res,
                                )
                            )

    # Path G: Standard Named Concatenated Topologies Search
    if cfg.enable_concatenated and cfg.enable_viterbi and cfg.enable_reed_solomon and len(proc_bits) >= 64:
        for topo in STANDARD_CONCATENATED_TOPOLOGIES:
            concat_res = execute_concatenated_decode(
                received_bits=proc_bits,
                topology=topo,
                soft_bits=soft_bits,
                enable_erasures=True,
                max_correction_fraction=cfg.max_correction_fraction,
            )
            if concat_res.valid:
                concat_fec_hyp = next(
                    (h for h in STANDARD_FEC_CONFIGURATIONS if h.code_family == FECCodeFamily.CONCATENATED and topo.name in h.code_name),
                    FECHypothesis(
                        code_family=FECCodeFamily.CONCATENATED,
                        code_name=topo.name,
                        rate=topo.outer_fec.rate * topo.inner_fec.rate,
                        constraint_length=topo.inner_fec.constraint_length,
                        generator_polynomials=topo.inner_fec.generator_polynomials,
                        block_size=topo.outer_fec.block_size,
                        confidence=topo.confidence_prior,
                        valid=True,
                    ),
                )
                paths.append(
                    _evaluate_bits(
                        concat_res.decoded_bits,
                        interleaver_h=topo.interleaver,
                        fec_h=concat_fec_hyp,
                        fec_dec=concat_res.outer_result,
                        inner_fec_dec=concat_res.inner_result,
                        is_concat=True,
                    )
                )

    # Separate standalone vs concatenated paths for Occam's selection gate
    standalone_paths = [p for p in paths if not p.is_concatenated]
    concatenated_paths = [p for p in paths if p.is_concatenated]

    def _path_sort_key(p: _PipelinePath) -> tuple[int, bool, float]:
        return (p.integrity.valid_frame_count, p.is_periodic, p.score)

    standalone_paths.sort(key=_path_sort_key, reverse=True)
    concatenated_paths.sort(key=_path_sort_key, reverse=True)

    best_standalone = standalone_paths[0] if standalone_paths else paths[0]
    best_concat = concatenated_paths[0] if concatenated_paths else None

    # Occam's Razor Selection Gate:
    # A concatenated path is only selected over standalone if it provides strictly superior structural evidence
    # (e.g. recovers valid frames when standalone has none, or improves CRC valid fraction by >= 0.15 margin)
    if best_concat is not None:
        standalone_crc = best_standalone.integrity.crc_valid_fraction
        concat_crc = best_concat.integrity.crc_valid_fraction
        standalone_valid_cnt = best_standalone.integrity.valid_frame_count
        concat_valid_cnt = best_concat.integrity.valid_frame_count

        if standalone_valid_cnt == 0 and concat_valid_cnt > 0 and concat_crc >= 0.50:
            best_path = best_concat
        elif concat_valid_cnt > standalone_valid_cnt and (concat_crc - standalone_crc >= 0.15):
            best_path = best_concat
        elif concat_crc >= 0.50 and not best_standalone.is_periodic and best_concat.is_periodic and concat_crc > standalone_crc:
            best_path = best_concat
        else:
            best_path = best_standalone
    else:
        best_path = best_standalone

    # Complexity penalty
    complexity = 0.0
    if bit_hyp.epistemic_status == EpistemicStatus.INFERRED or bit_hyp.phase_rotation_deg != 0.0:
        complexity += 0.03
    if bit_hyp.bit_offset != 0:
        complexity += 0.02
    if bit_hyp.line_code != LineCodeType.NONE:
        complexity += 0.05
    if best_path.interleaver and best_path.interleaver.interleaver_type != InterleaverType.NONE:
        complexity += 0.05
    if best_path.scrambler and best_path.scrambler.scrambler_type != ScramblerType.NONE:
        complexity += 0.05
    corr_frac = best_path.fec_decode.correction_fraction if best_path.fec_decode else 0.0
    if best_path.fec_hyp and best_path.fec_hyp.code_family != FECCodeFamily.NONE:
        complexity += 0.05 + 0.15 * corr_frac
    if best_path.is_concatenated:
        # Extra composite topology penalty reflecting stronger multi-layer claim
        complexity += 0.10
        if best_path.inner_fec_decode:
            inner_corr = best_path.inner_fec_decode.correction_fraction
            complexity += 0.05 + 0.15 * inner_corr

    comp_score = float(np.clip(best_path.score - (cfg.complexity_weight * complexity), 0.0, 1.0))

    # Payload extraction
    payload_chunks: list[bytes] = []
    for f in best_path.frames:
        if f.decoded_payload:
            payload_chunks.append(f.decoded_payload)
    recovered_payload = b"".join(payload_chunks)

    corr_bits_total = best_path.fec_decode.corrected_bit_count if best_path.fec_decode else 0
    mean_corr = float(corr_bits_total / max(1, len(best_path.frames)))

    corr_quality = CorrectionQuality(
        input_error_estimate=corr_frac,
        corrected_bits_total=corr_bits_total,
        mean_corrected_bits_per_frame=round(mean_corr, 2),
        median_correction_fraction=round(corr_frac, 4),
        decoder_metric=best_path.fec_decode.path_metric if best_path.fec_decode else 0.0,
        crc_before_fec_fraction=round(best_path.integrity.crc_valid_fraction, 4),
        crc_after_fec_fraction=round(best_path.integrity.crc_valid_fraction, 4),
        structural_consistency_score=1.0 if best_path.is_periodic else 0.5,
    )

    if best_path.integrity.valid_frame_count >= 1 and best_path.integrity.crc_valid_fraction >= 0.50:
        if best_path.fec_decode and best_path.fec_decode.corrected_bit_count > 0:
            epistemic = EpistemicStatus.CORRECTED
        else:
            epistemic = EpistemicStatus.INFERRED
        q_level = DataQualityLevel.HIGH if comp_score >= 0.60 else DataQualityLevel.MEDIUM
    elif best_path.preamble and best_path.is_periodic:
        epistemic = EpistemicStatus.INFERRED
        q_level = DataQualityLevel.MEDIUM if comp_score >= 0.50 else DataQualityLevel.LOW
    else:
        epistemic = EpistemicStatus.UNKNOWN
        q_level = DataQualityLevel.VERY_LOW

    return ReconstructionCandidate(
        candidate_id=candidate_id,
        bit_hypothesis=bit_hyp,
        preamble=best_path.preamble,
        frames=tuple(best_path.frames),
        line_code=bit_hyp.line_code if bit_hyp.line_code != LineCodeType.NONE else None,
        interleaver=best_path.interleaver if (best_path.interleaver and best_path.interleaver.interleaver_type != InterleaverType.NONE) else None,
        scrambler=best_path.scrambler if (best_path.scrambler and best_path.scrambler.scrambler_type != ScramblerType.NONE) else None,
        fec=best_path.fec_hyp,
        fec_decode=best_path.fec_decode,
        integrity=best_path.integrity,
        correction_quality=corr_quality,
        recovered_payload_bytes=recovered_payload,
        composite_score=round(comp_score, 3),
        complexity_penalty=round(complexity, 3),
        data_quality_level=q_level,
        epistemic_status=epistemic,
    )
