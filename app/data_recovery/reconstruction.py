from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .crc import search_crc_presets
from .fec_decode import decode_hamming_7_4, viterbi_decode
from .fec_models import STANDARD_FEC_CONFIGURATIONS
from .framing import detect_frame_boundaries, slice_frames
from .integrity import evaluate_multi_frame_integrity
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
    LineCodeType,
    PreambleCandidate,
    ReconstructionCandidate,
    ScramblerHypothesis,
    ScramblerType,
)
from .scrambling import STANDARD_LFSR_POLYNOMIALS, descramble_lfsr, evaluate_scrambler_hypotheses
from .synchronization import detect_preamble_candidates

@dataclass
class _PipelinePath:
    working_bits: np.ndarray
    preamble: PreambleCandidate | None
    frames: list[FrameCandidate]
    scrambler: ScramblerHypothesis | None
    fec_hyp: FECHypothesis | None
    fec_decode: FECDecodeResult | None
    integrity: IntegrityResult
    is_periodic: bool
    score: float

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
        scrambler_h: ScramblerHypothesis | None = None,
        fec_h: FECHypothesis | None = None,
        fec_dec: FECDecodeResult | None = None,
    ) -> _PipelinePath:
        preambles = detect_preamble_candidates(bits_to_eval)
        best_p = preambles[0] if preambles else None
        boundaries, p_info = detect_frame_boundaries(bits_to_eval, preamble=best_p)
        frames = slice_frames(bits_to_eval, boundaries)
        integ = evaluate_multi_frame_integrity(frames)
        is_per = bool(p_info.get("is_periodic", False))

        s_framing = 0.80 if (best_p and is_per) else (0.40 if best_p else 0.10)
        s_integrity = integ.crc_valid_fraction
        s_fec = 0.80 if (fec_dec and fec_dec.valid) else (0.50 if (fec_h and fec_h.code_family == FECCodeFamily.NONE) else 0.10)
        s_scram = scrambler_h.confidence if scrambler_h else 0.50

        # Score path
        path_score = 0.35 * s_framing + 0.35 * s_integrity + 0.15 * s_fec + 0.15 * s_scram
        return _PipelinePath(
            working_bits=bits_to_eval,
            preamble=best_p,
            frames=frames,
            scrambler=scrambler_h,
            fec_hyp=fec_h or STANDARD_FEC_CONFIGURATIONS[0],
            fec_decode=fec_dec,
            integrity=integ,
            is_periodic=is_per,
            score=path_score,
        )

    paths: list[_PipelinePath] = []

    # Path 1: Direct Raw/Uncoded
    paths.append(_evaluate_bits(proc_bits))

    # Path 2: Candidate Descramblers
    for name, taps, deg in STANDARD_LFSR_POLYNOMIALS[:2]:  # Test ITU-V29 (7, 4) and IEEE 802.11
        desc_bits = descramble_lfsr(proc_bits, taps)
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
        paths.append(_evaluate_bits(desc_bits, scrambler_h=scram_h))

    # Path 3: Convolutional FEC (Viterbi K=7, R=1/2)
    if len(proc_bits) >= 64:
        vit_res = viterbi_decode(
            proc_bits,
            soft_bits=soft_bits,
            k=7,
            g1=0o133,
            g2=0o171,
            max_correction_fraction=cfg.max_correction_fraction,
        )
        if vit_res.valid:
            paths.append(
                _evaluate_bits(
                    vit_res.decoded_bits,
                    fec_h=STANDARD_FEC_CONFIGURATIONS[1],
                    fec_dec=vit_res,
                )
            )
            # Path 4: FEC Decoded + Descrambled (Protocol E)
            for name, taps, deg in STANDARD_LFSR_POLYNOMIALS[:1]:
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
                        scrambler_h=scram_h,
                        fec_h=STANDARD_FEC_CONFIGURATIONS[1],
                        fec_dec=vit_res,
                    )
                )

    # Sort paths by: CRC valid count -> is_periodic -> score
    def _path_sort_key(p: _PipelinePath) -> tuple[int, bool, float]:
        return (p.integrity.valid_frame_count, p.is_periodic, p.score)

    paths.sort(key=_path_sort_key, reverse=True)
    best_path = paths[0]

    # Complexity penalty
    complexity = 0.0
    if bit_hyp.polarity == EpistemicStatus.INFERRED or bit_hyp.phase_rotation_deg != 0.0:
        complexity += 0.03
    if bit_hyp.bit_offset != 0:
        complexity += 0.02
    if bit_hyp.line_code != LineCodeType.NONE:
        complexity += 0.05
    if best_path.scrambler and best_path.scrambler.scrambler_type != ScramblerType.NONE:
        complexity += 0.05
    corr_frac = best_path.fec_decode.correction_fraction if best_path.fec_decode else 0.0
    if best_path.fec_hyp and best_path.fec_hyp.code_family != FECCodeFamily.NONE:
        complexity += 0.05 + 0.15 * corr_frac

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
