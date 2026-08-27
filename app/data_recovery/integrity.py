from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from .crc import CRC_PRESETS, CRCParam, evaluate_frame_crc
from .models import CRCResult, FrameCandidate, IntegrityResult

def _binomial_tail(n: int, k: int, p: float) -> float:
    """Calculate P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    prob_sum = 0.0
    for j in range(k, n + 1):
        try:
            coeff = math.comb(n, j)
            prob_sum += coeff * (p ** j) * ((1.0 - p) ** (n - j))
        except (ValueError, OverflowError):
            break
    return float(min(1.0, max(0.0, prob_sum)))

def evaluate_multi_frame_integrity(
    frames: list[FrameCandidate],
    candidate_crc: CRCParam | None = None,
) -> IntegrityResult:
    """
    Validate CRC consistency across multiple frames and compute multi-hypothesis p-value.

    Parameters
    ----------
    frames : list[FrameCandidate]
    candidate_crc : CRCParam | None

    Returns
    -------
    IntegrityResult
    """
    total_frames = len(frames)
    if total_frames == 0:
        return IntegrityResult(
            crc_results=(),
            valid_frame_count=0,
            total_frame_count=0,
            crc_valid_fraction=0.0,
            multi_frame_p_value=1.0,
            before_fec_valid_count=0,
            after_fec_valid_count=0,
            valid=False,
        )

    # If candidate CRC not specified, search all presets and pick the one with most valid frames
    best_param = candidate_crc
    best_valid_count = 0
    best_results: list[CRCResult] = []

    params_to_test = [candidate_crc] if candidate_crc is not None else CRC_PRESETS

    for param in params_to_test:
        for offset_bits in (0, 16, 32, 64):
            frame_results: list[CRCResult] = []
            valid_cnt = 0

            for f in frames:
                if len(f.raw_bits) > offset_bits + param.width:
                    candidate_bits = f.raw_bits[offset_bits:]
                    n_bytes = len(candidate_bits) // 8
                    if n_bytes > (param.width // 8):
                        frame_bytes = bytes(np.packbits(candidate_bits[: n_bytes * 8]))
                        crc_res = evaluate_frame_crc(frame_bytes, param)
                        frame_results.append(crc_res)
                        if crc_res.is_valid:
                            valid_cnt += 1

            if valid_cnt > best_valid_count:
                best_valid_count = valid_cnt
                best_param = param
                best_results = frame_results

    if not best_results and params_to_test:
        best_param = params_to_test[0]
        best_results = []

    width = best_param.width if best_param else 16
    single_p = 2.0 ** (-width)
    multi_p = _binomial_tail(total_frames, best_valid_count, single_p)
    valid_fraction = float(best_valid_count / total_frames) if total_frames > 0 else 0.0

    return IntegrityResult(
        crc_results=tuple(best_results),
        valid_frame_count=best_valid_count,
        total_frame_count=total_frames,
        crc_valid_fraction=round(valid_fraction, 4),
        multi_frame_p_value=multi_p,
        before_fec_valid_count=best_valid_count,
        after_fec_valid_count=best_valid_count,
        valid=bool(best_valid_count > 0 and multi_p < 0.05),
    )
