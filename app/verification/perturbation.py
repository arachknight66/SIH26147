from __future__ import annotations
from typing import Any
import numpy as np
from app.data_recovery.crc import search_crc_presets
from app.data_recovery.models import FrameCandidate, ReconstructionCandidate

def evaluate_boundary_perturbations(
    candidate: ReconstructionCandidate,
    delta_offsets: tuple[int, ...] = (-4, -2, -1, 1, 2, 4),
) -> tuple[bool, float, dict[str, Any]]:
    """
    Deliberately perturb frame boundaries by +/-1, +/-2, +/-4 bits and verify evidence collapse.

    Parameters
    ----------
    candidate : ReconstructionCandidate
    delta_offsets : tuple[int, ...]

    Returns
    -------
    passed : bool
    score : float
    details : dict[str, Any]
    """
    frames = list(candidate.frames)
    if not frames:
        return False, 0.0, {"reason": "no_frames"}

    # Baseline valid frame count at offset 0
    baseline_valid = candidate.integrity.valid_frame_count if candidate.integrity else 0
    raw_bits = candidate.bit_hypothesis.bitstream.hard_bits

    perturb_results: dict[int, int] = {}
    collapsed = True

    if baseline_valid > 0:
        for delta in delta_offsets:
            shifted_valid = 0
            for f in frames:
                s_bit = max(0, f.start_bit + delta)
                e_bit = min(len(raw_bits), f.end_bit + delta)
                if e_bit - s_bit >= 32:
                    shifted_bytes = bytes(np.packbits(raw_bits[s_bit : (e_bit // 8) * 8]))
                    crc_matches = [r for r in search_crc_presets(shifted_bytes) if r.is_valid]
                    if crc_matches:
                        shifted_valid += 1

            perturb_results[delta] = shifted_valid
            if shifted_valid >= baseline_valid:
                collapsed = False

        pass_score = 1.0 if collapsed else 0.20
        return collapsed, pass_score, {"baseline_valid": baseline_valid, "perturb_results": perturb_results}

    else:
        # Preamble-based boundary test
        if candidate.preamble and candidate.preamble.is_periodic:
            pass_score = 0.90
            return True, pass_score, {"status": "periodic_framing_without_crc"}
        return True, 0.50, {"status": "inconclusive"}
