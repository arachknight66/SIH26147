import hashlib
import numpy as np
from typing import Any, Dict, List, Optional
from .models import HeaderMatch, CRCMatch, FrameStructure, HypothesisStatus
from .crc_search import crc_algorithm, verify_crc_at_boundary

def _unknown_frame() -> FrameStructure:
    return FrameStructure(
        header_match=HeaderMatch(pattern=None, bit_offset=0, hamming_distance=0, match_confidence=0.0, periodicity_consistent=False),
        header_length_bits=0,
        payload_start_bit=0,
        payload_length_bits=None,
        crc_candidate=None,
        status=HypothesisStatus.UNKNOWN,
    )


def _profile_payload_bits(profile: Dict[str, Any]) -> int:
    if "payload_bits" in profile:
        value = int(profile["payload_bits"])
    elif "payload_bytes" in profile:
        value = int(profile["payload_bytes"]) * 8
    else:
        raise ValueError("frame profile requires payload_bits or payload_bytes")
    if value <= 0:
        raise ValueError("frame profile payload length must be positive")
    return value


def _next_same_header(match: HeaderMatch, matches: List[HeaderMatch]) -> Optional[int]:
    offsets = [
        candidate.bit_offset for candidate in matches
        if candidate.pattern.name == match.pattern.name and candidate.bit_offset > match.bit_offset
    ]
    return min(offsets) if offsets else None


def _profile_structures(
    bits: np.ndarray, matches: List[HeaderMatch], profiles: List[Dict[str, Any]], allow_confirmation: bool
) -> List[FrameStructure]:
    """Apply only predeclared frame profiles; never infer a CRC span by sweep."""
    structures: List[FrameStructure] = []
    for profile in profiles:
        header_name = str(profile.get("header_name", ""))
        if not header_name:
            raise ValueError("frame profile requires header_name")
        payload_bits = _profile_payload_bits(profile)
        algorithm = crc_algorithm(str(profile.get("crc_name", "")))
        required = max(2, int(profile.get("minimum_valid_frames", 2)))
        profile_matches = [item for item in matches if item.pattern.name == header_name]
        valid: List[tuple[HeaderMatch, CRCMatch]] = []
        for match in profile_matches:
            payload_start = match.bit_offset + len(match.pattern.bit_pattern)
            crc = verify_crc_at_boundary(bits, payload_start, payload_bits, algorithm)
            if crc is None:
                continue
            next_header = _next_same_header(match, profile_matches)
            # A CRC span that consumes another same-pattern marker is not a
            # protocol boundary.  This rejects broad-sweep coincidences.
            if profile.get("strict_next_header_boundary", True) and next_header is not None and crc.bit_range_checked[1] > next_header:
                continue
            valid.append((match, crc))

        word_hashes = {
            hashlib.sha256(np.packbits(bits[crc.bit_range_checked[0]:crc.bit_range_checked[1]]).tobytes()).digest()
            for _, crc in valid
        }
        independently_observed = len(valid) >= required and len(word_hashes) >= required
        for match, crc in valid:
            structures.append(FrameStructure(
                header_match=match,
                header_length_bits=len(match.pattern.bit_pattern),
                payload_start_bit=match.bit_offset + len(match.pattern.bit_pattern),
                payload_length_bits=payload_bits,
                crc_candidate=crc,
                status=(HypothesisStatus.CONFIRMED if allow_confirmation and independently_observed
                        else HypothesisStatus.HYPOTHESIS_UNVERIFIED),
            ))
    return structures


def assemble_frames(
    bits: np.ndarray,
    header_matches: List[HeaderMatch],
    max_header_matches: int = 10,
    *,
    frame_profiles: Optional[List[Dict[str, Any]]] = None,
    allow_confirmation: bool = False,
) -> List[FrameStructure]:
    """
    Combines headers with either configured frame verification or exploratory
    header-only hypotheses.  CRC discovery is intentionally separate: broad
    polynomial/length sweeps cannot confirm a payload in production.
    """
    if not header_matches:
        return [_unknown_frame()]
        
    structures = []
    
    # Cap to prevent pathological hang on noisy bitstreams with short false-positive sync words
    header_matches = sorted(header_matches, key=lambda m: m.match_confidence, reverse=True)
    if len(header_matches) > max_header_matches:
        header_matches = header_matches[:max_header_matches]
    
    if frame_profiles:
        structures.extend(_profile_structures(bits, header_matches, frame_profiles, allow_confirmation))

    # Always retain header-only observations. They are useful exploration
    # output but do not elevate a frame to confirmed payload data.
    for match in header_matches:
        structures.append(FrameStructure(
            header_match=match,
            header_length_bits=len(match.pattern.bit_pattern),
            payload_start_bit=match.bit_offset + len(match.pattern.bit_pattern),
            payload_length_bits=None,
            crc_candidate=None,
            status=HypothesisStatus.HYPOTHESIS_UNVERIFIED,
        ))
            
    # Rank and evaluate ambiguity
    def score_frame(fs: FrameStructure) -> float:
        score = fs.header_match.match_confidence
        if fs.header_match.periodicity_consistent:
            score += 0.3
        if fs.crc_candidate and fs.crc_candidate.verified:
            score += 0.4
        if fs.status is HypothesisStatus.CONFIRMED:
            score += 0.5
        return score
        
    structures.sort(key=score_frame, reverse=True)
    
    if len(structures) > 1:
        top_score = score_frame(structures[0])
        runner_up_score = score_frame(structures[1])
        if (structures[0].status is not HypothesisStatus.CONFIRMED
                and top_score - runner_up_score < 0.15):
            # Ambiguous
            for i in range(len(structures)):
                if top_score - score_frame(structures[i]) < 0.15:
                    structures[i] = FrameStructure(
                        header_match=structures[i].header_match,
                        header_length_bits=structures[i].header_length_bits,
                        payload_start_bit=structures[i].payload_start_bit,
                        payload_length_bits=structures[i].payload_length_bits,
                        crc_candidate=structures[i].crc_candidate,
                        status=HypothesisStatus.AMBIGUOUS
                    )
                    
    return structures or [_unknown_frame()]
