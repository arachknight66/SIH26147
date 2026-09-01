import numpy as np
from typing import List, Tuple
from .models import HeaderMatch, CRCMatch, FrameStructure, HypothesisStatus
from .crc_search import search_crcs

def assemble_frames(bits: np.ndarray, header_matches: List[HeaderMatch], max_header_matches: int = 10) -> List[FrameStructure]:
    """
    Combines Header matches and CRC searches into FrameStructure hypotheses.
    Reuses the ambiguity ranking logic pattern.
    Limits to `max_header_matches` top matches to cap O(N) CRC search sweeps.
    """
    if not header_matches:
        # Honest fallback to UNKNOWN
        return [FrameStructure(
            header_match=HeaderMatch(pattern=None, bit_offset=0, hamming_distance=0, match_confidence=0.0, periodicity_consistent=False),
            header_length_bits=0,
            payload_start_bit=0,
            payload_length_bits=None,
            crc_candidate=None,
            status=HypothesisStatus.UNKNOWN
        )]
        
    structures = []
    
    # Cap to prevent pathological hang on noisy bitstreams with short false-positive sync words
    header_matches = sorted(header_matches, key=lambda m: m.match_confidence, reverse=True)
    if len(header_matches) > max_header_matches:
        header_matches = header_matches[:max_header_matches]
    
    for match in header_matches:
        # A single match. Try to find a CRC starting right after the header
        header_len = len(match.pattern.bit_pattern)
        payload_start = match.bit_offset + header_len
        
        crc_matches = search_crcs(bits, payload_start)
        
        if crc_matches:
            # We found one or more CRCs! This boosts confidence massively.
            for crc in crc_matches:
                payload_len = crc.bit_range_checked[1] - crc.bit_range_checked[0] - (crc.bit_range_checked[1] - payload_start)
                # Actually crc.bit_range_checked is (payload_start, payload_start + payload_len + crc_len)
                # So payload_len is just the difference minus crc_len
                # It's easier: crc_len is known by the poly width, but we didn't export it in CRCMatch.
                # Let's just use the verified CRC.
                structures.append(FrameStructure(
                    header_match=match,
                    header_length_bits=header_len,
                    payload_start_bit=payload_start,
                    payload_length_bits=crc.bit_range_checked[1] - payload_start, # payload + crc
                    crc_candidate=crc,
                    status=HypothesisStatus.HYPOTHESIS_UNVERIFIED # upgraded below
                ))
        else:
            # No CRC found, but we have a header
            structures.append(FrameStructure(
                header_match=match,
                header_length_bits=header_len,
                payload_start_bit=payload_start,
                payload_length_bits=None,
                crc_candidate=None,
                status=HypothesisStatus.HYPOTHESIS_UNVERIFIED
            ))
            
    # Rank and evaluate ambiguity
    def score_frame(fs: FrameStructure) -> float:
        score = fs.header_match.match_confidence
        if fs.header_match.periodicity_consistent:
            score += 0.3
        if fs.crc_candidate and fs.crc_candidate.verified:
            score += 0.5
        return score
        
    structures.sort(key=score_frame, reverse=True)
    
    if len(structures) > 1:
        top_score = score_frame(structures[0])
        runner_up_score = score_frame(structures[1])
        if top_score - runner_up_score < 0.15:
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
                    
    return structures
