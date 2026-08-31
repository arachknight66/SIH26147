import numpy as np
from typing import List, Optional, Tuple
from numpy.lib.stride_tricks import sliding_window_view
from .models import SyncWordPattern, HeaderMatch, DeinterleavingResult

# Built-in library of Sync Words
BUILTIN_SYNC_WORDS = [
    SyncWordPattern(
        name="HDLC_FLAG",
        bit_pattern=np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8),
        description="Standard HDLC framing flag (0x7E).",
        source="built-in library",
        reference="ISO/IEC 13239 High-level data link control (HDLC) procedures"
    ),
    SyncWordPattern(
        name="CCSDS_ASM_32",
        bit_pattern=np.unpackbits(np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8)),
        description="CCSDS Attached Sync Marker (32-bit).",
        source="built-in library",
        reference="CCSDS 131.0-B-3 TM Synchronization and Channel Coding"
    ),
    SyncWordPattern(
        name="BARKER_11",
        bit_pattern=np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),
        description="11-bit Barker Code.",
        source="built-in library",
        reference="Barker, R. H. (1953). Group Synchronizing of Binary Digital Systems"
    )
]

def correlate_sync_words(
    bits: np.ndarray, 
    llrs: np.ndarray, 
    patterns: List[SyncWordPattern],
    max_hamming_fraction: float = 0.15
) -> List[HeaderMatch]:
    """
    Sliding window correlation of bit patterns against the recovered bitstream.
    Computes Hamming distance and local LLR-based confidence.
    """
    if len(bits) == 0:
        return []
        
    all_matches = []
    
    for pattern in patterns:
        pat_len = len(pattern.bit_pattern)
        if len(bits) < pat_len:
            continue
            
        # Vectorized sliding window for Hamming distance
        bit_windows = sliding_window_view(bits, pat_len)
        hamming_distances = np.sum(bit_windows != pattern.bit_pattern, axis=1)
        
        # Max allowed bit errors for this pattern length
        # For short patterns, we require exact matches to avoid massive false positives
        max_errors = int(pat_len * max_hamming_fraction)
        if pat_len < 16:
            max_errors = 0
        
        # Find indices passing the threshold
        candidate_indices = np.where(hamming_distances <= max_errors)[0]
        
        if len(candidate_indices) == 0:
            continue
            
        # Get LLR windows for confidence weighting
        llr_windows = sliding_window_view(np.abs(llrs), pat_len)
        
        # To compute periodicity, look at diffs between candidate indices
        periodicity_map = {}
        if len(candidate_indices) >= 3:
            all_diffs = []
            pair_map = []
            for i in range(len(candidate_indices)):
                for j in range(i+1, min(i+10, len(candidate_indices))):
                    d = candidate_indices[j] - candidate_indices[i]
                    if d > pat_len:
                        all_diffs.append(d)
                        pair_map.append((d, candidate_indices[i], candidate_indices[j]))
                        
            if all_diffs:
                values, counts = np.unique(all_diffs, return_counts=True)
                best_diff = values[np.argmax(counts)]
                best_count = np.max(counts)
                # If a spacing repeats at least twice, flag the involved indices
                if best_count >= 2:
                    for d, idx1, idx2 in pair_map:
                        if d == best_diff:
                            periodicity_map[idx1] = True
                            periodicity_map[idx2] = True
                            
        for idx in candidate_indices:
            dist = hamming_distances[idx]
            local_llrs = llr_windows[idx]
            
            # Confidence Calculation:
            # 1. Base distance metric (1.0 for perfect match, down to 0.0 at threshold)
            base_score = 1.0 - (dist / max(1, max_errors + 1))
            
            # 2. Epistemic LLR weight. A perfect match on noise (LLR ~ 0) is weak.
            # Max-Log LLRs can be arbitrarily large. We normalize by the median of the whole stream.
            global_median = np.median(np.abs(llrs))
            if global_median < 1e-9:
                llr_weight = 0.5  # Unknown scale
            else:
                local_mean = np.mean(local_llrs)
                llr_weight = np.clip(local_mean / global_median, 0.0, 1.0)
                
            confidence = base_score * llr_weight
            
            # Boost confidence significantly if periodic
            is_periodic = periodicity_map.get(idx, False)
            if is_periodic:
                confidence = np.clip(confidence + 0.3, 0.0, 1.0)
                
            all_matches.append(HeaderMatch(
                pattern=pattern,
                bit_offset=int(idx),
                hamming_distance=int(dist),
                match_confidence=float(confidence),
                periodicity_consistent=is_periodic
            ))
            
    # Sort by confidence descending
    all_matches.sort(key=lambda m: m.match_confidence, reverse=True)
    return all_matches
