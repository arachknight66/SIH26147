import numpy as np
from typing import List, Optional, Tuple
from numpy.lib.stride_tricks import sliding_window_view
from .models import SyncWordPattern, HeaderMatch, DeinterleavingResult
from .native import require_native

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
    max_hamming_fraction: float = 0.15,
    *,
    backend: str = "cpu",
) -> List[HeaderMatch]:
    """
    Sliding window correlation of bit patterns against the recovered bitstream.
    Computes Hamming distance and local LLR-based confidence.
    """
    if len(bits) == 0:
        return []
    from .acceleration import should_use_gpu
    by_name = {pattern.name: pattern for pattern in patterns}
    if should_use_gpu(backend):
        from .gpu_dsp import gpu_correlate
        raw_matches = gpu_correlate(
            np.ascontiguousarray(bits, dtype=np.uint8),
            np.ascontiguousarray(llrs, dtype=np.float32) if len(llrs) == len(bits) else np.zeros(0, dtype=np.float32),
            [(pattern.name, pattern.bit_pattern) for pattern in patterns], float(max_hamming_fraction),
        )
        # Keep the existing periodic-evidence definition byte-for-byte aligned
        # with the native implementation; only window scoring runs on CUDA.
        enriched = []
        for name, offset, distance, confidence in raw_matches:
            periodic = False
            for other_name, other_offset, _, _ in raw_matches:
                if name != other_name or other_offset <= offset or other_offset - offset < 16:
                    continue
                period = other_offset - offset
                if any(candidate_name == name and candidate_offset == other_offset + period
                       for candidate_name, candidate_offset, _, _ in raw_matches):
                    periodic = True
                    confidence = min(1.0, confidence + 0.3)
                    break
            if not periodic:
                for previous_name, previous_offset, _, _ in raw_matches:
                    if previous_name != name or previous_offset >= offset:
                        continue
                    period = offset - previous_offset
                    if any(candidate_name == name and candidate_offset == offset + period
                           for candidate_name, candidate_offset, _, _ in raw_matches):
                        periodic = True
                        confidence = min(1.0, confidence + 0.3)
                        break
            enriched.append(HeaderMatch(by_name[name], offset, distance, confidence, periodic))
        return sorted(enriched, key=lambda item: item.match_confidence, reverse=True)

    native = require_native()
    native_patterns = []
    for pattern in patterns:
        candidate = native.CorrelationPattern()
        candidate.name = pattern.name
        candidate.bits = np.ascontiguousarray(pattern.bit_pattern, dtype=np.uint8).tolist()
        native_patterns.append(candidate)
    native_matches = native.correlate_bits(
        np.ascontiguousarray(bits, dtype=np.uint8),
        np.ascontiguousarray(llrs, dtype=np.float32) if len(llrs) == len(bits) else np.zeros(0, dtype=np.float32),
        native_patterns,
        float(max_hamming_fraction),
    )
    return [HeaderMatch(
        pattern=by_name[match.pattern_name], bit_offset=int(match.bit_offset),
        hamming_distance=int(match.hamming_distance), match_confidence=float(match.confidence),
        periodicity_consistent=bool(match.periodic),
    ) for match in native_matches]
