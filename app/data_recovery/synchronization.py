from __future__ import annotations
import numpy as np
from .models import PreambleCandidate

# Standard sync-word and preamble pattern library
STANDARD_SYNC_PATTERNS: list[tuple[str, str]] = [
    ("CCSDS_32", "1acffc1d"),          # 32-bit CCSDS Attached Sync Marker
    ("BLUETOOTH_16", "2dd4"),           # 16-bit Sync Word
    ("BARKER_13", "1f35"),              # 13-bit Barker code: 1111100110101 -> padded to hex
    ("BARKER_11", "0712"),              # 11-bit Barker: 11100010010
    ("BARKER_7", "72"),                 # 7-bit Barker: 1110010
    ("GENERIC_16A", "faf0"),            # 16-bit Test sync
    ("GENERIC_16B", "aa55"),            # Alternating preamble / sync
    ("GENERIC_16C", "55aa"),            # Alternating preamble
    ("GENERIC_8A", "aa"),               # 8-bit preamble
    ("GENERIC_8B", "55"),               # 8-bit preamble
    ("IEEE_802154", "a7"),              # 8-bit SFD
]

def hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert hex string into 1D uint8 binary array."""
    data = bytes.fromhex(hex_str)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def find_pattern_matches(
    bits: np.ndarray,
    pattern: np.ndarray,
    max_hamming_distance: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Search bitstream for pattern matches within maximum Hamming distance.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary array.
    pattern : np.ndarray
        1D uint8 binary pattern.
    max_hamming_distance : int
        Maximum allowed bit errors in pattern.

    Returns
    -------
    match_indices : list[int]
    hamming_distances : list[int]
    """
    n_bits = len(bits)
    pat_len = len(pattern)
    if n_bits < pat_len or pat_len == 0:
        return [], []

    match_indices: list[int] = []
    distances: list[int] = []

    # Slide window
    for idx in range(n_bits - pat_len + 1):
        segment = bits[idx : idx + pat_len]
        dist = int(np.sum(segment != pattern))
        if dist <= max_hamming_distance:
            match_indices.append(idx)
            distances.append(dist)

    return match_indices, distances

def _extract_periodic_subset(matches: list[int], max_jitter: float = 4.0) -> tuple[list[int], float, float, bool]:
    if len(matches) < 3:
        if len(matches) == 2:
            return matches, float(matches[1] - matches[0]), 0.0, False
        return matches, 0.0, 0.0, False

    # 1. First test full matches
    spacings = np.diff(matches)
    var_sp = float(np.var(spacings))
    mean_sp = float(np.mean(spacings))
    max_j = float(np.max(np.abs(spacings - mean_sp)))
    if var_sp < 8.0 and max_j <= max_jitter and mean_sp >= 32.0:
        return matches, mean_sp, var_sp, True

    # 2. Search candidate period S from pairwise differences
    n = len(matches)
    best_subset: list[int] = []
    best_period = 0.0

    candidate_steps: set[int] = set()
    for i in range(n):
        for j in range(i + 1, min(n, i + 6)):
            step = matches[j] - matches[i]
            if step >= 32:
                candidate_steps.add(step)

    for step in candidate_steps:
        for start_idx in range(n):
            curr_subset = [matches[start_idx]]
            curr_pos = matches[start_idx]
            for next_idx in range(start_idx + 1, n):
                expected = curr_pos + step
                if abs(matches[next_idx] - expected) <= max_jitter:
                    curr_subset.append(matches[next_idx])
                    curr_pos = matches[next_idx]

            if len(curr_subset) > len(best_subset):
                best_subset = curr_subset
                best_period = float(step)

    if len(best_subset) >= 3:
        sub_spacings = np.diff(best_subset)
        sub_var = float(np.var(sub_spacings))
        sub_mean = float(np.mean(sub_spacings))
        return best_subset, sub_mean, sub_var, True

    return matches, mean_sp, var_sp, False

def detect_preamble_candidates(
    bits: np.ndarray,
    max_hamming_distance: int = 1,
    custom_patterns: list[np.ndarray] | None = None,
) -> list[PreambleCandidate]:
    """
    Search for known and periodic preamble / sync-word candidates.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 bitstream.
    max_hamming_distance : int
        Tolerance for noisy matches.
    custom_patterns : list[np.ndarray] | None
        Additional patterns to evaluate.

    Returns
    -------
    candidates : list[PreambleCandidate]
    """
    candidates: list[PreambleCandidate] = []
    if len(bits) < 16 or np.all(bits == 0) or np.all(bits == 1):
        return []

    # 1. Evaluate standard library patterns
    patterns_to_test: list[tuple[str, np.ndarray]] = []
    for name, hex_str in STANDARD_SYNC_PATTERNS:
        p_bits = hex_to_bits(hex_str)
        if len(p_bits) <= len(bits):
            patterns_to_test.append((hex_str, p_bits))

    if custom_patterns:
        for p in custom_patterns:
            patterns_to_test.append((p.tobytes().hex(), p))

    for hex_repr, pat in patterns_to_test:
        matches, dists = find_pattern_matches(bits, pat, max_hamming_distance=max_hamming_distance)
        match_count = len(matches)

        if match_count >= 1:
            if match_count >= 2:
                filtered_matches, mean_sp, var_sp, is_periodic = _extract_periodic_subset(matches)
                f_count = len(filtered_matches)
                
                # Confidence scoring
                period_bonus = 0.50 if is_periodic else 0.0
                count_bonus = min(0.30, (f_count - 1) * 0.10)
                len_bonus = min(0.20, len(pat) / 32.0)
                conf = min(1.0, period_bonus + count_bonus + len_bonus)
                final_matches = filtered_matches
            else:
                mean_sp = float(len(bits))
                var_sp = 0.0
                is_periodic = False
                conf = 0.20
                final_matches = matches

            candidates.append(
                PreambleCandidate(
                    pattern_bits=pat,
                    pattern_hex=hex_repr,
                    length_bits=len(pat),
                    match_indices=tuple(final_matches),
                    match_count=len(final_matches),
                    mean_spacing=round(mean_sp, 2),
                    spacing_variance=round(var_sp, 2),
                    hamming_distance_dist=tuple(float(d) for d in dists[: len(final_matches)]),
                    is_periodic=is_periodic,
                    confidence=round(conf, 3),
                    valid=True,
                )
            )

    # Sort by periodicity, variance, length, and confidence descending
    candidates.sort(key=lambda c: (c.is_periodic, -c.spacing_variance, c.length_bits, c.confidence, c.match_count), reverse=True)
    return candidates
