import numpy as np
import pytest
from app.data_recovery.synchronization import detect_preamble_candidates, find_pattern_matches, hex_to_bits

def test_find_pattern_matches_exact():
    # 0x2DD4 pattern: 00101101 11010100
    pat = hex_to_bits("2dd4")
    # Stream with two occurrences spaced by 64 bits
    stream = np.concatenate((pat, np.zeros(64, dtype=np.uint8), pat))
    matches, dists = find_pattern_matches(stream, pat, max_hamming_distance=0)
    assert len(matches) == 2
    assert matches[0] == 0
    assert matches[1] == len(pat) + 64
    assert dists == [0, 0]

def test_detect_preamble_candidates_periodic():
    pat = hex_to_bits("1acffc1d")  # 32-bit CCSDS
    frame_len = 256
    # Construct 4 periodic frames
    frames = [np.concatenate((pat, np.random.randint(0, 2, frame_len - len(pat), dtype=np.uint8))) for _ in range(4)]
    stream = np.concatenate(frames)

    cands = detect_preamble_candidates(stream)
    assert len(cands) >= 1
    best = cands[0]
    assert best.pattern_hex == "1acffc1d"
    assert best.match_count == 4
    assert best.is_periodic is True
    assert best.confidence >= 0.80
