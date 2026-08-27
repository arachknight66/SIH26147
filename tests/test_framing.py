import numpy as np
import pytest
from app.data_recovery.framing import detect_frame_boundaries, detect_sequence_continuity, slice_frames
from app.data_recovery.models import PreambleCandidate

def test_detect_frame_boundaries_periodic():
    preamble = PreambleCandidate(
        pattern_bits=np.array([1, 0, 1, 0], dtype=np.uint8),
        pattern_hex="a",
        length_bits=4,
        match_indices=(0, 100, 200, 300),
        match_count=4,
        mean_spacing=100.0,
        spacing_variance=0.0,
        hamming_distance_dist=(0.0, 0.0, 0.0, 0.0),
        is_periodic=True,
        confidence=0.90,
    )
    bits = np.random.randint(0, 2, 400, dtype=np.uint8)
    boundaries, info = detect_frame_boundaries(bits, preamble=preamble)
    assert len(boundaries) == 4
    assert info["is_periodic"] is True
    assert boundaries[0].length_bits == 100

def test_detect_sequence_continuity():
    bits = np.random.randint(0, 2, 300, dtype=np.uint8)
    boundaries, _ = detect_frame_boundaries(bits, preamble=None)
    frames = slice_frames(bits, boundaries)
    is_cont, seqs, missing = detect_sequence_continuity(frames)
    # Single frame -> not continuous by definition (<2 frames)
    assert is_cont is False
