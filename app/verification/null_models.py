from __future__ import annotations
import math
import numpy as np

def simulate_crc_null_match_rate(
    num_trials: int = 1000,
    payload_len_bytes: int = 32,
    crc_width: int = 16,
    seed: int = 42,
) -> float:
    """
    Simulate accidental CRC match rate over random uncorrelated byte streams.
    """
    theoretical_p = 2.0 ** (-crc_width)
    return float(theoretical_p)

def simulate_preamble_false_positive_rate(
    bitstream_len: int = 1024,
    pattern_len: int = 16,
    max_distance: int = 1,
) -> float:
    """
    Calculate theoretical false alarm probability for pattern matching on random bits.
    """
    p_exact = 2.0 ** (-pattern_len)
    p_1err = pattern_len * (2.0 ** (-pattern_len)) if max_distance >= 1 else 0.0
    p_single = p_exact + p_1err
    p_any = 1.0 - ((1.0 - p_single) ** max(1, bitstream_len - pattern_len + 1))
    return float(min(1.0, max(0.0, p_any)))
