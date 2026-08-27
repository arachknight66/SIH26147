from __future__ import annotations
from typing import Sequence
import numpy as np
from .models import ScramblerHypothesis, ScramblerType

# Standard LFSR scrambler polynomials (taps from 1-indexed MSB down)
STANDARD_LFSR_POLYNOMIALS: list[tuple[str, tuple[int, ...], int]] = [
    ("ITU_V29_7", (7, 4), 7),                 # x^7 + x^4 + 1 (period 127)
    ("IEEE_80211_7", (7, 4), 7),              # x^7 + x^4 + 1
    ("CCSDS_8", (8, 7, 5, 3), 8),             # x^8 + x^7 + x^5 + x^3 + 1 (period 255)
    ("DVB_15", (15, 14), 15),                 # x^15 + x^14 + 1
    ("CCITT_9", (9, 5), 9),                   # x^9 + x^5 + 1 (period 511)
]

def berlekamp_massey(bits: np.ndarray) -> int:
    """
    Compute linear complexity of a binary sequence using the Berlekamp-Massey algorithm.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.

    Returns
    -------
    L : int
        Linear complexity (length of shortest LFSR generating bits).
    """
    n = len(bits)
    if n == 0:
        return 0

    c = np.zeros(n, dtype=np.uint8)
    b = np.zeros(n, dtype=np.uint8)
    c[0] = 1
    b[0] = 1
    l_deg = 0
    m = -1

    for i in range(n):
        # Discrepancy d = sum_{j=0}^L c_j * s_{i-j} mod 2
        d = bits[i]
        for j in range(1, l_deg + 1):
            d ^= (c[j] & bits[i - j])

        if d != 0:
            t = c.copy()
            p = i - m
            for j in range(n - p):
                c[j + p] ^= b[j]

            if l_deg <= i // 2:
                l_deg = i + 1 - l_deg
                m = i
                b = t

    return int(l_deg)

def generate_lfsr_sequence(
    taps: Sequence[int],
    length: int,
    init_state: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Generate pseudo-random binary sequence from a linear feedback shift register.

    Parameters
    ----------
    taps : Sequence[int]
        Feedback tap positions (e.g. (7, 4) for x^7 + x^4 + 1).
    length : int
        Number of output bits to generate.
    init_state : Sequence[int] | None
        Initial shift register state (default all 1s).

    Returns
    -------
    seq : np.ndarray
        1D uint8 array.
    """
    degree = max(taps)
    state = np.ones(degree, dtype=np.uint8) if init_state is None else np.array(init_state, dtype=np.uint8)[:degree]
    out = np.zeros(length, dtype=np.uint8)

    for i in range(length):
        out[i] = state[-1]
        # Feedback bit = XOR of tap positions (1-indexed from MSB)
        fb = 0
        for tap in taps:
            fb ^= state[tap - 1]
        state = np.roll(state, 1)
        state[0] = fb

    return out

def descramble_lfsr(
    bits: np.ndarray,
    taps: Sequence[int],
    init_state: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Descramble synchronous LFSR scrambled bitstream: output = bits XOR lfsr_sequence.

    Parameters
    ----------
    bits : np.ndarray
    taps : Sequence[int]
    init_state : Sequence[int] | None

    Returns
    -------
    descrambled : np.ndarray
    """
    prbs = generate_lfsr_sequence(taps, len(bits), init_state=init_state)
    return (bits ^ prbs).astype(np.uint8)

def evaluate_scrambler_hypotheses(
    bits: np.ndarray,
    reference_preambles: list[np.ndarray] | None = None,
) -> list[ScramblerHypothesis]:
    """
    Evaluate candidate descrambler hypotheses against preamble recovery and linear complexity metrics.

    Parameters
    ----------
    bits : np.ndarray
    reference_preambles : list[np.ndarray] | None

    Returns
    -------
    hypotheses : list[ScramblerHypothesis]
    """
    n = len(bits)
    if n < 32:
        return [
            ScramblerHypothesis(
                scrambler_type=ScramblerType.NONE,
                polynomial_name="none",
                polynomial_bits=(),
                initial_state=(),
                period=0,
                linear_complexity=0,
                entropy_improvement=0.0,
                crc_improvement=0.0,
                confidence=1.0,
                valid=True,
            )
        ]

    hyps: list[ScramblerHypothesis] = []

    # 1. Baseline: No Scrambler
    lin_comp = berlekamp_massey(bits[: min(512, n)])
    hyps.append(
        ScramblerHypothesis(
            scrambler_type=ScramblerType.NONE,
            polynomial_name="none",
            polynomial_bits=(),
            initial_state=(),
            period=0,
            linear_complexity=lin_comp,
            entropy_improvement=0.0,
            crc_improvement=0.0,
            confidence=0.85,
            valid=True,
        )
    )

    # 2. Test standard LFSR candidates
    for name, taps, deg in STANDARD_LFSR_POLYNOMIALS:
        descrambled = descramble_lfsr(bits, taps)
        # Check linear complexity reduction or preamble recovery
        lc_desc = berlekamp_massey(descrambled[: min(512, n)])
        
        # Check if descrambling recovers any reference preambles
        preamble_found = False
        if reference_preambles:
            for p in reference_preambles:
                if len(p) <= len(descrambled):
                    # Sliding check
                    for i in range(len(descrambled) - len(p) + 1):
                        if np.sum(descrambled[i : i + len(p)] != p) == 0:
                            preamble_found = True
                            break
                if preamble_found:
                    break

        conf = 0.90 if preamble_found else (0.45 if lc_desc < lin_comp * 0.70 else 0.15)
        hyps.append(
            ScramblerHypothesis(
                scrambler_type=ScramblerType.LFSR_SYNCHRONOUS,
                polynomial_name=name,
                polynomial_bits=taps,
                initial_state=tuple([1] * deg),
                period=(1 << deg) - 1,
                linear_complexity=lc_desc,
                entropy_improvement=round(float((lin_comp - lc_desc) / max(1, lin_comp)), 3),
                crc_improvement=1.0 if preamble_found else 0.0,
                confidence=round(conf, 3),
                valid=bool(preamble_found or conf > 0.40),
            )
        )

    hyps.sort(key=lambda h: h.confidence, reverse=True)
    return hyps
