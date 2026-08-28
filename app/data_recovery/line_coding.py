from __future__ import annotations
import numpy as np
from .models import LineCodeHypothesis, LineCodeType

def decode_line_code(
    bits: np.ndarray,
    code_type: LineCodeType,
) -> tuple[np.ndarray, float]:
    """
    Decode bitstream under a given line code hypothesis.

    Parameters
    ----------
    bits : np.ndarray
        Input 1D uint8 binary stream.
    code_type : LineCodeType

    Returns
    -------
    decoded_bits : np.ndarray
    violation_rate : float
    """
    n = len(bits)
    if n == 0 or code_type in (LineCodeType.NONE, LineCodeType.NRZ):
        return bits.copy(), 0.0

    if code_type == LineCodeType.NRZI_TRANS_1:
        # Differential decode: transition (diff=1) indicates 1, no transition (diff=0) indicates 0
        diffs = (bits[1:] != bits[:-1]).astype(np.uint8)
        return diffs, 0.0

    elif code_type == LineCodeType.NRZI_TRANS_0:
        # Differential decode: transition (diff=1) indicates 0, no transition (diff=0) indicates 1
        diffs = (bits[1:] == bits[:-1]).astype(np.uint8)
        return diffs, 0.0

    elif code_type == LineCodeType.MANCHESTER:
        # Paired-bit decode: 01 -> 0, 10 -> 1 (IEEE 802.3)
        n_pairs = n // 2
        if n_pairs == 0:
            return np.array([], dtype=np.uint8), 1.0

        pairs = bits[: n_pairs * 2].reshape(-1, 2)
        p0 = pairs[:, 0]
        p1 = pairs[:, 1]

        # Valid transitions: (0,1) -> 0, (1,0) -> 1
        decoded = p0.copy()  # In IEEE 802.3, if (0,1) -> 0 (p0=0), if (1,0) -> 1 (p0=1)
        violations = np.sum(p0 == p1)
        violation_rate = float(violations / n_pairs)
        return decoded, violation_rate

    elif code_type == LineCodeType.DIFF_MANCHESTER:
        n_pairs = n // 2
        if n_pairs < 2:
            return np.array([], dtype=np.uint8), 1.0
        pairs = bits[: n_pairs * 2].reshape(-1, 2)
        # Transition at start of bit interval
        start_transitions = (pairs[1:, 0] != pairs[:-1, 1]).astype(np.uint8)
        return start_transitions, 0.0

    return bits.copy(), 0.0

def evaluate_line_code_hypotheses(
    bits: np.ndarray,
) -> list[LineCodeHypothesis]:
    """
    Evaluate candidate line-coding hypotheses against statistical transition properties.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.

    Returns
    -------
    hypotheses : list[LineCodeHypothesis]
    """
    n = len(bits)
    if n < 32:
        return [
            LineCodeHypothesis(
                code_type=LineCodeType.NONE,
                convention="direct_binary",
                transition_density=0.5,
                run_length_score=1.0,
                clock_consistency=1.0,
                confidence=1.0,
                valid=True,
            )
        ]

    hyps: list[LineCodeHypothesis] = []

    # 1. Direct / NRZ
    trans_prob = float(np.mean(bits[1:] != bits[:-1]))
    hyps.append(
        LineCodeHypothesis(
            code_type=LineCodeType.NONE,
            convention="direct_nrz",
            transition_density=round(trans_prob, 4),
            run_length_score=1.0,
            clock_consistency=1.0,
            confidence=0.85,
            valid=True,
        )
    )

    # 2. NRZI (Transition on 1)
    hyps.append(
        LineCodeHypothesis(
            code_type=LineCodeType.NRZI_TRANS_1,
            convention="nrzi_mark",
            transition_density=round(trans_prob, 4),
            run_length_score=0.9,
            clock_consistency=0.8,
            confidence=0.50,
            valid=True,
        )
    )

    # 3. Manchester
    if n >= 64:
        _, man_violation = decode_line_code(bits, LineCodeType.MANCHESTER)
        # If violation rate is low (< 0.05), Manchester is a strong candidate
        man_conf = max(0.0, 1.0 - (man_violation / 0.10))
        hyps.append(
            LineCodeHypothesis(
                code_type=LineCodeType.MANCHESTER,
                convention="ieee_802_3",
                transition_density=round(trans_prob, 4),
                run_length_score=round(1.0 - man_violation, 4),
                clock_consistency=round(1.0 - man_violation, 4),
                confidence=round(man_conf, 3),
                valid=bool(man_violation < 0.20),
            )
        )

    hyps.sort(key=lambda h: h.confidence, reverse=True)
    return hyps
