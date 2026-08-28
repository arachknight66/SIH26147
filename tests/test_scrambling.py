import numpy as np
import pytest
from app.data_recovery.models import ScramblerType
from app.data_recovery.scrambling import (
    berlekamp_massey,
    descramble_lfsr,
    evaluate_scrambler_hypotheses,
    generate_lfsr_sequence,
)

def test_lfsr_generation_and_descrambling():
    taps = (7, 4)
    length = 127
    seq = generate_lfsr_sequence(taps, length)
    assert len(seq) == 127
    # Sequence of length 127 from degree 7 LFSR must have linear complexity 7
    lc = berlekamp_massey(seq)
    assert lc == 7

def test_descramble_reversibility():
    payload = np.random.randint(0, 2, 256, dtype=np.uint8)
    taps = (7, 4)
    scrambled = descramble_lfsr(payload, taps)
    descrambled = descramble_lfsr(scrambled, taps)
    assert np.array_equal(payload, descrambled)

def test_evaluate_scrambler_hypotheses():
    bits = np.random.randint(0, 2, 256, dtype=np.uint8)
    hyps = evaluate_scrambler_hypotheses(bits)
    assert len(hyps) >= 2
    assert hyps[0].scrambler_type == ScramblerType.NONE
