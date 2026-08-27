import numpy as np
import pytest
from app.data_recovery.line_coding import decode_line_code, evaluate_line_code_hypotheses
from app.data_recovery.models import LineCodeType

def test_manchester_decode():
    # Valid Manchester: 01 -> 0, 10 -> 1, 01 -> 0, 10 -> 1
    man_bits = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    decoded, viol = decode_line_code(man_bits, LineCodeType.MANCHESTER)
    assert viol == 0.0
    assert np.array_equal(decoded, np.array([0, 1, 0, 1], dtype=np.uint8))

def test_nrzi_decode():
    bits = np.array([0, 1, 1, 0, 0, 1], dtype=np.uint8)
    # diffs = [1, 0, 1, 0, 1]
    decoded, viol = decode_line_code(bits, LineCodeType.NRZI_TRANS_1)
    assert len(decoded) == len(bits) - 1
    assert np.array_equal(decoded, np.array([1, 0, 1, 0, 1], dtype=np.uint8))

def test_evaluate_line_code_hypotheses():
    bits = np.random.randint(0, 2, 128, dtype=np.uint8)
    hyps = evaluate_line_code_hypotheses(bits)
    assert len(hyps) >= 2
    assert hyps[0].valid is True
