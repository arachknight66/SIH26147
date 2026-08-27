import numpy as np
import pytest
from app.recovery.demodulation import demodulate_8psk, demodulate_bpsk, demodulate_qpsk
from app.recovery.models import BitStreamStatus

def test_bpsk_demodulation():
    symbols = np.array([1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.complex64)
    res = demodulate_bpsk(symbols)
    assert res.valid is True
    assert res.bit_stream_status == BitStreamStatus.AVAILABLE
    assert np.array_equal(res.hard_bits, np.array([1, 0, 1, 1, 0], dtype=np.uint8))
    assert len(res.soft_decisions) == 5

def test_qpsk_demodulation():
    symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    res = demodulate_qpsk(symbols)
    assert res.valid is True
    assert len(res.hard_bits) == 8
    # Gray mapping: (1,1)->00, (-1,1)->01, (-1,-1)->11, (1,-1)->10
    expected_bits = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    assert np.array_equal(res.hard_bits, expected_bits)

def test_8psk_demodulation():
    angles = np.arange(8) * (2.0 * np.pi / 8.0)
    symbols = np.exp(1j * angles).astype(np.complex64)
    res = demodulate_8psk(symbols)
    assert res.valid is True
    assert len(res.hard_bits) == 24
