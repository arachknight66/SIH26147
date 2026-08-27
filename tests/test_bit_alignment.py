import numpy as np
import pytest
from app.data_recovery.bit_alignment import convert_bits_to_bytes, generate_byte_stream_candidates
from app.data_recovery.models import BitOrder

def test_convert_bits_to_bytes_exact():
    # 0x41 ('A') = 01000001, 0x42 ('B') = 01000010
    bits = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
    cand_msb = convert_bits_to_bytes(bits, bit_offset=0, bit_order=BitOrder.MSB_FIRST)
    assert cand_msb.bytes_data == b"AB"
    assert cand_msb.printable_ratio == 1.0
    assert cand_msb.bit_count == 16

def test_convert_bits_to_bytes_offset():
    # Prepend 3 dummy bits (1, 1, 1) to 'AB'
    dummy = np.array([1, 1, 1], dtype=np.uint8)
    bits = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
    shifted = np.concatenate((dummy, bits))

    cand = convert_bits_to_bytes(shifted, bit_offset=3, bit_order=BitOrder.MSB_FIRST)
    assert cand.bytes_data == b"AB"
    assert cand.printable_ratio == 1.0

def test_generate_byte_stream_candidates():
    bits = np.random.randint(0, 2, 64, dtype=np.uint8)
    cands = generate_byte_stream_candidates(bits)
    # 8 offsets * 2 bit orders = 16 candidates
    assert len(cands) == 16
