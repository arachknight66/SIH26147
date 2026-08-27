import numpy as np
import pytest
from app.data_recovery.fec_decode import (
    decode_hamming_7_4,
    decode_repetition,
    encode_convolutional,
    viterbi_decode,
)
from app.data_recovery.models import FECCodeFamily

def test_viterbi_decode_clean():
    info_bits = np.random.randint(0, 2, 64, dtype=np.uint8)
    encoded = encode_convolutional(info_bits, k=7, g1=0o133, g2=0o171)
    
    res = viterbi_decode(encoded, k=7, g1=0o133, g2=0o171)
    assert res.valid is True
    assert res.corrected_bit_count == 0
    assert np.array_equal(res.decoded_bits, info_bits)

def test_viterbi_decode_with_errors():
    info_bits = np.random.randint(0, 2, 64, dtype=np.uint8)
    encoded = encode_convolutional(info_bits, k=7, g1=0o133, g2=0o171)
    
    # Flip 2 bits in the encoded stream
    corrupted = encoded.copy()
    corrupted[10] ^= 1
    corrupted[30] ^= 1

    res = viterbi_decode(corrupted, k=7, g1=0o133, g2=0o171)
    assert res.valid is True
    assert res.corrected_bit_count == 2
    assert np.array_equal(res.decoded_bits, info_bits)
    assert res.correction_mask[10] is True or res.correction_mask[10] == 1
    assert res.correction_mask[30] is True or res.correction_mask[30] == 1

def test_viterbi_overcorrection_detection():
    # If 40% of bits are corrupted, should declare over-correction
    info_bits = np.random.randint(0, 2, 64, dtype=np.uint8)
    corrupted = np.random.randint(0, 2, len(info_bits) * 2 + 12, dtype=np.uint8)
    res = viterbi_decode(corrupted, k=7, max_correction_fraction=0.10)
    assert res.is_overcorrected is True
    assert res.valid is False

def test_hamming_7_4_clean_and_error():
    # 4 info bits
    info = np.array([1, 0, 1, 1], dtype=np.uint8)
    # Generator matrix encoding for (7, 4)
    # Data bits at positions 2, 4, 5, 6
    # p0 = d0 ^ d1 ^ d3 = 1 ^ 0 ^ 1 = 0
    # p1 = d0 ^ d2 ^ d3 = 1 ^ 1 ^ 1 = 1
    # p2 = d1 ^ d2 ^ d3 = 0 ^ 1 ^ 1 = 0
    # Codeword: [p0, p1, d0, p2, d1, d2, d3] = [0, 1, 1, 0, 0, 1, 1]
    codeword = np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
    
    # 1. Clean decode
    res_clean = decode_hamming_7_4(codeword)
    assert np.array_equal(res_clean.decoded_bits, info)
    assert res_clean.corrected_bit_count == 0

    # 2. 1-bit error at position 2 (flip d0 from 1 to 0)
    corrupted = codeword.copy()
    corrupted[2] ^= 1
    res_corr = decode_hamming_7_4(corrupted)
    assert np.array_equal(res_corr.decoded_bits, info)
    assert res_corr.corrected_bit_count == 1
    assert res_corr.correction_mask[2] == True
