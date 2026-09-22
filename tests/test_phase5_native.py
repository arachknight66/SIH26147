from __future__ import annotations

import numpy as np
import pytest

from signal_analysis.native import (
    decode_ldpc,
    decode_reed_solomon,
    deinterleave,
    encode_reed_solomon,
    interleave_bits,
    require_native,
)


def _config(family: str):
    native = require_native()
    config = native.NativeInterleaverConfig()
    config.family = getattr(native.InterleaverFamily, family)
    return config


@pytest.mark.parametrize("family", ["BLOCK", "CONVOLUTIONAL", "DIAGONAL", "PSEUDO_RANDOM"])
def test_configured_interleavers_round_trip_bits_and_llrs(family):
    rng = np.random.default_rng(26147)
    bits = np.ascontiguousarray(rng.integers(0, 2, 192, dtype=np.uint8))
    llrs = np.ascontiguousarray(np.where(bits == 1, 3.0, -3.0), dtype=np.float32)
    config = _config(family)
    if family in {"BLOCK", "DIAGONAL"}:
        config.rows, config.columns = 8, 24
        config.delay = 3
    elif family == "CONVOLUTIONAL":
        config.branches, config.delay = 8, 24
    else:
        config.seed = 26147
    transmitted = interleave_bits(bits, config=config)
    transmitted_llrs = np.asarray(require_native().interleave_llrs(llrs, config), dtype=np.float32)
    received = deinterleave(transmitted, transmitted_llrs, config=config)
    np.testing.assert_array_equal(received.bits, bits)
    np.testing.assert_array_equal(received.llrs, llrs)
    assert received.residual_bits == 0


def test_interleaver_preserves_incomplete_tail_with_evidence():
    bits = np.arange(67, dtype=np.uint8) & 1
    llrs = np.where(bits == 1, 2.0, -2.0).astype(np.float32)
    config = _config("BLOCK")
    config.rows, config.columns = 4, 16
    result = deinterleave(np.ascontiguousarray(bits), np.ascontiguousarray(llrs), config=config)
    assert result.residual_bits == 3
    np.testing.assert_array_equal(result.bits[-3:], bits[-3:])
    assert result.diagnostics[0].code == "INTERLEAVER_RESIDUAL_BITS"


@pytest.mark.parametrize("k", [223, 239])
def test_native_rs_profiles_correct_bound_errors_and_reject_residual(k):
    native = require_native()
    config = native.ReedSolomonConfig()
    config.n, config.k = 255, k
    message = np.arange(k, dtype=np.uint8)
    word = encode_reed_solomon(np.ascontiguousarray(message), config=config)
    corrupted = word.copy()
    for index in range((255 - k) // 2):
        corrupted[index] ^= 0x5A
    result = decode_reed_solomon(np.ascontiguousarray(corrupted), config=config)
    assert result.success
    assert result.corrected_symbols == (255 - k) // 2
    np.testing.assert_array_equal(result.decoded_bytes, message)
    over_capacity = word.copy()
    for index in range((255 - k) // 2 + 1):
        over_capacity[index] ^= 0xA5
    rejected = decode_reed_solomon(np.ascontiguousarray(over_capacity), config=config)
    assert not rejected.success
    assert rejected.diagnostics
    residual = decode_reed_solomon(np.ascontiguousarray(np.append(corrupted, 1).astype(np.uint8)), config=config)
    assert not residual.success
    assert residual.residual_bytes == 1


def test_native_ldpc_validates_syndrome_and_reports_non_convergence():
    native = require_native()
    matrix = native.LdpcMatrix()
    matrix.variable_count = 3
    matrix.checks = [[0, 1], [1, 2]]
    valid = decode_ldpc(np.ascontiguousarray(np.array([-4.0, -3.0, -2.0], np.float32)), matrix=matrix)
    assert valid.converged
    np.testing.assert_array_equal(valid.decoded_bits, np.zeros(3, np.uint8))
    config = native.LdpcConfig()
    config.maximum_iterations = 1
    invalid = decode_ldpc(np.ascontiguousarray(np.array([4.0, -4.0, -4.0], np.float32)), matrix=matrix, config=config)
    assert not invalid.converged
    assert invalid.diagnostics[0].code == "LDPC_MAX_ITERATIONS"


def test_native_correlation_and_crc_use_real_bit_evidence():
    native = require_native()
    bits = np.zeros(128, dtype=np.uint8)
    pattern = np.array([1, 0, 1, 1, 0, 1, 0, 0], np.uint8)
    bits[[8, 10, 11, 13, 32, 34, 35, 37, 56, 58, 59, 61]] = 1
    llrs = np.where(bits == 1, 5.0, -5.0).astype(np.float32)
    item = native.CorrelationPattern()
    item.name, item.bits = "test", pattern.tolist()
    matches = native.correlate_bits(np.ascontiguousarray(bits), np.ascontiguousarray(llrs), [item])
    assert any(match.bit_offset == 8 and match.periodic for match in matches)
    crc = native.CrcConfig()
    crc.name, crc.width, crc.polynomial, crc.initial, crc.xor_output = "CRC-8", 8, 0x07, 0, 0
    assert native.crc_bits(np.ascontiguousarray(pattern), crc) == 0x05
