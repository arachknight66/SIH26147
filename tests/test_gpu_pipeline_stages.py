"""CUDA stage vectors: exact simple controls plus CPU/GPU evidence parity."""

import numpy as np
import pytest

from signal_analysis.correlation import correlate_sync_words
from signal_analysis.demodulation import attempt_synchronization
from signal_analysis.fec_convolutional import viterbi_decode_soft
from signal_analysis.fec_reed_solomon import decode_reed_solomon
from signal_analysis.gpu_dsp import gpu_rs_zero_syndrome
from signal_analysis.models import (
    DeinterleaverFamily,
    DeinterleaverHypothesis,
    DeinterleavingResult,
    HypothesisStatus,
    SyncWordPattern,
)
from signal_analysis.native import encode_reed_solomon, require_native
from tests.test_sync_demod import make_hypothesis, make_recording
from tests.test_synthesis import generate_synthetic_signal


@pytest.fixture(scope="module", autouse=True)
def cuda_available():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device unavailable")


def _encoded_k7(bits: np.ndarray) -> np.ndarray:
    out = []
    state = 0
    for bit in bits:
        state = (int(bit) << 6) | state
        out.extend([(state & 0o171).bit_count() & 1, (state & 0o133).bit_count() & 1])
        state >>= 1
    return np.asarray(out, dtype=np.uint8)


def test_gpu_viterbi_recovers_noiseless_k7_reference_vector():
    message = np.random.default_rng(26147).integers(0, 2, 2048, dtype=np.uint8)
    coded = _encoded_k7(message)
    llrs = np.where(coded == 1, 7.0, -7.0).astype(np.float32)
    hypothesis = DeinterleaverHypothesis(DeinterleaverFamily.NONE, {}, 0.0, [], HypothesisStatus.HYPOTHESIS_UNVERIFIED)
    result = viterbi_decode_soft(DeinterleavingResult(coded, llrs, hypothesis, 0.0), backend="gpu")
    np.testing.assert_array_equal(result.decoded_bits, message)
    assert any(item.code == "GPU_VITERBI_K7_R12" for item in result.diagnostics)


def test_gpu_receiver_decisions_preserve_llr_sign_contract():
    samples, _ = generate_synthetic_signal("QPSK", n_symbols=1000, sps=4, snr_db=30.0, pulse_shape="rrc", return_bits=True)
    result = attempt_synchronization(make_recording(samples, 4), make_hypothesis("QPSK", 4), {"compute_backend": "gpu"})
    assert result.hypothesis_confirmed
    assert np.all((result.soft_llrs > 0) == (result.hard_bits == 1))
    assert any(item.code == "GPU_RECEIVER_DECISIONS" for item in result.sync_result.diagnostics)


def test_gpu_correlation_matches_cpu_evidence():
    pattern = SyncWordPattern("test", np.array([1, 0, 1, 1, 0, 1, 0, 0], np.uint8), "", "", "")
    bits = np.zeros(128, dtype=np.uint8)
    bits[[8, 10, 11, 13, 32, 34, 35, 37, 56, 58, 59, 61]] = 1
    llrs = np.where(bits == 1, 5.0, -5.0).astype(np.float32)
    cpu = correlate_sync_words(bits, llrs, [pattern], backend="cpu")
    gpu = correlate_sync_words(bits, llrs, [pattern], backend="gpu")
    assert [(item.bit_offset, item.hamming_distance, item.periodicity_consistent) for item in gpu] == [
        (item.bit_offset, item.hamming_distance, item.periodicity_consistent) for item in cpu
    ]


def test_gpu_rs_zero_syndrome_screen_accepts_clean_and_rejects_corrupt_word():
    native = require_native()
    config = native.ReedSolomonConfig()
    config.n, config.k = 255, 223
    word = encode_reed_solomon(np.arange(223, dtype=np.uint8), config=config)
    np.testing.assert_array_equal(gpu_rs_zero_syndrome(word, 255, 223), np.array([True]))
    word[7] ^= 0x55
    np.testing.assert_array_equal(gpu_rs_zero_syndrome(word, 255, 223), np.array([False]))


def test_gpu_rs_profile_uses_screen_for_clean_codeword():
    native = require_native()
    config = native.ReedSolomonConfig()
    config.n, config.k = 255, 223
    word = encode_reed_solomon(np.arange(223, dtype=np.uint8), config=config)
    hypothesis = DeinterleaverHypothesis(DeinterleaverFamily.NONE, {}, 0.0, [], HypothesisStatus.HYPOTHESIS_UNVERIFIED)
    result = decode_reed_solomon(
        DeinterleavingResult(np.unpackbits(word), np.zeros(word.size * 8, np.float32), hypothesis, 0.0), backend="gpu"
    )
    assert result.decode_success
    assert any(item.code == "GPU_RS_SYNDROME" for item in result.diagnostics)
