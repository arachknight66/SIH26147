import numpy as np
import pytest
from app.dsp.autocorrelation import compute_autocorrelation
from app.dsp.rate_estimation import estimate_symbol_rate_candidates
from app.models.metadata import MetadataStatus

def test_rate_estimation_pulsed_signal():
    np.random.seed(42)
    # Generate BPSK-like signal with 8 samples per symbol (pulse shaped)
    sps = 8
    n_symbols = 512
    bits = 2 * np.random.randint(0, 2, n_symbols) - 1.0
    upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = bits
    h = np.hanning(sps * 2).astype(np.float32)
    signal = np.convolve(upsampled, h, mode="same").astype(np.complex64)

    autocorr = compute_autocorrelation(signal, max_lag=64)
    candidates = estimate_symbol_rate_candidates(signal, autocorr_result=autocorr)

    assert len(candidates) > 0
    top_cand = candidates[0]
    assert top_cand.status == MetadataStatus.AMBIGUOUS
    # Normalized rate should be near 1/8 = 0.125
    assert np.isclose(top_cand.normalized_rate, 1.0 / sps, atol=0.01)
    assert np.isclose(top_cand.estimated_samples_per_symbol, sps, atol=0.5)
