import numpy as np
import pytest
import scipy.signal as signal
from app.dsp.autocorrelation import compute_autocorrelation
from app.dsp.rate_estimation import (
    estimate_rate_autocorrelation,
    estimate_rate_envelope_spectrum,
    estimate_rate_squared_magnitude,
    estimate_rate_transition_energy,
    estimate_symbol_rate_candidates,
    estimate_symbol_rate_consensus,
    METHOD_TRANSITION_ENERGY,
    METHOD_SQUARED_MAGNITUDE,
    METHOD_ENVELOPE,
    METHOD_AUTOCORRELATION,
)
from app.models.metadata import MetadataStatus


def _generate_synthetic_psk(
    sps: float = 8.0,
    n_symbols: int = 512,
    snr_db: float = 20.0,
    cfo_normalized: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic pulse-shaped BPSK/QPSK signal with configurable SPS, SNR, and CFO."""
    rng = np.random.default_rng(seed)
    
    # Generate random QPSK symbols
    syms = (rng.choice([-1.0, 1.0], size=n_symbols) + 1j * rng.choice([-1.0, 1.0], size=n_symbols)) / np.sqrt(2.0)
    
    total_samples = int(np.round(n_symbols * sps))
    upsampled = np.zeros(total_samples, dtype=np.complex64)
    for i, sym in enumerate(syms):
        idx = int(np.round(i * sps))
        if idx < total_samples:
            upsampled[idx] = sym

    pulse_len = max(4, int(np.round(sps * 2)))
    if pulse_len % 2 == 0:
        pulse_len += 1
    h = np.hanning(pulse_len).astype(np.float32)
    
    shaped = np.convolve(upsampled, h, mode="same").astype(np.complex64)
    
    # Apply CFO
    if abs(cfo_normalized) > 1e-9:
        t = np.arange(len(shaped))
        shaped *= np.exp(2j * np.pi * cfo_normalized * t).astype(np.complex64)
        
    # Add AWGN
    sig_power = float(np.mean(np.abs(shaped) ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = (rng.normal(0, np.sqrt(noise_power / 2.0), len(shaped)) + 
             1j * rng.normal(0, np.sqrt(noise_power / 2.0), len(shaped))).astype(np.complex64)
             
    return (shaped + noise).astype(np.complex64)


# -------------------------------------------------------------
# 1. Ground Truth Recovery across SPS and SNR
# -------------------------------------------------------------
@pytest.mark.parametrize("sps", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("snr_db", [10.0, 20.0])
def test_rate_estimation_ground_truth_recovery(sps, snr_db):
    sig = _generate_synthetic_psk(sps=sps, n_symbols=512, snr_db=snr_db, seed=int(sps * 10 + snr_db))
    autocorr = compute_autocorrelation(sig, max_lag=min(512, len(sig) - 1))
    candidates = estimate_symbol_rate_candidates(sig, autocorr_result=autocorr)

    assert len(candidates) > 0
    top_cand = candidates[0]
    expected_rate = 1.0 / sps

    # Cross-validated across independent estimators
    assert top_cand.status == MetadataStatus.ESTIMATED
    # SPS=4.0 receives coarse oversampling penalty (0.75x)
    min_expected_conf = 0.65 if sps <= 4.0 else 0.75
    assert top_cand.confidence >= min_expected_conf
    assert top_cand.normalized_rate is not None
    assert np.isclose(top_cand.normalized_rate, expected_rate, atol=0.015)
    assert np.isclose(top_cand.estimated_samples_per_symbol, sps, atol=0.8)
    # Uncertainty must be a valid positive number
    assert top_cand.uncertainty is not None
    assert top_cand.uncertainty > 0.0


# -------------------------------------------------------------
# 2. Calibration Test: True Rate inside Reported Uncertainty Interval
# -------------------------------------------------------------
def test_uncertainty_calibration_coverage():
    """
    Verify that across multiple independent trials, the true normalized symbol rate
    falls within [R_est - 2.5*sigma, R_est + 2.5*sigma] at expected statistical proportion.
    """
    sps_true = 8.0
    rate_true = 1.0 / sps_true
    n_trials = 20
    in_interval_count = 0

    for trial in range(n_trials):
        sig = _generate_synthetic_psk(sps=sps_true, n_symbols=384, snr_db=15.0, seed=1000 + trial)
        candidates = estimate_symbol_rate_candidates(sig)
        if candidates and candidates[0].normalized_rate is not None and candidates[0].uncertainty is not None:
            r_est = candidates[0].normalized_rate
            unc = max(candidates[0].uncertainty, 0.0005)
            # 2.5-sigma bounds (~98.7% theoretical normal coverage)
            if abs(r_est - rate_true) <= 2.5 * unc + 0.002:
                in_interval_count += 1

    coverage = in_interval_count / n_trials
    assert coverage >= 0.80, f"Uncertainty calibration failed: empirical coverage {coverage*100:.1f}% < 80%"


# -------------------------------------------------------------
# 3. Harmonic Aliasing Rejection
# -------------------------------------------------------------
def test_harmonic_aliasing_rejection():
    """
    Construct a candidate set containing fundamental (f0) and strong harmonic (2*f0, 3*f0)
    components; verify harmonic collapse selects the fundamental.
    """
    candidates_by_family = {
        METHOD_TRANSITION_ENERGY: [
            (0.125, 0.75, 0.002, ["Transition energy at fundamental."]),
            (0.250, 0.85, 0.003, ["Strong transition harmonic."]),  # 2x harmonic with higher raw score
        ],
        METHOD_SQUARED_MAGNITUDE: [
            (0.125, 0.70, 0.002, ["Squared magnitude baud line."]),
        ],
    }

    consensus = estimate_symbol_rate_consensus(candidates_by_family)
    assert len(consensus) > 0
    top = consensus[0]

    # Fundamental (0.125) must be primary, NOT the 2x harmonic (0.250)
    assert np.isclose(top.normalized_rate, 0.125, atol=0.005)
    # Assumptions must document the collapsed harmonic
    assert any("harmonic" in a.lower() for a in top.assumptions)


# -------------------------------------------------------------
# 4. Null Test: Pure Noise and Unmodulated Carrier
# -------------------------------------------------------------
def test_null_test_pure_noise():
    """Pure AWGN must yield no candidates clearing significance threshold."""
    rng = np.random.default_rng(42)
    noise = (rng.normal(0, 1.0, 4096) + 1j * rng.normal(0, 1.0, 4096)).astype(np.complex64)
    candidates = estimate_symbol_rate_candidates(noise, significance_thresh_db=4.0)
    assert len(candidates) == 0, f"False positive candidates detected on pure noise: {candidates}"


def test_null_test_unmodulated_cw_tone():
    """Unmodulated pure sinusoidal carrier must yield no symbol rate candidates."""
    t = np.arange(4096)
    cw = np.exp(2j * np.pi * 0.15 * t).astype(np.complex64)
    candidates = estimate_symbol_rate_candidates(cw, significance_thresh_db=4.0)
    assert len(candidates) == 0, f"False positive candidates detected on CW tone: {candidates}"


# -------------------------------------------------------------
# 5. Non-Integer Oversampling Test
# -------------------------------------------------------------
@pytest.mark.parametrize("sps_target", [5.333, 6.4])
def test_non_integer_oversampling(sps_target):
    """Confirm estimator recovers continuous non-integer rates without integer quantization."""
    sig = _generate_synthetic_psk(sps=sps_target, n_symbols=512, snr_db=20.0, seed=77)
    candidates = estimate_symbol_rate_candidates(sig)

    assert len(candidates) > 0
    top = candidates[0]
    expected_rate = 1.0 / sps_target
    assert np.isclose(top.normalized_rate, expected_rate, atol=0.01)
    assert np.isclose(top.estimated_samples_per_symbol, sps_target, atol=0.5)


# -------------------------------------------------------------
# 6. CFO Invariance Test
# -------------------------------------------------------------
def test_cfo_invariance_envelope_and_squared_magnitude():
    """Envelope and squared magnitude estimators must recover baud rate despite large CFO."""
    sps = 8.0
    cfo = 0.18  # Large carrier frequency offset
    sig = _generate_synthetic_psk(sps=sps, n_symbols=512, snr_db=20.0, cfo_normalized=cfo, seed=99)

    env_lines = estimate_rate_envelope_spectrum(sig)
    sq_lines = estimate_rate_squared_magnitude(sig)

    expected_rate = 1.0 / sps
    assert len(env_lines) > 0
    assert np.isclose(env_lines[0][0], expected_rate, atol=0.015)

    assert len(sq_lines) > 0
    assert np.isclose(sq_lines[0][0], expected_rate, atol=0.015)


# -------------------------------------------------------------
# 7. Sample Rate Propagation and Quadrature Uncertainty
# -------------------------------------------------------------
def test_sample_rate_propagation_and_quadrature_uncertainty():
    sig = _generate_synthetic_psk(sps=8.0, n_symbols=512, snr_db=20.0, seed=42)
    fs = 2_000_000.0  # 2 MHz

    # Case A: Known sample rate with full confidence (1.0)
    cands_full_conf = estimate_symbol_rate_candidates(sig, sample_rate_hz=fs, sample_rate_confidence=1.0)
    assert len(cands_full_conf) > 0
    top_a = cands_full_conf[0]
    assert top_a.rate_hz is not None
    assert np.isclose(top_a.rate_hz, 250_000.0, atol=5000.0)
    assert top_a.rate_hz_uncertainty is not None

    # Case B: Sample rate with partial confidence (0.90)
    cands_part_conf = estimate_symbol_rate_candidates(sig, sample_rate_hz=fs, sample_rate_confidence=0.90)
    top_b = cands_part_conf[0]
    # Quadrature combination with sample rate uncertainty must increase total Hz uncertainty
    assert top_b.rate_hz_uncertainty is not None
    assert top_b.rate_hz_uncertainty > top_a.rate_hz_uncertainty
    assert any("quadrature" in a.lower() for a in top_b.assumptions)

    # Case C: Metadata-free (sample_rate_hz is None) -> No Hz manufactured
    cands_no_fs = estimate_symbol_rate_candidates(sig, sample_rate_hz=None)
    top_c = cands_no_fs[0]
    assert top_c.rate_hz is None
    assert top_c.rate_hz_uncertainty is None


# -------------------------------------------------------------
# 8. Physical Plausibility & Bandwidth Gating
# -------------------------------------------------------------
def test_bandwidth_cross_check_penalty():
    """If candidate rate wildly exceeds 99% OBW, confidence is penalized and flagged."""
    candidates_by_family = {
        METHOD_TRANSITION_ENERGY: [(0.25, 0.85, 0.002, ["Transition line at 0.25."])],
        METHOD_SQUARED_MAGNITUDE: [(0.25, 0.80, 0.002, ["Squared magnitude at 0.25."])],
    }
    # Provide OBW of only 0.10 (much smaller than 0.25)
    cands = estimate_symbol_rate_consensus(candidates_by_family, occupied_bandwidth_normalized=0.10)
    assert len(cands) > 0
    top = cands[0]
    # Must be penalized and capped at AMBIGUOUS
    assert top.status == MetadataStatus.AMBIGUOUS
    assert top.confidence < 0.60
    assert any("exceeds measured 99% occupied bandwidth" in a for a in top.assumptions)

