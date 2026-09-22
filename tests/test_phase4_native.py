from __future__ import annotations

import numpy as np
import pytest

from signal_analysis.native import demodulate, require_native


def _profiles():
    root2 = np.sqrt(2.0)
    qpsk_points = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / root2
    qpsk_bits = np.array([[1, 1], [0, 1], [0, 0], [1, 0]], np.uint8)
    return {
        "BPSK": (np.array([-1, 1], complex), np.array([[0], [1]], np.uint8)),
        "QPSK": (qpsk_points, qpsk_bits),
        "8PSK": (
            np.exp(2j * np.pi * np.arange(8) / 8),
            np.array([[int(bit) for bit in f"{index ^ (index >> 1):03b}"] for index in range(8)], np.uint8),
        ),
    }


def _qam_profile(side: int):
    axis_bits = int(np.log2(side))
    points = []
    labels = []
    for q_row in range(side):
        for i_column in range(side):
            points.append(2 * i_column - side + 1 + 1j * (side - 1 - 2 * q_row))
            q = f"{q_row ^ (q_row >> 1):0{axis_bits}b}"
            i = f"{i_column ^ (i_column >> 1):0{axis_bits}b}"
            labels.append([int(value) for pair in zip(q, i) for value in pair])
    points = np.asarray(points, complex)
    points /= np.sqrt(np.mean(np.abs(points) ** 2))
    return points, np.asarray(labels, np.uint8)


def _linear_fixture(modulation: str, seed: int, *, symbols: int = 700, sps: int = 4,
                    cfo: float = 0.027, snr_db: float | None = None):
    rng = np.random.default_rng(seed)
    profiles = _profiles()
    if modulation in profiles:
        points, labels = profiles[modulation]
    else:
        points, labels = _qam_profile({"16-QAM": 4, "64-QAM": 8, "256-QAM": 16}[modulation])
    choices = rng.integers(len(points), size=symbols)
    transmitted = points[choices]
    samples = np.repeat(transmitted, sps) * np.exp(2j * np.pi * cfo * np.arange(symbols * sps))
    if snr_db is not None:
        sigma = 10 ** (-snr_db / 20) / np.sqrt(2)
        samples += sigma * (rng.standard_normal(samples.size) + 1j * rng.standard_normal(samples.size))
    return np.ascontiguousarray(samples, dtype=np.complex64), labels[choices].reshape(-1)


def _receiver_config(modulation: str, *, phase_reference=True, acquisition=12):
    native = require_native()
    config = native.ReceiverConfig()
    config.modulation = modulation
    config.samples_per_symbol = 4.0
    config.acquisition_symbols = acquisition
    config.carrier_reference_cycles_per_sample = 0.027
    if phase_reference:
        config.phase_reference_radians = 0.0
    return config


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "256-QAM"])
def test_noiseless_linear_profiles_are_exact_after_declared_acquisition(modulation):
    samples, expected = _linear_fixture(modulation, 100 + len(modulation))
    config = _receiver_config(modulation)
    result = demodulate(samples, config=config)
    assert result.acquisition_status.name == "LOCKED"
    assert result.mapping_status.name == "VERIFIED"
    np.testing.assert_array_equal(result.hard_bits, expected[config.acquisition_symbols * result.bits_per_symbol :])
    assert np.all((result.soft_llrs > 0) == (result.hard_bits == 1))


def test_noisy_receiver_matches_independent_known_clock_reference():
    samples, expected = _linear_fixture("QPSK", 818, snr_db=12)
    config = _receiver_config("QPSK")
    result = demodulate(samples, config=config)
    native_bits = np.asarray(result.hard_bits)
    expected = expected[config.acquisition_symbols * 2 : config.acquisition_symbols * 2 + native_bits.size]
    native_ber = np.mean(native_bits != expected)

    points, labels = _profiles()["QPSK"]
    positions = np.arange(config.acquisition_symbols, 700) * 4
    reference_symbols = samples[positions] * np.exp(-2j * np.pi * 0.027 * positions)
    nearest = np.argmin(np.abs(reference_symbols[:, None] - points[None, :]), axis=1)
    reference_bits = labels[nearest].reshape(-1)[: expected.size]
    reference_ber = np.mean(reference_bits != expected)
    assert native_ber == pytest.approx(reference_ber, abs=0.002)


def test_unresolved_carrier_rotations_are_retained_and_not_marked_verified():
    samples, _ = _linear_fixture("QPSK", 222)
    result = demodulate(samples, config=_receiver_config("QPSK", phase_reference=False))
    assert result.acquisition_status.name == "LOCKED"
    assert result.mapping_status.name == "UNVERIFIED"
    assert len(result.unresolved_phase_rotations) == 4
    assert any(item.code == "PHASE_AMBIGUITY_UNRESOLVED" for item in result.diagnostics)


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_fractional_timing_offsets_lock_and_recover_after_alignment(fraction):
    samples, expected = _linear_fixture("QPSK", 889, snr_db=25)
    delay = fraction * 4
    frequencies = np.fft.fftfreq(samples.size)
    delayed = np.fft.ifft(np.fft.fft(samples) * np.exp(-2j * np.pi * frequencies * delay))
    config = _receiver_config("QPSK")
    config.phase_reference_radians = -2 * np.pi * 0.027 * delay
    result = demodulate(np.ascontiguousarray(delayed, dtype=np.complex64), config=config)
    assert result.acquisition_status.name == "LOCKED"
    # A circular fractional delay can shift the declared acquisition boundary by one symbol.
    observed = np.asarray(result.hard_bits)
    error_rates = []
    for symbol_shift in [-1, 0, 1]:
        start = (config.acquisition_symbols + symbol_shift) * 2
        target = expected[max(start, 0) : max(start, 0) + observed.size]
        error_rates.append(np.mean(observed[: target.size] != target))
    assert min(error_rates) == 0.0


def test_receiver_session_is_chunk_invariant_and_owns_result_storage():
    native = require_native()
    samples, _ = _linear_fixture("BPSK", 333, snr_db=18)
    config = _receiver_config("BPSK")
    direct = demodulate(samples, config=config)
    session = native.ReceiverSession(config)
    offset = 0
    for size in [1, 31, 700, samples.size - 732]:
        assert session.process(samples[offset : offset + size]) == size
        offset += size
    chunked = session.flush()
    np.testing.assert_array_equal(chunked.hard_bits, direct.hard_bits)
    np.testing.assert_array_equal(chunked.soft_llrs, direct.soft_llrs)
    retained = np.asarray(chunked.hard_bits)
    del chunked, session
    assert retained.flags.owndata is False
    assert retained.size > 0


@pytest.mark.parametrize("states", [2, 4, 8])
def test_fsk_correlator_profiles_recover_declared_tone_mapping(states):
    rng = np.random.default_rng(900 + states)
    choices = rng.integers(states, size=800)
    tones = (choices - (states - 1) / 2) * 0.08
    phase = np.cumsum(np.repeat(tones, 4) * 2 * np.pi)
    samples = np.ascontiguousarray(np.exp(1j * phase), dtype=np.complex64)
    config = _receiver_config(f"{states}-FSK", phase_reference=False, acquisition=12)
    result = demodulate(samples, config=config)
    assert result.acquisition_status.name == "LOCKED"
    # The first averaged frequency interval starts at the next symbol; sample offsets make this explicit.
    first_symbol = int((result.sample_offsets[0] + 1) // 4)
    width = int(np.log2(states))
    expected = []
    for choice in choices[first_symbol : first_symbol + len(result.symbols)]:
        expected.extend(int(bit) for bit in f"{choice ^ (choice >> 1):0{width}b}")
    np.testing.assert_array_equal(result.hard_bits, np.asarray(expected, np.uint8))


@pytest.mark.parametrize("profile", ["GMSK", "GFSK", "CPM", "PI4-DQPSK"])
def test_specialized_profiles_fail_explicitly_until_implemented(profile):
    samples, _ = _linear_fixture("BPSK", 444)
    config = _receiver_config(profile)
    result = demodulate(samples, config=config)
    assert result.acquisition_status.name == "UNSUPPORTED"
    assert result.hard_bits.size == 0
    assert result.diagnostics[0].code == "RECEIVER_PROFILE_UNSUPPORTED"


@pytest.mark.parametrize(("profile", "states"), [("DBPSK", 2), ("DQPSK", 4)])
def test_differential_profiles_remove_constant_phase_ambiguity(profile, states):
    rng = np.random.default_rng(1200 + states)
    increments = rng.integers(states, size=700)
    phase = np.cumsum(increments * 2 * np.pi / states)
    symbols = np.exp(1j * phase)
    samples = np.repeat(symbols, 4) * np.exp(2j * np.pi * 0.027 * np.arange(symbols.size * 4))
    config = _receiver_config(profile, phase_reference=False)
    result = demodulate(np.ascontiguousarray(samples, dtype=np.complex64), config=config)
    assert result.acquisition_status.name == "LOCKED"
    expected = []
    width = int(np.log2(states))
    for increment in increments[config.acquisition_symbols :]:
        expected.extend(int(bit) for bit in f"{increment ^ (increment >> 1):0{width}b}")
    np.testing.assert_array_equal(result.hard_bits, np.asarray(expected, np.uint8))
    assert result.unresolved_phase_rotations == []


def test_msk_uses_frequency_receiver():
    rng = np.random.default_rng(1400)
    bits = rng.integers(2, size=800)
    tones = np.where(bits == 1, 0.0625, -0.0625)
    tones_up = np.repeat(tones, 4)
    samples = np.exp(1j * np.cumsum(tones_up) * 2 * np.pi)
    config = _receiver_config("MSK", phase_reference=False)
    result = demodulate(np.ascontiguousarray(samples, dtype=np.complex64), config=config)
    assert result.acquisition_status.name == "LOCKED"
    first_symbol = int((result.sample_offsets[0] + 1) // 4)
    expected = bits[first_symbol : first_symbol + result.hard_bits.size]
    assert np.mean(result.hard_bits != expected) < 0.01


def test_low_snr_does_not_claim_lock_and_buffer_limit_is_enforced():
    samples, _ = _linear_fixture("QPSK", 555, snr_db=0)
    result = demodulate(samples, config=_receiver_config("QPSK"))
    assert result.acquisition_status.name == "UNLOCKED"

    native = require_native()
    config = _receiver_config("BPSK")
    config.maximum_buffered_samples = 100
    session = native.ReceiverSession(config)
    with pytest.raises(ValueError, match="buffer limit"):
        session.process(np.ones(101, np.complex64))
