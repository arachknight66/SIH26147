from __future__ import annotations

import numpy as np
import pytest

from signal_analysis.analysis import analyze_modulation
from signal_analysis.models import MetadataStatus, MetadataValue, SignalRecording, SourceFormat
from signal_analysis.native import analyze_window, require_native


def _awgn(values: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = np.mean(np.abs(values) ** 2)
    sigma = np.sqrt(power * 10 ** (-snr_db / 10) / 2)
    return values + sigma * (rng.standard_normal(values.size) + 1j * rng.standard_normal(values.size))


def _linear_signal(kind: str, seed: int, *, sps: int = 4, snr_db: float = 15, cfo: float = 0.025) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = 2400
    if kind.endswith("PSK"):
        order = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[kind]
        symbols = np.exp(2j * np.pi * rng.integers(order, size=count) / order)
    else:
        side = {"16-QAM": 4, "64-QAM": 8, "256-QAM": 16}[kind]
        levels = np.arange(-(side - 1), side, 2)
        symbols = rng.choice(levels, count) + 1j * rng.choice(levels, count)
        symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))
    values = np.repeat(symbols, sps)
    values *= np.exp(2j * np.pi * cfo * np.arange(values.size))
    return np.ascontiguousarray(_awgn(values, snr_db, rng), dtype=np.complex64)


def _fsk_signal(states: int, seed: int, *, sps: int = 4, snr_db: float = 15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    choices = rng.integers(states, size=2400)
    tones = (choices - (states - 1) / 2) * 0.08
    phase = np.cumsum(np.repeat(tones, sps) * 2 * np.pi)
    values = np.exp(1j * phase) * np.exp(2j * np.pi * 0.01 * np.arange(phase.size))
    return np.ascontiguousarray(_awgn(values, snr_db, rng), dtype=np.complex64)


def _config(*, windows: int = 1):
    config = require_native().AnalysisConfig()
    config.temporal_windows = windows
    return config


@pytest.mark.parametrize("kind", ["BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "256-QAM"])
@pytest.mark.parametrize("seed", [101, 203])
def test_supported_linear_families_and_rate_on_held_out_seeds(kind, seed):
    result = analyze_window(_linear_signal(kind, seed), config=_config())
    expected_family = "QAM" if "QAM" in kind else "PSK"
    assert result.modulation_candidates[0].family == expected_family
    assert result.status.name in {"CANDIDATE", "AMBIGUOUS"}
    assert result.symbol_rate.validity.name in {"VALID", "UNRELIABLE"}
    assert result.symbol_rate.value == pytest.approx(0.25, rel=0.02)
    assert abs(result.carrier_offset.value - 0.025) <= max(result.carrier_offset.uncertainty, 0.002)


@pytest.mark.parametrize("states", [2, 4, 8])
def test_fsk_family_tone_clustering_and_rate(states):
    result = analyze_window(_fsk_signal(states, 700 + states), config=_config())
    assert result.modulation_candidates[0].family == "FSK"
    assert result.symbol_rate.value == pytest.approx(0.25, rel=0.02)


def test_measured_units_uncertainty_and_python_pipeline_adapter():
    samples = _linear_signal("QPSK", 991, snr_db=22)
    recording = SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(48_000.0, "test header", MetadataStatus.KNOWN),
        center_frequency_hz=MetadataValue(None, "missing", MetadataStatus.MISSING),
        provenance={},
        diagnostics=[],
    )
    hypotheses, selected, ambiguous, unknown, result = analyze_modulation(recording, {"window_count": 2})
    assert not unknown
    assert selected is not None or ambiguous
    parameters = hypotheses[0].candidate_parameters
    assert parameters.symbol_rate == pytest.approx(12_000.0, rel=0.02)
    assert parameters.symbol_rate_unit == "Hz"
    assert parameters.bandwidth_unit == "Hz"
    assert parameters.carrier_offset_unit == "Hz"
    assert parameters.snr_db is not None
    assert result.processed_samples == samples.size
    assert 0.0 <= result.temporal_consistency <= 1.0


def test_noise_silence_and_short_inputs_do_not_become_confident_candidates():
    rng = np.random.default_rng(4321)
    noise = np.ascontiguousarray(rng.standard_normal(8192) + 1j * rng.standard_normal(8192), dtype=np.complex64)
    noise_result = analyze_window(noise, config=_config())
    assert noise_result.status.name in {"UNKNOWN", "AMBIGUOUS"}
    assert not noise_result.modulation_candidates or noise_result.modulation_candidates[0].score < 0.8

    silence = analyze_window(np.zeros(1024, np.complex64), config=_config())
    assert silence.status.name == "UNKNOWN"
    assert not silence.modulation_candidates
    short = analyze_window(np.ones(128, np.complex64), config=_config())
    assert short.status.name == "UNKNOWN"
    assert "INSUFFICIENT_SAMPLES" in short.diagnostics


def test_real_input_reports_parameters_without_complex_modulation_claim():
    native = require_native()
    config = native.AnalysisConfig()
    config.complex_input = False
    config.sample_rate_hz = 48_000.0
    time = np.arange(8192) / 48_000.0
    samples = np.ascontiguousarray(np.sin(2 * np.pi * 3000 * time), dtype=np.complex64)
    result = analyze_window(samples, config=config)
    assert result.status.name == "UNKNOWN"
    assert not result.modulation_candidates
    assert result.occupied_bandwidth.unit == "Hz"
    assert result.carrier_offset.validity.name == "UNAVAILABLE"


def test_binding_rejects_implicit_dtype_rank_and_stride_conversion():
    with pytest.raises(TypeError):
        analyze_window(np.ones(1024, np.complex128), config=_config())
    with pytest.raises(ValueError):
        analyze_window(np.ones((32, 32), np.complex64), config=_config())
    with pytest.raises(ValueError):
        analyze_window(np.ones(2048, np.complex64)[::2], config=_config())
