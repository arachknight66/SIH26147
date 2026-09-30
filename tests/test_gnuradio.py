from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signal_analysis.gnuradio import (
    GNUradioPreprocessConfig,
    _discover_gnuradio_runtime,
    gnuradio_runtime,
    preprocess_complex64,
)
from signal_analysis.loaders import RawIQConfig
from signal_analysis.models import MetadataStatus
from signal_analysis.workflow import AnalysisRequest, load_recording


@pytest.fixture(scope="module")
def runtime():
    value = gnuradio_runtime()
    if not value.available:
        pytest.skip(value.reason)
    return value


def test_gnuradio_sidecar_resamples_configured_complex64_stream(tmp_path, runtime):
    source = tmp_path / "source.cf32"
    np.exp(2j * np.pi * 1_000 * np.arange(4_000) / 20_000).astype(np.complex64).tofile(source)
    output = preprocess_complex64(source, GNUradioPreprocessConfig(20_000, 10_000))
    try:
        samples = np.fromfile(output, dtype=np.complex64)
    finally:
        output.unlink(missing_ok=True)
    assert 1_900 <= len(samples) <= 2_000
    assert np.isfinite(samples).all()


def test_gnuradio_runtime_probe_is_cached(runtime):
    _discover_gnuradio_runtime.cache_clear()
    first = gnuradio_runtime()
    second = gnuradio_runtime()
    assert first == second
    assert _discover_gnuradio_runtime.cache_info().hits == 1


def test_workflow_records_gnuradio_rate_as_assumed_configuration(tmp_path, runtime):
    source = tmp_path / "source.cf32"
    np.exp(2j * np.pi * 1_000 * np.arange(4_000) / 20_000).astype(np.complex64).tofile(source)
    request = AnalysisRequest(
        source,
        raw_iq_config=RawIQConfig("complex64", "IQ", "little", 20_000),
        gnuradio_preprocess=GNUradioPreprocessConfig(20_000, 10_000),
    )
    recording = load_recording(request)
    assert recording.sample_rate_hz.value == 10_000
    assert recording.sample_rate_hz.status is MetadataStatus.ASSUMED
    assert recording.provenance["loader"] == "GNUradioDirectRaw -> RawIQReader"
    assert 1_900 <= len(recording.samples) <= 2_000
    assert np.isfinite(recording.samples).all()


def test_direct_int16_iq_path_matches_unscaled_loader_values(tmp_path, runtime):
    source = tmp_path / "source.ci16"
    expected = np.array([100 + 200j, -300 + 400j, 500 - 600j], dtype=np.complex64)
    np.column_stack((expected.real, expected.imag)).astype("<i2").tofile(source)
    recording = load_recording(
        AnalysisRequest(source, raw_iq_config=RawIQConfig("int16", "IQ", "little", 10_000))
    )
    assert recording.provenance["loader"] == "GNUradioDirectRaw -> RawIQReader"
    assert recording.original_dtype == "int16"
    assert np.array_equal(recording.samples, expected)
