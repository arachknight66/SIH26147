import numpy as np
import pytest
from app.models.metadata import MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import SignalRecording, SourceFormat
from app.modulation.analyzer import analyze_all_regions, analyze_modulation
from app.modulation.models import (
    HypothesisStatus,
    ModulationAnalysisConfig,
    ModulationFamily,
)
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_pipeline_single_carrier_qpsk():
    samples, manifest = generate_modulated_signal("QPSK", snr_db=20.0, seed=42)
    rec = SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )

    result = analyze_modulation(rec)

    assert result.selected_hypothesis is not None
    assert result.selected_hypothesis.family == ModulationFamily.PSK
    assert result.selected_hypothesis.order == 4
    assert result.selected_hypothesis.quality in ("HIGH", "MODERATE")
    assert not result.is_unknown
    assert not result.is_ambiguous
    assert result.window_consistency >= 0.75
    assert "provenance" in result.__dict__
    assert result.raw_distribution is not None
    assert len(result.raw_distribution.sample_subset_i) > 0
    assert result.provenance["ml_metadata"] is None

def test_pipeline_enforces_hypothesis_limit_without_ml():
    samples, _ = generate_modulated_signal("QPSK", snr_db=20.0, seed=42)
    rec = SignalRecording(samples=samples, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
    result = analyze_modulation(rec, config=ModulationAnalysisConfig(enable_ml=False, max_hypotheses=2))
    assert len(result.hypotheses) <= 2
    assert result.provenance["ml_metadata"] is None

def test_pipeline_frequency_offset_shifted_region():
    # Signal with 0.10 carrier frequency offset
    cfo = 0.10
    samples, _ = generate_modulated_signal("BPSK", cfo_normalized=cfo, snr_db=20.0, seed=42)
    rec = SignalRecording(
        samples=samples,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )

    result = analyze_modulation(rec)
    assert result.selected_hypothesis is not None
    assert result.selected_hypothesis.family == ModulationFamily.PSK
    assert result.selected_hypothesis.order == 2

def test_pipeline_multi_region_batch():
    # Generate multi-tone / multi-signal recording
    s1, _ = generate_modulated_signal("BPSK", cfo_normalized=-0.20, snr_db=25.0, seed=42)
    s2, _ = generate_modulated_signal("QPSK", cfo_normalized=+0.20, snr_db=25.0, seed=43)
    mix = (s1 + s2).astype(np.complex64)
    rec = SignalRecording(
        samples=mix,
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
    )

    results = analyze_all_regions(rec)
    assert len(results) >= 1
