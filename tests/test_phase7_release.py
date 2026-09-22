import json

import numpy as np

from benchmarks.native_benchmark import run
from signal_analysis.release import PROFILE_CATALOG_VERSION, build_run_metadata
from signal_analysis.workflow import AnalysisRequest, ProductionAnalysisResult, WorkflowStatus


def test_run_metadata_is_versioned_and_excludes_request_path(tmp_path):
    request = AnalysisRequest(tmp_path / "private_capture.wav", pipeline_config={"fec_profile": "UNCODED"})
    result = ProductionAnalysisResult(request, None, None, WorkflowStatus.CANCELLED, 0, 3)

    metadata = build_run_metadata(result)

    assert metadata["schema_version"] == 1
    assert metadata["profile_catalog_version"] == PROFILE_CATALOG_VERSION
    assert metadata["processing"] == {
        "completed_stages": 0,
        "total_stages": 3,
        "input_samples": None,
        "input_coverage": "not_loaded",
    }
    assert str(request.path) not in json.dumps(metadata)


def test_native_benchmark_report_is_deterministic_in_shape():
    report = run(samples=2048, iterations=1, seed=9)

    assert report["benchmark_schema_version"] == 1
    assert report["native"]["api_version"] == 5
    assert report["measurements"]["spectral_seconds_median"] > 0
    assert report["measurements"]["viterbi_llrs_per_second"] > 0


def test_negative_noise_windows_do_not_create_confirmed_frames():
    """Small seeded smoke corpus: unknown/noise inputs must not claim a confirmed frame."""
    from signal_analysis.models import PipelineStageStatus
    from signal_analysis.pipeline import run_full_pipeline
    from signal_analysis.models import (
        MetadataStatus,
        MetadataValue,
        SignalRecording,
        SourceFormat,
    )

    rng = np.random.default_rng(26147)
    for _ in range(16):
        samples = np.ascontiguousarray(
            (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
        )
        recording = SignalRecording(
            samples=samples,
            source_format=SourceFormat.RAW_IQ,
            original_dtype="complex64",
            semantic_type="complex_iq",
            sample_rate_hz=MetadataValue(1_000_000.0, "test", MetadataStatus.KNOWN),
            center_frequency_hz=MetadataValue(None, "test", MetadataStatus.MISSING),
            provenance={},
            diagnostics=[],
        )
        pipeline = run_full_pipeline(recording)
        assert pipeline.framing_status is not PipelineStageStatus.COMPLETED or pipeline.frame_structure is None
