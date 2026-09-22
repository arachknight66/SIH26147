"""Versioned, JSON-safe analysis provenance for beta validation and reporting."""

from __future__ import annotations

from typing import Any

from .native import runtime_info
from .workflow import ProductionAnalysisResult


REPORT_SCHEMA_VERSION = 1
PROFILE_CATALOG_VERSION = "2026-09-18"


def build_run_metadata(result: ProductionAnalysisResult) -> dict[str, Any]:
    """Return stable provenance without serializing samples or hidden demo truth."""
    runtime = runtime_info()
    recording = result.recording
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "engine": {
            "native_available": runtime.available,
            "native_version": runtime.version,
            "native_api_version": runtime.api_version,
        },
        "execution_status": result.execution_status.value,
        "profile_catalog_version": PROFILE_CATALOG_VERSION,
        "selected_profiles": dict(result.request.pipeline_config),
        "processing": {
            "completed_stages": result.processed_stages,
            "total_stages": result.total_stages,
            "input_samples": None if recording is None else len(recording.samples),
            "input_coverage": "complete_in_memory" if recording is not None else "not_loaded",
        },
        "source": {
            "origin": result.request.origin,
            "format": None if recording is None else recording.source_format.value,
            "wav_stereo_mode": result.request.wav_stereo_mode,
        },
    }
