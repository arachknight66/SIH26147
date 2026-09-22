"""Shared production loading and analysis workflow for GUI, CLI, and Demo Mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .loaders import RawIQConfig, RawIQReader, WavReader, read_sigmf
from .models import SignalRecording
from .pipeline import run_full_pipeline


class WorkflowStatus(Enum):
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class AnalysisRequest:
    """User-visible import and processing choices; no demo truth is carried here."""

    path: Path
    wav_stereo_mode: str = "unresolved"
    raw_iq_config: RawIQConfig | None = None
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    origin: str = "file"


@dataclass(frozen=True)
class ProductionAnalysisResult:
    request: AnalysisRequest
    recording: SignalRecording | None
    pipeline_result: Any | None
    execution_status: WorkflowStatus
    processed_stages: int
    total_stages: int


def load_recording(request: AnalysisRequest) -> SignalRecording:
    """The sole import adapter used by CLI, GUI, and Demo Mode."""
    path = request.path
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return WavReader(str(path), mode=request.wav_stereo_mode).read()
    if str(path).lower().endswith(".sigmf-meta"):
        return read_sigmf(str(path))
    if request.raw_iq_config is None:
        raise ValueError("raw IQ input requires explicit import parameters")
    return RawIQReader(str(path), request.raw_iq_config).read()


def run_production_analysis(
    request: AnalysisRequest,
    cancellation: Any | None = None,
    progress: Any | None = None,
) -> ProductionAnalysisResult:
    """Execute one ordinary analysis request with coarse cancellation boundaries."""
    total = 3
    if progress is not None:
        progress.update(0, total)
    if cancellation is not None and cancellation.is_cancelled:
        return ProductionAnalysisResult(request, None, None, WorkflowStatus.CANCELLED, 0, total)
    recording = load_recording(request)
    if progress is not None:
        progress.update(1, total)
    if cancellation is not None and cancellation.is_cancelled:
        return ProductionAnalysisResult(request, recording, None, WorkflowStatus.CANCELLED, 1, total)
    pipeline_result = run_full_pipeline(recording, dict(request.pipeline_config))
    if progress is not None:
        progress.update(3, total)
    status = WorkflowStatus.CANCELLED if cancellation is not None and cancellation.is_cancelled else WorkflowStatus.COMPLETED
    return ProductionAnalysisResult(request, recording, pipeline_result, status, 3, total)
