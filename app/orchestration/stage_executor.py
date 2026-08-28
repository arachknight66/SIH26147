from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Generic, TypeVar
from app.models.metadata import Diagnostic, DiagnosticSeverity
from .cancellation import CancellationToken
from .failure_recovery import FailureCategory, PipelineFailure, classify_stage_failure

T = TypeVar("T")

@dataclass(frozen=True)
class StageResult(Generic[T]):
    stage_name: str
    phase_number: int
    success: bool
    output: T | None
    duration_seconds: float
    diagnostics: list[Diagnostic] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failure: PipelineFailure | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

def execute_stage(
    stage_name: str,
    phase_number: int,
    func: Callable[[], T],
    cancel_token: CancellationToken | None = None,
) -> StageResult[T]:
    """Execute a single pipeline stage with execution timing, error trapping, and cancellation checks."""
    if cancel_token:
        cancel_token.check()

    t0 = time.perf_counter()
    try:
        output = func()
        duration = time.perf_counter() - t0
        if cancel_token:
            cancel_token.check()

        return StageResult(
            stage_name=stage_name,
            phase_number=phase_number,
            success=True,
            output=output,
            duration_seconds=round(duration, 4),
        )
    except Exception as e:
        duration = time.perf_counter() - t0
        failure = classify_stage_failure(stage_name, e)
        diag = Diagnostic(
            severity=DiagnosticSeverity.ERROR,
            code=f"STAGE_ERROR_{stage_name.upper()}",
            message=str(e),
            evidence=f"Phase {phase_number} failure after {duration:.2f}s",
        )
        return StageResult(
            stage_name=stage_name,
            phase_number=phase_number,
            success=False,
            output=None,
            duration_seconds=round(duration, 4),
            diagnostics=[diag],
            warnings=[str(e)],
            failure=failure,
        )
