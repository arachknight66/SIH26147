from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Callable

@dataclass(frozen=True)
class ProgressUpdate:
    phase_number: int
    total_phases: int
    phase_name: str
    operation: str
    progress_fraction: float
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    warnings: tuple[str, ...] = ()

ProgressCallback = Callable[[ProgressUpdate], None]

class ProgressTracker:
    def __init__(self, total_phases: int = 6, callback: ProgressCallback | None = None) -> None:
        self.total_phases = total_phases
        self.callback = callback
        self.start_time = time.perf_counter()
        self._warnings: list[str] = []

    def update(
        self,
        phase_number: int,
        phase_name: str,
        operation: str,
        progress_fraction: float = 0.0,
        warning: str | None = None,
    ) -> ProgressUpdate:
        if warning:
            self._warnings.append(warning)

        now = time.perf_counter()
        elapsed = now - self.start_time

        # Calculate ETA only when progress is non-trivial and >= 10%
        overall_progress = (max(0, phase_number - 1) + max(0.0, min(1.0, progress_fraction))) / self.total_phases
        if overall_progress >= 0.10 and elapsed > 0.5:
            estimated_total = elapsed / overall_progress
            eta: float | None = max(0.0, estimated_total - elapsed)
        else:
            eta = None

        up = ProgressUpdate(
            phase_number=phase_number,
            total_phases=self.total_phases,
            phase_name=phase_name,
            operation=operation,
            progress_fraction=progress_fraction,
            elapsed_seconds=round(elapsed, 2),
            estimated_remaining_seconds=round(eta, 2) if eta is not None else None,
            warnings=tuple(self._warnings),
        )

        if self.callback:
            try:
                self.callback(up)
            except Exception:
                pass

        return up
