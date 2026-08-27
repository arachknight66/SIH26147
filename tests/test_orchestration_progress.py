from __future__ import annotations
import pytest
from app.orchestration.progress import (
    ProgressTracker,
    ProgressUpdate,
)

def test_progress_tracker_updates():
    updates = []
    tracker = ProgressTracker(total_phases=6, callback=lambda u: updates.append(u))

    u1 = tracker.update(1, "Phase 1: Ingestion", "Reading raw IQ samples...", 0.5)
    assert u1.phase_number == 1
    assert u1.phase_name == "Phase 1: Ingestion"
    assert u1.progress_fraction == 0.5
    assert len(updates) == 1

def test_progress_tracker_with_warning():
    tracker = ProgressTracker(total_phases=6)
    u = tracker.update(2, "Phase 2", "Estimating SNR", 0.8, warning="SNR low")
    assert "SNR low" in u.warnings
