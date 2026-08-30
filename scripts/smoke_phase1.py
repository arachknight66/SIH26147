"""Fast, dependency-light smoke test for Phase 1 signal ingestion."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from app.io.loader import load_signal
from app.io.raw_iq import RawIQConfig


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sih26147-phase1-") as directory:
        path = Path(directory) / "smoke.iq"
        np.array([[100, -50], [-100, 50]], dtype="<i2").tofile(path)
        recording = load_signal(
            path,
            raw_config=RawIQConfig(
                "int16", sample_rate_hz=48_000.0, center_frequency_hz=433_920_000.0, compute_hash=True
            ),
        )
        assert recording.samples.dtype == np.complex64
        assert recording.sample_rate_hz.value == 48_000.0
        assert recording.center_frequency_hz.value == 433_920_000.0
        assert recording.provenance["sha256"]
    print("Phase 1 smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
