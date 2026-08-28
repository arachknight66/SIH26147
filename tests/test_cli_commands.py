from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from app.models.signal import SignalRecording, SourceFormat
from tests.test_phase6_cases import _make_rec_sig
from scripts.generate_digital_dataset import generate_digital_stream

def test_cli_analyze_command(tmp_path: Path):
    rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    iq_path = tmp_path / "test_sig.iq"
    np_samples = np.repeat(
        ((np.where(rx[0::2] == 0, 1.0, -1.0) + 1j * np.where(rx[1::2] == 0, 1.0, -1.0)) / 1.414).astype("complex64"),
        4
    )
    np_samples.tofile(iq_path)

    cmd = [sys.executable, "-m", "scripts.sih26147", "analyze", str(iq_path), "--preset", "fast"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "FINAL SCIENTIFIC ASSESSMENT" in res.stdout

def test_cli_benchmark_command():
    cmd = [sys.executable, "-m", "scripts.sih26147", "benchmark"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "COMPREHENSIVE END-TO-END SYSTEM BENCHMARK" in res.stdout
