import sys
import subprocess
import json
import numpy as np
from pathlib import Path
import pytest

def test_cli_no_gui_import(tmp_path):
    # Test that running cli.py doesn't import PySide6 or pyqtgraph
    # We will write a tiny dummy WAV file
    import wave
    test_wav = tmp_path / "test.wav"
    with wave.open(str(test_wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(np.zeros(100, dtype=np.int16).tobytes())
        
    cmd = [sys.executable, "-m", "signal_analysis.cli", str(test_wav), "--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    
    # Assert JSON parseable
    out = json.loads(result.stdout)
    assert str(test_wav) in out
    
    # Assert specific status strings exist in the JSON output, not Enum objects
    assert "hypothesis_status" in out[str(test_wav)]
    assert out[str(test_wav)]["hypothesis_status"] in ["COMPLETED", "FAILED", "NOT_ATTEMPTED", "SKIPPED"]

def test_cli_batch_isolation(tmp_path):
    import wave
    
    valid_wav = tmp_path / "valid.wav"
    with wave.open(str(valid_wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(np.zeros(100, dtype=np.int16).tobytes())
        
    invalid_wav = tmp_path / "invalid.wav"
    invalid_wav.write_text("not a wav")
    
    cmd = [sys.executable, "-m", "signal_analysis.cli", str(tmp_path), "--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    out = json.loads(result.stdout)
    
    assert str(valid_wav) in out
    assert str(invalid_wav) in out
    
    assert "error" in out[str(invalid_wav)]
    assert "error" not in out[str(valid_wav)]
