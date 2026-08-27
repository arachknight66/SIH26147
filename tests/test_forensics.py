import numpy as np
from app.io.forensic import inspect_raw_iq
def test_forensics_returns_deterministic_candidates(tmp_path):
    path = tmp_path / "a.iq"; np.arange(32, dtype="i2").tofile(path)
    assert inspect_raw_iq(path) == inspect_raw_iq(path)
