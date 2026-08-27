import numpy as np
import pytest
from app.exceptions import UnsupportedFileFormatError
from app.io.loader import load_signal
from app.io.raw_iq import RawIQConfig

def test_raw_requires_explicit_interpretation(tmp_path):
    path = tmp_path / "unknown.iq"; np.array([1, 2], dtype="i1").tofile(path)
    with pytest.raises(UnsupportedFileFormatError): load_signal(path)
    assert load_signal(path, raw_config=RawIQConfig("int8")).samples[0] == 1+2j
