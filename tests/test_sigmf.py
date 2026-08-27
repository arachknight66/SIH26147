import json
import numpy as np
from app.io.loader import load_signal
from app.models.signal import SourceFormat

def test_sigmf_ci16(tmp_path):
    data = tmp_path / "capture.sigmf-data"; np.array([[10, -20]], dtype="<i2").tofile(data)
    meta = tmp_path / "capture.sigmf-meta"
    meta.write_text(json.dumps({"global": {"core:datatype": "ci16_le", "core:sample_rate": 2400000}, "captures": [{"core:frequency": 433920000}]}), encoding="utf8")
    result = load_signal(meta)
    assert result.source_format == SourceFormat.SIGMF
    assert result.samples[0] == 10-20j
    assert result.sample_rate_hz.value == 2400000
