from __future__ import annotations

import numpy as np
import pytest

from app.io.raw_iq import RawIQConfig, RawIQReader
from app.models.metadata import DiagnosticSeverity


def test_raw_iq_records_interpretation_and_hash(tmp_path):
    path = tmp_path / "capture.iq"
    np.array([[3, -4]], dtype="<i2").tofile(path)
    rec = RawIQReader(path, RawIQConfig("int16", sample_rate_hz=48_000.0, center_frequency_hz=433_920_000.0, compute_hash=True)).read()
    assert rec.sample_rate_hz.value == 48_000.0
    assert rec.center_frequency_hz.value == 433_920_000.0
    assert rec.provenance["sha256"]
    assert rec.provenance["input_configuration"]["dtype"] == "int16"


@pytest.mark.parametrize("kwargs", [{"sample_rate_hz": 0}, {"center_frequency_hz": -1}])
def test_raw_iq_rejects_invalid_physical_metadata(kwargs):
    with pytest.raises(ValueError):
        RawIQConfig("int16", **kwargs)


def test_raw_iq_reports_non_finite_complex_capture(tmp_path):
    path = tmp_path / "nonfinite.iq"
    np.array([complex(float("nan"), 0)], dtype="<c8").tofile(path)
    rec = RawIQReader(path, RawIQConfig("complex64")).read()
    assert any(d.code == "NON_FINITE_SAMPLES" and d.severity == DiagnosticSeverity.ERROR for d in rec.diagnostics)
