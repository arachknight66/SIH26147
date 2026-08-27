import numpy as np
import pytest
from app.exceptions import InvalidSampleCountError
from app.io.raw_iq import RawIQConfig, RawIQReader
from app.models.signal import Endian, IQOrder

def test_int16_iq_roundtrip(tmp_path):
    values = np.array([[100, -200], [-300, 400]], dtype="<i2"); path = tmp_path / "x.iq"; values.tofile(path)
    record = RawIQReader(path, RawIQConfig("int16")).read()
    np.testing.assert_array_equal(record.samples, np.array([100-200j, -300+400j], np.complex64))
    assert record.sample_rate_hz.value is None

def test_qi_and_big_endian(tmp_path):
    path = tmp_path / "x.iq"; np.array([[2, 1]], dtype=">i2").tofile(path)
    assert RawIQReader(path, RawIQConfig("int16", IQOrder.QI, Endian.BIG)).read().samples[0] == 1+2j

def test_odd_pair_rejected(tmp_path):
    path = tmp_path / "bad.iq"; np.array([1, 2, 3], dtype="i2").tofile(path)
    with pytest.raises(InvalidSampleCountError): RawIQReader(path, RawIQConfig("int16"))

def test_chunking(tmp_path):
    path = tmp_path / "x.iq"; np.arange(20, dtype="i1").tofile(path)
    reader = RawIQReader(path, RawIQConfig("int8")); assert reader.sample_count == 10; assert len(reader.read_chunk(2, 3)) == 3
