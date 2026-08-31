import os
import wave
import json
import pytest
import numpy as np
from signal_analysis.loaders import RawIQConfig, RawIQReader, WavReader, read_sigmf
from signal_analysis.models import SourceFormat, MetadataStatus, Severity

def test_raw_iq_roundtrip_complex64(tmp_path):
    path = tmp_path / "test.iq"
    data = np.array([1+2j, 3+4j], dtype=np.complex64)
    data.tofile(path)
    
    config = RawIQConfig(dtype="complex64", iq_order="IQ", endian="little")
    reader = RawIQReader(str(path), config)
    rec = reader.read()
    
    assert rec.source_format == SourceFormat.RAW_IQ
    assert np.allclose(rec.samples, data)
    assert rec.semantic_type == "complex_iq"
    assert rec.original_dtype == "complex64"

def test_raw_iq_roundtrip_int16_qi_big_endian(tmp_path):
    path = tmp_path / "test.iq"
    # Data: Q, I pairs, big endian int16
    # 2+1j, 4+3j -> Q=2, I=1, Q=4, I=3
    arr = np.array([2, 1, 4, 3], dtype=">i2")
    arr.tofile(path)
    
    config = RawIQConfig(dtype="int16", iq_order="QI", endian="big")
    reader = RawIQReader(str(path), config)
    rec = reader.read()
    
    assert rec.original_dtype == "int16"
    assert np.allclose(rec.samples, np.array([1+2j, 3+4j], dtype=np.complex64))

def test_raw_iq_malformed(tmp_path):
    path = tmp_path / "test.iq"
    # Odd scalar count (3 bytes for uint8)
    arr = np.array([1, 2, 3], dtype=np.uint8)
    arr.tofile(path)
    
    config = RawIQConfig(dtype="uint8", iq_order="IQ", endian="little")
    with pytest.raises(ValueError, match="not a multiple of the IQ pair size"):
        reader = RawIQReader(str(path), config)
        reader.read()

def test_raw_iq_chunk_reading(tmp_path):
    path = tmp_path / "test.iq"
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    arr.tofile(path) # 4 pairs
    
    config = RawIQConfig(dtype="float32", iq_order="IQ", endian="little")
    reader = RawIQReader(str(path), config)
    
    full = reader.read().samples
    assert len(full) == 4
    
    chunk = reader.read_chunk(1, 2)
    assert len(chunk) == 2
    assert np.allclose(chunk, full[1:3])

def test_wav_mono(tmp_path):
    path = tmp_path / "mono.wav"
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        data = np.array([100, 200, 300], dtype=np.int16)
        w.writeframes(data.tobytes())
        
    reader = WavReader(str(path))
    rec = reader.read()
    
    assert rec.source_format == SourceFormat.WAV
    assert rec.semantic_type == "mono_real"
    assert rec.sample_rate_hz.value == 44100.0
    assert rec.sample_rate_hz.status == MetadataStatus.KNOWN
    assert np.allclose(rec.samples, np.array([100, 200, 300], dtype=np.complex64))

def test_wav_stereo_unresolved(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        data = np.array([100, 200, 300, 400], dtype=np.int16)
        w.writeframes(data.tobytes())
        
    reader = WavReader(str(path), mode="unresolved")
    rec = reader.read()
    
    assert rec.semantic_type == "stereo_real"
    assert len(rec.diagnostics) == 1
    assert rec.diagnostics[0].code == "UNRESOLVED_STEREO"
    assert rec.diagnostics[0].severity == Severity.WARNING
    assert rec.samples.shape == (2, 2)

def test_wav_stereo_iq(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        data = np.array([100, 200, 300, 400], dtype=np.int16)
        w.writeframes(data.tobytes())
        
    reader = WavReader(str(path), mode="stereo_iq")
    rec = reader.read()
    
    assert rec.semantic_type == "complex_iq"
    assert len(rec.diagnostics) == 0
    assert np.allclose(rec.samples, np.array([100+200j, 300+400j], dtype=np.complex64))

def test_sigmf_read(tmp_path):
    meta_path = tmp_path / "test.sigmf-meta"
    data_path = tmp_path / "test.sigmf-data"
    
    meta = {
        "global": {
            "core:datatype": "ci16_le",
            "core:sample_rate": 1e6
        },
        "captures": [
            {"core:frequency": 2.4e9}
        ]
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
        
    data = np.array([100, 200], dtype="<i2")
    data.tofile(data_path)
    
    rec = read_sigmf(str(meta_path))
    assert rec.source_format == SourceFormat.SIGMF
    assert rec.sample_rate_hz.value == 1e6
    assert rec.center_frequency_hz.value == 2.4e9
    assert np.allclose(rec.samples, np.array([100+200j], dtype=np.complex64))
