import wave
import numpy as np
from app.io.wav import WavReader

def _write(path, channels, data):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels); output.setsampwidth(2); output.setframerate(8000); output.writeframes(np.asarray(data, dtype="<i2").tobytes())
def test_stereo_not_silently_iq(tmp_path):
    path = tmp_path / "stereo.wav"; _write(path, 2, [[100, 200], [300, 400]])
    record = WavReader(path).read(); assert record.semantic_type == "stereo_real"; assert record.sample_rate_hz.value == 8000
    iq = WavReader(path, "stereo_iq").read(); np.testing.assert_array_equal(iq.samples, np.array([100+200j, 300+400j], np.complex64))
def test_mono(tmp_path):
    path = tmp_path / "mono.wav"; _write(path, 1, [1, 2, 3]); record = WavReader(path).read(); assert record.semantic_type == "mono_real"; assert np.all(record.samples.imag == 0)
