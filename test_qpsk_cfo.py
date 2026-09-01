import numpy as np
from tests.test_synthesis import generate_synthetic_signal
from signal_analysis.pipeline import run_full_pipeline
from signal_analysis.loaders import WavReader
import wave

sig = generate_synthetic_signal("16-QAM", n_symbols=2000, snr_db=30, cfo_norm=0.002, return_bits=False)
i = np.clip(sig.real * 32767, -32768, 32767).astype(np.int16)
q = np.clip(sig.imag * 32767, -32768, 32767).astype(np.int16)
stereo = np.empty((len(i), 2), dtype=np.int16)
stereo[:, 0] = i
stereo[:, 1] = q
with wave.open("test_16qam_cfo.wav", 'wb') as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(1000000)
    wf.writeframes(stereo.tobytes())

rec = WavReader("test_16qam_cfo.wav", mode="stereo_iq").read()
res = run_full_pipeline(rec)
print("Top hyp:", res.top_hypothesis.label if res.top_hypothesis else "None")
print("Sync:", res.sync_status)
