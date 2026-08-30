import io
import urllib.request
import json
import wave
import numpy as np
from pathlib import Path

# Create a valid test WAV recording (10,000 samples of 100 kHz tone at 2.4 MS/s)
fs = 2_400_000
t = np.arange(10000) / fs
i_samples = (np.cos(2 * np.pi * 100000 * t) * 16000).astype(np.int16)
q_samples = (np.sin(2 * np.pi * 100000 * t) * 16000).astype(np.int16)
interleaved = np.column_stack((i_samples, q_samples)).flatten().tobytes()

wav_io = io.BytesIO()
with wave.open(wav_io, "wb") as wav:
    wav.setnchannels(2)
    wav.setsampwidth(2)
    wav.setframerate(fs)
    wav.writeframes(interleaved)

wav_bytes = wav_io.getvalue()

boundary = "----WebKitFormBoundaryWavTest"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="tone_100k.wav"\r\n'
    f"Content-Type: audio/wav\r\n\r\n"
).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8050/api/upload",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("WAV Upload success:", res.get("is_success"))
        print("Input path:", res.get("input", {}).get("source_path"))
        print("Format:", res.get("input", {}).get("format"))
        print("Sample rate (MS/s):", (res.get("input", {}).get("sample_rate_hz") or 0) / 1e6)
        print("Sample count:", res.get("input", {}).get("sample_count"))
        print("PSD points:", len(res.get("plots", {}).get("psd_p", [])))
        print("Spectrogram available:", res.get("plots", {}).get("spectrogram", {}).get("available"))
        if res.get("plots", {}).get("spectrogram", {}).get("available"):
            print("Spectrogram matrix shape:", len(res.get("plots", {}).get("spectrogram", {}).get("matrix")), "x", len(res.get("plots", {}).get("spectrogram", {}).get("matrix")[0]))
        print("Final assessment:", res.get("final_assessment"))
except Exception as e:
    print("Upload failed with error:", e)
