import io
import urllib.request
import json
from pathlib import Path

# Read an actual binary IQ file
with open("examples/clean_qpsk.iq", "rb") as f:
    samples = f.read()

boundary = "----WebKitFormBoundaryX7gK8e4kL"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="clean_qpsk.iq"\r\n'
    f"Content-Type: application/octet-stream\r\n\r\n"
).encode("utf-8") + samples + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8050/api/upload",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Upload test success:", res.get("is_success"))
        print("Input path:", res.get("input", {}).get("source_path"))
        print("Sample count:", res.get("input", {}).get("sample_count"))
        print("Modulation winner:", res.get("phase3_modulation", {}).get("winner"))
        print("Constellation points count:", len(res.get("plots", {}).get("const_i", [])))
        print("Spectrogram available:", res.get("plots", {}).get("spectrogram", {}).get("available"))
        print("Final assessment:", res.get("final_assessment"))
except Exception as e:
    print("Upload failed with error:", e)
