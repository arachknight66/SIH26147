"""Generate deterministic, small Phase-1 fixture recordings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import wave
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("output", type=Path); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    t = np.arange(256, dtype=np.float32); iq = np.exp(2j * np.pi * .0625 * t).astype(np.complex64)
    iq.tofile(args.output / "tone_cf32.iq")
    pairs16 = np.column_stack((np.rint(iq.real * 30000), np.rint(iq.imag * 30000))).astype("<i2"); pairs16.tofile(args.output / "tone_ci16.iq")
    pairs8 = np.column_stack((np.rint(iq.real * 100 + 128), np.rint(iq.imag * 100 + 128))).astype("u1"); pairs8.tofile(args.output / "tone_cu8.iq")
    with wave.open(str(args.output / "tone_stereo.wav"), "wb") as output:
        output.setnchannels(2); output.setsampwidth(2); output.setframerate(8000); output.writeframes(pairs16.tobytes())
    with wave.open(str(args.output / "tone_mono.wav"), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(8000); output.writeframes(pairs16[:, :1].tobytes())
    (args.output / "tone_ci16.sigmf-meta").write_text(json.dumps({"global": {"core:datatype": "ci16_le", "core:sample_rate": 8000}, "captures": [{"core:frequency": 433920000}]}), encoding="utf8")
    pairs16.tofile(args.output / "tone_ci16.sigmf-data")
if __name__ == "__main__": main()
