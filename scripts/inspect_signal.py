from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from app.io.forensic import inspect_raw_iq
from app.io.loader import load_signal
from app.io.raw_iq import RawIQConfig
from app.models.signal import Endian, IQOrder

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26147 scientific signal-input audit")
    parser.add_argument("path"); parser.add_argument("--dtype", choices=["complex64", "float32", "int8", "int16", "uint8"]); parser.add_argument("--iq-order", choices=["IQ", "QI"], default="IQ"); parser.add_argument("--endian", choices=["little", "big"], default="little"); parser.add_argument("--stereo-iq", action="store_true"); parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args(); path = Path(args.path)
    raw = RawIQConfig(args.dtype, IQOrder(args.iq_order), Endian(args.endian), compute_hash=args.sha256) if args.dtype else None
    if path.suffix.lower() not in {".wav", ".sigmf-meta"} and raw is None:
        print("RAW-IQ FORMAT PLAUSIBILITY CANDIDATES (not identification)")
        for candidate in inspect_raw_iq(path)[:8]: print(f"  {candidate.dtype:9} {candidate.iq_order.value} {candidate.endian.value:6} score={candidate.score:.3f}  {candidate.evidence[0]}")
        return
    recording = load_signal(path, raw_config=raw, wav_mode="stereo_iq" if args.stereo_iq else "unresolved")
    finite = float(np.isfinite(recording.samples).mean()) * 100
    print("=" * 60 + "\nSIH26147 SIGNAL INPUT AUDIT\n" + "=" * 60)
    print(f"File: {path}\nFormat: {recording.source_format.value}\nSamples: {len(recording.samples):,}\nSemantic type: {recording.semantic_type}\nDatatype: {recording.original_dtype}\nSample rate: {recording.sample_rate_hz.value if recording.sample_rate_hz.value is not None else 'UNKNOWN'}\nCenter frequency: {recording.center_frequency_hz.value if recording.center_frequency_hz.value is not None else 'UNKNOWN'}\nFinite samples: {finite:.2f}%")
    print("Warnings:")
    for item in recording.diagnostics: print(f"  [{item.severity.value.upper()}] {item.message}")
    print("=" * 60)
if __name__ == "__main__": main()
