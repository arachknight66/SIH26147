from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from app.analysis.analyzer import analyze_signal
from app.data_recovery.analyzer import recover_data
from app.data_recovery.models import DataRecoveryConfig, DataRecoveryStatus
from app.io.loader import load_signal
from app.io.raw_iq import RawIQConfig
from app.models.signal import SourceFormat
from app.modulation.analyzer import analyze_modulation
from app.recovery.analyzer import recover_signal

def print_data_recovery_report(
    analysis,
    dump_bits: bool = False,
    dump_frames: bool = False,
    dump_corrections: bool = False,
    dump_payload: bool = False,
) -> None:
    """Print scientific data recovery report matching Section 87 specification."""
    print("=" * 60)
    print("SIH26147 PHASE 5 DATA RECOVERY")
    print("=" * 60)
    print()

    selected = analysis.selected_candidate
    if selected is None:
        print("STATUS: INSUFFICIENT_STRUCTURE / AMBIGUOUS")
        print(f"Reason: {analysis.failure_reason}")
        if analysis.reconstruction_candidates:
            print("\nAttempted Candidates:")
            for cand in analysis.reconstruction_candidates:
                print(f"  - Candidate #{cand.candidate_id} (score={cand.composite_score:.2f}, status={cand.epistemic_status.value})")
        print("=" * 60)
        return

    bit_hyp = selected.bit_hypothesis
    print("INPUT")
    print(f"    modulation: {bit_hyp.bitstream.source_candidate}")
    print(f"    bits: {bit_hyp.bitstream.length:,}")
    print(f"    hard bits: AVAILABLE")
    print(f"    soft decisions: {'AVAILABLE' if bit_hyp.bitstream.soft_bits is not None else 'UNAVAILABLE'}")
    print()
    print("-" * 60)
    print("BITSTREAM")
    print()
    print(f"bit polarity:\n    {bit_hyp.polarity.value}")
    print()
    print(f"byte alignment:\n    offset = {bit_hyp.bit_offset} bits")
    print()
    print(f"selected orientation:\n    phase rotation = {bit_hyp.phase_rotation_deg:.1f} deg")
    print()
    print("-" * 60)
    print("FRAME STRUCTURE")
    print()

    if selected.preamble:
        print(f"candidate frame length:\n    {selected.frames[0].end_bit - selected.frames[0].start_bit if selected.frames else 'unknown'} bits")
        print()
        print(f"preamble:\n    {selected.preamble.length_bits} bits (hex: 0x{selected.preamble.pattern_hex})")
        print()
        print(f"frames detected:\n    {len(selected.frames)}")
        print()
        print(f"frame consistency:\n    {'HIGH' if selected.preamble.is_periodic else 'MODERATE'}")
        print()
    else:
        print("No repeating preamble pattern detected.")
        print(f"frames detected: {len(selected.frames)}")
        print()

    print("-" * 60)
    print("ERROR CORRECTION")
    print()

    if selected.fec:
        print(f"FEC:\n    {selected.fec.code_name}")
        print(f"rate:\n    {selected.fec.rate:.2f}")
        if selected.fec_decode:
            dec = selected.fec_decode
            print()
            print(f"corrected bits:\n    {dec.corrected_bit_count}")
            print()
            print(f"correction fraction:\n    {dec.correction_fraction * 100:.2f}%")
            print()
            print(f"over-correction risk:\n    {dec.is_overcorrected}")
            print()
    else:
        print("FEC: UNCODED\n")

    print("-" * 60)
    print("INTEGRITY")
    print()

    if selected.integrity and selected.integrity.valid_frame_count > 0:
        integ = selected.integrity
        crc_name = integ.crc_results[0].crc_name if integ.crc_results else "CRC-16"
        print(f"CRC:\n    {crc_name}")
        print()
        print(f"CRC-valid frames:\n    {integ.valid_frame_count} / {integ.total_frame_count}")
        print()
        print(f"multi-frame false-alarm p-value:\n    {integ.multi_frame_p_value:.2e}")
        print()
    else:
        print("CRC: UNRESOLVED / NO MATCH\n")

    print("-" * 60)
    print("PAYLOAD")
    print()

    payload_len = len(selected.recovered_payload_bytes)
    printable_chars = sum(1 for b in selected.recovered_payload_bytes if 32 <= b <= 126 or b in (9, 10, 13))
    print_ratio = (printable_chars / payload_len * 100.0) if payload_len > 0 else 0.0

    print(f"payload:\n    {payload_len} bytes structurally recovered")
    print()
    print(f"printability:\n    {print_ratio:.1f}%")
    print()
    print("NOTE:\n    Printability is supporting evidence only.")
    print()

    print("-" * 60)
    print("ASSESSMENT")
    print()
    print(f"STATUS:\n    {analysis.status.value.upper()}")
    print()
    print(f"QUALITY:\n    {analysis.quality_level.value}")
    print()
    print("The candidate reconstruction is supported\nby structural evidence and coding hypothesis.")
    print()
    print("Independent verification has NOT yet been\nperformed.")
    print("Phase 6 required.")
    print("=" * 60)

    if dump_payload and selected.recovered_payload_bytes:
        print("\n--- RECOVERED PAYLOAD BYTES (HEX DUMP) ---")
        print(selected.recovered_payload_bytes.hex())
        print("\n--- RECOVERED PAYLOAD BYTES (ASCII PREVIEW) ---")
        preview = "".join(chr(b) if 32 <= b <= 126 else "." for b in selected.recovered_payload_bytes[:256])
        print(preview)

    if dump_frames and selected.frames:
        print("\n--- DETECTED FRAMES BREAKDOWN ---")
        for f in selected.frames[:10]:
            print(f"Frame #{f.frame_index}: [{f.start_bit}:{f.end_bit}] len={f.end_bit - f.start_bit} bits | CRC valid: {f.is_crc_valid}")

    if dump_corrections and selected.fec_decode:
        corr_indices = np.where(selected.fec_decode.correction_mask)[0]
        print(f"\n--- CORRECTED BIT INDICES (TOTAL {len(corr_indices)}) ---")
        print(corr_indices[:64])
        if len(corr_indices) > 64:
            print(f"... ({len(corr_indices) - 64} more)")

def main() -> int:
    parser = argparse.ArgumentParser(description="SIH26147 Phase 5 Data Recovery CLI")
    parser.add_argument("file_path", type=str, help="Path to input signal recording (.iq or .wav)")
    parser.add_argument("--dtype", type=str, default="complex64", help="Raw sample dtype (default: complex64)")
    parser.add_argument("--sample-rate", type=float, default=None, help="Declared sample rate in Hz")
    parser.add_argument("--dump-bits", action="store_true", help="Print reconstructed bits")
    parser.add_argument("--dump-frames", action="store_true", help="Print frame breakdown")
    parser.add_argument("--dump-corrections", action="store_true", help="Print FEC correction indices")
    parser.add_argument("--dump-payload", action="store_true", help="Print recovered payload bytes")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    args = parser.parse_args()

    path = Path(args.file_path)
    if not path.exists():
        print(f"Error: file '{path}' does not exist.", file=sys.stderr)
        return 1

    if path.suffix.lower() == ".iq":
        raw_cfg = RawIQConfig(dtype=args.dtype, sample_rate_hz=args.sample_rate)
        recording = load_signal(str(path), raw_config=raw_cfg)
    else:
        recording = load_signal(str(path))

    analysis = analyze_signal(recording)
    mod_analysis = analyze_modulation(recording, analysis=analysis)
    rec_analysis = recover_signal(recording, analysis=analysis, modulation_analysis=mod_analysis)
    data_analysis = recover_data(rec_analysis)

    if args.json:
        out_dict = {
            "status": data_analysis.status.value,
            "quality_level": data_analysis.quality_level.value,
            "is_recovered": data_analysis.is_recovered,
            "selected_candidate_id": data_analysis.selected_candidate.candidate_id if data_analysis.selected_candidate else None,
            "num_frames": len(data_analysis.selected_candidate.frames) if data_analysis.selected_candidate else 0,
            "diagnostics": [d.code for d in data_analysis.diagnostics],
        }
        print(json.dumps(out_dict, indent=2))
    else:
        print_data_recovery_report(
            data_analysis,
            dump_bits=args.dump_bits,
            dump_frames=args.dump_frames,
            dump_corrections=args.dump_corrections,
            dump_payload=args.dump_payload,
        )

    return 0

if __name__ == "__main__":
    sys.exit(main())
