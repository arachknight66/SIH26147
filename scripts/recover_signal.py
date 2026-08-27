from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from app.analysis.analyzer import analyze_signal
from app.io.loader import load_signal
from app.models.signal import SourceFormat
from app.modulation.analyzer import analyze_modulation
from app.recovery.analyzer import recover_signal
from app.recovery.models import RecoveryConfig, RecoveryStatus

def print_recovery_report(analysis, dump_bits: bool = False, dump_symbols: bool = False) -> None:
    """Print scientific recovery report matching SIH26147 Phase 4 specification."""
    print("=" * 60)
    print("SIH26147 PHASE 4 SIGNAL RECOVERY")
    print("=" * 60)
    print()

    selected = analysis.selected_candidate
    if selected is None:
        print("STATUS: RECOVERY INCONCLUSIVE")
        print(f"Reason: {analysis.failure_reason}")
        if analysis.candidates:
            print("\nAttempted Candidates:")
            for cand in analysis.candidates:
                print(f"  - {cand.label} (score={cand.quality.composite_score:.2f}, status={cand.status.value})")
        print("=" * 60)
        return

    print("Candidate:")
    print(f"    {selected.label}")
    print(f"    symbol rate: {selected.symbol_rate_normalized:.6f} cycles/sample")
    print()
    print("-" * 60)
    print("SYNCHRONIZATION")
    print()

    if selected.synchronization:
        freq = selected.synchronization.frequency
        carrier = selected.synchronization.carrier
        timing = selected.synchronization.timing

        print("Coarse CFO:")
        print(f"    {freq.coarse_cfo_normalized:+.6f} cycles/sample")
        print()
        print("Residual CFO:")
        print(f"    {carrier.residual_cfo_normalized:+.6f} cycles/sample")
        print()
        print("Carrier:")
        print(f"    {carrier.lock_status.value.upper()}")
        print()
        print("Phase error RMS:")
        print(f"    {carrier.phase_error_rms_rad:.4f} rad")
        print()
        print("Timing:")
        print(f"    {timing.lock_status.value.upper()}")
        print()
        print("Estimated timing offset:")
        print(f"    {timing.timing_offset_samples:.2f} samples")
        print()
        print("TED RMS:")
        print(f"    {np.sqrt(timing.ted_variance):.4f}")
        print()

    print("-" * 60)
    print("CONSTELLATION")
    print()

    if selected.constellation:
        const = selected.constellation
        print(f"Clusters:\n    {len(const.cluster_centroids)}")
        print()
        print(f"EVM:\n    {const.evm_percent:.1f} % ({const.evm_db:.1f} dB)")
        print()
        print(f"Decision margin:\n    {const.decision_margin:.2f}")
        print()
        print(f"Constellation quality:\n    {selected.quality.quality_level.value}")
        print()

    print("-" * 60)
    print("DEMODULATION")
    print()

    if selected.demodulation:
        demod = selected.demodulation
        print(f"Symbols recovered:\n    {len(demod.symbol_indices):,}")
        print()
        print("Hard decisions:\n    available")
        print()
        print("Soft decisions:\n    available")
        print()
        print(f"Bit polarity:\n    {demod.bit_polarity}")
        print()
        rot_amb = selected.constellation.rotational_ambiguity_deg if selected.constellation else (0.0,)
        print(f"Rotational ambiguity:\n    {rot_amb}")
        print()

    print("-" * 60)
    print("RECOVERY ASSESSMENT")
    print()
    print(f"Status:\n    {selected.status.value.upper()}")
    print()
    print(f"Quality:\n    {selected.quality.quality_level.value}")
    print()
    if selected.status == RecoveryStatus.RECOVERED:
        print(f"Receiver evidence strongly supports\nthe {selected.label} hypothesis.")
    else:
        print("Receiver evidence is inconclusive.")
    print()
    print("Bit-level correctness has not yet been\nindependently verified in Phase 5/6.")
    print("=" * 60)

    if dump_bits and selected.demodulation:
        print("\n--- RECOVERED HARD BITS ---")
        print("".join(str(b) for b in selected.demodulation.hard_bits[:1024]))
        if len(selected.demodulation.hard_bits) > 1024:
            print(f"... ({len(selected.demodulation.hard_bits) - 1024} more bits)")

    if dump_symbols and selected.constellation:
        print("\n--- RECOVERED CONSTELLATION SYMBOLS (FIRST 32) ---")
        for i, s in enumerate(selected.constellation.symbols[:32]):
            print(f"[{i:02d}] {s.real:+.4f} {s.imag:+.4f}j")

def main() -> int:
    parser = argparse.ArgumentParser(description="SIH26147 Phase 4 Signal Recovery CLI")
    parser.add_argument("file_path", type=str, help="Path to input signal file")
    parser.add_argument("--dtype", type=str, default="complex64", help="Raw sample dtype (default: complex64)")
    parser.add_argument("--sample-rate", type=float, default=None, help="Declared sample rate in Hz")
    parser.add_argument("--dump-bits", action="store_true", help="Print recovered hard bits")
    parser.add_argument("--dump-symbols", action="store_true", help="Print recovered constellation symbols")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    path = Path(args.file_path)
    if not path.exists():
        print(f"Error: file '{path}' does not exist.", file=sys.stderr)
        return 1

    from app.io.raw_iq import RawIQConfig

    if path.suffix.lower() == ".iq":
        raw_cfg = RawIQConfig(dtype=args.dtype, sample_rate_hz=args.sample_rate)
        recording = load_signal(str(path), raw_config=raw_cfg)
    else:
        recording = load_signal(str(path))

    analysis = analyze_signal(recording)
    mod_analysis = analyze_modulation(recording, analysis=analysis)
    rec_analysis = recover_signal(recording, analysis=analysis, modulation_analysis=mod_analysis)

    if args.json:
        out_dict = {
            "is_recovered": rec_analysis.is_recovered,
            "selected_candidate": rec_analysis.selected_candidate.label if rec_analysis.selected_candidate else None,
            "status": rec_analysis.selected_candidate.status.value if rec_analysis.selected_candidate else "inconclusive",
            "evm_percent": rec_analysis.selected_candidate.constellation.evm_percent if (rec_analysis.selected_candidate and rec_analysis.selected_candidate.constellation) else None,
            "diagnostics": [d.code for d in rec_analysis.diagnostics],
        }
        print(json.dumps(out_dict, indent=2))
    else:
        print_recovery_report(rec_analysis, dump_bits=args.dump_bits, dump_symbols=args.dump_symbols)

    return 0

if __name__ == "__main__":
    sys.exit(main())
