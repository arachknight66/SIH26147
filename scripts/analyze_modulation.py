from __future__ import annotations
import argparse
from pathlib import Path
from app.io.forensic import inspect_raw_iq
from app.io.loader import load_signal
from app.io.raw_iq import RawIQConfig
from app.models.signal import Endian, IQOrder
from app.modulation.analyzer import analyze_modulation
from app.modulation.models import HypothesisStatus, ModulationAnalysisConfig

def _score_to_word(val: float) -> str:
    if val >= 0.70:
        return "strong"
    elif val >= 0.40:
        return "moderate"
    elif val >= 0.20:
        return "weak"
    return "incompatible"

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26147 Scientific Modulation Identification & Hypothesis Analysis")
    parser.add_argument("path", help="Path to recording (WAV, SigMF, or Raw IQ)")
    parser.add_argument("--dtype", choices=["complex64", "float32", "int8", "int16", "uint8"], help="Raw IQ datatype")
    parser.add_argument("--iq-order", choices=["IQ", "QI"], default="IQ", help="Raw IQ channel ordering")
    parser.add_argument("--endian", choices=["little", "big"], default="little", help="Raw IQ byte endianness")
    parser.add_argument("--sample-rate", type=float, default=None, help="Explicit sample rate in Hz")
    parser.add_argument("--center-freq", type=float, default=None, help="Explicit center frequency in Hz")
    parser.add_argument("--stereo-iq", action="store_true", help="Interpret 2-channel WAV as stereo IQ")
    parser.add_argument("--unknown-threshold", type=float, default=0.45, help="Unknown rejection threshold")
    parser.add_argument("--ambiguity-margin", type=float, default=0.08, help="Ambiguity margin")
    args = parser.parse_args()

    path = Path(args.path)
    raw = None
    if args.dtype:
        raw = RawIQConfig(
            args.dtype,
            IQOrder(args.iq_order),
            Endian(args.endian),
            sample_rate_hz=args.sample_rate,
            center_frequency_hz=args.center_freq,
        )

    if path.suffix.lower() not in {".wav", ".sigmf-meta"} and raw is None:
        print("RAW-IQ FORMAT PLAUSIBILITY CANDIDATES (not identification)")
        for candidate in inspect_raw_iq(path)[:8]:
            print(f"  {candidate.dtype:9} {candidate.iq_order.value} {candidate.endian.value:6} score={candidate.score:.3f}  {candidate.evidence[0]}")
        return

    recording = load_signal(path, raw_config=raw, wav_mode="stereo_iq" if args.stereo_iq else "unresolved")
    
    # Run Modulation Analysis
    config = ModulationAnalysisConfig(
        unknown_threshold=args.unknown_threshold,
        ambiguity_margin=args.ambiguity_margin,
    )
    result = analyze_modulation(recording, config=config)

    print("=" * 60)
    print("SIH26147 MODULATION ANALYSIS")
    print("=" * 60)

    # Signal Region Summary
    reg = result.signal_region
    print("\nSignal region:")
    if reg:
        cf_str = f"{reg.center_freq_hz / 1e3:+.1f} kHz" if reg.center_freq_hz is not None else f"{reg.center_freq_normalized:+.4f} cycles/sample"
        bw_str = f"{reg.bandwidth_hz / 1e3:.1f} kHz" if reg.bandwidth_hz is not None else f"{reg.bandwidth_normalized:.4f} cycles/sample"
        snr_str = f"{reg.estimated_snr_db:.1f} dB"
        print(f"    center: {cf_str}")
        print(f"    bandwidth: {bw_str}")
        print(f"    SNR: {snr_str}")
    else:
        print("    Full recording bandwidth analyzed")

    print("\n" + "-" * 60)
    print("TOP MODULATION HYPOTHESES")

    if result.hypotheses:
        for idx, h in enumerate(result.hypotheses[:3], start=1):
            print(f"\n#{idx} {h.label}")
            print(f"    score: {h.score:.2f}")
            print(f"    quality: {h.quality}")

            ev = h.evidence
            print("\n    Evidence:")
            print(f"      amplitude:     {_score_to_word(ev.amplitude_score)}")
            print(f"      phase:         {_score_to_word(ev.phase_score)}")
            print(f"      frequency:     {_score_to_word(ev.frequency_score)}")
            print(f"      cumulants:     {_score_to_word(ev.cumulant_score)}")
            print(f"      spectral:      {_score_to_word(ev.spectral_score)}")
            print(f"      periodicity:   {_score_to_word(ev.periodicity_score)}")
            print(f"      classical:     {_score_to_word(ev.classical_model_score)}")
            print(f"      ML:            {_score_to_word(ev.ml_score)}")

            if ev.supporting_evidence:
                print("\n    Supporting observations:")
                for note in ev.supporting_evidence[:2]:
                    print(f"      • {note}")

    print("\n" + "-" * 60)
    print("STATUS\n")

    if result.is_unknown:
        print("MODULATION UNKNOWN / OUT-OF-DISTRIBUTION")
        print("No candidate hypothesis exceeded the scientific confidence threshold.")
    elif result.is_ambiguous:
        print("MODULATION AMBIGUOUS")
        top_labels = ", ".join([h.label for h in result.hypotheses[:2]])
        print(f"Competing evidence between candidates: {top_labels}.")
    elif result.selected_hypothesis:
        print(f"{result.selected_hypothesis.label} is the strongest candidate.")
    else:
        print("INSUFFICIENT EVIDENCE")

    print("\nThis result is a modulation hypothesis.")
    print("Receiver synchronization and demodulation have not yet been performed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
