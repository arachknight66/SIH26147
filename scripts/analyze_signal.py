from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from app.analysis.analyzer import AnalysisConfig, analyze_signal
from app.io.forensic import inspect_raw_iq
from app.io.loader import load_signal
from app.io.raw_iq import RawIQConfig
from app.models.metadata import MetadataStatus
from app.models.signal import Endian, IQOrder

def _format_freq(val_hz: float | None, val_norm: float | None, unit: str = "kHz") -> str:
    if val_hz is not None:
        if abs(val_hz) >= 1e6:
            return f"{val_hz / 1e6:+.3f} MHz"
        elif abs(val_hz) >= 1e3:
            return f"{val_hz / 1e3:+.1f} kHz"
        else:
            return f"{val_hz:+.1f} Hz"
    elif val_norm is not None:
        return f"{val_norm:+.4f} cycles/sample"
    return "N/A"

def _format_bw(val_hz: float | None, val_norm: float | None) -> str:
    if val_hz is not None:
        if val_hz >= 1e6:
            return f"{val_hz / 1e6:.3f} MHz"
        elif val_hz >= 1e3:
            return f"{val_hz / 1e3:.1f} kHz"
        else:
            return f"{val_hz:.1f} Hz"
    elif val_norm is not None:
        return f"{val_norm:.4f} cycles/sample"
    return "N/A"

def _format_rate(cand) -> str:
    if cand.rate_hz is not None:
        if cand.rate_hz >= 1e6:
            r_str = f"{cand.rate_hz / 1e6:.3f} Msym/s"
        elif cand.rate_hz >= 1e3:
            r_str = f"{cand.rate_hz / 1e3:.1f} ksym/s"
        else:
            r_str = f"{cand.rate_hz:.1f} sym/s"
    elif cand.normalized_rate is not None:
        r_str = f"{cand.normalized_rate:.4f} symbols/sample"
    else:
        r_str = "N/A"
    
    sps_str = f"{cand.estimated_samples_per_symbol:.1f} samples/symbol" if cand.estimated_samples_per_symbol else ""
    return f"{r_str}\n    {sps_str}\n    method: {cand.method} [score={cand.score:.2f}, {cand.status.value}]"

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26147 Quantitative Signal Analysis & Parameter Extraction")
    parser.add_argument("path", help="Path to recording (WAV, SigMF, or Raw IQ)")
    parser.add_argument("--dtype", choices=["complex64", "float32", "int8", "int16", "uint8"], help="Raw IQ datatype")
    parser.add_argument("--iq-order", choices=["IQ", "QI"], default="IQ", help="Raw IQ channel ordering")
    parser.add_argument("--endian", choices=["little", "big"], default="little", help="Raw IQ byte endianness")
    parser.add_argument("--sample-rate", type=float, default=None, help="Explicit sample rate in Hz")
    parser.add_argument("--center-freq", type=float, default=None, help="Explicit center frequency in Hz")
    parser.add_argument("--stereo-iq", action="store_true", help="Interpret 2-channel WAV as stereo IQ")
    parser.add_argument("--fft-size", type=int, default=4096, help="FFT size")
    parser.add_argument("--threshold-db", type=float, default=10.0, help="Detection threshold in dB above noise")
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
    
    # Run analysis
    config = AnalysisConfig(fft_size=args.fft_size, detection_threshold_db=args.threshold_db)
    analysis = analyze_signal(recording, config=config)

    # Format terminal output
    print("=" * 60)
    print("SIH26147 SIGNAL ANALYSIS")
    print("=" * 60)
    print(f"Input:\n    {path.name}")
    print(f"\nSamples:\n    {analysis.sample_count:,}")

    sr_str = "UNKNOWN"
    if analysis.sample_rate_hz.value is not None:
        sr_val = analysis.sample_rate_hz.value
        sr_str = f"{sr_val / 1e6:.3f} MHz" if sr_val >= 1e6 else f"{sr_val / 1e3:.1f} kHz"
        sr_str += f" [{analysis.sample_rate_hz.source.value}]"
    print(f"\nSample rate:\n    {sr_str}")

    dur_str = f"{analysis.duration_seconds:.3f} s" if analysis.duration_seconds is not None else "N/A"
    print(f"\nSignal duration:\n    {dur_str}")
    print(f"\nSemantic type:\n    {analysis.semantic_type}")

    # Spectral Analysis
    print("\n" + "-" * 60)
    print("SPECTRAL ANALYSIS")
    
    dom_freq = analysis.frequency_candidates[0] if analysis.frequency_candidates else None
    if dom_freq:
        print(f"\nPeak frequency:\n    {_format_freq(dom_freq.frequency_hz, dom_freq.normalized_frequency)} [{dom_freq.method}]")
    
    bw_est = next((b for b in analysis.bandwidth_candidates if b.method == "power_containment_99pct"), None)
    if bw_est:
        print(f"\nOccupied bandwidth:\n    {_format_bw(bw_est.occupied_bandwidth_hz, bw_est.occupied_bandwidth_normalized)} [99%-power]")

    nf_str = f"{analysis.noise_estimate.noise_floor_db:.1f} dB [relative]" if analysis.noise_estimate.noise_floor_db is not None else "N/A"
    print(f"\nNoise floor:\n    {nf_str}")

    snr_est = next((s for s in analysis.snr_candidates if s.method == "spectral_noise_floor"), None)
    snr_str = f"{snr_est.snr_db:.1f} dB" if (snr_est and snr_est.snr_db is not None) else "N/A"
    print(f"\nEstimated SNR:\n    {snr_str}")

    # Signal Detection
    print("\n" + "-" * 60)
    print("SIGNAL DETECTION")
    print(f"\nCandidate regions:\n    {len(analysis.detected_regions)}")
    for reg in analysis.detected_regions[:5]:
        print(f"\nRegion #{reg.region_id}:")
        print(f"    center: {_format_freq(reg.center_freq_hz, reg.center_freq_normalized)}")
        print(f"    bandwidth: {_format_bw(reg.bandwidth_hz, reg.bandwidth_normalized)}")
        print(f"    peak power: {reg.peak_power_db:.1f} dB")
        print(f"    estimated SNR: {reg.estimated_snr_db:.1f} dB")
        print(f"    detection score: {reg.detection_score:.2f} (confidence: {reg.confidence:.2f})")

    # Symbol-Rate Candidates
    print("\n" + "-" * 60)
    print("SYMBOL-RATE CANDIDATES")
    if analysis.symbol_rate_candidates:
        for idx, cand in enumerate(analysis.symbol_rate_candidates[:3], start=1):
            print(f"\n#{idx}:\n    {_format_rate(cand)}")
    else:
        print("\n    No prominent periodicity / rate candidate detected.")

    # Diagnostics
    print("\n" + "-" * 60)
    print("DIAGNOSTICS")
    if analysis.diagnostics:
        for diag in analysis.diagnostics:
            print(f"[{diag.severity.value.upper()}] {diag.code}: {diag.message}")
    else:
        print("  [INFO] No anomalous diagnostics recorded.")
    print("=" * 60)

if __name__ == "__main__":
    main()
