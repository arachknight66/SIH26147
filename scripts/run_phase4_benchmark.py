from __future__ import annotations
import time
from typing import Any
import numpy as np
from app.analysis.analyzer import analyze_signal
from app.models.metadata import MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import SignalRecording, SourceFormat
from app.modulation.analyzer import analyze_modulation
from app.recovery.analyzer import recover_signal
from app.recovery.models import (
    LockStatus,
    ModulationFamily,
    RecoveryConfig,
    RecoveryQualityLevel,
    RecoveryStatus,
)
from scripts.generate_modulated_dataset import generate_modulated_signal

def _make_rec(samples: np.ndarray, sample_rate: float = 100000.0) -> SignalRecording:
    meta_sr = MetadataValue(
        value=sample_rate,
        source=MetadataSource.USER_INPUT,
        status=MetadataStatus.KNOWN,
        confidence=1.0,
        evidence=["benchmark"],
    )
    return SignalRecording(
        samples=samples.astype(np.complex64),
        source_format=SourceFormat.RAW_IQ,
        original_dtype="complex64",
        channels=2,
        semantic_type="complex_iq",
        sample_rate_hz=meta_sr,
    )

def compute_synthetic_ser(recovered_syms: np.ndarray, tx_syms: np.ndarray, order: int = 4) -> float:
    """Compute Symbol Error Rate resolving circular phase ambiguity."""
    if len(recovered_syms) == 0 or len(tx_syms) == 0:
        return 1.0
    
    n = min(len(recovered_syms), len(tx_syms))
    r = recovered_syms[:n]
    t = tx_syms[:n]

    # Test all possible rotational offsets: k * 2*pi / order
    best_ser = 1.0
    for k in range(order):
        rot = np.exp(-1j * k * (2.0 * np.pi / order))
        rotated_r = r * rot
        errs = np.sum(np.abs(rotated_r - t) > 0.50)
        ser = float(errs / n)
        if ser < best_ser:
            best_ser = ser
    return float(best_ser)

def run_phase4_benchmark() -> None:
    print("=" * 65)
    print("SIH26147 PHASE 4 SIGNAL RECOVERY SCIENTIFIC BENCHMARK")
    print("=" * 65)
    print()

    # -------------------------------------------------------------------------
    # EXPERIMENT A — CLEAN RECOVERY (20 dB SNR)
    # -------------------------------------------------------------------------
    print("1. EXPERIMENT A — CLEAN SIGNAL RECOVERY (20 dB SNR)")
    print("-" * 50)
    targets = ["BPSK", "QPSK", "8PSK", "BFSK", "16QAM"]
    n_trials = 10
    success_count = 0
    total_trials = len(targets) * n_trials

    for mod in targets:
        mod_success = 0
        evm_list = []
        for i in range(n_trials):
            s, m = generate_modulated_signal(mod, snr_db=20.0, seed=100 + i)
            rec = _make_rec(s)
            an = analyze_signal(rec)
            mod_an = analyze_modulation(rec, analysis=an)
            rec_an = recover_signal(rec, analysis=an, modulation_analysis=mod_an)

            if rec_an.is_recovered and rec_an.selected_candidate:
                mod_success += 1
                if rec_an.selected_candidate.constellation:
                    evm_list.append(rec_an.selected_candidate.constellation.evm_percent)

        success_count += mod_success
        avg_evm = float(np.mean(evm_list)) if evm_list else 100.0
        print(f"Modulation: {mod:6s} | Recovery Success: {mod_success:2d}/{n_trials:2d} ({mod_success/n_trials*100:5.1f}%) | Mean EVM: {avg_evm:5.1f}%")

    print(f"\nOverall Clean Recovery Success Rate: {success_count}/{total_trials} ({success_count/total_trials*100:.1f}%)\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT B & C — CFO & TIMING OFFSET SWEEPS
    # -------------------------------------------------------------------------
    print("2. EXPERIMENTS B & C — CFO & TIMING OFFSET SWEEPS (QPSK @ 18 dB SNR)")
    print("-" * 50)
    cfo_values = [0.0, 0.001, 0.005, 0.015]
    for cfo in cfo_values:
        s, _ = generate_modulated_signal("QPSK", cfo_normalized=cfo, snr_db=18.0, seed=42)
        rec = _make_rec(s)
        an = analyze_signal(rec)
        mod_an = analyze_modulation(rec, analysis=an)
        rec_an = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
        
        status = rec_an.selected_candidate.status.value.upper() if rec_an.selected_candidate else "FAILED"
        cfo_rec = rec_an.selected_candidate.synchronization.frequency.coarse_cfo_normalized if (rec_an.selected_candidate and rec_an.selected_candidate.synchronization) else 0.0
        print(f"True CFO: {cfo:+.4f} | Est CFO: {cfo_rec:+.4f} | Status: {status}")

    timing_values = [0.0, 0.15, 0.35, 0.65]
    for timing in timing_values:
        s, _ = generate_modulated_signal("QPSK", timing_offset=timing, snr_db=18.0, seed=42)
        rec = _make_rec(s)
        an = analyze_signal(rec)
        mod_an = analyze_modulation(rec, analysis=an)
        rec_an = recover_signal(rec, analysis=an, modulation_analysis=mod_an)
        status = rec_an.selected_candidate.status.value.upper() if rec_an.selected_candidate else "FAILED"
        evm = rec_an.selected_candidate.constellation.evm_percent if (rec_an.selected_candidate and rec_an.selected_candidate.constellation) else 100.0
        print(f"Timing Offset: {timing:.2f} samples | Status: {status} | EVM: {evm:.1f}%")
    print()

    # -------------------------------------------------------------------------
    # EXPERIMENT D — SNR SWEEP
    # -------------------------------------------------------------------------
    print("3. EXPERIMENT D — SNR SWEEP (Degradation Analysis on QPSK)")
    print("-" * 50)
    snr_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    for snr in snr_levels:
        s, _ = generate_modulated_signal("QPSK", snr_db=snr, seed=42)
        rec = _make_rec(s)
        an = analyze_signal(rec)
        mod_an = analyze_modulation(rec, analysis=an)
        rec_an = recover_signal(rec, analysis=an, modulation_analysis=mod_an)

        is_rec = rec_an.is_recovered
        evm = rec_an.selected_candidate.constellation.evm_percent if (rec_an.selected_candidate and rec_an.selected_candidate.constellation) else 100.0
        print(f"SNR: {snr:4.1f} dB | Recovered: {str(is_rec):5s} | EVM: {evm:5.1f}%")
    print()

    # -------------------------------------------------------------------------
    # EXPERIMENT H — OUT-OF-DISTRIBUTION REJECTION
    # -------------------------------------------------------------------------
    print("4. EXPERIMENT H — OUT-OF-DISTRIBUTION REJECTION")
    print("-" * 50)
    ood_signals = ["AM", "FM", "GMSK", "OFDM", "NOISE"]
    rejections = 0
    for ood in ood_signals:
        s, _ = generate_modulated_signal(ood, seed=42)
        rec = _make_rec(s)
        an = analyze_signal(rec)
        mod_an = analyze_modulation(rec, analysis=an)
        rec_an = recover_signal(rec, analysis=an, modulation_analysis=mod_an)

        is_rej = rec_an.is_inconclusive or (rec_an.selected_candidate and rec_an.selected_candidate.quality.quality_level == RecoveryQualityLevel.REJECTED)
        if is_rej:
            rejections += 1
        print(f"OOD Target: {ood:6s} | Correctly Inconclusive/Rejected: {str(is_rej):5s} | Diags: {[d.code for d in rec_an.diagnostics]}")

    print(f"\nOOD Non-forced Rejection Rate: {rejections}/{len(ood_signals)} ({rejections/len(ood_signals)*100:.1f}%)\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT I — WRONG PHASE 3 HYPOTHESIS PROMOTION
    # -------------------------------------------------------------------------
    print("5. EXPERIMENT I — WRONG PHASE 3 HYPOTHESIS DETECTION")
    print("-" * 50)
    # Generate 16-QAM signal
    s_qam, _ = generate_modulated_signal("16QAM", snr_db=20.0, seed=42)
    rec_qam = _make_rec(s_qam)
    an = analyze_signal(rec_qam)
    mod_an = analyze_modulation(rec_qam, analysis=an)
    rec_an = recover_signal(rec_qam, analysis=an, modulation_analysis=mod_an)

    print(f"Signal: 16QAM | Selected Candidate: {rec_an.selected_candidate.label if rec_an.selected_candidate else 'None'}")
    print(f"Wrong Hypothesis Handled: {rec_an.is_recovered and rec_an.selected_candidate.family == ModulationFamily.QAM}")
    print()

    # -------------------------------------------------------------------------
    # EXECUTION SPEED
    # -------------------------------------------------------------------------
    print("6. EXECUTION SPEED BENCHMARKS (16,384 samples)")
    print("-" * 50)
    s_bench, _ = generate_modulated_signal("QPSK", n_symbols=2048, samples_per_symbol=8, snr_db=20.0, seed=42)
    rec_bench = _make_rec(s_bench)
    an_bench = analyze_signal(rec_bench)
    mod_bench = analyze_modulation(rec_bench, analysis=an_bench)

    t0 = time.perf_counter()
    for _ in range(5):
        _ = recover_signal(rec_bench, analysis=an_bench, modulation_analysis=mod_bench)
    t1 = time.perf_counter()
    avg_ms = ((t1 - t0) / 5.0) * 1000.0
    print(f"Full Phase 4 Receiver Pipeline: {avg_ms:.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    run_phase4_benchmark()
