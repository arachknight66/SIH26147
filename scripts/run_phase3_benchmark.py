from __future__ import annotations
import time
from collections import defaultdict
import numpy as np
from app.models.metadata import MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import SignalRecording, SourceFormat
from app.modulation.analyzer import analyze_modulation
from app.modulation.models import ModulationAnalysisConfig
from scripts.generate_modulated_dataset import generate_modulated_signal

def run_phase3_benchmark() -> None:
    print("=" * 65)
    print("SIH26147 PHASE 3 MODULATION IDENTIFICATION SCIENTIFIC BENCHMARK")
    print("=" * 65)

    target_classes = ["BFSK", "BPSK", "QPSK", "8PSK", "16QAM"]
    cfg = ModulationAnalysisConfig()

    # -------------------------------------------------------------
    # Experiment A: Clean Signals Benchmark (20 dB SNR)
    # -------------------------------------------------------------
    print("\n1. EXPERIMENT A — CLEAN SIGNALS (20 dB SNR)")
    print("-" * 50)
    correct_a = 0
    total_a = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for mod in target_classes:
        for seed in range(20):
            samples, manifest = generate_modulated_signal(mod, snr_db=20.0, seed=100 + seed)
            rec = SignalRecording(
                samples=samples,
                source_format=SourceFormat.RAW_IQ,
                original_dtype="complex64",
                channels=2,
                semantic_type="complex_iq",
            )
            res = analyze_modulation(rec, config=cfg)
            pred_label = res.selected_hypothesis.label if res.selected_hypothesis else ("AMBIGUOUS" if res.is_ambiguous else "UNKNOWN")
            confusion[mod][pred_label] += 1
            if pred_label == mod:
                correct_a += 1
            total_a += 1

    acc_a = (correct_a / total_a) * 100.0
    print(f"Top-1 Accuracy on Clean Signals: {acc_a:.1f}% ({correct_a}/{total_a})")
    print("\nConfusion Matrix (Rows: True Class, Cols: Predicted):")
    all_preds = sorted(list({p for row in confusion.values() for p in row.keys()}))
    header = f"{'True Class':10} | " + " | ".join([f"{p:8}" for p in all_preds])
    print(header)
    print("-" * len(header))
    for true_cls in target_classes:
        row_str = f"{true_cls:10} | " + " | ".join([f"{confusion[true_cls].get(p, 0):8d}" for p in all_preds])
        print(row_str)

    # -------------------------------------------------------------
    # Experiment B: SNR Sweep Evaluation (0, 5, 10, 15, 20 dB)
    # -------------------------------------------------------------
    print("\n2. EXPERIMENT B — SNR SWEEP (Degradation Analysis)")
    print("-" * 50)
    snr_levels = [0.0, 5.0, 10.0, 15.0, 20.0]
    for snr in snr_levels:
        correct_s = 0
        total_s = 0
        for mod in target_classes:
            for seed in range(15):
                samples, _ = generate_modulated_signal(mod, snr_db=snr, seed=200 + seed)
                rec = SignalRecording(samples=samples, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
                res = analyze_modulation(rec, config=cfg)
                pred = res.selected_hypothesis.label if res.selected_hypothesis else ("AMBIGUOUS" if res.is_ambiguous else "UNKNOWN")
                if pred == mod:
                    correct_s += 1
                total_s += 1
        print(f"SNR: {snr:4.1f} dB | Top-1 Accuracy: {(correct_s/total_s)*100.0:5.1f}% ({correct_s}/{total_s})")

    # -------------------------------------------------------------
    # Experiment C & D: CFO & Timing Offset Robustness
    # -------------------------------------------------------------
    print("\n3. EXPERIMENTS C & D — CFO & TIMING OFFSET ROBUSTNESS (15 dB SNR)")
    print("-" * 50)
    cfo_offsets = [0.001, 0.005, 0.015]
    for cfo in cfo_offsets:
        c_cor = 0
        for mod in target_classes:
            for seed in range(10):
                samples, _ = generate_modulated_signal(mod, snr_db=15.0, cfo_normalized=cfo, seed=300 + seed)
                rec = SignalRecording(samples=samples, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
                res = analyze_modulation(rec, config=cfg)
                pred = res.selected_hypothesis.label if res.selected_hypothesis else "OTHER"
                if pred == mod:
                    c_cor += 1
        print(f"CFO: {cfo:+.4f} cycles/sample | Top-1 Accuracy: {(c_cor/50.0)*100.0:5.1f}%")

    # -------------------------------------------------------------
    # Experiment E: Fading Channels (Rayleigh & Rician)
    # -------------------------------------------------------------
    print("\n4. EXPERIMENT E — CHANNEL FADING (15 dB SNR)")
    print("-" * 50)
    for f_mode in ("rician", "rayleigh"):
        f_cor = 0
        for mod in target_classes:
            for seed in range(10):
                samples, _ = generate_modulated_signal(mod, snr_db=15.0, fading=f_mode, seed=400 + seed)
                rec = SignalRecording(samples=samples, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
                res = analyze_modulation(rec, config=cfg)
                pred = res.selected_hypothesis.label if res.selected_hypothesis else "OTHER"
                if pred == mod:
                    f_cor += 1
        print(f"Fading: {f_mode:8} | Top-1 Accuracy: {(f_cor/50.0)*100.0:5.1f}%")

    # -------------------------------------------------------------
    # Experiment F: Out-of-Distribution (OOD) Rejection
    # -------------------------------------------------------------
    print("\n5. EXPERIMENT F — OUT-OF-DISTRIBUTION REJECTION")
    print("-" * 50)
    ood_classes = ["AM", "FM", "GMSK", "OFDM", "NOISE"]
    ood_rejected = 0
    ood_total = 0

    for ood in ood_classes:
        for seed in range(10):
            samples, _ = generate_modulated_signal(ood, snr_db=20.0, seed=500 + seed)
            rec = SignalRecording(samples=samples, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")
            res = analyze_modulation(rec, config=cfg)
            if res.is_unknown or res.is_ambiguous or (res.selected_hypothesis and res.selected_hypothesis.quality == "LOW"):
                ood_rejected += 1
            ood_total += 1

    ood_rej_rate = (ood_rejected / ood_total) * 100.0
    print(f"OOD Non-forced Rejection / Ambiguity Rate: {ood_rej_rate:.1f}% ({ood_rejected}/{ood_total})")
    print(f"OOD False Positive Classification Rate:    {100.0 - ood_rej_rate:.1f}%")

    # -------------------------------------------------------------
    # Performance Execution Timings
    # -------------------------------------------------------------
    print("\n6. EXECUTION SPEED BENCHMARKS (16,384 samples)")
    print("-" * 50)
    bench_samples, _ = generate_modulated_signal("QPSK", n_symbols=2048, samples_per_symbol=8)
    rec = SignalRecording(samples=bench_samples, source_format=SourceFormat.RAW_IQ, original_dtype="complex64", channels=2, semantic_type="complex_iq")

    t0 = time.perf_counter()
    res_full = analyze_modulation(rec, config=cfg)
    t_full = (time.perf_counter() - t0) * 1000.0

    print(f"Full Modulation Analysis Pipeline: {t_full:6.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    run_phase3_benchmark()
