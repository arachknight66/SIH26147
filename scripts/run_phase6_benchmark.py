from __future__ import annotations
import time
from typing import Any
import numpy as np
from app.data_recovery.analyzer import recover_data
from app.verification.analyzer import verify_result
from app.verification.models import VerificationStatus
from scripts.generate_digital_dataset import generate_digital_stream
from tests.test_phase6_cases import _make_rec_sig

def run_all_benchmarks() -> None:
    print("=" * 65)
    print("SIH26147 PHASE 6 SCIENTIFIC VERIFICATION BENCHMARK")
    print("=" * 65)
    print("")

    # 1. Experiment A: Clean Known Signals (Protocols A through E)
    print("1. EXPERIMENT A — CLEAN SIGNAL INDEPENDENT VERIFICATION")
    print("-" * 50)
    protocols = ["PROTOCOL_A", "PROTOCOL_B", "PROTOCOL_C", "PROTOCOL_D", "PROTOCOL_E"]
    exp_a_pass = 0
    total_a = 0

    for prot in protocols:
        prot_pass = 0
        trials = 10
        for seed in range(trials):
            rx_b, rx_s, _ = generate_digital_stream(protocol=prot, num_frames=5, seed=42 + seed)
            rec = _make_rec_sig(rx_b, rx_s)
            p5 = recover_data(rec)
            p6 = verify_result(phase5_result=p5, phase4_result=rec)
            if p6.status == VerificationStatus.INDEPENDENTLY_VERIFIED:
                prot_pass += 1

        exp_a_pass += prot_pass
        total_a += trials
        print(f"Protocol: {prot:10s} | Verified: {prot_pass:2d}/{trials} ({prot_pass/trials*100:5.1f}%)")

    print(f"\nOverall Independent Verification Rate: {exp_a_pass}/{total_a} ({exp_a_pass/total_a*100:.1f}%)\n")

    # 2. Experiment B: SNR Sweep
    print("2. EXPERIMENT B — SNR SWEEP VERIFICATION")
    print("-" * 50)
    for snr in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
        rec = _make_rec_sig(rx_b, rx_s, snr_db=snr)
        p5 = recover_data(rec)
        p6 = verify_result(phase5_result=p5, phase4_result=rec)
        print(f"SNR: {snr:4.1f} dB | Status: {p6.status.value.upper():22s} | Verified: {p6.is_verified}")

    print("")

    # 3. Experiment E & H: FEC Information Gain
    print("3. EXPERIMENTS E & H — FEC ERROR CORRECTION & HELD-OUT CROSS-VALIDATION")
    print("-" * 50)
    for ber in (0.000, 0.001, 0.005, 0.010):
        rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=6, ber=ber, seed=42)
        rec = _make_rec_sig(rx_b, rx_s)
        p5 = recover_data(rec)
        p6 = verify_result(phase5_result=p5, phase4_result=rec)
        fec_a = p6.fec_audit
        corr_bits = p5.selected_candidate.fec_decode.corrected_bit_count if (p5.selected_candidate and p5.selected_candidate.fec_decode) else 0
        print(f"Channel BER: {ber:5.3f} | Status: {p6.status.value.upper():22s} | Corrected: {corr_bits:3d} | Held-out: {fec_a.held_out_validation_passed if fec_a else False}")

    print("")

    # 4. Experiments I & K: Adversarial & OOD Rejection
    print("4. EXPERIMENTS I, K & L — ADVERSARIAL & OUT-OF-DISTRIBUTION REJECTION")
    print("-" * 50)
    ood_rejections = 0
    trials_ood = 10
    for seed in range(trials_ood):
        rx_b, rx_s, _ = generate_digital_stream(protocol="OOD_RANDOM", num_frames=5, seed=100 + seed)
        rec = _make_rec_sig(rx_b, rx_s)
        p5 = recover_data(rec)
        p6 = verify_result(phase5_result=p5, phase4_result=rec)
        if not p6.is_verified:
            ood_rejections += 1

    print(f"OOD / Noise Non-Verification Rate: {ood_rejections}/{trials_ood} ({ood_rejections/trials_ood*100:.1f}%)")

    # 5. Reproducibility & Performance
    print("\n5. EXPERIMENTS P — DETERMINISTIC REPRODUCIBILITY & TIMING")
    print("-" * 50)
    t0 = time.perf_counter()
    rx_b, rx_s, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    rec = _make_rec_sig(rx_b, rx_s)
    p5 = recover_data(rec)
    p6_a = verify_result(phase5_result=p5, phase4_result=rec)
    p6_b = verify_result(phase5_result=p5, phase4_result=rec)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    hash_match = (p6_a.handoff.reproducibility_hash == p6_b.handoff.reproducibility_hash) if (p6_a.handoff and p6_b.handoff) else False
    print(f"Reproducibility Hash Match: {hash_match}")
    print(f"Execution Latency: {elapsed_ms:.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    run_all_benchmarks()
