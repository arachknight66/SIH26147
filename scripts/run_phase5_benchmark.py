from __future__ import annotations
import time
from typing import Any
import numpy as np
from app.data_recovery.analyzer import recover_data
from app.data_recovery.models import (
    BitOrder,
    BitPolarity,
    BitStream,
    DataQualityLevel,
    DataRecoveryConfig,
    DataRecoveryStatus,
)
from app.recovery.models import ModulationFamily, RecoveredSignal
from scripts.generate_digital_dataset import generate_digital_stream

def _make_recovered_signal(
    hard_bits: np.ndarray,
    soft_bits: np.ndarray | None = None,
) -> RecoveredSignal:
    """Wrap raw bits into RecoveredSignal contract for Phase 5 testing."""
    n_bits = len(hard_bits)
    dummy_syms = np.ones(max(1, n_bits // 2), dtype=np.complex64)
    dummy_indices = np.arange(len(dummy_syms), dtype=np.int32)
    sample_indices = np.arange(len(dummy_syms), dtype=np.float64) * 8.0

    return RecoveredSignal(
        symbols=dummy_syms,
        hard_bits=hard_bits.astype(np.uint8),
        soft_bits=soft_bits.astype(np.float32) if soft_bits is not None else np.where(hard_bits == 1, 1.0, -1.0).astype(np.float32),
        symbol_indices=dummy_indices,
        sample_indices=sample_indices,
        modulation_family=ModulationFamily.PSK,
        modulation_order=4,
        symbol_rate_normalized=0.125,
        samples_per_symbol=8.0,
        cfo_normalized=0.0,
        carrier_phase_rad=0.0,
        evm_percent=5.0,
        decision_margin=0.95,
        rotational_ambiguities_deg=(0.0, 90.0, 180.0, 270.0),
        bit_polarity_status="unresolved",
        provenance={"benchmark": True},
    )

def run_phase5_benchmark() -> None:
    print("=" * 65)
    print("SIH26147 PHASE 5 DATA RECOVERY SCIENTIFIC BENCHMARK")
    print("=" * 65)
    print()

    # -------------------------------------------------------------------------
    # EXPERIMENT A — CLEAN STRUCTURAL RECOVERY (Protocols A through E)
    # -------------------------------------------------------------------------
    print("1. EXPERIMENT A — CLEAN STRUCTURAL RECOVERY (Protocols A through E)")
    print("-" * 50)
    protocols = ["PROTOCOL_A", "PROTOCOL_B", "PROTOCOL_C", "PROTOCOL_D", "PROTOCOL_E"]
    n_trials = 10
    total_success = 0
    total_trials = len(protocols) * n_trials

    for prot in protocols:
        prot_success = 0
        for i in range(n_trials):
            rx_bits, rx_soft, manifest = generate_digital_stream(protocol=prot, num_frames=5, seed=100 + i)
            rec_sig = _make_recovered_signal(rx_bits, rx_soft)
            res = recover_data(rec_sig)

            if res.is_recovered and res.selected_candidate is not None:
                # Check if CRC or framing was confirmed
                if res.selected_candidate.integrity and res.selected_candidate.integrity.valid_frame_count >= 1:
                    prot_success += 1
                elif res.selected_candidate.preamble and res.selected_candidate.preamble.is_periodic:
                    prot_success += 1

        total_success += prot_success
        print(f"Protocol: {prot:10s} | Recovery Success: {prot_success:2d}/{n_trials:2d} ({prot_success/n_trials*100:5.1f}%)")

    print(f"\nOverall Clean Structural Recovery Rate: {total_success}/{total_trials} ({total_success/total_trials*100:.1f}%)\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT B — BIT-OFFSET SWEEP (0 to 7 bits)
    # -------------------------------------------------------------------------
    print("2. EXPERIMENT B — BIT-OFFSET SWEEP (Protocol A)")
    print("-" * 50)
    offset_success = 0
    for offset in range(8):
        rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, bit_offset=offset, seed=42 + offset)
        rec_sig = _make_recovered_signal(rx_bits, rx_soft)
        res = recover_data(rec_sig)

        is_ok = (res.is_recovered and res.selected_candidate is not None and res.selected_candidate.integrity.valid_frame_count >= 1)
        if is_ok:
            offset_success += 1
        print(f"Injected Bit Offset: {offset} bits | Status: {res.status.value.upper():20s} | Recovered: {str(is_ok):5s}")

    print(f"Bit-Offset Alignment Recovery Rate: {offset_success}/8 ({offset_success/8*100:.1f}%)\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT C — POLARITY INVERSION
    # -------------------------------------------------------------------------
    print("3. EXPERIMENT C — POLARITY INVERSION (Protocol A)")
    print("-" * 50)
    for inv in (False, True):
        rx_bits, rx_soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, invert_polarity=inv, seed=42)
        rec_sig = _make_recovered_signal(rx_bits, rx_soft)
        res = recover_data(rec_sig)
        is_ok = (res.is_recovered and res.selected_candidate is not None and res.selected_candidate.integrity.valid_frame_count >= 1)
        print(f"Inverted Polarity: {str(inv):5s} | Status: {res.status.value.upper():20s} | Recovered: {str(is_ok):5s}")
    print()

    # -------------------------------------------------------------------------
    # EXPERIMENT E — FEC ERROR CORRECTION & INFORMATION GAIN
    # -------------------------------------------------------------------------
    print("4. EXPERIMENT E — FEC ERROR CORRECTION & INFORMATION GAIN (Protocol C)")
    print("-" * 50)
    ber_levels = [0.0, 0.001, 0.01, 0.03]
    for ber in ber_levels:
        rx_bits, rx_soft, manifest = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, ber=ber, seed=42)
        rec_sig = _make_recovered_signal(rx_bits, rx_soft)
        res = recover_data(rec_sig)

        sel = res.selected_candidate
        corr_count = sel.fec_decode.corrected_bit_count if (sel and sel.fec_decode) else 0
        is_ok = (res.is_recovered and sel is not None)
        print(f"Channel BER: {ber:5.3f} | Recovered: {str(is_ok):5s} | Status: {res.status.value.upper():10s} | Corrected Bits: {corr_count:3d}")
    print()

    # -------------------------------------------------------------------------
    # EXPERIMENT I & J — OUT-OF-DISTRIBUTION & ADVERSARIAL REJECTION
    # -------------------------------------------------------------------------
    print("5. EXPERIMENTS I & J — OUT-OF-DISTRIBUTION & ADVERSARIAL REJECTION")
    print("-" * 50)
    ood_trials = 10
    ood_rejections = 0
    for i in range(ood_trials):
        rx_bits, rx_soft, _ = generate_digital_stream(protocol="OOD_RANDOM", num_frames=5, seed=500 + i)
        rec_sig = _make_recovered_signal(rx_bits, rx_soft)
        res = recover_data(rec_sig)

        is_rej = (res.is_inconclusive or res.status in (DataRecoveryStatus.INSUFFICIENT_DATA, DataRecoveryStatus.AMBIGUOUS))
        if is_rej:
            ood_rejections += 1

    print(f"OOD Non-forced Rejection Rate: {ood_rejections}/{ood_trials} ({ood_rejections/ood_trials*100:.1f}%)\n")

    # -------------------------------------------------------------------------
    # EXECUTION SPEED BENCHMARK (10,000 bits)
    # -------------------------------------------------------------------------
    print("6. EXECUTION SPEED BENCHMARKS (10,000 bits)")
    print("-" * 50)
    rx_bits_bench, rx_soft_bench, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=20, payload_len_bytes=60, seed=42)
    rec_bench = _make_recovered_signal(rx_bits_bench, rx_soft_bench)

    t0 = time.perf_counter()
    for _ in range(5):
        _ = recover_data(rec_bench)
    t1 = time.perf_counter()
    avg_ms = ((t1 - t0) / 5.0) * 1000.0
    print(f"Full Phase 5 Reconstruction & Correction: {avg_ms:.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    run_phase5_benchmark()
