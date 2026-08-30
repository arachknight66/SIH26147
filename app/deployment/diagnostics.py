from __future__ import annotations
from typing import Any
import numpy as np
from app.models.signal import SignalRecording, SourceFormat
from app.analysis.analyzer import analyze_signal
from app.modulation.analyzer import analyze_modulation
from app.recovery.analyzer import recover_signal
from app.data_recovery.analyzer import recover_data
from app.verification.analyzer import verify_result
from scripts.generate_digital_dataset import generate_digital_stream
from app.modulation.models import ModulationFamily
from app.recovery.models import RecoveredSignal


def _make_diagnostic_recovered_signal(hard_bits: np.ndarray, soft_bits: np.ndarray | None = None) -> RecoveredSignal:
    """Build a deterministic QPSK recovery fixture without importing test code."""
    n_symbols = max(16, (len(hard_bits) + 1) // 2)
    padded = np.pad(hard_bits, (0, max(0, 2 * n_symbols - len(hard_bits))))[: 2 * n_symbols]
    pairs = padded.reshape(-1, 2)
    symbols = (np.where(pairs[:, 0] == 0, 1.0, -1.0) + 1j * np.where(pairs[:, 1] == 0, 1.0, -1.0)).astype(np.complex64) / np.sqrt(2.0)
    return RecoveredSignal(
        symbols=symbols,
        hard_bits=hard_bits.astype(np.uint8),
        soft_bits=soft_bits.astype(np.float32) if soft_bits is not None else np.where(hard_bits == 1, 1.0, -1.0).astype(np.float32),
        symbol_indices=np.arange(n_symbols, dtype=np.int32),
        sample_indices=np.arange(n_symbols, dtype=np.float64) * 8.0,
        modulation_family=ModulationFamily.PSK,
        modulation_order=4,
        symbol_rate_normalized=0.125,
        samples_per_symbol=8.0,
        cfo_normalized=0.0,
        carrier_phase_rad=0.0,
        evm_percent=1.0,
        decision_margin=0.95,
        rotational_ambiguities_deg=(0.0, 90.0, 180.0, 270.0),
        bit_polarity_status="unresolved",
    )

def run_self_diagnostics() -> dict[str, Any]:
    """
    Run an end-to-end self-diagnostic test across all computational modules
    using a synthetic QPSK signal.
    """
    results: dict[str, Any] = {}
    try:
        rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=3, seed=42)
        rec_sig = _make_diagnostic_recovered_signal(rx, soft)

        samples = np.repeat(rec_sig.symbols, 4)
        rec = SignalRecording(
            samples=samples,
            source_format=SourceFormat.RAW_IQ,
            original_dtype="complex64",
            channels=1,
            semantic_type="iq",
        )

        p2 = analyze_signal(rec)
        results["phase2_measurement"] = "PASS" if (p2 and len(p2.snr_candidates) > 0) else "FAIL"

        p3 = analyze_modulation(rec, p2)
        results["phase3_modulation"] = "PASS" if (p3 and p3.selected_hypothesis is not None) else "FAIL"

        p4 = recover_signal(rec, analysis=p2, modulation_analysis=p3)
        results["phase4_recovery"] = "PASS" if (p4 and p4.recovered_signal is not None) else "FAIL"

        p5 = recover_data(rec_sig)
        results["phase5_data_recovery"] = "PASS" if p5 is not None else "FAIL"

        p6 = verify_result(p5, rec_sig)
        results["phase6_verification"] = "PASS" if p6 is not None else "FAIL"

        results["overall_health"] = "HEALTHY" if all(v == "PASS" for v in results.values()) else "DEGRADED"
    except Exception as e:
        results["overall_health"] = "FAILED"
        results["error"] = str(e)

    return results
