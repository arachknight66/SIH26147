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
from tests.test_phase6_cases import _make_rec_sig

def run_self_diagnostics() -> dict[str, Any]:
    """
    Run an end-to-end self-diagnostic test across all computational modules
    using a synthetic QPSK signal.
    """
    results: dict[str, Any] = {}
    try:
        rx, soft, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=3, seed=42)
        rec_sig = _make_rec_sig(rx, soft)

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
