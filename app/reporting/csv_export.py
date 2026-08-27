from __future__ import annotations
import csv
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult

def export_frames_csv(result: PipelineResult, file_path: str) -> None:
    """Export frame reconstruction table to CSV."""
    p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None
    sel_cand = p5.selected_candidate if p5 else None
    frames = sel_cand.frames if sel_cand else ()

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "start_bit",
            "end_bit",
            "total_bits",
            "sequence_number",
            "length_field",
            "is_crc_valid",
            "is_fec_corrected",
            "payload_bytes_len",
        ])
        for fr in frames:
            writer.writerow([
                fr.frame_index,
                fr.start_bit,
                fr.end_bit,
                len(fr.raw_bits),
                fr.sequence_number if fr.sequence_number is not None else "N/A",
                fr.length_field_value if fr.length_field_value is not None else "N/A",
                fr.is_crc_valid,
                fr.is_fec_corrected,
                len(fr.decoded_payload) if fr.decoded_payload else 0,
            ])

def export_parameters_csv(result: PipelineResult, file_path: str) -> None:
    """Export extracted signal and demodulation parameters to CSV."""
    p2 = result.phase2_result.output if (result.phase2_result and result.phase2_result.output) else None
    p3 = result.phase3_result.output if (result.phase3_result and result.phase3_result.output) else None
    p4 = result.phase4_result.output if (result.phase4_result and result.phase4_result.output) else None
    p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None
    p6 = result.phase6_result.output if (result.phase6_result and result.phase6_result.output) else None

    params = [
        ("Modulation Winner", p3.selected_hypothesis.label if (p3 and p3.selected_hypothesis) else "UNKNOWN", "dimensionless", "INFERRED"),
        ("Modulation Confidence", f"{p3.selected_hypothesis.score:.4f}" if (p3 and p3.selected_hypothesis) else "0.0", "score", "INFERRED"),
        ("Samples Per Symbol", f"{p4.recovered_signal.samples_per_symbol:.2f}" if (p4 and p4.recovered_signal) else "UNKNOWN", "samples/symbol", "INFERRED"),
        ("EVM RMS", f"{p4.recovered_signal.evm_percent:.2f}" if (p4 and p4.recovered_signal) else "UNKNOWN", "%", "MEASURED"),
        ("Residual CFO", f"{p4.recovered_signal.cfo_normalized:.6f}" if (p4 and p4.recovered_signal) else "UNKNOWN", "normalized", "MEASURED"),
        ("SNR Estimate", f"{p2.snr_candidates[0].snr_db:.2f}" if (p2 and p2.snr_candidates) else "UNKNOWN", "dB", "ESTIMATED"),
        ("Occupied Bandwidth", f"{p2.bandwidth_candidates[0].occupied_bandwidth_hz:.1f}" if (p2 and p2.bandwidth_candidates and p2.bandwidth_candidates[0].occupied_bandwidth_hz is not None) else "UNKNOWN", "Hz", "ESTIMATED"),
        ("FEC Code", p5.selected_candidate.fec.code_name if (p5 and p5.selected_candidate and p5.selected_candidate.fec) else "NONE", "string", "INFERRED"),
        ("Integrity CRC", p5.selected_candidate.integrity.crc_results[0].crc_name if (p5 and p5.selected_candidate and p5.selected_candidate.integrity and p5.selected_candidate.integrity.crc_results) else "NONE", "string", "INFERRED"),
        ("Verification Status", p6.status.value.upper() if p6 else "UNKNOWN", "status", "VERIFIED" if (p6 and p6.is_verified) else "UNVERIFIED"),
    ]

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter_name", "value", "unit", "epistemic_status"])
        for p in params:
            writer.writerow(p)
