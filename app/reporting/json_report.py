from __future__ import annotations
import json
from typing import Any
from app.orchestration.pipeline_runner import PipelineResult

def build_json_report(result: PipelineResult) -> dict[str, Any]:
    """
    Construct a versioned schema v1.0 JSON-serializable dictionary from a PipelineResult.
    """
    rec = result.input_recording
    p2 = result.phase2_result.output if (result.phase2_result and result.phase2_result.output) else None
    p3 = result.phase3_result.output if (result.phase3_result and result.phase3_result.output) else None
    p4 = result.phase4_result.output if (result.phase4_result and result.phase4_result.output) else None
    p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None
    p6 = result.phase6_result.output if (result.phase6_result and result.phase6_result.output) else None

    # Input section
    input_data = {
        "source_path": result.input_path or "in_memory",
        "sha256": result.input_sha256,
        "sample_count": len(rec.samples) if rec else 0,
        "format": rec.source_format.value if rec else "unknown",
        "dtype": rec.original_dtype if rec else "unknown",
    }

    # Physical measurements section
    p2_data = {
        "snr_db": p2.snr_candidates[0].snr_db if (p2 and p2.snr_candidates) else None,
        "bandwidth_hz": p2.bandwidth_candidates[0].occupied_bandwidth_hz if (p2 and p2.bandwidth_candidates) else None,
        "noise_floor_db": p2.noise_estimate.noise_floor_db if p2 else None,
        "detected_regions_count": len(p2.detected_regions) if p2 else 0,
    }

    # Modulation hypotheses section
    p3_data = {
        "winner": p3.selected_hypothesis.label if (p3 and p3.selected_hypothesis) else None,
        "winner_score": p3.selected_hypothesis.score if (p3 and p3.selected_hypothesis) else None,
        "hypotheses": [
            {
                "label": h.label,
                "score": round(h.score, 4),
                "family": h.family.value,
                "order": h.order,
            }
            for h in (p3.hypotheses if p3 else [])
        ],
    }

    # Synchronization section
    p4_data = {
        "lock_status": p4.selected_candidate.status.value if (p4 and p4.selected_candidate) else "unknown",
        "evm_percent": p4.recovered_signal.evm_percent if (p4 and p4.recovered_signal) else None,
        "cfo_normalized": p4.recovered_signal.cfo_normalized if (p4 and p4.recovered_signal) else None,
        "samples_per_symbol": p4.recovered_signal.samples_per_symbol if (p4 and p4.recovered_signal) else None,
    }

    # Data recovery section
    sel_cand = p5.selected_candidate if p5 else None
    p5_data = {
        "status": p5.status.value if p5 else "unknown",
        "frames_recovered": len(sel_cand.frames) if sel_cand else 0,
        "fec_code": sel_cand.fec.code_name if (sel_cand and sel_cand.fec) else "NONE",
        "fec_corrected_bits": sel_cand.fec_decode.corrected_bit_count if (sel_cand and sel_cand.fec_decode) else 0,
        "crc_name": sel_cand.integrity.crc_results[0].crc_name if (sel_cand and sel_cand.integrity and sel_cand.integrity.crc_results) else "NONE",
        "payload_bytes_length": len(sel_cand.recovered_payload_bytes) if sel_cand else 0,
    }

    # Verification section
    p6_data = {
        "status": p6.status.value if p6 else "unknown",
        "is_verified": p6.is_verified if p6 else False,
        "claims": [
            {
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "status": c.status.value,
                "confidence": c.confidence,
                "independence": c.independence_level.value,
            }
            for c in (p6.claims if p6 else [])
        ],
        "error_budget": p6.error_budget.__dict__ if (p6 and p6.error_budget) else None,
        "reproducibility_hash": p6.handoff.reproducibility_hash if (p6 and p6.handoff) else None,
    }

    # Provenance section
    prov = result.provenance
    prov_data = prov.__dict__ if prov else {}

    return {
        "schema_version": "1.0",
        "project": "SIH26147 Signal Recovery & Scientific Verification Engine",
        "is_success": result.is_success,
        "is_verified": result.is_verified,
        "final_assessment": result.final_assessment_text,
        "input": input_data,
        "phase2_physical": p2_data,
        "phase3_modulation": p3_data,
        "phase4_recovery": p4_data,
        "phase5_data": p5_data,
        "phase6_verification": p6_data,
        "provenance": prov_data,
        "durations_seconds": {
            "total": result.total_duration_seconds,
            "phase1": result.phase1_result.duration_seconds,
            "phase2": result.phase2_result.duration_seconds if result.phase2_result else 0.0,
            "phase3": result.phase3_result.duration_seconds if result.phase3_result else 0.0,
            "phase4": result.phase4_result.duration_seconds if result.phase4_result else 0.0,
            "phase5": result.phase5_result.duration_seconds if result.phase5_result else 0.0,
            "phase6": result.phase6_result.duration_seconds if result.phase6_result else 0.0,
        },
    }

def export_json_report(result: PipelineResult, file_path: str) -> None:
    data = build_json_report(result)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
