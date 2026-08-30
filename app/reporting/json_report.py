from __future__ import annotations
import json
from typing import Any
import numpy as np
from app.orchestration.pipeline_runner import PipelineResult

def build_json_report(result: PipelineResult) -> dict[str, Any]:
    """
    Construct a versioned schema v1.0 JSON-serializable dictionary from a PipelineResult,
    including genuine downsampled arrays for interactive visualization.
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
        "activity": {
            "duty_cycle": p2.activity_metrics.duty_cycle if (p2 and p2.activity_metrics) else None,
            "burst_count": p2.activity_metrics.burst_count if (p2 and p2.activity_metrics) else None,
            "active_sample_count": p2.activity_metrics.active_sample_count if (p2 and p2.activity_metrics) else None,
            "method": p2.activity_metrics.method if (p2 and p2.activity_metrics) else None,
            "evidence": p2.activity_metrics.evidence if (p2 and p2.activity_metrics) else None,
        },
    }

    # Modulation hypotheses section
    p3_data = {
        "winner": p3.selected_hypothesis.label if (p3 and p3.selected_hypothesis) else None,
        "winner_score": p3.selected_hypothesis.score if (p3 and p3.selected_hypothesis) else None,
        "is_ambiguous": p3.is_ambiguous if p3 else False,
        "is_unknown": p3.is_unknown if p3 else True,
        "window_consistency": p3.window_consistency if p3 else None,
        "hypotheses": [
            {
                "label": h.label,
                "score": round(h.score, 4),
                "family": h.family.value,
                "order": h.order,
                "status": h.status.value,
                "quality": h.quality,
                "evidence": {
                    "amplitude": h.evidence.amplitude_score,
                    "phase": h.evidence.phase_score,
                    "frequency": h.evidence.frequency_score,
                    "cumulants": h.evidence.cumulant_score,
                    "spectral": h.evidence.spectral_score,
                    "periodicity": h.evidence.periodicity_score,
                    "contradiction_penalty": h.evidence.contradiction_penalty,
                    "supporting_notes": list(h.evidence.supporting_evidence),
                },
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
        "quality": {
            "composite_score": p4.selected_candidate.quality.composite_score if (p4 and p4.selected_candidate) else None,
            "quality_level": p4.selected_candidate.quality.quality_level.value if (p4 and p4.selected_candidate) else None,
            "evm_score": p4.selected_candidate.quality.evm_score if (p4 and p4.selected_candidate) else None,
            "timing_lock_score": p4.selected_candidate.quality.timing_lock_score if (p4 and p4.selected_candidate) else None,
            "carrier_lock_score": p4.selected_candidate.quality.carrier_lock_score if (p4 and p4.selected_candidate) else None,
            "window_consistency_score": p4.selected_candidate.quality.window_consistency_score if (p4 and p4.selected_candidate) else None,
        },
        "synchronization": {
            "is_locked": p4.selected_candidate.synchronization.is_locked if (p4 and p4.selected_candidate and p4.selected_candidate.synchronization) else None,
            "coarse_cfo_normalized": p4.selected_candidate.synchronization.frequency.coarse_cfo_normalized if (p4 and p4.selected_candidate and p4.selected_candidate.synchronization) else None,
            "residual_cfo_normalized": p4.selected_candidate.synchronization.frequency.residual_cfo_normalized if (p4 and p4.selected_candidate and p4.selected_candidate.synchronization) else None,
        },
    }

    # Data recovery section
    sel_cand = p5.selected_candidate if p5 else None
    frames_list = []
    if sel_cand and sel_cand.frames:
        for f in sel_cand.frames:
            payload_b = np.packbits(f.payload_bits).tobytes() if len(f.payload_bits) > 0 else (f.decoded_payload or b"")
            frames_list.append({
                "frame_index": f.frame_index,
                "start_bit": f.start_bit,
                "end_bit": f.end_bit,
                "length_bits": len(f.raw_bits) if hasattr(f, "raw_bits") else (f.end_bit - f.start_bit),
                "is_crc_valid": f.is_crc_valid,
                "payload_hex": payload_b.hex(),
                "payload_ascii": "".join(chr(b) if 32 <= b <= 126 else "." for b in payload_b),
            })

    fec_mask_info: dict[str, Any] = {}
    if sel_cand and sel_cand.fec_decode:
        fd = sel_cand.fec_decode
        fec_mask_info = {
            "corrected_bit_count": fd.corrected_bit_count,
            "correction_fraction": round(fd.correction_fraction, 4),
            "modified_bit_indices": [int(i) for i in np.where(fd.correction_mask)[0][:128]] if fd.correction_mask is not None else [],
        }

    p5_data = {
        "status": p5.status.value if p5 else "unknown",
        "quality_level": p5.quality_level.value if p5 else None,
        "is_ambiguous": p5.is_ambiguous if p5 else False,
        "frames_recovered": len(sel_cand.frames) if sel_cand else 0,
        "fec_code": sel_cand.fec.code_name if (sel_cand and sel_cand.fec) else "NONE",
        "fec_corrected_bits": sel_cand.fec_decode.corrected_bit_count if (sel_cand and sel_cand.fec_decode) else 0,
        "crc_name": sel_cand.integrity.crc_results[0].crc_name if (sel_cand and sel_cand.integrity and sel_cand.integrity.crc_results) else "NONE",
        "payload_bytes_length": len(sel_cand.recovered_payload_bytes) if sel_cand else 0,
        "frames_list": frames_list,
        "fec_mask": fec_mask_info,
        "candidate": {
            "composite_score": sel_cand.composite_score if sel_cand else None,
            "complexity_penalty": sel_cand.complexity_penalty if sel_cand else None,
            "phase_rotation_deg": sel_cand.bit_hypothesis.phase_rotation_deg if sel_cand else None,
            "polarity": sel_cand.bit_hypothesis.polarity.value if sel_cand else None,
            "bit_offset": sel_cand.bit_hypothesis.bit_offset if sel_cand else None,
            "preamble_hex": sel_cand.preamble.pattern_hex if (sel_cand and sel_cand.preamble) else None,
            "preamble_periodic": sel_cand.preamble.is_periodic if (sel_cand and sel_cand.preamble) else None,
            "crc_valid_fraction": sel_cand.integrity.crc_valid_fraction if (sel_cand and sel_cand.integrity) else None,
            "corrected_bit_count": sel_cand.fec_decode.corrected_bit_count if (sel_cand and sel_cand.fec_decode) else 0,
        },
    }

    # Verification section
    all_tests = []
    if p6 and p6.claims:
        for c in p6.claims:
            for t in c.tests:
                all_tests.append({
                    "test_id": t.test_id,
                    "name": t.name,
                    "category": t.category,
                    "status": t.status.value,
                    "score": round(t.score, 4),
                    "details": t.details,
                    "counter_evidence": t.counter_evidence,
                    "is_critical": t.is_critical,
                })

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
        "tests": all_tests,
        "error_budget": p6.error_budget.__dict__ if (p6 and p6.error_budget) else None,
        "reproducibility_hash": p6.handoff.reproducibility_hash if (p6 and p6.handoff) else None,
    }

    # Genuine signal plotting data
    waveform_i: list[float] = []
    waveform_q: list[float] = []
    if rec is not None and len(rec.samples) > 0:
        step_w = max(1, len(rec.samples) // 500)
        sub_samples = rec.samples[::step_w][:500]
        waveform_i = [round(float(s.real), 4) for s in sub_samples]
        waveform_q = [round(float(s.imag), 4) for s in sub_samples]

    psd_f: list[float] = []
    psd_p: list[float] = []
    if p2 is not None and p2.psd is not None and len(p2.psd.psd_db) > 0:
        step_p = max(1, len(p2.psd.psd_db) // 300)
        sub_f = p2.psd.frequencies_normalized[::step_p][:300]
        sub_p = p2.psd.psd_db[::step_p][:300]
        psd_f = [round(float(f), 4) for f in sub_f]
        psd_p = [round(float(p), 2) for p in sub_p]

    const_i: list[float] = []
    const_q: list[float] = []
    if p4 is not None and p4.recovered_signal is not None and len(p4.recovered_signal.symbols) > 0:
        step_c = max(1, len(p4.recovered_signal.symbols) // 600)
        sub_syms = p4.recovered_signal.symbols[::step_c][:600]
        const_i = [round(float(s.real), 4) for s in sub_syms]
        const_q = [round(float(s.imag), 4) for s in sub_syms]

    plots_data = {
        "waveform_i": waveform_i,
        "waveform_q": waveform_q,
        "psd_f": psd_f,
        "psd_p": psd_p,
        "noise_floor_dbfs": p2.noise_estimate.noise_floor_db if (p2 and p2.noise_estimate) else -60.0,
        "const_i": const_i,
        "const_q": const_q,
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
        "plots": plots_data,
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
