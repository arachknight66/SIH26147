from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from typing import Any
import numpy as np

from app.analysis.analyzer import analyze_signal
from app.data_recovery.analyzer import recover_data
from app.data_recovery.models import DataRecoveryAnalysis
from app.io.loader import load_signal
from app.models.analysis import SignalAnalysis
from app.models.metadata import Diagnostic, DiagnosticSeverity
from app.models.signal import SignalRecording, SourceFormat
from app.modulation.analyzer import analyze_modulation
from app.modulation.models import ModulationAnalysis
from app.recovery.analyzer import recover_signal
from app.recovery.models import (
    RecoveredSignal,
    RecoveryAnalysis,
    RecoveryCandidate,
    RecoveryQuality,
    RecoveryQualityLevel,
    RecoveryStatus,
)
from app.verification.analyzer import verify_result
from app.verification.models import VerificationAnalysis, VerificationConfig, VerificationStatus

from .cancellation import CancellationToken, PipelineCancelledError
from .failure_recovery import FailureCategory, PipelineFailure, classify_stage_failure
from .pipeline_config import PipelineConfig, PresetName, get_preset_config
from .progress import ProgressCallback, ProgressTracker
from .provenance import ProvenanceManifest, build_provenance_manifest
from .stage_executor import StageResult, execute_stage
from .state_machine import PipelineState, PipelineStateMachine

@dataclass
class PipelineResult:
    input_recording: SignalRecording | None
    input_path: str | None
    input_sha256: str
    phase1_result: StageResult[SignalRecording]
    phase2_result: StageResult[SignalAnalysis] | None
    phase3_result: StageResult[ModulationAnalysis] | None
    phase4_result: StageResult[RecoveryAnalysis] | None
    phase5_result: StageResult[DataRecoveryAnalysis] | None
    phase6_result: StageResult[VerificationAnalysis] | None
    is_success: bool
    is_verified: bool
    state: PipelineState
    total_duration_seconds: float
    provenance: ProvenanceManifest | None
    failure: PipelineFailure | None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def final_assessment_text(self) -> str:
        if self.phase6_result and self.phase6_result.output:
            v = self.phase6_result.output
            return f"Status: {v.status.value.upper()} | Quality: {v.quality_level.value} | Verified: {v.is_verified}"
        elif self.failure:
            return f"Failed: {self.failure.category.value.upper()} ({self.failure.message})"
        return "Analysis Inconclusive"

def run_pipeline(
    input_source: str | Path | SignalRecording | RecoveredSignal,
    config: PipelineConfig | None = None,
    cancel_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """
    Execute the unified 6-phase SIH26147 scientific signal analysis pipeline.
    """
    cfg = config or get_preset_config(PresetName.STANDARD_ANALYSIS)
    sm = PipelineStateMachine()
    tracker = ProgressTracker(total_phases=6, callback=progress_callback)
    token = cancel_token or CancellationToken()
    t_start = time.perf_counter()
    stage_durations: dict[str, float] = {}
    stage_hashes: dict[str, str] = {}

    input_path_str = str(input_source) if isinstance(input_source, (str, Path)) else None
    recording: SignalRecording | None = None
    input_sha256 = ""
    direct_recovered_signal: RecoveredSignal | None = None

    if isinstance(input_source, RecoveredSignal):
        direct_recovered_signal = input_source

    # Phase 1: Ingest & Validate
    sm.transition_to(PipelineState.LOADING)
    tracker.update(1, "Phase 1: Ingestion", "Ingesting and validating canonical signal format...", 0.1)

    def _exec_phase1() -> SignalRecording:
        nonlocal input_sha256
        if isinstance(input_source, SignalRecording):
            rec = input_source
            input_sha256 = hashlib.sha256(rec.samples.tobytes()).hexdigest()
            return rec
        elif isinstance(input_source, RecoveredSignal):
            samples = np.repeat(input_source.symbols, 4)
            rec = SignalRecording(
                samples=samples,
                source_format=SourceFormat.RAW_IQ,
                original_dtype="complex64",
                channels=1,
                semantic_type="iq",
            )
            input_sha256 = hashlib.sha256(rec.samples.tobytes()).hexdigest()
            return rec
        else:
            p = Path(input_source)
            suffix = p.suffix.lower()
            if suffix in (".iq", ".raw", ".bin"):
                from app.io.raw_iq import RawIQConfig
                from app.models.signal import Endian, IQOrder
                iq_str = str(cfg.user_overrides.get("iq_order", "IQ")).upper()
                end_str = str(cfg.user_overrides.get("endianness", "little")).lower()
                raw_cfg = RawIQConfig(
                    dtype=cfg.user_overrides.get("raw_dtype", "complex64"),
                    iq_order=IQOrder(iq_str) if iq_str in ("IQ", "QI") else IQOrder.IQ,
                    endian=Endian(end_str) if end_str in ("little", "big") else Endian.LITTLE,
                )
                rec = load_signal(p, raw_config=raw_cfg)
            else:
                rec = load_signal(p)
            input_sha256 = hashlib.sha256(rec.samples.tobytes()).hexdigest()
            return rec

    res_p1 = execute_stage("phase1_ingest", 1, _exec_phase1, token)
    stage_durations["phase1"] = res_p1.duration_seconds
    if not res_p1.success or res_p1.output is None:
        sm.transition_to(PipelineState.FAILED)
        return PipelineResult(
            input_recording=None,
            input_path=input_path_str,
            input_sha256=input_sha256 or "unknown",
            phase1_result=res_p1,
            phase2_result=None,
            phase3_result=None,
            phase4_result=None,
            phase5_result=None,
            phase6_result=None,
            is_success=False,
            is_verified=False,
            state=sm.current_state,
            total_duration_seconds=round(time.perf_counter() - t_start, 4),
            provenance=None,
            failure=res_p1.failure,
            diagnostics=res_p1.diagnostics,
        )

    recording = res_p1.output
    stage_hashes["phase1"] = hashlib.sha256(recording.samples.tobytes()).hexdigest()

    # Phase 2: Quantitative Physical Measurement
    sm.transition_to(PipelineState.VALIDATING)
    sm.transition_to(PipelineState.ANALYZING)
    tracker.update(2, "Phase 2: Physical Analysis", "Computing FFT, Welch PSD, noise floor, SNR, and signal regions...", 0.2)

    res_p2 = execute_stage("phase2_analysis", 2, lambda: analyze_signal(recording), token)
    stage_durations["phase2"] = res_p2.duration_seconds
    if not res_p2.success or res_p2.output is None:
        sm.transition_to(PipelineState.FAILED)
        return PipelineResult(
            input_recording=recording,
            input_path=input_path_str,
            input_sha256=input_sha256,
            phase1_result=res_p1,
            phase2_result=res_p2,
            phase3_result=None,
            phase4_result=None,
            phase5_result=None,
            phase6_result=None,
            is_success=False,
            is_verified=False,
            state=sm.current_state,
            total_duration_seconds=round(time.perf_counter() - t_start, 4),
            provenance=None,
            failure=res_p2.failure,
            diagnostics=res_p2.diagnostics,
        )

    analysis_p2 = res_p2.output
    stage_hashes["phase2"] = hashlib.sha256(str(analysis_p2.sample_count).encode("utf-8")).hexdigest()

    # Phase 3: Modulation Hypothesis Generation
    sm.transition_to(PipelineState.CLASSIFYING)
    tracker.update(3, "Phase 3: Modulation Analysis", "Extracting cumulant/spectral features and ranking hypotheses...", 0.4)

    res_p3 = execute_stage("phase3_modulation", 3, lambda: analyze_modulation(recording, analysis_p2), token)
    stage_durations["phase3"] = res_p3.duration_seconds
    if not res_p3.success or res_p3.output is None:
        sm.transition_to(PipelineState.FAILED)
        return PipelineResult(
            input_recording=recording,
            input_path=input_path_str,
            input_sha256=input_sha256,
            phase1_result=res_p1,
            phase2_result=res_p2,
            phase3_result=res_p3,
            phase4_result=None,
            phase5_result=None,
            phase6_result=None,
            is_success=False,
            is_verified=False,
            state=sm.current_state,
            total_duration_seconds=round(time.perf_counter() - t_start, 4),
            provenance=None,
            failure=res_p3.failure,
            diagnostics=res_p3.diagnostics,
        )

    analysis_p3 = res_p3.output
    stage_hashes["phase3"] = hashlib.sha256(str(analysis_p3.selected_hypothesis.label if analysis_p3.selected_hypothesis else "none").encode("utf-8")).hexdigest()

    # Phase 4: Synchronization & Demodulation
    sm.transition_to(PipelineState.SYNCHRONIZING)
    tracker.update(4, "Phase 4: Synchronization", "Synchronizing carrier/timing and demodulating 1-SPS constellation...", 0.6)

    def _exec_phase4() -> RecoveryAnalysis:
        if direct_recovered_signal is not None:
            cand = RecoveryCandidate(
                candidate_id=1,
                family=direct_recovered_signal.modulation_family,
                order=direct_recovered_signal.modulation_order,
                symbol_rate_normalized=direct_recovered_signal.symbol_rate_normalized,
                samples_per_symbol=direct_recovered_signal.samples_per_symbol,
                phase3_score=0.95,
                status=RecoveryStatus.RECOVERED,
                quality=RecoveryQuality(
                    composite_score=0.95,
                    evm_score=0.95,
                    timing_lock_score=0.95,
                    carrier_lock_score=0.95,
                    constellation_score=0.95,
                    decision_margin_score=0.95,
                    window_consistency_score=0.95,
                    quality_level=RecoveryQualityLevel.HIGH,
                ),
            )
            return RecoveryAnalysis(
                recording_reference="in_memory",
                signal_region=None,
                candidates=[cand],
                selected_candidate=cand,
                recovered_signal=direct_recovered_signal,
                is_recovered=True,
                is_inconclusive=False,
            )
        else:
            return recover_signal(recording, analysis=analysis_p2, modulation_analysis=analysis_p3)

    res_p4 = execute_stage("phase4_recovery", 4, _exec_phase4, token)
    stage_durations["phase4"] = res_p4.duration_seconds
    if not res_p4.success or res_p4.output is None:
        sm.transition_to(PipelineState.FAILED)
        return PipelineResult(
            input_recording=recording,
            input_path=input_path_str,
            input_sha256=input_sha256,
            phase1_result=res_p1,
            phase2_result=res_p2,
            phase3_result=res_p3,
            phase4_result=res_p4,
            phase5_result=None,
            phase6_result=None,
            is_success=False,
            is_verified=False,
            state=sm.current_state,
            total_duration_seconds=round(time.perf_counter() - t_start, 4),
            provenance=None,
            failure=res_p4.failure,
            diagnostics=res_p4.diagnostics,
        )

    recovery_p4 = res_p4.output
    selected_candidate_p4 = recovery_p4.selected_candidate
    recovered_sig = recovery_p4.recovered_signal

    if selected_candidate_p4 is None or recovered_sig is None:
        # Graceful degradation: Phase 4 could not lock
        sm.transition_to(PipelineState.FAILED)
        fail = PipelineFailure(
            category=FailureCategory.SYNCHRONIZATION_FAILURE,
            stage_name="phase4_recovery",
            message="No synchronization candidate locked successfully.",
            remediation_suggestion="Inspect modulation order, SNR, and CFO search range.",
        )
        return PipelineResult(
            input_recording=recording,
            input_path=input_path_str,
            input_sha256=input_sha256,
            phase1_result=res_p1,
            phase2_result=res_p2,
            phase3_result=res_p3,
            phase4_result=res_p4,
            phase5_result=None,
            phase6_result=None,
            is_success=False,
            is_verified=False,
            state=sm.current_state,
            total_duration_seconds=round(time.perf_counter() - t_start, 4),
            provenance=None,
            failure=fail,
            diagnostics=recovery_p4.diagnostics,
        )

    stage_hashes["phase4"] = hashlib.sha256(recovered_sig.hard_bits.tobytes()).hexdigest()

    # Phase 5: Data Recovery, Framing & Error Correction
    sm.transition_to(PipelineState.RECONSTRUCTING)
    tracker.update(5, "Phase 5: Data Recovery", "Reconstructing frames, evaluating descrambler, Viterbi FEC, and CRC...", 0.8)

    res_p5 = execute_stage("phase5_data_recovery", 5, lambda: recover_data(recovery_p4), token)
    stage_durations["phase5"] = res_p5.duration_seconds
    if not res_p5.success or res_p5.output is None:
        sm.transition_to(PipelineState.FAILED)
        return PipelineResult(
            input_recording=recording,
            input_path=input_path_str,
            input_sha256=input_sha256,
            phase1_result=res_p1,
            phase2_result=res_p2,
            phase3_result=res_p3,
            phase4_result=res_p4,
            phase5_result=res_p5,
            phase6_result=None,
            is_success=False,
            is_verified=False,
            state=sm.current_state,
            total_duration_seconds=round(time.perf_counter() - t_start, 4),
            provenance=None,
            failure=res_p5.failure,
            diagnostics=res_p5.diagnostics,
        )

    data_rec_p5 = res_p5.output
    stage_hashes["phase5"] = hashlib.sha256(data_rec_p5.phase6_handoff.corrected_bits.tobytes() if data_rec_p5.phase6_handoff else b"").hexdigest()

    # Phase 6: Independent Scientific Verification & Falsification
    sm.transition_to(PipelineState.VERIFYING)
    tracker.update(6, "Phase 6: Independent Verification", "Auditing 7-claim matrix, falsification disproofs, and error budget...", 0.95)

    v_cfg = VerificationConfig(
        min_window_consistency_fraction=cfg.verification.min_window_consistency,
        max_allowable_correction_fraction=cfg.data_recovery.max_fec_correction_fraction,
        held_out_split_ratio=cfg.verification.held_out_ratio,
        multiple_testing_alpha=cfg.verification.multiple_testing_alpha,
        random_seed=cfg.random_seed,
        strict_falsification=cfg.verification.strict_falsification,
    )

    res_p6 = execute_stage(
        "phase6_verification",
        6,
        lambda: verify_result(
            phase5_result=data_rec_p5,
            phase4_result=recovered_sig,
            phase3_result=analysis_p3 if direct_recovered_signal is None else None,
            phase2_result=analysis_p2 if direct_recovered_signal is None else None,
            phase1_result=recording if direct_recovered_signal is None else None,
            config=v_cfg,
        ),
        token,
    )
    stage_durations["phase6"] = res_p6.duration_seconds
    analysis_p6 = res_p6.output

    if analysis_p6 is not None:
        stage_hashes["phase6"] = analysis_p6.handoff.reproducibility_hash if analysis_p6.handoff else ""

    # Build Provenance Manifest
    utc_now = datetime.now(timezone.utc).isoformat()
    manifest = build_provenance_manifest(
        input_sha256=input_sha256,
        config=cfg,
        stage_hashes=stage_hashes,
        stage_durations=stage_durations,
        timestamp_utc=utc_now,
        software_version="0.7.0",
    )

    sm.transition_to(PipelineState.REPORTING)
    sm.transition_to(PipelineState.COMPLETED)
    tracker.update(6, "Phase 6: Verification", "Analysis Complete.", 1.0)

    is_ver = analysis_p6.is_verified if analysis_p6 else False

    return PipelineResult(
        input_recording=recording,
        input_path=input_path_str,
        input_sha256=input_sha256,
        phase1_result=res_p1,
        phase2_result=res_p2,
        phase3_result=res_p3,
        phase4_result=res_p4,
        phase5_result=res_p5,
        phase6_result=res_p6,
        is_success=True,
        is_verified=is_ver,
        state=sm.current_state,
        total_duration_seconds=round(time.perf_counter() - t_start, 4),
        provenance=manifest,
        failure=None if is_ver else (PipelineFailure(
            category=FailureCategory.VERIFICATION_FAILURE if (analysis_p6 and analysis_p6.is_falsified) else FailureCategory.INSUFFICIENT_EVIDENCE,
            stage_name="phase6_verification",
            message=f"Verification status: {analysis_p6.status.value.upper() if analysis_p6 else 'UNKNOWN'}",
        )),
        diagnostics=analysis_p6.diagnostics if analysis_p6 else [],
    )
