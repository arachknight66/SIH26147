from typing import Optional, Dict, Any
from .models import (
    SignalRecording,
    PipelineResult,
    PipelineStageStatus,
    HypothesisStatus,
    ModulationHypothesis,
    DemodulationResult,
    DeinterleavingResult,
    FECDecodeResult,
    FrameStructure
)
from .hypotheses import evaluate_and_rank_hypotheses
from .features import extract_all_features
from .classifier import compute_classical_scores
from .demodulation import attempt_synchronization_multi_hypothesis
from .fec_concatenated import decode_concatenated
from .correlation import correlate_sync_words, BUILTIN_SYNC_WORDS
from .framing import assemble_frames

def run_full_pipeline(recording: SignalRecording, config: Dict[str, Any] = None) -> PipelineResult:
    """
    Executes the full pipeline explicitly tracking boundaries and epistemic status.
    Early-terminates cleanly if upstream stages fail or return UNKNOWN.
    """
    if config is None:
        config = {}
        
    # Default Result State
    res = PipelineResult(
        recording=recording,
        hypothesis_status=PipelineStageStatus.NOT_ATTEMPTED,
        top_hypothesis=None,
        all_hypotheses=[],
        sync_status=PipelineStageStatus.NOT_ATTEMPTED,
        demod_result=None,
        fec_status=PipelineStageStatus.NOT_ATTEMPTED,
        deint_result=None,
        fec_result=None,
        framing_status=PipelineStageStatus.NOT_ATTEMPTED,
        frame_structure=None
    )
    
    # --- Stage 2: Hypothesis ---
    res = PipelineResult(**{**res.__dict__, 'hypothesis_status': PipelineStageStatus.COMPLETED})
    
    if recording.semantic_type != "complex_iq":
        from .models import Diagnostic, Severity
        diag = Diagnostic("NON_COMPLEX_PIPELINE", f"Pipeline ran on {recording.semantic_type}. Phase/cumulant metrics are compromised.", Severity.WARNING)
        res = PipelineResult(**{**res.__dict__, 'diagnostics': res.diagnostics + [diag]})

    
    if recording.samples.ndim > 1:
        import dataclasses
        rec_1d = dataclasses.replace(recording, samples=recording.samples[:, 0])
    else:
        rec_1d = recording
        
    fv = extract_all_features(rec_1d)
    c_scores = compute_classical_scores(fv)
    snr_est = 20.0
    
    hyps, selected, is_ambig, is_unk = evaluate_and_rank_hypotheses(fv, c_scores, snr_est, {}, rec_1d)

    res = PipelineResult(**{**res.__dict__, 'all_hypotheses': hyps})
    
    if not hyps:
        res = PipelineResult(**{**res.__dict__, 'hypothesis_status': PipelineStageStatus.FAILED})
        return res
        
    top_hyp = hyps[0]
    res = PipelineResult(**{**res.__dict__, 'top_hypothesis': top_hyp})
    
    if top_hyp.status in [HypothesisStatus.UNKNOWN, HypothesisStatus.INSUFFICIENT_EVIDENCE]:
        # Early termination boundary. Do not force downstream sync on unclassifiable signals.
        return res
        
    # --- Stage 3: Sync & Demod ---
    sync_results = attempt_synchronization_multi_hypothesis(rec_1d, hyps, config)
    if not sync_results or not sync_results[0].hypothesis_confirmed:
        res = PipelineResult(**{**res.__dict__, 'sync_status': PipelineStageStatus.FAILED})
        return res
        
    demod = sync_results[0]
    res = PipelineResult(**{**res.__dict__, 'sync_status': PipelineStageStatus.COMPLETED, 'demod_result': demod})
    
    # --- Stage 4: Deinterleave & FEC ---
    vit_res, rs_res, deint_res = decode_concatenated(demod)
    res = PipelineResult(**{**res.__dict__, 'fec_status': PipelineStageStatus.COMPLETED, 'deint_result': deint_res, 'fec_result': rs_res})
    
    # Check FEC boundary exception: "If FECDecodeResult.decode_success is False, correlation may still be attempted"
    final_bits = rs_res.decoded_bits
    final_llrs = deint_res.llrs_reordered[:len(final_bits)] if len(deint_res.llrs_reordered) >= len(final_bits) else deint_res.llrs_reordered
    
    # --- Stage 5: Correlation & Framing ---
    patterns = config.get("sync_patterns", BUILTIN_SYNC_WORDS)
    matches = correlate_sync_words(final_bits, final_llrs, patterns)
    
    frame_structures = assemble_frames(final_bits, matches)
    top_frame = frame_structures[0] if frame_structures else None
    
    # Mark framing status
    status = PipelineStageStatus.COMPLETED
    if not frame_structures or top_frame.status == HypothesisStatus.UNKNOWN:
        status = PipelineStageStatus.FAILED
        
    res = PipelineResult(**{**res.__dict__, 'framing_status': status, 'frame_structure': top_frame})
    
    return res
