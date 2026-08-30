from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from app.dsp.pipeline import AnalysisConfig, run_dsp_pipeline
from app.models.analysis import SignalAnalysis
from app.models.metadata import Diagnostic, DiagnosticSeverity, MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import SignalRecording

__ANALYSIS_VERSION__ = "0.2.0"

def analyze_signal(
    recording: SignalRecording,
    config: AnalysisConfig | None = None,
) -> SignalAnalysis:
    """
    Transform a validated SignalRecording into a quantitatively characterized SignalAnalysis.

    Parameters
    ----------
    recording : SignalRecording
        Validated Phase 1 signal recording contract.
    config : AnalysisConfig | None
        Analysis engine configuration parameters.

    Returns
    -------
    SignalAnalysis
        Quantitative measurement object containing time, frequency, time-frequency,
        noise, detection, bandwidth, SNR, frequency, and rate estimates with full provenance.
    """
    cfg = config or AnalysisConfig()
    
    # Extract sample rate if present
    sample_rate_hz: float | None = None
    if recording.sample_rate_hz.status in (MetadataStatus.KNOWN, MetadataStatus.INFERRED, MetadataStatus.ASSUMED, MetadataStatus.MEASURED):
        sample_rate_hz = recording.sample_rate_hz.value

    # Extract center frequency if present
    center_freq_hz: float | None = None
    if recording.center_frequency_hz.status in (MetadataStatus.KNOWN, MetadataStatus.INFERRED, MetadataStatus.ASSUMED, MetadataStatus.MEASURED):
        center_freq_hz = recording.center_frequency_hz.value

    is_complex = (recording.semantic_type == "complex_iq")
    n_samples = len(recording.samples)

    # Compute duration if sample rate known
    duration_s = (n_samples / sample_rate_hz) if (sample_rate_hz and sample_rate_hz > 0) else None

    # Run DSP pipeline
    dsp_result = run_dsp_pipeline(
        recording.samples,
        sample_rate_hz=sample_rate_hz,
        sample_rate_confidence=recording.sample_rate_hz.confidence,
        center_frequency_hz=center_freq_hz,
        is_complex=is_complex,
        original_dtype=recording.original_dtype,
        semantic_type=recording.semantic_type,
        config=cfg,
    )

    # Combine recording diagnostics with DSP diagnostics
    combined_diagnostics: list[Diagnostic] = list(recording.diagnostics) + list(dsp_result.diagnostics)

    # Compile comprehensive provenance
    provenance: dict[str, Any] = {
        "analysis_version": __ANALYSIS_VERSION__,
        "input_recording_source": recording.provenance.get("source_path", "in_memory"),
        "input_format": recording.source_format.value,
        "input_dtype": recording.original_dtype,
        "semantic_type": recording.semantic_type,
        "sample_count": n_samples,
        "sample_rate_status": recording.sample_rate_hz.status.value,
        "configuration": {
            "fft_size": cfg.fft_size,
            "window": cfg.window,
            "psd_segment_length": cfg.psd_segment_length,
            "psd_overlap": cfg.psd_overlap,
            "spectrogram_fft_size": cfg.spectrogram_fft_size,
            "spectrogram_overlap": cfg.spectrogram_overlap,
            "detection_threshold_db": cfg.detection_threshold_db,
            "noise_method": cfg.noise_method,
            "noise_percentile": cfg.noise_percentile,
            "obw_fractions": list(cfg.obw_fractions),
            "max_autocorrelation_lag": cfg.max_autocorrelation_lag,
            "max_samples_for_analysis": cfg.max_samples_for_analysis,
        },
    }

    ref = str(recording.provenance.get("source_path", "in_memory_recording"))

    return SignalAnalysis(
        recording_reference=ref,
        sample_count=n_samples,
        duration_seconds=round(duration_s, 6) if duration_s is not None else None,
        sample_rate_hz=recording.sample_rate_hz,
        center_frequency_hz=recording.center_frequency_hz,
        semantic_type=recording.semantic_type,
        time_statistics=dsp_result.time_statistics,
        dc_offset=dsp_result.dc_offset,
        clipping_diagnostics=dsp_result.clipping_diagnostics,
        spectrum=dsp_result.spectrum,
        psd=dsp_result.psd,
        spectrogram=dsp_result.spectrogram,
        autocorrelation=dsp_result.autocorrelation,
        noise_estimate=dsp_result.noise_estimate,
        detected_regions=dsp_result.detected_regions,
        bandwidth_candidates=dsp_result.bandwidth_candidates,
        snr_candidates=dsp_result.snr_candidates,
        frequency_candidates=dsp_result.frequency_candidates,
        symbol_rate_candidates=dsp_result.symbol_rate_candidates,
        activity_metrics=dsp_result.activity_metrics,
        diagnostics=combined_diagnostics,
        provenance=provenance,
    )
