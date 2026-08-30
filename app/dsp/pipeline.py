from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from app.models.analysis import (
    AutocorrelationResult,
    BandwidthEstimate,
    ClippingDiagnostics,
    DCOffsetEstimate,
    DetectedRegion,
    FrequencyEstimate,
    NoiseEstimate,
    PSDResult,
    SNREstimate,
    SpectrogramResult,
    SpectrumResult,
    SymbolRateCandidate,
    TimeStatistics,
    ActivityMetrics,
)
from app.models.metadata import Diagnostic, DiagnosticSeverity, MetadataStatus
from .autocorrelation import compute_autocorrelation
from .bandwidth import compute_all_bandwidth_estimates
from .detection import detect_burst_regions_time, detect_signal_regions_spectral
from .frequency import compute_all_frequency_estimates
from .noise import estimate_noise_floor
from .psd import compute_psd
from .rate_estimation import estimate_symbol_rate_candidates
from .snr import compute_all_snr_estimates
from .spectrogram import compute_spectrogram
from .spectrum import compute_spectrum
from .statistics import compute_dc_offset, compute_time_statistics, detect_clipping

@dataclass(frozen=True)
class AnalysisConfig:
    """Explicit, fully parameterized configuration for signal analysis pipeline."""
    fft_size: int = 4096
    window: str = "hann"
    psd_segment_length: int = 4096
    psd_overlap: float = 0.5
    psd_scaling: str = "density"
    spectrogram_fft_size: int = 2048
    spectrogram_window_length: int = 2048
    spectrogram_overlap: float = 0.5
    detection_threshold_db: float = 10.0
    detection_min_bandwidth_bins: int = 3
    detection_merge_gap_bins: int = 2
    noise_method: str = "iterative_sigma_clip"
    noise_percentile: float = 25.0
    obw_fractions: tuple[float, ...] = (0.99, 0.95)
    max_autocorrelation_lag: int = 2048
    max_samples_for_analysis: int = 1_048_576
    enable_symbol_rate_candidates: bool = True
    enable_spectrogram: bool = True
    db_floor: float = -120.0

@dataclass
class DSPPipelineResult:
    time_statistics: TimeStatistics
    dc_offset: DCOffsetEstimate
    clipping_diagnostics: ClippingDiagnostics
    spectrum: SpectrumResult | None
    psd: PSDResult | None
    spectrogram: SpectrogramResult | None
    autocorrelation: AutocorrelationResult | None
    noise_estimate: NoiseEstimate
    detected_regions: list[DetectedRegion]
    bandwidth_candidates: list[BandwidthEstimate]
    snr_candidates: list[SNREstimate]
    frequency_candidates: list[FrequencyEstimate]
    symbol_rate_candidates: list[SymbolRateCandidate]
    activity_metrics: ActivityMetrics
    diagnostics: list[Diagnostic] = field(default_factory=list)

def run_dsp_pipeline(
    samples: np.ndarray,
    *,
    sample_rate_hz: float | None = None,
    sample_rate_confidence: float = 1.0,
    center_frequency_hz: float | None = None,
    is_complex: bool = True,
    original_dtype: str = "complex64",
    semantic_type: str = "complex_iq",
    config: AnalysisConfig | None = None,
) -> DSPPipelineResult:
    """
    Run complete research-grade DSP measurement and parameter extraction pipeline.
    """
    cfg = config or AnalysisConfig()
    diagnostics: list[Diagnostic] = []

    n_total = len(samples)
    if n_total == 0:
        diagnostics.append(Diagnostic(DiagnosticSeverity.ERROR, "INSUFFICIENT_DATA", "Recording contains zero samples."))
        raise ValueError("Cannot analyze an empty signal array.")

    # Check finite samples
    finite_mask = np.isfinite(samples)
    finite_ratio = float(np.mean(finite_mask))
    if finite_ratio < 1.0:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.WARNING,
                "NON_FINITE_SAMPLES",
                f"Recording contains {(1.0 - finite_ratio)*100:.2f}% non-finite (NaN/Inf) samples.",
                "Non-finite samples replaced with zero for analysis.",
            )
        )
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    # Subsample / slice if recording exceeds max_samples_for_analysis
    if n_total > cfg.max_samples_for_analysis:
        analysis_samples = samples[: cfg.max_samples_for_analysis]
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.INFO,
                "SAMPLED_ANALYSIS_WINDOW",
                f"Signal exceeds {cfg.max_samples_for_analysis:,} samples; analyzing first window ({cfg.max_samples_for_analysis:,} samples).",
            )
        )
    else:
        analysis_samples = samples

    n_samples = len(analysis_samples)
    if n_samples < 64:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.WARNING,
                "SHORT_RECORDING",
                f"Recording is very short ({n_samples} samples). Frequency resolution and statistical estimators have reduced precision.",
            )
        )

    # 1. Time statistics and DC offset
    time_stats = compute_time_statistics(analysis_samples)
    dc_offset = compute_dc_offset(analysis_samples)
    
    if dc_offset.magnitude > 0.05 * (time_stats.rms_amplitude or 1.0):
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.INFO,
                "DC_OFFSET_DETECTED",
                f"Measurable DC offset detected: I={dc_offset.i_offset:+.4g}, Q={dc_offset.q_offset:+.4g} (magnitude {dc_offset.magnitude:.4g}).",
                "DC offset is reported as an observation; not automatically filtered in Phase 2.",
            )
        )

    # 2. Clipping diagnostics
    clipping = detect_clipping(analysis_samples, original_dtype=original_dtype)
    if clipping.is_clipped:
        diagnostics.append(
            Diagnostic(
                clipping.severity,
                "CLIPPING_DETECTED",
                f"Possible clipping/saturation detected: {clipping.fraction_near_extrema*100:.2f}% of samples near extrema.",
                clipping.evidence,
            )
        )

    # 3. Spectrum
    fft_size = min(cfg.fft_size, n_samples)
    spectrum = compute_spectrum(
        analysis_samples,
        fft_size=fft_size,
        window_name=cfg.window,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=center_frequency_hz,
        is_complex=is_complex,
        db_floor=cfg.db_floor,
    )

    # 4. Welch PSD
    psd_seg = min(cfg.psd_segment_length, n_samples)
    psd = compute_psd(
        analysis_samples,
        sample_rate_hz=sample_rate_hz,
        segment_length=psd_seg,
        overlap=cfg.psd_overlap,
        window_name=cfg.window,
        fft_size=fft_size,
        scaling=cfg.psd_scaling,
        is_complex=is_complex,
    )

    # 5. Spectrogram
    if cfg.enable_spectrogram and n_samples >= 32:
        spec_win = min(cfg.spectrogram_window_length, n_samples)
        spectrogram = compute_spectrogram(
            analysis_samples,
            sample_rate_hz=sample_rate_hz,
            window_length=spec_win,
            fft_size=min(cfg.spectrogram_fft_size, spec_win),
            overlap=cfg.spectrogram_overlap,
            window_name=cfg.window,
            is_complex=is_complex,
        )
    else:
        spectrogram = None

    # 6. Noise-Floor Estimation
    noise = estimate_noise_floor(
        psd.psd,
        method=cfg.noise_method,
        percentile=cfg.noise_percentile,
    )
    if noise.is_signal_dominated:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.WARNING,
                "NOISE_ESTIMATE_UNCERTAIN",
                "Spectrum appears signal-dominated; noise floor estimate may have elevated uncertainty.",
                noise.evidence,
            )
        )

    # 7. Signal Detection (Spectral & Burst)
    spectral_regions = detect_signal_regions_spectral(
        psd,
        noise,
        threshold_db_offset=cfg.detection_threshold_db,
        min_bins=cfg.detection_min_bandwidth_bins,
        merge_gap_bins=cfg.detection_merge_gap_bins,
        sample_rate_hz=sample_rate_hz,
    )
    
    burst_regions = detect_burst_regions_time(
        analysis_samples,
        sample_rate_hz=sample_rate_hz,
    )
    
    all_regions: list[DetectedRegion] = []
    reg_id = 1
    for r in spectral_regions:
        all_regions.append(
            DetectedRegion(
                region_id=reg_id,
                start_sample=r.start_sample,
                end_sample=r.end_sample,
                start_time_s=r.start_time_s,
                end_time_s=r.end_time_s,
                start_freq_normalized=r.start_freq_normalized,
                end_freq_normalized=r.end_freq_normalized,
                center_freq_normalized=r.center_freq_normalized,
                bandwidth_normalized=r.bandwidth_normalized,
                center_freq_hz=r.center_freq_hz,
                bandwidth_hz=r.bandwidth_hz,
                peak_power_db=r.peak_power_db,
                estimated_snr_db=r.estimated_snr_db,
                detection_score=r.detection_score,
                method=r.method,
                confidence=r.confidence,
                assumptions=r.assumptions,
            )
        )
        reg_id += 1
    for r in burst_regions:
        all_regions.append(
            DetectedRegion(
                region_id=reg_id,
                start_sample=r.start_sample,
                end_sample=r.end_sample,
                start_time_s=r.start_time_s,
                end_time_s=r.end_time_s,
                start_freq_normalized=r.start_freq_normalized,
                end_freq_normalized=r.end_freq_normalized,
                center_freq_normalized=r.center_freq_normalized,
                bandwidth_normalized=r.bandwidth_normalized,
                center_freq_hz=r.center_freq_hz,
                bandwidth_hz=r.bandwidth_hz,
                peak_power_db=r.peak_power_db,
                estimated_snr_db=r.estimated_snr_db,
                detection_score=r.detection_score,
                method=r.method,
                confidence=r.confidence,
                assumptions=r.assumptions,
            )
        )
        reg_id += 1
    detected_regions = all_regions

    # Spectral regions describe occupied frequency. Duty cycle must instead be
    # computed from the union of energy-based time-domain burst intervals.
    active_mask = np.zeros(n_samples, dtype=bool)
    for region in burst_regions:
        if region.start_sample is not None and region.end_sample is not None:
            active_mask[max(0, region.start_sample) : min(n_samples, region.end_sample + 1)] = True
    active_count = int(np.count_nonzero(active_mask))
    duty_cycle = active_count / n_samples
    if burst_regions:
        activity_quality = float(np.clip(np.mean([r.confidence for r in burst_regions]), 0.0, 1.0))
        activity_evidence = f"{len(burst_regions)} energy burst(s), {active_count}/{n_samples} active samples."
    else:
        activity_quality = 0.35
        activity_evidence = "No time-domain burst exceeded the configured energy threshold; continuous or low-SNR signals remain possible."
    activity_metrics = ActivityMetrics(
        active_sample_count=active_count,
        total_sample_count=n_samples,
        duty_cycle=round(duty_cycle, 6),
        burst_count=len(burst_regions),
        method="time_domain_power_envelope",
        quality_score=round(activity_quality, 3),
        evidence=activity_evidence,
    )
    
    if not detected_regions:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.INFO,
                "NO_SIGNAL_DETECTED",
                "No candidate signal region exceeded the detection threshold above estimated noise.",
            )
        )
    elif len(detected_regions) > 1:
        diagnostics.append(
            Diagnostic(
                DiagnosticSeverity.INFO,
                "MULTIPLE_SIGNAL_REGIONS",
                f"Detected {len(detected_regions)} distinct candidate signal regions in spectrum.",
            )
        )

    # 8. Occupied Bandwidth
    bandwidth_candidates = compute_all_bandwidth_estimates(
        psd,
        noise,
        power_fractions=cfg.obw_fractions,
        threshold_db_offset=6.0,
        sample_rate_hz=sample_rate_hz,
    )

    # Check bandwidth consistency
    bw_99 = next((b for b in bandwidth_candidates if b.method == "power_containment_99pct"), None)
    bw_thresh = next((b for b in bandwidth_candidates if b.method == "noise_threshold"), None)
    if bw_99 and bw_thresh and bw_99.occupied_bandwidth_normalized and bw_thresh.occupied_bandwidth_normalized:
        diff = abs(bw_99.occupied_bandwidth_normalized - bw_thresh.occupied_bandwidth_normalized)
        if diff > 0.20 * bw_99.occupied_bandwidth_normalized and diff > 0.05:
            diagnostics.append(
                Diagnostic(
                    DiagnosticSeverity.INFO,
                    "BANDWIDTH_ESTIMATE_UNCERTAIN",
                    f"Power containment (99%) and threshold bandwidth estimates differ by {diff:.4g} cycles/sample.",
                    "Expected when signal has gradual roll-off or low SNR.",
                )
            )

    # 9. SNR Candidates
    snr_candidates = compute_all_snr_estimates(
        analysis_samples,
        psd,
        noise,
        detected_regions=spectral_regions,
    )
    for snr_est in snr_candidates:
        if snr_est.snr_db is not None and snr_est.snr_db < 3.0:
            diagnostics.append(
                Diagnostic(
                    DiagnosticSeverity.WARNING,
                    "LOW_SIGNAL_TO_NOISE",
                    f"Estimated SNR is low ({snr_est.snr_db:.1f} dB via {snr_est.method}). Parameter extraction accuracy may degrade.",
                )
            )
            break

    # 10. Frequency Estimates
    frequency_candidates = compute_all_frequency_estimates(
        analysis_samples,
        psd,
        sample_rate_hz=sample_rate_hz,
    )

    # 11. Autocorrelation
    autocorr = compute_autocorrelation(
        analysis_samples,
        max_lag=min(cfg.max_autocorrelation_lag, n_samples - 1),
    )

    # 12. Symbol-rate candidates
    if cfg.enable_symbol_rate_candidates and n_samples >= 64:
        obw_99_val = bw_99.occupied_bandwidth_normalized if (bw_99 and bw_99.occupied_bandwidth_normalized) else None
        symbol_rate_candidates = estimate_symbol_rate_candidates(
            analysis_samples,
            autocorr_result=autocorr,
            sample_rate_hz=sample_rate_hz,
            sample_rate_confidence=sample_rate_confidence,
            occupied_bandwidth_normalized=obw_99_val,
        )
        if symbol_rate_candidates:
            top_status = symbol_rate_candidates[0].status
            status_desc = "cross-validated" if top_status == MetadataStatus.ESTIMATED else "preliminary"
            diagnostics.append(
                Diagnostic(
                    DiagnosticSeverity.INFO,
                    "SYMBOL_RATE_CANDIDATES_AVAILABLE",
                    f"Generated {len(symbol_rate_candidates)} {status_desc} symbol-rate candidate(s).",
                    "Candidate rates subject to downstream AMC and clock recovery in Phase 3/4.",
                )
            )
    else:
        symbol_rate_candidates = []

    # I/Q Imbalance Diagnostic
    if is_complex and time_stats.variance_i > 0 and time_stats.variance_q > 0:
        var_ratio = time_stats.variance_i / time_stats.variance_q
        if abs(10.0 * np.log10(var_ratio)) > 3.0:
            diagnostics.append(
                Diagnostic(
                    DiagnosticSeverity.INFO,
                    "IQ_IMBALANCE_INDICATOR",
                    f"I/Q power ratio is {var_ratio:.2f} ({10.0*np.log10(var_ratio):+.1f} dB). Quadrature correlation is {time_stats.iq_correlation:.3f}.",
                    "Diagnostic indicator only; modulated signals may also exhibit non-circular statistics.",
                )
            )

    return DSPPipelineResult(
        time_statistics=time_stats,
        dc_offset=dc_offset,
        clipping_diagnostics=clipping,
        spectrum=spectrum,
        psd=psd,
        spectrogram=spectrogram,
        autocorrelation=autocorr,
        noise_estimate=noise,
        detected_regions=detected_regions,
        bandwidth_candidates=bandwidth_candidates,
        snr_candidates=snr_candidates,
        frequency_candidates=frequency_candidates,
        symbol_rate_candidates=symbol_rate_candidates,
        activity_metrics=activity_metrics,
        diagnostics=diagnostics,
    )
