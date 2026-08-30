from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar
import numpy as np
from .metadata import Diagnostic, DiagnosticSeverity, MetadataStatus, MetadataValue

T = TypeVar("T")

@dataclass(frozen=True)
class Estimate(Generic[T]):
    value: T | None
    unit: str
    method: str
    status: MetadataStatus
    quality_score: float
    uncertainty: float | None = None
    evidence: str = ""
    assumptions: list[str] = field(default_factory=list)
    supporting_data: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TimeStatistics:
    mean_i: float
    mean_q: float
    mean_complex: complex
    variance_i: float
    variance_q: float
    iq_covariance: float
    iq_correlation: float
    mean_amplitude: float
    median_amplitude: float
    std_amplitude: float
    rms_amplitude: float
    peak_amplitude: float
    mean_power: float
    median_power: float
    variance_power: float
    peak_power: float
    peak_to_rms_ratio: float
    crest_factor: float
    dynamic_range_db: float
    phase_mean: float
    phase_variance: float
    phase_valid_fraction: float

@dataclass(frozen=True)
class DCOffsetEstimate:
    i_offset: float
    q_offset: float
    magnitude: float
    phase_rad: float
    status: MetadataStatus
    quality_score: float
    evidence: str = ""

@dataclass(frozen=True)
class ClippingDiagnostics:
    is_clipped: bool
    fraction_near_extrema: float
    sample_range_min: float
    sample_range_max: float
    clipping_threshold: float
    evidence: str
    severity: DiagnosticSeverity

@dataclass(frozen=True)
class SpectrumResult:
    frequencies: np.ndarray
    frequencies_normalized: np.ndarray
    magnitude_spectrum: np.ndarray
    power_spectrum_db: np.ndarray
    complex_spectrum: np.ndarray | None
    fft_size: int
    window: str
    coherent_gain: float
    noise_power_gain: float
    db_floor: float
    frequency_unit: str
    frequency_reference: str
    is_complex: bool
    bin_resolution: float
    bin_resolution_unit: str

@dataclass(frozen=True)
class PSDResult:
    frequencies: np.ndarray
    frequencies_normalized: np.ndarray
    psd: np.ndarray
    psd_db: np.ndarray
    segment_length: int
    overlap: float
    window: str
    fft_size: int
    detrend: str
    scaling: str
    frequency_unit: str
    is_two_sided: bool
    bin_resolution: float

@dataclass(frozen=True)
class SpectrogramResult:
    time_axis: np.ndarray
    time_unit: str
    frequency_axis: np.ndarray
    frequency_axis_normalized: np.ndarray
    power_matrix_db: np.ndarray
    window_length: int
    fft_size: int
    hop_size: int
    window: str
    frequency_unit: str

@dataclass(frozen=True)
class NoiseEstimate:
    noise_floor_db: float | None
    noise_power_linear: float | None
    method: str
    unit: str
    quality_score: float
    uncertainty_db: float | None
    is_signal_dominated: bool
    evidence: str

@dataclass(frozen=True)
class DetectedRegion:
    region_id: int
    start_sample: int | None = None
    end_sample: int | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None
    start_freq_normalized: float = 0.0
    end_freq_normalized: float = 0.0
    center_freq_normalized: float = 0.0
    bandwidth_normalized: float = 0.0
    center_freq_hz: float | None = None
    bandwidth_hz: float | None = None
    peak_power_db: float = 0.0
    estimated_snr_db: float = 0.0
    detection_score: float = 0.0
    method: str = ""
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class BandwidthEstimate:
    occupied_bandwidth_normalized: float | None
    occupied_bandwidth_hz: float | None
    method: str
    fraction: float | None = None
    threshold_db_above_noise: float | None = None
    status: MetadataStatus = MetadataStatus.ESTIMATED
    quality_score: float = 0.0
    uncertainty: float | None = None
    evidence: str = ""

@dataclass(frozen=True)
class SNREstimate:
    snr_db: float | None
    method: str
    status: MetadataStatus = MetadataStatus.ESTIMATED
    quality_score: float = 0.0
    uncertainty_db: float | None = None
    evidence: str = ""
    assumptions: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class FrequencyEstimate:
    normalized_frequency: float | None
    frequency_hz: float | None
    method: str
    status: MetadataStatus = MetadataStatus.ESTIMATED
    quality_score: float = 0.0
    uncertainty: float | None = None
    evidence: str = ""

@dataclass(frozen=True)
class AutocorrelationResult:
    lags: np.ndarray
    complex_autocorrelation: np.ndarray
    normalized_magnitude: np.ndarray
    max_lag: int

@dataclass(frozen=True)
class SymbolRateCandidate:
    normalized_rate: float | None
    estimated_samples_per_symbol: float | None
    rate_hz: float | None = None
    method: str = ""
    score: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: MetadataStatus = MetadataStatus.AMBIGUOUS
    uncertainty: float | None = None
    rate_hz_uncertainty: float | None = None

@dataclass(frozen=True)
class ActivityMetrics:
    """Time-domain occupancy derived from the energy-based burst detector."""
    active_sample_count: int
    total_sample_count: int
    duty_cycle: float
    burst_count: int
    method: str
    quality_score: float
    evidence: str

@dataclass
class SignalAnalysis:
    recording_reference: str
    sample_count: int
    duration_seconds: float | None
    sample_rate_hz: MetadataValue[float]
    center_frequency_hz: MetadataValue[float]
    semantic_type: str
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
    activity_metrics: ActivityMetrics | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
