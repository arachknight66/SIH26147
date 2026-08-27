from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence
import numpy as np
from app.models.analysis import DetectedRegion
from app.models.metadata import Diagnostic, DiagnosticSeverity
from app.modulation.models import ModulationFamily, ModulationHypothesis

class RecoveryStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    PREPROCESSING_FAILED = "preprocessing_failed"
    CFO_ESTIMATION_FAILED = "cfo_estimation_failed"
    CARRIER_UNLOCKED = "carrier_unlocked"
    TIMING_UNLOCKED = "timing_unlocked"
    CONSTELLATION_INVALID = "constellation_invalid"
    DEMODULATION_FAILED = "demodulation_failed"
    RECOVERY_INCONCLUSIVE = "recovery_inconclusive"
    RECOVERED = "recovered"

class LockStatus(str, Enum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    AMBIGUOUS = "ambiguous"
    SEARCHING = "searching"
    UNSUPPORTED = "unsupported"

class BitStreamStatus(str, Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    UNRELIABLE = "unreliable"
    UNAVAILABLE = "unavailable"

class RecoveryQualityLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    REJECTED = "REJECTED"

@dataclass(frozen=True)
class FrequencySyncResult:
    coarse_cfo_normalized: float
    residual_cfo_normalized: float
    cfo_variance: float
    capture_bandwidth: float
    method: str
    ambiguity_set: tuple[float, ...] = ()
    is_ambiguous: bool = False
    valid: bool = True

@dataclass(frozen=True)
class CarrierSyncResult:
    phase_estimate_rad: float
    phase_error_var: float
    phase_error_rms_rad: float
    residual_cfo_normalized: float
    lock_status: LockStatus
    lock_duration_fraction: float
    loop_bandwidth: float
    damping_factor: float
    settling_symbols: int
    valid: bool = True

@dataclass(frozen=True)
class TimingSyncResult:
    estimated_sps: float
    timing_offset_samples: float
    timing_drift: float
    ted_variance: float
    ted_mean: float
    lock_status: LockStatus
    eye_opening_proxy: float
    interpolation_method: str
    valid: bool = True

@dataclass(frozen=True)
class SynchronizationResult:
    frequency: FrequencySyncResult
    carrier: CarrierSyncResult
    timing: TimingSyncResult
    is_locked: bool

@dataclass(frozen=True)
class ConstellationResult:
    symbols: np.ndarray                 # Complex64 1-SPS normalized symbol points
    cluster_centroids: np.ndarray       # Estimated cluster centroids
    cluster_variances: np.ndarray       # Variance around each centroid
    rms_radius: float
    evm_linear: float
    evm_percent: float
    evm_db: float
    decision_margin: float              # Normalized margin to nearest decision boundary
    phase_error_rms_rad: float
    amplitude_error_rms: float
    rotational_ambiguity_deg: tuple[float, ...]  # e.g. (0.0, 90.0, 180.0, 270.0)
    valid: bool = True

@dataclass(frozen=True)
class DemodulationResult:
    hard_bits: np.ndarray               # 1D uint8 array of binary decisions
    soft_decisions: np.ndarray          # 1D float32 LLR / distance soft metrics
    symbol_indices: np.ndarray          # 1D int32 symbol constellation indices
    bit_stream_status: BitStreamStatus
    bit_polarity: str = "unresolved"    # Invariant: preserved for Phase 5 protocol resolution
    fec_status: str = "not_applied"     # Invariant: FEC belongs to Phase 5
    mapping_scheme: str = "gray"
    valid: bool = True

@dataclass(frozen=True)
class RecoveryQuality:
    composite_score: float              # S_rec in [0.0, 1.0]
    evm_score: float
    timing_lock_score: float
    carrier_lock_score: float
    constellation_score: float
    decision_margin_score: float
    window_consistency_score: float
    quality_level: RecoveryQualityLevel
    pre_sync_snr_db: float | None = None
    post_sync_snr_db: float | None = None

@dataclass(frozen=True)
class RecoveryCandidate:
    candidate_id: int
    family: ModulationFamily
    order: int | None
    symbol_rate_normalized: float
    samples_per_symbol: float
    phase3_score: float
    status: RecoveryStatus
    quality: RecoveryQuality
    synchronization: SynchronizationResult | None = None
    constellation: ConstellationResult | None = None
    demodulation: DemodulationResult | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.family == ModulationFamily.FSK:
            return f"{self.order}-FSK" if self.order else "FSK"
        elif self.family == ModulationFamily.PSK:
            if self.order == 2:
                return "BPSK"
            elif self.order == 4:
                return "QPSK"
            elif self.order == 8:
                return "8PSK"
            return f"{self.order}-PSK" if self.order else "PSK"
        elif self.family == ModulationFamily.QAM:
            return f"{self.order}QAM" if self.order else "QAM"
        return self.family.value

@dataclass(frozen=True)
class RecoveredSignal:
    symbols: np.ndarray                 # 1-SPS complex64 constellation symbols
    hard_bits: np.ndarray               # 1D uint8 binary stream
    soft_bits: np.ndarray               # 1D float32 soft LLR decisions
    symbol_indices: np.ndarray          # 1D int32 symbol constellation indices
    sample_indices: np.ndarray          # Original sample index strobes
    modulation_family: ModulationFamily
    modulation_order: int | None
    symbol_rate_normalized: float
    samples_per_symbol: float
    cfo_normalized: float
    carrier_phase_rad: float
    evm_percent: float
    decision_margin: float
    rotational_ambiguities_deg: tuple[float, ...]
    bit_polarity_status: str
    provenance: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RecoveryConfig:
    max_candidates: int = 3
    max_symbol_rate_candidates: int = 5
    sps_search_delta: float = 0.5
    sps_search_step: float = 0.25
    cfo_search_range: float = 0.10
    cfo_search_steps: int = 21
    loop_bandwidth: float = 0.015
    damping_factor: float = 0.707
    rrc_rolloffs: tuple[float, ...] = (0.25, 0.35, 0.50)
    filter_span_symbols: int = 8
    max_recovery_samples: int = 16384
    min_recovery_symbols: int = 32
    evm_threshold_high: float = 0.15     # EVM < 15% -> High quality
    evm_threshold_max: float = 0.35      # EVM > 35% -> Rejection / Inconclusive
    window_count: int = 4
    random_seed: int = 42

@dataclass(frozen=True)
class RecoveryAnalysis:
    recording_reference: str
    signal_region: DetectedRegion | None
    candidates: list[RecoveryCandidate]
    selected_candidate: RecoveryCandidate | None
    recovered_signal: RecoveredSignal | None
    is_recovered: bool
    is_inconclusive: bool
    wrong_hypothesis_detected: bool = False
    failure_reason: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
