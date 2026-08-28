from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import hashlib
import json

class PresetName(str, Enum):
    FAST_SCREENING = "fast_screening"
    STANDARD_ANALYSIS = "standard_analysis"
    DEEP_ANALYSIS = "deep_analysis"
    FORENSIC_ANALYSIS = "forensic_analysis"

@dataclass(frozen=True)
class IngestConfig:
    max_samples: int | None = 10_000_000
    chunk_size: int = 1_000_000
    allow_memory_map: bool = True
    strict_finiteness: bool = True

@dataclass(frozen=True)
class AnalysisConfig:
    fft_size: int = 4096
    welch_segment_length: int = 2048
    welch_overlap: float = 0.50
    welch_window: str = "hann"
    noise_estimation_percentile: float = 25.0
    detection_threshold_db: float = 6.0
    obw_fractions: tuple[float, ...] = (0.99, 0.95)
    max_autocorrelation_lag: int = 2048

@dataclass(frozen=True)
class ModulationConfig:
    max_hypotheses: int = 5
    min_confidence_threshold: float = 0.15
    enable_cumulant_features: bool = True
    enable_spectral_features: bool = True
    enable_cyclic_features: bool = True

@dataclass(frozen=True)
class RecoveryConfig:
    max_candidates: int = 3
    sps_search_range: tuple[float, float] = (1.5, 32.0)
    cfo_search_range: float = 0.10
    loop_bandwidth: float = 0.015
    damping_factor: float = 0.707
    max_ted_iterations: int = 2000

@dataclass(frozen=True)
class DataRecoveryConfig:
    max_reconstruction_candidates: int = 3
    eval_all_bit_offsets: bool = True
    eval_polarity_inversion: bool = True
    eval_rotational_ambiguities: bool = True
    max_fec_correction_fraction: float = 0.10
    enable_viterbi: bool = True
    enable_descrambler: bool = True

@dataclass(frozen=True)
class VerificationPipelineConfig:
    temporal_windows: int = 8
    min_window_consistency: float = 0.80
    held_out_ratio: float = 0.30
    multiple_testing_alpha: float = 0.01
    boundary_perturbation_offsets: tuple[int, ...] = (-8, -4, -2, -1, 1, 2, 4, 8)
    strict_falsification: bool = True

@dataclass(frozen=True)
class ResourceLimitConfig:
    max_preview_samples: int = 100_000
    max_analysis_samples: int = 5_000_000
    max_runtime_seconds: float = 300.0
    max_memory_mb: int = 4096

@dataclass(frozen=True)
class PipelineConfig:
    preset: PresetName = PresetName.STANDARD_ANALYSIS
    random_seed: int = 42
    ingest: IngestConfig = field(default_factory=IngestConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    data_recovery: DataRecoveryConfig = field(default_factory=DataRecoveryConfig)
    verification: VerificationPipelineConfig = field(default_factory=VerificationPipelineConfig)
    limits: ResourceLimitConfig = field(default_factory=ResourceLimitConfig)
    user_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset.value,
            "random_seed": self.random_seed,
            "ingest": self.ingest.__dict__,
            "analysis": self.analysis.__dict__,
            "modulation": self.modulation.__dict__,
            "recovery": self.recovery.__dict__,
            "data_recovery": self.data_recovery.__dict__,
            "verification": self.verification.__dict__,
            "limits": self.limits.__dict__,
            "user_overrides": self.user_overrides,
        }

    def compute_hash(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

def get_preset_config(preset: PresetName | str, seed: int = 42) -> PipelineConfig:
    if isinstance(preset, str):
        preset = PresetName(preset.lower())

    if preset == PresetName.FAST_SCREENING:
        return PipelineConfig(
            preset=PresetName.FAST_SCREENING,
            random_seed=seed,
            analysis=AnalysisConfig(fft_size=2048, welch_segment_length=1024),
            verification=VerificationPipelineConfig(temporal_windows=4, boundary_perturbation_offsets=(-2, -1, 1, 2)),
            limits=ResourceLimitConfig(max_analysis_samples=500_000, max_runtime_seconds=60.0),
        )
    elif preset == PresetName.DEEP_ANALYSIS:
        return PipelineConfig(
            preset=PresetName.DEEP_ANALYSIS,
            random_seed=seed,
            analysis=AnalysisConfig(fft_size=8192, welch_segment_length=4096),
            modulation=ModulationConfig(max_hypotheses=10, min_confidence_threshold=0.05),
            recovery=RecoveryConfig(max_candidates=5),
            data_recovery=DataRecoveryConfig(max_reconstruction_candidates=5),
            verification=VerificationPipelineConfig(temporal_windows=16),
            limits=ResourceLimitConfig(max_analysis_samples=10_000_000, max_runtime_seconds=600.0),
        )
    elif preset == PresetName.FORENSIC_ANALYSIS:
        return PipelineConfig(
            preset=PresetName.FORENSIC_ANALYSIS,
            random_seed=seed,
            analysis=AnalysisConfig(fft_size=8192, welch_segment_length=4096),
            modulation=ModulationConfig(max_hypotheses=10),
            recovery=RecoveryConfig(max_candidates=5),
            data_recovery=DataRecoveryConfig(max_reconstruction_candidates=10),
            verification=VerificationPipelineConfig(
                temporal_windows=16,
                boundary_perturbation_offsets=(-16, -8, -4, -2, -1, 1, 2, 4, 8, 16),
            ),
            limits=ResourceLimitConfig(max_analysis_samples=20_000_000, max_runtime_seconds=1200.0),
        )
    else:
        return PipelineConfig(preset=PresetName.STANDARD_ANALYSIS, random_seed=seed)
