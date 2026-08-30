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
    unknown_threshold: float = 0.45
    ambiguity_margin: float = 0.08
    window_count: int = 4
    max_analysis_samples: int = 65_536

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

    def to_measurement_config(self):
        """Translate pipeline settings into the Phase 2 DSP configuration."""
        from app.dsp.pipeline import AnalysisConfig as MeasurementConfig

        return MeasurementConfig(
            fft_size=self.analysis.fft_size,
            window=self.analysis.welch_window,
            psd_segment_length=self.analysis.welch_segment_length,
            psd_overlap=self.analysis.welch_overlap,
            spectrogram_fft_size=min(self.analysis.fft_size, 4096),
            spectrogram_window_length=min(self.analysis.welch_segment_length, 4096),
            detection_threshold_db=self.analysis.detection_threshold_db,
            noise_percentile=self.analysis.noise_estimation_percentile,
            obw_fractions=self.analysis.obw_fractions,
            max_autocorrelation_lag=self.analysis.max_autocorrelation_lag,
            max_samples_for_analysis=self.limits.max_analysis_samples,
        )

    def to_modulation_config(self):
        """Translate orchestration settings into the deterministic Phase 3 config."""
        from app.modulation.models import ModulationAnalysisConfig

        return ModulationAnalysisConfig(
            min_snr_db=3.0,
            unknown_threshold=self.modulation.unknown_threshold,
            ambiguity_margin=self.modulation.ambiguity_margin,
            window_count=self.modulation.window_count,
            enable_ml=False,
            random_seed=self.random_seed,
            max_analysis_samples=self.modulation.max_analysis_samples,
            max_hypotheses=self.modulation.max_hypotheses,
        )

    def to_recovery_config(self):
        """Translate bounded orchestration settings into Phase 4 receiver controls."""
        from app.recovery.models import RecoveryConfig

        sps_low, sps_high = self.recovery.sps_search_range
        span = max(sps_high - sps_low, 0.5)
        step = min(0.25, span / 4.0)
        return RecoveryConfig(
            max_candidates=self.recovery.max_candidates,
            max_symbol_rate_candidates=5,
            sps_search_delta=span / 2.0,
            sps_search_step=step,
            cfo_search_range=self.recovery.cfo_search_range,
            loop_bandwidth=self.recovery.loop_bandwidth,
            damping_factor=self.recovery.damping_factor,
            max_recovery_samples=self.limits.max_analysis_samples,
            random_seed=self.random_seed,
        )

    def to_data_recovery_config(self):
        """Translate Phase 5 search constraints into reconstruction controls."""
        from app.data_recovery.models import DataRecoveryConfig

        return DataRecoveryConfig(
            max_bit_hypotheses=self.data_recovery.max_reconstruction_candidates * 4,
            max_frame_hypotheses=self.data_recovery.max_reconstruction_candidates,
            max_correction_fraction=self.data_recovery.max_fec_correction_fraction,
            evaluate_all_bit_offsets=self.data_recovery.eval_all_bit_offsets,
            evaluate_polarity_inversion=self.data_recovery.eval_polarity_inversion,
            evaluate_rotational_ambiguities=self.data_recovery.eval_rotational_ambiguities,
            enable_viterbi=self.data_recovery.enable_viterbi,
            enable_descrambler=self.data_recovery.enable_descrambler,
            random_seed=self.random_seed,
        )

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
