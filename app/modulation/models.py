from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np
from app.models.analysis import DetectedRegion
from app.models.metadata import Diagnostic, DiagnosticSeverity, MetadataStatus

class ModulationFamily(str, Enum):
    FSK = "FSK"
    PSK = "PSK"
    QAM = "QAM"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"

class ModulationOrder(int, Enum):
    ORDER_2 = 2
    ORDER_4 = 4
    ORDER_8 = 8
    ORDER_16 = 16
    ORDER_64 = 64

class HypothesisStatus(str, Enum):
    HYPOTHESIS_UNVERIFIED = "hypothesis_unverified"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"

class FeatureValidity(str, Enum):
    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    UNRELIABLE = "unreliable"
    UNAVAILABLE = "unavailable"

@dataclass(frozen=True)
class ModulationEvidence:
    amplitude_score: float = 0.0
    phase_score: float = 0.0
    frequency_score: float = 0.0
    cumulant_score: float = 0.0
    spectral_score: float = 0.0
    periodicity_score: float = 0.0
    classical_model_score: float = 0.0
    ml_score: float = 0.0
    snr_quality: float = 1.0
    contradiction_penalty: float = 0.0
    supporting_evidence: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

@dataclass(frozen=True)
class ModulationHypothesis:
    family: ModulationFamily
    order: int | None
    score: float
    family_score: float
    order_score: float
    quality: str
    evidence: ModulationEvidence
    status: HypothesisStatus = HypothesisStatus.HYPOTHESIS_UNVERIFIED
    candidate_parameters: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)

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
class AmplitudeFeatures:
    mean: float
    rms: float
    variance: float
    coeff_var: float
    kurtosis: float
    skewness: float
    peak_to_rms: float
    norm_variance: float
    validity: FeatureValidity = FeatureValidity.VALID

@dataclass(frozen=True)
class PhaseFeatures:
    phase_inc_mean: float
    phase_inc_var: float
    phase_inc_kurtosis: float
    var_phase_sq: float     # Var(angle(x^2)) - collapses BPSK
    var_phase_4th: float    # Var(angle(x^4)) - collapses QPSK
    var_phase_8th: float    # Var(angle(x^8)) - collapses 8PSK
    valid_fraction: float
    validity: FeatureValidity = FeatureValidity.VALID

@dataclass(frozen=True)
class FrequencyFeatures:
    inst_freq_mean: float
    inst_freq_var: float
    inst_freq_median: float
    inst_freq_mad: float
    bimodal_separation: float | None
    bimodal_prominence: float
    state_occupancy_ratio: float
    validity: FeatureValidity = FeatureValidity.VALID

@dataclass(frozen=True)
class CumulantFeatures:
    c20: complex
    c21: float
    c40: complex
    c41: complex
    c42: float
    f20: float  # |C20| / C21
    f40: float  # |C40| / C21^2
    f41: float  # |C41| / C21^2
    f42: float  # |C42| / C21^2
    validity: FeatureValidity = FeatureValidity.VALID

@dataclass(frozen=True)
class SpectralFeatures:
    spectral_centroid: float
    spectral_spread: float
    spectral_kurtosis: float
    spectral_flatness: float
    spectral_asymmetry: float
    peak_count: int
    occupied_bandwidth: float | None
    validity: FeatureValidity = FeatureValidity.VALID

@dataclass(frozen=True)
class CyclostationaryFeatures:
    periodicity_score: float
    top_candidate_sps: float | None
    top_candidate_rate: float | None
    validity: FeatureValidity = FeatureValidity.VALID

@dataclass(frozen=True)
class ModulationFeatureVector:
    amplitude: AmplitudeFeatures
    phase: PhaseFeatures
    frequency: FrequencyFeatures
    cumulants: CumulantFeatures
    spectral: SpectralFeatures
    cyclostationary: CyclostationaryFeatures
    overall_validity: FeatureValidity = FeatureValidity.VALID

@dataclass(frozen=True)
class RawComplexPlaneDistribution:
    sample_subset_i: np.ndarray
    sample_subset_q: np.ndarray
    radii: np.ndarray
    phases: np.ndarray

@dataclass(frozen=True)
class ModulationAnalysisConfig:
    min_snr_db: float = 3.0
    min_samples: int = 128
    unknown_threshold: float = 0.45
    ambiguity_margin: float = 0.08
    window_count: int = 4
    enable_ml: bool = True
    random_seed: int = 42
    max_analysis_samples: int = 65536

@dataclass
class ModulationAnalysis:
    recording_reference: str
    signal_region: DetectedRegion | None
    hypotheses: list[ModulationHypothesis]
    selected_hypothesis: ModulationHypothesis | None
    feature_vector: ModulationFeatureVector
    raw_distribution: RawComplexPlaneDistribution | None
    window_consistency: float
    is_ambiguous: bool
    is_unknown: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
