from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity

class VerificationStatus(str, Enum):
    INDEPENDENTLY_VERIFIED = "independently_verified"
    STRONGLY_SUPPORTED = "strongly_supported"
    PARTIALLY_VERIFIED = "partially_verified"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FALSIFIED = "falsified"
    REJECTED = "rejected"

class VerificationQualityLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    STRONGLY_SUPPORTED = "strongly_supported"
    WEAKLY_SUPPORTED = "weakly_supported"
    NEUTRAL = "neutral"
    AMBIGUOUS = "ambiguous"
    CONTRADICTED = "contradicted"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"

class IndependenceLevel(str, Enum):
    DIRECT = "direct"
    DERIVED = "derived"
    REUSED = "reused"
    INDEPENDENT = "independent"

class AuditResultStatus(str, Enum):
    PASS = "PASS"
    WEAK_PASS = "WEAK_PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    SKIPPED = "SKIPPED"

# Tell pytest not to collect
AuditResultStatus.__test__ = False  # type: ignore[attr-defined]

# Alias for backwards compatibility
TestResultStatus = AuditResultStatus
VerificationTestStatus = AuditResultStatus

class FalsificationOutcome(str, Enum):
    NOT_FALSIFIED = "not_falsified"
    PARTIALLY_FALSIFIED = "partially_falsified"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"

@dataclass(frozen=True)
class VerificationTest:
    test_id: str
    name: str
    category: str
    description: str
    status: AuditResultStatus
    score: float
    p_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    counter_evidence: str | None = None
    is_critical: bool = False

VerificationTest.__test__ = False  # type: ignore[attr-defined]

@dataclass(frozen=True)
class VerificationClaim:
    claim_id: int
    claim_text: str
    status: ClaimStatus
    evidence_category: str
    tests: tuple[VerificationTest, ...]
    confidence: float
    independence_level: IndependenceLevel
    counter_evidence: str | None = None
    limitations: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

VerificationClaim.__test__ = False  # type: ignore[attr-defined]

@dataclass(frozen=True)
class PhysicalAuditResult:
    is_finite: bool
    rms_power: float
    clipping_fraction: float
    dc_offset_magnitude: float
    estimated_snr_db: float
    occupied_bandwidth_hz: float
    measurement_consistent: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ModulationAuditResult:
    modulation_name: str
    evm_percent: float
    cluster_variance: float
    mth_power_concentration: float
    decision_margin: float
    runner_up_name: str | None
    runner_up_margin: float
    is_consistent: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SyncAuditResult:
    residual_cfo_hz: float
    phase_variance_rad2: float
    ted_variance: float
    window_count: int
    passed_window_count: int
    window_consistency_fraction: float
    is_stable: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BitstreamAuditResult:
    bit_balance: float
    transition_probability: float
    byte_entropy: float
    selected_offset_score: float
    alternative_offsets_score: float
    is_alignment_unique: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FrameAuditResult:
    preamble_name: str
    frame_length_bits: int
    total_frames: int
    interval_mean: float
    interval_std: float
    interval_cv: float
    sequence_is_continuous: bool
    missing_sequences: tuple[int, ...]
    boundary_perturbation_passed: bool
    is_structurally_sound: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FECAuditResult:
    code_name: str
    ber_before: float
    ber_after: float
    information_gain: float
    correction_fraction: float
    anti_overcorrection_passed: bool
    held_out_validation_passed: bool
    is_beneficial: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class IntegrityAuditResult:
    crc_name: str
    selection_frames_count: int
    selection_valid_count: int
    validation_frames_count: int
    validation_valid_count: int
    validation_success_rate: float
    raw_p_value: float
    multiple_testing_corrected_p_value: float
    is_statistically_significant: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ScramblerAuditResult:
    polynomial_name: str
    is_reproducible: bool
    improves_framing: bool
    improves_integrity: bool
    is_verified: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class InterleaverAuditResult:
    interleaver_type: str
    parameter_perturbation_passed: bool
    held_out_validation_passed: bool
    improves_framing: bool
    improves_integrity: bool
    is_verified: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RobustnessAuditResult:
    bit_flip_tolerance_score: float
    burst_error_tolerance_score: float
    boundary_perturbation_score: float
    leave_one_out_stable: bool
    high_leverage_frame_detected: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FalsificationAuditResult:
    total_falsification_tests: int
    falsified_test_count: int
    critical_failure_count: int
    major_contradictions: tuple[str, ...]
    outcome: FalsificationOutcome

@dataclass(frozen=True)
class ErrorBudget:
    carrier_uncertainty: float
    timing_uncertainty: float
    bit_error_rate_proxy: float
    fec_residual_uncertainty: float
    total_composite_uncertainty: float
    summary: str

@dataclass(frozen=True)
class VerificationConfig:
    min_window_consistency_fraction: float = 0.80
    max_allowable_correction_fraction: float = 0.10
    max_allowable_interval_cv: float = 0.05
    held_out_split_ratio: float = 0.30
    bootstrap_iterations: int = 50
    multiple_testing_alpha: float = 0.01
    random_seed: int = 42
    strict_falsification: bool = True

@dataclass(frozen=True)
class VerificationHandoff:
    is_verified: bool
    verified_payload: bytes
    status: VerificationStatus
    quality_level: VerificationQualityLevel
    claims_summary: dict[str, str]
    error_budget: ErrorBudget
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    reproducibility_hash: str

@dataclass(frozen=True)
class VerificationAnalysis:
    status: VerificationStatus
    quality_level: VerificationQualityLevel
    is_verified: bool
    is_falsified: bool
    is_ambiguous: bool
    claims: list[VerificationClaim]
    physical_audit: PhysicalAuditResult | None
    modulation_audit: ModulationAuditResult | None
    sync_audit: SyncAuditResult | None
    bitstream_audit: BitstreamAuditResult | None
    frame_audit: FrameAuditResult | None
    fec_audit: FECAuditResult | None
    integrity_audit: IntegrityAuditResult | None
    scrambler_audit: ScramblerAuditResult | None
    robustness_audit: RobustnessAuditResult | None
    falsification_audit: FalsificationAuditResult | None
    error_budget: ErrorBudget | None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    handoff: VerificationHandoff | None = None
