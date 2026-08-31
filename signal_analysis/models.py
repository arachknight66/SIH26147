from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Dict, List, Tuple
import numpy as np

class MetadataStatus(Enum):
    KNOWN = "KNOWN"
    ASSUMED = "ASSUMED"
    MISSING = "MISSING"

@dataclass(frozen=True)
class MetadataValue:
    value: Optional[float]
    source: str
    status: MetadataStatus
    confidence: float = 1.0
    evidence: str = ""

class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass(frozen=True)
class Diagnostic:
    severity: Severity
    code: str
    message: str
    evidence: str

class SourceFormat(Enum):
    RAW_IQ = "RAW_IQ"
    WAV = "WAV"
    SIGMF = "SIGMF"

@dataclass(frozen=True)
class SignalRecording:
    """
    Single in-memory representation of a signal recording.
    """
    samples: np.ndarray  # Always complex64
    source_format: SourceFormat
    original_dtype: str
    semantic_type: str  # e.g., "complex_iq", "mono_real", "stereo_real"
    sample_rate_hz: MetadataValue
    center_frequency_hz: MetadataValue
    provenance: Dict[str, Any]
    diagnostics: List[Diagnostic]

@dataclass(frozen=True)
class FormatCandidate:
    """Forensic triage for raw IQ format."""
    dtype: str
    iq_order: str
    endian: str
    score: float
    evidence: str

class FeatureValidity(Enum):
    VALID = "VALID"
    PARTIALLY_VALID = "PARTIALLY_VALID"
    UNRELIABLE = "UNRELIABLE"
    UNAVAILABLE = "UNAVAILABLE"

class HypothesisStatus(Enum):
    HYPOTHESIS_UNVERIFIED = "HYPOTHESIS_UNVERIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

@dataclass(frozen=True)
class CandidateParameters:
    symbol_rate: Optional[float]
    symbol_rate_unit: str  # "Hz" or "symbols/sample"
    samples_per_symbol: Optional[float]
    center_frequency_hz: Optional[float]
    bandwidth_hz: Optional[float]

@dataclass(frozen=True)
class ModulationHypothesis:
    label: str
    status: HypothesisStatus
    score: float
    quality_tier: str  # HIGH, MODERATE, LOW
    candidate_parameters: CandidateParameters
    evidence: Dict[str, float]
    contradictions: List[str]


@dataclass(frozen=True)
class SynchronizationResult:
    cfo_estimate: float
    cfo_unit: str  # "Hz" or "cycles/sample"
    timing_offset_fractional_symbols: float
    symbol_clock_locked: bool
    carrier_locked: bool
    lock_quality_metric: float
    evm_percent: float
    diagnostics: List[Diagnostic]

@dataclass(frozen=True)
class DemodulationResult:
    hard_bits: np.ndarray  # uint8
    soft_llrs: np.ndarray  # float32, positive means bit=1
    bits_per_symbol: int
    symbol_decisions: np.ndarray  # complex64
    sync_result: SynchronizationResult
    source_hypothesis_label: str
    hypothesis_confirmed: bool


class DeinterleaverFamily(Enum):
    BLOCK = "BLOCK"
    CONVOLUTIONAL = "CONVOLUTIONAL"
    DIAGONAL = "DIAGONAL"
    PSEUDO_RANDOM = "PSEUDO_RANDOM"
    NONE = "NONE"

@dataclass(frozen=True)
class DeinterleaverHypothesis:
    family: DeinterleaverFamily
    parameters: Dict[str, Any]
    score: float
    falsification_evidence: List[str]
    status: HypothesisStatus

@dataclass(frozen=True)
class DeinterleavingResult:
    bits: np.ndarray  # uint8
    llrs_reordered: np.ndarray  # float32
    hypothesis: DeinterleaverHypothesis
    cross_validation_score: float

@dataclass(frozen=True)
class FECDecodeResult:
    decoded_bits: np.ndarray
    corrected_bit_count: int
    corrected_bit_fraction: float
    decode_success: bool
    codec_name: str
    pre_correction_metric: float  # e.g. syndrome weight or path metric margin
    diagnostics: List[Diagnostic]

class PipelineStageStatus(Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    SKIPPED = "SKIPPED"

@dataclass(frozen=True)
class SyncWordPattern:
    name: str
    bit_pattern: np.ndarray  # uint8 array
    description: str
    source: str
    reference: str

@dataclass(frozen=True)
class HeaderMatch:
    pattern: SyncWordPattern
    bit_offset: int
    hamming_distance: int
    match_confidence: float
    periodicity_consistent: bool

@dataclass(frozen=True)
class CRCMatch:
    polynomial_hex: str
    polynomial_name: str
    bit_range_checked: Tuple[int, int]
    verified: bool

@dataclass(frozen=True)
class FrameStructure:
    header_match: HeaderMatch
    header_length_bits: int
    payload_start_bit: int
    payload_length_bits: Optional[int]
    crc_candidate: Optional[CRCMatch]
    status: HypothesisStatus

@dataclass(frozen=True)
class PipelineResult:
    recording: SignalRecording
    # Stage 2: Hypothesis
    hypothesis_status: PipelineStageStatus
    top_hypothesis: Optional[ModulationHypothesis]
    all_hypotheses: List[ModulationHypothesis]
    # Stage 3: Sync & Demod
    sync_status: PipelineStageStatus
    demod_result: Optional[DemodulationResult]
    # Stage 4: Deinterleave & FEC
    fec_status: PipelineStageStatus
    deint_result: Optional[DeinterleavingResult]
    fec_result: Optional[FECDecodeResult]
    # Stage 5: Framing
    framing_status: PipelineStageStatus
    frame_structure: Optional[FrameStructure]
    diagnostics: List[Diagnostic] = field(default_factory=list)
