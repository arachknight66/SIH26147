from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity

class DataRecoveryStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    INSUFFICIENT_DATA = "insufficient_data"
    BITSTREAM_UNRESOLVED = "bitstream_unresolved"
    FRAME_STRUCTURE_UNRESOLVED = "frame_structure_unresolved"
    FEC_UNRESOLVED = "fec_unresolved"
    CRC_UNRESOLVED = "crc_unresolved"
    AMBIGUOUS = "ambiguous"
    STRUCTURALLY_SUPPORTED = "structurally_supported"
    CORRECTED = "corrected"
    INTEGRITY_SUPPORTED = "integrity_supported"
    RECOVERY_FAILED = "recovery_failed"

class DataQualityLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class EpistemicStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    CORRECTED = "corrected"
    ASSUMED = "assumed"
    VERIFIED = "verified"
    UNKNOWN = "unknown"

class BitOrder(str, Enum):
    MSB_FIRST = "msb_first"
    LSB_FIRST = "lsb_first"
    UNKNOWN = "unknown"

class BitPolarity(str, Enum):
    NORMAL = "normal"
    INVERTED = "inverted"
    UNRESOLVED = "unresolved"

class LineCodeType(str, Enum):
    NONE = "none"
    NRZ = "nrz"
    NRZI_TRANS_0 = "nrzi_transition_on_0"
    NRZI_TRANS_1 = "nrzi_transition_on_1"
    MANCHESTER = "manchester"
    DIFF_MANCHESTER = "differential_manchester"

class FECCodeFamily(str, Enum):
    NONE = "none"
    REPETITION = "repetition"
    PARITY = "parity"
    HAMMING = "hamming"
    CONVOLUTIONAL = "convolutional"

class ScramblerType(str, Enum):
    NONE = "none"
    LFSR_SYNCHRONOUS = "lfsr_synchronous"
    SELF_SYNCHRONIZING = "self_synchronizing"
    XOR_PERIODIC = "xor_periodic"

@dataclass(frozen=True)
class BitStream:
    hard_bits: np.ndarray               # 1D uint8 array
    soft_bits: np.ndarray | None        # 1D float32 LLR array if available
    symbol_indices: np.ndarray | None   # 1D int32 array
    bit_order: BitOrder = BitOrder.UNKNOWN
    bit_polarity: BitPolarity = BitPolarity.UNRESOLVED
    bit_offset: int = 0
    source_candidate: str = "unknown"
    sample_indices: np.ndarray | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.hard_bits)

@dataclass(frozen=True)
class BitHypothesis:
    hypothesis_id: int
    bitstream: BitStream
    phase_rotation_deg: float
    polarity: BitPolarity
    line_code: LineCodeType
    bit_order: BitOrder
    bit_offset: int
    score: float = 0.0
    epistemic_status: EpistemicStatus = EpistemicStatus.INFERRED

@dataclass(frozen=True)
class ByteStreamCandidate:
    bytes_data: bytes
    bit_offset: int
    bit_order: BitOrder
    bit_count: int
    entropy: float
    bit_balance: float
    printable_ratio: float
    provenance: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PreambleCandidate:
    pattern_bits: np.ndarray            # 1D uint8
    pattern_hex: str
    length_bits: int
    match_indices: tuple[int, ...]
    match_count: int
    mean_spacing: float
    spacing_variance: float
    hamming_distance_dist: tuple[float, ...]
    is_periodic: bool
    confidence: float
    valid: bool = True

@dataclass(frozen=True)
class FrameBoundary:
    frame_index: int
    start_bit: int
    end_bit: int
    length_bits: int
    preamble_match: bool
    is_valid_interval: bool = True

@dataclass(frozen=True)
class FrameCandidate:
    frame_index: int
    raw_bits: np.ndarray
    header_bits: np.ndarray
    payload_bits: np.ndarray
    crc_bits: np.ndarray
    fec_bits: np.ndarray
    start_bit: int
    end_bit: int
    is_length_consistent: bool
    is_crc_valid: bool
    is_fec_corrected: bool
    sequence_number: int | None = None
    length_field_value: int | None = None
    decoded_payload: bytes | None = None

@dataclass(frozen=True)
class LineCodeHypothesis:
    code_type: LineCodeType
    convention: str
    transition_density: float
    run_length_score: float
    clock_consistency: float
    confidence: float
    valid: bool = True

@dataclass(frozen=True)
class ScramblerHypothesis:
    scrambler_type: ScramblerType
    polynomial_name: str
    polynomial_bits: tuple[int, ...]
    initial_state: tuple[int, ...]
    period: int
    linear_complexity: int
    entropy_improvement: float
    crc_improvement: float
    confidence: float
    valid: bool = True

@dataclass(frozen=True)
class FECHypothesis:
    code_family: FECCodeFamily
    code_name: str
    rate: float
    constraint_length: int | None
    generator_polynomials: tuple[int, ...]
    block_size: int | None
    assumptions: tuple[str, ...] = ()
    confidence: float = 0.0
    valid: bool = True

@dataclass(frozen=True)
class FECDecodeResult:
    input_bits: np.ndarray              # 1D uint8 original input bits
    decoded_bits: np.ndarray            # 1D uint8 error-corrected decoded bits
    correction_mask: np.ndarray         # 1D bool mask where True indicates corrected bit
    corrected_bit_count: int
    correction_fraction: float
    path_metric: float
    normalized_path_metric: float
    is_overcorrected: bool              # True if correction_fraction > max budget
    code_family: FECCodeFamily
    valid: bool = True

@dataclass(frozen=True)
class CRCResult:
    crc_name: str
    width: int
    polynomial: int
    init_value: int
    xor_out: int
    reflect_in: bool
    reflect_out: bool
    calculated_crc: int
    expected_crc: int
    is_valid: bool
    false_positive_p_value: float = 0.0

@dataclass(frozen=True)
class IntegrityResult:
    crc_results: tuple[CRCResult, ...]
    valid_frame_count: int
    total_frame_count: int
    crc_valid_fraction: float
    multi_frame_p_value: float
    before_fec_valid_count: int
    after_fec_valid_count: int
    valid: bool = True

@dataclass(frozen=True)
class CorrectionQuality:
    input_error_estimate: float
    corrected_bits_total: int
    mean_corrected_bits_per_frame: float
    median_correction_fraction: float
    decoder_metric: float
    crc_before_fec_fraction: float
    crc_after_fec_fraction: float
    structural_consistency_score: float

@dataclass(frozen=True)
class ReconstructionCandidate:
    candidate_id: int
    bit_hypothesis: BitHypothesis
    preamble: PreambleCandidate | None
    frames: tuple[FrameCandidate, ...]
    line_code: LineCodeHypothesis | None
    scrambler: ScramblerHypothesis | None
    fec: FECHypothesis | None
    fec_decode: FECDecodeResult | None
    integrity: IntegrityResult | None
    correction_quality: CorrectionQuality | None
    recovered_payload_bytes: bytes
    composite_score: float              # S_recon in [0.0, 1.0]
    complexity_penalty: float           # Occam's razor penalty
    data_quality_level: DataQualityLevel
    epistemic_status: EpistemicStatus
    diagnostics: list[Diagnostic] = field(default_factory=list)

@dataclass(frozen=True)
class Phase6Handoff:
    raw_bits: np.ndarray
    corrected_bits: np.ndarray
    payload_bytes: bytes
    frame_boundaries: tuple[FrameBoundary, ...]
    fec_parameters: dict[str, Any]
    scrambler_parameters: dict[str, Any]
    crc_parameters: dict[str, Any]
    correction_masks: tuple[np.ndarray, ...]
    structural_evidence: dict[str, Any]
    candidate_ranking_provenance: dict[str, Any]
    assumptions: tuple[str, ...]
    uncertainties: tuple[str, ...]

@dataclass(frozen=True)
class DataRecoveryConfig:
    max_bit_hypotheses: int = 8
    max_frame_hypotheses: int = 5
    max_crc_candidates: int = 16
    max_fec_candidates: int = 8
    max_scrambler_candidates: int = 8
    max_frames_to_analyze: int = 100
    max_correction_fraction: float = 0.10   # Max 10% bit alteration allowed
    min_frames_for_periodicity: int = 3
    complexity_weight: float = 0.15
    random_seed: int = 42
    max_runtime_seconds: float = 10.0

@dataclass(frozen=True)
class DataRecoveryAnalysis:
    recording_reference: str
    bitstream_candidates: list[BitHypothesis]
    reconstruction_candidates: list[ReconstructionCandidate]
    selected_candidate: ReconstructionCandidate | None
    status: DataRecoveryStatus
    quality_level: DataQualityLevel
    is_recovered: bool
    is_inconclusive: bool
    is_ambiguous: bool
    failure_reason: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    phase6_handoff: Phase6Handoff | None = None
