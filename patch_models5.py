import re

with open('signal_analysis/models.py', 'r') as f:
    content = f.read()

new_classes = """
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
"""

# Need to ensure Tuple is imported. 
# It is imported as `from typing import Optional, Any, Dict, List` -> add Tuple
content = content.replace("from typing import Optional, Any, Dict, List", "from typing import Optional, Any, Dict, List, Tuple")
content += new_classes

with open('signal_analysis/models.py', 'w') as f:
    f.write(content)
