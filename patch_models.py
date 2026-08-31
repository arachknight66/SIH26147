import re

with open('signal_analysis/models.py', 'r') as f:
    content = f.read()

new_classes = """

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
"""

content += new_classes

with open('signal_analysis/models.py', 'w') as f:
    f.write(content)
