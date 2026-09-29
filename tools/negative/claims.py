"""Pure claim-ladder predicates; these do not modify production pipeline behaviour."""
from __future__ import annotations
from dataclasses import dataclass
from signal_analysis.models import HypothesisStatus, PipelineResult, PipelineStageStatus

UNKNOWN_THRESHOLD = .55
LEVELS = ("L1_modulation", "L2_receiver", "L3_fec", "L4_frame", "L5_confirmed_frame")
@dataclass(frozen=True)
class ClaimResult:
    levels: dict[str, bool]
    uncoded_success: bool
    detail: str

def evaluate_claims(result: PipelineResult, unknown_threshold: float = UNKNOWN_THRESHOLD) -> ClaimResult:
    top=result.top_hypothesis
    l1=bool(top and top.status is HypothesisStatus.HYPOTHESIS_UNVERIFIED and top.score >= unknown_threshold)
    demod=result.demod_result
    l2=bool(result.sync_status is PipelineStageStatus.COMPLETED and demod and demod.hypothesis_confirmed and demod.sync_result.acquisition_status == "LOCKED")
    fec=result.fec_result; uncoded=bool(fec and fec.decode_success and fec.codec_name.upper() == "UNCODED")
    l3=bool(fec and fec.decode_success and not uncoded)
    frame=result.frame_structure
    l4=bool(result.framing_status is PipelineStageStatus.COMPLETED and frame and frame.status is not HypothesisStatus.UNKNOWN)
    crc=frame.crc_candidate if frame else None
    l5=bool(l4 and frame.header_match.periodicity_consistent and crc and crc.verified and crc.polynomial_name != "CRC-8")
    return ClaimResult(dict(zip(LEVELS,(l1,l2,l3,l4,l5))),uncoded, "held-out corroboration unavailable in current pipeline")
