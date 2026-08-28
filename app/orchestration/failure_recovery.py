from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class FailureCategory(str, Enum):
    NONE = "none"
    INPUT_FAILURE = "input_failure"
    FORMAT_AMBIGUITY = "format_ambiguity"
    SIGNAL_NOT_DETECTED = "signal_not_detected"
    LOW_SNR = "low_snr"
    MODULATION_AMBIGUITY = "modulation_ambiguity"
    SYNCHRONIZATION_FAILURE = "synchronization_failure"
    DEMODULATION_FAILURE = "demodulation_failure"
    BIT_ALIGNMENT_FAILURE = "bit_alignment_failure"
    FRAME_FAILURE = "frame_failure"
    FEC_FAILURE = "fec_failure"
    CRC_FAILURE = "crc_failure"
    VERIFICATION_FAILURE = "verification_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    UNKNOWN_ERROR = "unknown_error"

@dataclass(frozen=True)
class PipelineFailure:
    category: FailureCategory
    stage_name: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation_suggestion: str | None = None
    is_recoverable: bool = False

def classify_stage_failure(stage_name: str, error: Exception) -> PipelineFailure:
    msg = str(error)
    msg_lower = msg.lower()

    if "timeout" in msg_lower or isinstance(error, TimeoutError) or "time limit" in msg_lower:
        return PipelineFailure(
            category=FailureCategory.TIMEOUT,
            stage_name=stage_name,
            message=msg,
            remediation_suggestion="Increase resource limit timeout or downsample input recording.",
        )
    elif "memory" in msg_lower or "resource" in msg_lower or isinstance(error, MemoryError):
        return PipelineFailure(
            category=FailureCategory.RESOURCE_LIMIT,
            stage_name=stage_name,
            message=msg,
            remediation_suggestion="Use chunked memory-mapped ingestion or reduce maximum sample limits.",
        )
    elif "snr" in msg_lower or "noise" in msg_lower:
        return PipelineFailure(
            category=FailureCategory.LOW_SNR,
            stage_name=stage_name,
            message=msg,
            remediation_suggestion="Signal energy is near or below the noise floor; recovery cannot proceed reliably.",
        )
    elif "sync" in msg_lower or "carrier" in msg_lower or "timing" in msg_lower or "costas" in msg_lower:
        return PipelineFailure(
            category=FailureCategory.SYNCHRONIZATION_FAILURE,
            stage_name=stage_name,
            message=msg,
            remediation_suggestion="Carrier frequency or timing offset could not be locked. Check modulation order or CFO search range.",
        )
    elif "crc" in msg_lower or "checksum" in msg_lower or "parity" in msg_lower:
        return PipelineFailure(
            category=FailureCategory.CRC_FAILURE,
            stage_name=stage_name,
            message=msg,
            remediation_suggestion="CRC parity mismatch. Check bit order, polynomial selection, or FEC decoding.",
        )
    elif "frame" in msg_lower or "preamble" in msg_lower:
        return PipelineFailure(
            category=FailureCategory.FRAME_FAILURE,
            stage_name=stage_name,
            message=msg,
            remediation_suggestion="Preamble sync word not detected or frame intervals are non-stationary.",
        )
    elif "format" in msg_lower or "header" in msg_lower or "endian" in msg_lower:
        return PipelineFailure(
            category=FailureCategory.FORMAT_AMBIGUITY,
            stage_name=stage_name,
            message=msg,
            remediation_suggestion="Specify raw IQ format, endianness, or sample rate explicitly.",
        )
    else:
        return PipelineFailure(
            category=FailureCategory.UNKNOWN_ERROR,
            stage_name=stage_name,
            message=msg,
            remediation_suggestion="Inspect system logs and stack trace.",
        )
