from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")

class MetadataSource(str, Enum):
    FILE_HEADER = "file_header"
    SIGMF_METADATA = "sigmf_metadata"
    USER_INPUT = "user_input"
    FILENAME = "filename"
    STRUCTURAL_INFERENCE = "structural_inference"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"

class MetadataStatus(str, Enum):
    KNOWN = "known"
    MEASURED = "measured"
    ESTIMATED = "estimated"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"

class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass(frozen=True)
class MetadataValue(Generic[T]):
    value: T | None
    source: MetadataSource
    status: MetadataStatus
    confidence: float
    evidence: str

    @classmethod
    def unknown(cls, evidence: str) -> "MetadataValue[None]":
        return cls(None, MetadataSource.UNKNOWN, MetadataStatus.MISSING, 0.0, evidence)

@dataclass(frozen=True)
class Diagnostic:
    severity: DiagnosticSeverity
    code: str
    message: str
    evidence: str = ""
