from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np
from .metadata import Diagnostic, MetadataValue

class SourceFormat(str, Enum): WAV = "wav"; RAW_IQ = "raw_iq"; SIGMF = "sigmf"
class IQOrder(str, Enum): IQ = "IQ"; QI = "QI"; NOT_APPLICABLE = "not_applicable"
class Endian(str, Enum): LITTLE = "little"; BIG = "big"; NOT_APPLICABLE = "not_applicable"

@dataclass
class SignalRecording:
    """Canonical samples are complex64; real streams have an all-zero imaginary part."""
    samples: np.ndarray
    source_format: SourceFormat
    original_dtype: str
    channels: int
    semantic_type: str
    iq_order: IQOrder = IQOrder.NOT_APPLICABLE
    endian: Endian = Endian.NOT_APPLICABLE
    sample_rate_hz: MetadataValue[float] = field(default_factory=lambda: MetadataValue.unknown("No sample-rate metadata."))
    center_frequency_hz: MetadataValue[float] = field(default_factory=lambda: MetadataValue.unknown("No center-frequency metadata."))
    bandwidth_hz: MetadataValue[float] = field(default_factory=lambda: MetadataValue.unknown("No bandwidth metadata."))
    timestamp: MetadataValue[str] = field(default_factory=lambda: MetadataValue.unknown("No timestamp metadata."))
    metadata: dict[str, MetadataValue[Any]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.samples.dtype != np.complex64:
            self.samples = self.samples.astype(np.complex64, copy=False)
