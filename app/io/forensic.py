from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from app.models.signal import Endian, IQOrder
from .raw_iq import RawIQConfig, RawIQReader

@dataclass(frozen=True)
class FormatCandidate:
    dtype: str; iq_order: IQOrder; endian: Endian; score: float; evidence: tuple[str, ...]

def inspect_raw_iq(path: str | Path) -> list[FormatCandidate]:
    """Deterministic plausibility ranking, deliberately not format identification."""
    path = Path(path); size = path.stat().st_size; candidates: list[FormatCandidate] = []
    for dtype in ("complex64", "float32", "int16", "int8", "uint8"):
        for endian in (Endian.LITTLE, Endian.BIG):
            try: reader = RawIQReader(path, RawIQConfig(dtype, endian=endian))
            except Exception: continue
            chunk = reader.read_chunk(0, min(reader.sample_count, 16384)); finite = float(np.isfinite(chunk).mean())
            variance = float(np.var(chunk.real) + np.var(chunk.imag))
            score = 0.35 + 0.35 * finite + (0.20 if variance > 0 else 0.0)
            if dtype == "float32" and finite < 0.99: score -= 0.25
            evidence = (f"file-size compatible ({size} bytes)", f"finite complex samples: {finite:.3f}", f"combined I/Q variance: {variance:.4g}")
            for order in ((IQOrder.IQ,) if dtype == "complex64" else (IQOrder.IQ, IQOrder.QI)):
                candidates.append(FormatCandidate(dtype, order, endian, round(max(0.0, min(score, .99)), 3), evidence))
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.dtype, candidate.endian.value, candidate.iq_order.value))
