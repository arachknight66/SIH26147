from __future__ import annotations
import json
from pathlib import Path
from app.exceptions import CorruptSigMFMetadataError
from app.models.metadata import MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import SourceFormat
from .raw_iq import RawIQConfig, RawIQReader

_SIGMF_TYPES = {"cf32_le": "complex64", "ci16_le": "int16", "ci8": "int8", "cu8": "uint8"}
def load_sigmf(meta_path: str | Path):
    meta_path = Path(meta_path)
    try: meta = json.loads(meta_path.read_text(encoding="utf-8")); global_ = meta["global"]; datatype = global_["core:datatype"]
    except (OSError, KeyError, json.JSONDecodeError) as exc: raise CorruptSigMFMetadataError(f"Invalid SigMF metadata: {exc}") from exc
    if datatype not in _SIGMF_TYPES: raise CorruptSigMFMetadataError(f"Unsupported SigMF datatype '{datatype}'.")
    data_path = meta_path.with_suffix(".sigmf-data")
    if not data_path.exists(): raise CorruptSigMFMetadataError(f"SigMF data file is missing: {data_path}")
    reader = RawIQReader(data_path, RawIQConfig(_SIGMF_TYPES[datatype], sample_rate_hz=global_.get("core:sample_rate"), center_frequency_hz=(meta.get("captures") or [{}])[0].get("core:frequency")))
    recording = reader.read(); recording.source_format = SourceFormat.SIGMF; recording.provenance["loader"] = "SigMF adapter"; recording.provenance["sigmf_metadata_path"] = str(meta_path)
    return recording
