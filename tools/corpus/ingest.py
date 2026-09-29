"""Hash-checked corpus ingestion using only production import adapters."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from signal_analysis.loaders import RawIQConfig
from signal_analysis.sources import open_raw_iq_source, open_sigmf_source, open_wav_source
from signal_analysis.workflow import AnalysisRequest, load_recording

class IngestionError(ValueError): pass

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()

def _path(record: dict, root: Path) -> Path:
    path = (root / record["path"]).resolve()
    if not path.is_file(): raise IngestionError(f"{record['capture_id']}: capture unavailable at {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]: raise IngestionError(f"{record['capture_id']}: sha256 mismatch (refusing import)")
    return path

def raw_config(record: dict) -> RawIQConfig:
    config = record["import_config"]
    required = {"dtype", "iq_order", "endian"}
    if not required <= config.keys(): raise IngestionError("RAW_IQ import_config requires dtype, iq_order, and endian")
    rate = record["sample_rate_hz"]
    center = record["center_frequency_hz"]
    return RawIQConfig(config["dtype"], config["iq_order"], config["endian"], rate["value"], center["value"])

def production_request(record: dict, root: Path) -> AnalysisRequest:
    path = _path(record, root)
    fmt = record["format"]
    if fmt == "RAW_IQ": return AnalysisRequest(path, raw_iq_config=raw_config(record), origin="corpus")
    if fmt == "WAV":
        mode = record["import_config"].get("stereo_mode")
        if not mode: raise IngestionError("WAV import_config requires explicit stereo_mode")
        return AnalysisRequest(path, wav_stereo_mode=mode, origin="corpus")
    if fmt == "SIGMF": return AnalysisRequest(path, origin="corpus")
    raise IngestionError(f"unsupported corpus format: {fmt}")

def load_capture(record: dict, root: Path):
    """Load via workflow.load_recording. Truth is intentionally never accepted here."""
    return load_recording(production_request(record, root))

def open_capture_source(record: dict, root: Path):
    """Open via bounded production RecordingSource APIs for source-level checks."""
    path = _path(record, root)
    if record["format"] == "RAW_IQ": return open_raw_iq_source(path, raw_config(record))
    if record["format"] == "WAV": return open_wav_source(path, record["import_config"].get("stereo_mode"))
    if record["format"] == "SIGMF": return open_sigmf_source(path)
    raise IngestionError(f"unsupported corpus format: {record['format']}")
