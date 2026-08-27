from __future__ import annotations
from pathlib import Path
from app.exceptions import UnsupportedFileFormatError
from app.models.signal import Endian, IQOrder, SignalRecording
from .raw_iq import RawIQConfig, RawIQReader
from .sigmf import load_sigmf
from .wav import WavReader

def load_signal(path: str | Path, *, raw_config: RawIQConfig | None = None, wav_mode: str = "unresolved") -> SignalRecording:
    path = Path(path); suffix = path.suffix.lower()
    if suffix == ".wav": return WavReader(path, wav_mode).read()
    if suffix == ".sigmf-meta": return load_sigmf(path)
    if raw_config is not None: return RawIQReader(path, raw_config).read()
    raise UnsupportedFileFormatError("Cannot identify a metadata-free raw IQ file without an explicit RawIQConfig. Provide dtype, I/Q order, and endian; use format forensics only to rank plausible candidates.")
