import os
import json
import wave
import struct
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from .models import (
    SignalRecording, MetadataValue, MetadataStatus, Diagnostic, Severity,
    SourceFormat, FormatCandidate
)

@dataclass(frozen=True)
class RawIQConfig:
    dtype: str  # 'complex64', 'float32', 'int8', 'int16', 'uint8'
    iq_order: str  # 'IQ' or 'QI'
    endian: str  # 'little' or 'big'
    sample_rate_hz: Optional[float] = None
    center_frequency_hz: Optional[float] = None

def _get_numpy_dtype(config: RawIQConfig) -> np.dtype:
    """Map RawIQConfig to numpy dtype."""
    endian_char = '<' if config.endian == 'little' else '>'
    
    if config.dtype == 'complex64':
        return np.dtype(f"{endian_char}c8")
    elif config.dtype == 'float32':
        return np.dtype(f"{endian_char}f4")
    elif config.dtype == 'int16':
        return np.dtype(f"{endian_char}i2")
    elif config.dtype == 'int8':
        return np.dtype(f"{endian_char}i1")
    elif config.dtype == 'uint8':
        return np.dtype(f"{endian_char}u1")
    else:
        raise ValueError(f"Unsupported dtype: {config.dtype}")

def _read_raw_iq_chunk_from_file(path: str, config: RawIQConfig, start: int = 0, count: int = -1) -> np.ndarray:
    """Read a chunk of raw IQ data."""
    dt = _get_numpy_dtype(config)
    scalar_size = dt.itemsize
    
    # complex64 is already a pair. other dtypes represent scalars, so a pair is 2 * scalar_size
    pair_size = scalar_size if config.dtype == 'complex64' else 2 * scalar_size
    
    file_size = os.path.getsize(path)
    if file_size % pair_size != 0:
        raise ValueError(f"File size {file_size} is not a multiple of the IQ pair size {pair_size} bytes. "
                         "Likely wrong dtype or truncated capture.")
    
    total_pairs = file_size // pair_size
    
    if count == -1:
        count = total_pairs - start
    
    if start + count > total_pairs:
        count = total_pairs - start
        
    if count <= 0:
        return np.array([], dtype=np.complex64)
        
    offset = start * pair_size
    
    if config.dtype == 'complex64':
        # Direct memmap
        data = np.memmap(path, dtype=dt, mode='r', offset=offset, shape=(count,))
        if config.iq_order == 'QI':
            # Swap real and imaginary
            data = data.imag + 1j * data.real
    else:
        # Interleaved scalars
        data = np.memmap(path, dtype=dt, mode='r', offset=offset, shape=(count * 2,))
        
        # Convert to complex64
        # We don't scale it. "Loaders convert datatype only; the samples array must be numerically traceable"
        data_f32 = data.astype(np.float32)
        
        if config.iq_order == 'IQ':
            data = data_f32[0::2] + 1j * data_f32[1::2]
        else:
            data = data_f32[1::2] + 1j * data_f32[0::2]
            
    return np.array(data, dtype=np.complex64)

class RawIQReader:
    def __init__(self, path: str, config: RawIQConfig):
        self.path = path
        self.config = config
        
    def read_chunk(self, start: int, count: int) -> np.ndarray:
        return _read_raw_iq_chunk_from_file(self.path, self.config, start, count)
        
    def read(self) -> SignalRecording:
        samples = self.read_chunk(0, -1)
        
        # Metadata handling
        if self.config.sample_rate_hz is not None:
            sr = MetadataValue(self.config.sample_rate_hz, "user_input", MetadataStatus.KNOWN)
        else:
            sr = MetadataValue(None, "user_input", MetadataStatus.MISSING)
            
        if self.config.center_frequency_hz is not None:
            cf = MetadataValue(self.config.center_frequency_hz, "user_input", MetadataStatus.KNOWN)
        else:
            cf = MetadataValue(None, "user_input", MetadataStatus.MISSING)
            
        provenance = {
            "source_path": self.path,
            "file_size_bytes": os.path.getsize(self.path),
            "loader": "RawIQReader",
            "conversion_description": f"interleaved {self.config.dtype} scalar pairs ({self.config.iq_order}, {self.config.endian}-endian) -> complex64, no scaling"
        }
        
        return SignalRecording(
            samples=samples,
            source_format=SourceFormat.RAW_IQ,
            original_dtype=self.config.dtype,
            semantic_type="complex_iq",
            sample_rate_hz=sr,
            center_frequency_hz=cf,
            provenance=provenance,
            diagnostics=[]
        )

def inspect_raw_iq(path: str) -> List[FormatCandidate]:
    """Deterministic plausibility ranker for raw IQ format."""
    file_size = os.path.getsize(path)
    candidates = []
    
    dtypes = ['complex64', 'float32', 'int16', 'int8', 'uint8']
    endians = ['little', 'big']
    orders = ['IQ', 'QI']
    
    for dtype in dtypes:
        for endian in endians:
            dt = _get_numpy_dtype(RawIQConfig(dtype, 'IQ', endian))
            scalar_size = dt.itemsize
            pair_size = scalar_size if dtype == 'complex64' else 2 * scalar_size
            
            if file_size % pair_size != 0:
                continue # Impossible size for this dtype
                
            try:
                # Read a small chunk to test
                chunk = _read_raw_iq_chunk_from_file(
                    path, RawIQConfig(dtype, 'IQ', endian), start=0, count=min(1000, file_size // pair_size)
                )
                
                # Check for NaNs/Infs
                finite_ratio = float(np.isfinite(chunk).mean())
                if finite_ratio < 0.5:
                    continue # likely wrong type (e.g. interpreting int as float)
                    
                # Calculate score
                # This is just forensic triage. 
                var_i = float(np.var(chunk.real))
                var_q = float(np.var(chunk.imag))
                
                var_ratio = min(var_i, var_q) / max(var_i, var_q) if max(var_i, var_q) > 0 else 0.0
                
                score = finite_ratio * 10 + var_ratio
                
                for order in orders:
                    candidates.append(FormatCandidate(
                        dtype=dtype,
                        iq_order=order,
                        endian=endian,
                        score=float(score),
                        evidence=f"File size exact multiple. Finite ratio {finite_ratio:.2f}. I/Q var ratio {var_ratio:.2f}"
                    ))
            except Exception:
                pass
                
    # Sort descending by score
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates

class WavReader:
    """
    WAV Loader
    path: path to wav
    mode: "unresolved"|"mono_real"|"stereo_real"|"stereo_iq"
    """
    def __init__(self, path: str, mode: str = "unresolved"):
        self.path = path
        self.mode = mode
        
    def read(self) -> SignalRecording:
        diagnostics = []
        with wave.open(self.path, 'rb') as w:
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            n_frames = w.getnframes()
            
            raw_data = w.readframes(n_frames)
            
        if sampwidth == 1:
            dtype = np.uint8
            original_dtype = "uint8"
        elif sampwidth == 2:
            dtype = np.int16
            original_dtype = "int16"
        elif sampwidth == 3:
            # 24-bit PCM: convert to 32-bit int
            original_dtype = "int24"
            padded = np.zeros(n_frames * n_channels * 4, dtype=np.uint8)
            raw_np = np.frombuffer(raw_data, dtype=np.uint8)
            # Assuming little endian for WAV PCM
            padded[0::4] = raw_np[0::3]
            padded[1::4] = raw_np[1::3]
            padded[2::4] = raw_np[2::3]
            padded[3::4] = np.where(padded[2::4] >= 128, 255, 0)
            data = padded.view(np.int32)
            dtype = np.int32
        elif sampwidth == 4:
            dtype = np.int32
            original_dtype = "int32"
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
            
        if sampwidth != 3:
            data = np.frombuffer(raw_data, dtype=dtype)
            
        if n_channels == 1:
            semantic_type = "mono_real"
            samples = data.astype(np.complex64) # Imaginary part is 0
            conversion_description = f"mono {original_dtype} -> complex64 (real only)"
        elif n_channels == 2:
            data = data.reshape(-1, 2)
            if self.mode == "stereo_iq":
                semantic_type = "complex_iq"
                samples = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
                conversion_description = f"stereo {original_dtype} (ch0=I, ch1=Q) -> complex64"
            else:
                semantic_type = "stereo_real"
                samples = data.astype(np.complex64) 
                conversion_description = f"stereo {original_dtype} -> complex64 (real only, 2 channels)"
                
                if self.mode == "unresolved":
                    diagnostics.append(Diagnostic(
                        severity=Severity.WARNING,
                        code="UNRESOLVED_STEREO",
                        message="Stereo channels were NOT assumed to be I/Q. Pass mode='stereo_iq' to interpret as I/Q.",
                        evidence=f"{n_channels} channels found in WAV"
                    ))
        else:
            raise ValueError(f"Unsupported number of channels: {n_channels}")

        sr = MetadataValue(float(framerate), "wav_header", MetadataStatus.KNOWN)
        cf = MetadataValue(None, "wav_header", MetadataStatus.MISSING)
        
        provenance = {
            "source_path": self.path,
            "file_size_bytes": os.path.getsize(self.path),
            "loader": "WavReader",
            "conversion_description": conversion_description
        }
        
        return SignalRecording(
            samples=samples,
            source_format=SourceFormat.WAV,
            original_dtype=original_dtype,
            semantic_type=semantic_type,
            sample_rate_hz=sr,
            center_frequency_hz=cf,
            provenance=provenance,
            diagnostics=diagnostics
        )

def read_sigmf(meta_path: str) -> SignalRecording:
    """
    Parse .sigmf-meta JSON and load corresponding .sigmf-data.
    """
    if not meta_path.endswith('.sigmf-meta'):
        raise ValueError("SigMF meta path must end with .sigmf-meta")
        
    data_path = meta_path.replace('.sigmf-meta', '.sigmf-data')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing companion data file: {data_path}")
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    global_meta = meta.get("global", {})
    datatype = global_meta.get("core:datatype", "")
    sample_rate = global_meta.get("core:sample_rate")
    
    captures = meta.get("captures", [])
    center_frequency = None
    if captures:
        center_frequency = captures[0].get("core:frequency")
        
    if datatype == "cf32_le":
        dtype, endian = "float32", "little"
    elif datatype == "cf32_be":
        dtype, endian = "float32", "big"
    elif datatype == "ci16_le":
        dtype, endian = "int16", "little"
    elif datatype == "ci16_be":
        dtype, endian = "int16", "big"
    elif datatype == "ci8" or datatype == "ci8_le":
        dtype, endian = "int8", "little"
    elif datatype == "cu8" or datatype == "cu8_le":
        dtype, endian = "uint8", "little"
    else:
        raise ValueError(f"Unsupported core:datatype: {datatype}")
        
    config = RawIQConfig(
        dtype=dtype,
        iq_order='IQ',
        endian=endian,
        sample_rate_hz=float(sample_rate) if sample_rate else None,
        center_frequency_hz=float(center_frequency) if center_frequency else None
    )
    
    reader = RawIQReader(data_path, config)
    rec = reader.read()
    
    sr_status = MetadataStatus.KNOWN if sample_rate else MetadataStatus.MISSING
    cf_status = MetadataStatus.KNOWN if center_frequency else MetadataStatus.MISSING
    
    provenance = {
        "source_path": meta_path,
        "data_path": data_path,
        "file_size_bytes": os.path.getsize(data_path),
        "loader": "read_sigmf",
        "conversion_description": f"SigMF {datatype} -> complex64"
    }
    
    return SignalRecording(
        samples=rec.samples,
        source_format=SourceFormat.SIGMF,
        original_dtype=datatype,
        semantic_type="complex_iq",
        sample_rate_hz=MetadataValue(float(sample_rate) if sample_rate else None, "sigmf-meta", sr_status),
        center_frequency_hz=MetadataValue(float(center_frequency) if center_frequency else None, "sigmf-meta", cf_status),
        provenance=provenance,
        diagnostics=rec.diagnostics
    )
