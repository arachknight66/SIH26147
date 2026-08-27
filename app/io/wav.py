from __future__ import annotations
from pathlib import Path
import wave
import numpy as np
from app.exceptions import InvalidWavHeaderError
from app.models.metadata import Diagnostic, DiagnosticSeverity, MetadataSource, MetadataStatus, MetadataValue
from app.models.signal import IQOrder, SignalRecording, SourceFormat

class WavReader:
    def __init__(self, path: str | Path, mode: str = "unresolved"):
        self.path, self.mode = Path(path), mode
        if mode not in {"unresolved", "audio_mono", "real_baseband", "stereo_iq", "user_defined"}: raise ValueError("Invalid WAV mode.")
        try:
            with wave.open(str(self.path), "rb") as wav:
                self.channels, self.sample_width, self.sample_rate, self.frame_count = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
                self.comptype = wav.getcomptype()
        except (wave.Error, EOFError) as exc: raise InvalidWavHeaderError(f"Invalid WAV header: {exc}. Recovery: provide a valid RIFF/WAVE PCM file.") from exc
        if self.comptype != "NONE" or self.channels < 1 or self.sample_rate <= 0 or self.sample_width not in (1, 2, 3, 4): raise InvalidWavHeaderError("Unsupported or inconsistent WAV metadata (requires uncompressed PCM, 1–4 byte samples, and positive sample rate).")

    @property
    def sample_count(self) -> int: return self.frame_count

    def _decode(self, frames: bytes) -> np.ndarray:
        if self.sample_width == 1: data = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128.0)
        elif self.sample_width == 2: data = np.frombuffer(frames, "<i2").astype(np.float32)
        elif self.sample_width == 4: data = np.frombuffer(frames, "<i4").astype(np.float32)
        else:
            raw = np.frombuffer(frames, np.uint8).reshape(-1, 3)
            data = ((raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)) - ((raw[:, 2] & 128).astype(np.int32) << 24)).astype(np.float32)
        return data.reshape(-1, self.channels)

    def read_chunk(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > self.frame_count: raise ValueError("Chunk lies outside the WAV recording.")
        with wave.open(str(self.path), "rb") as wav:
            wav.setpos(start); frames = self._decode(wav.readframes(count))
        if self.mode == "stereo_iq":
            if self.channels != 2: raise InvalidWavHeaderError("stereo_iq mode requires exactly two WAV channels.")
            return (frames[:, 0] + 1j * frames[:, 1]).astype(np.complex64)
        return frames[:, 0].astype(np.complex64)

    def read(self) -> SignalRecording:
        samples = self.read_chunk(0, self.frame_count)
        is_iq = self.mode == "stereo_iq"
        semantic = "complex_iq" if is_iq else ("mono_real" if self.channels == 1 else "stereo_real")
        diags: list[Diagnostic] = []
        if self.channels == 2 and not is_iq: diags.append(Diagnostic(DiagnosticSeverity.WARNING, "WAV_STEREO_UNRESOLVED", "Stereo WAV is not assumed to be I/Q.", "Select stereo_iq explicitly if channel 0 is I and channel 1 is Q."))
        return SignalRecording(samples=samples, source_format=SourceFormat.WAV, original_dtype=f"pcm_s{self.sample_width * 8}le", channels=self.channels, semantic_type=semantic, iq_order=IQOrder.IQ if is_iq else IQOrder.NOT_APPLICABLE, sample_rate_hz=MetadataValue(float(self.sample_rate), MetadataSource.FILE_HEADER, MetadataStatus.KNOWN, 1.0, "WAV fmt chunk sample rate"), diagnostics=diags, metadata={"bits_per_sample": MetadataValue(self.sample_width * 8, MetadataSource.FILE_HEADER, MetadataStatus.KNOWN, 1.0, "WAV sample width"), "frame_count": MetadataValue(self.frame_count, MetadataSource.FILE_HEADER, MetadataStatus.KNOWN, 1.0, "WAV data chunk"), "duration_seconds": MetadataValue(self.frame_count / self.sample_rate, MetadataSource.STRUCTURAL_INFERENCE, MetadataStatus.INFERRED, 1.0, "frames / WAV header sample rate")}, provenance={"source_path": str(self.path), "file_size": self.path.stat().st_size, "loader": "WavReader", "conversion": "PCM values converted to complex64; no scaling, filtering, or resampling"})
