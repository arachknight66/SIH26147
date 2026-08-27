from __future__ import annotations
from typing import Any
import numpy as np
from app.models.analysis import DetectedRegion
from app.models.signal import SignalRecording
from .models import RecoveryConfig

def prepare_recovery_samples(
    recording: SignalRecording,
    region: DetectedRegion | None = None,
    config: RecoveryConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Non-destructively extract, condition, and normalize complex IQ samples for receiver processing.

    Parameters
    ----------
    recording : SignalRecording
        Input canonical complex64 signal recording.
    region : DetectedRegion | None
        Target spectral/temporal signal region from Phase 2.
    config : RecoveryConfig | None
        Recovery configuration.

    Returns
    -------
    samples : np.ndarray
        Prepared complex64 samples.
    provenance : dict[str, Any]
        Record of applied conditioning operations.
    """
    cfg = config or RecoveryConfig()
    raw = recording.samples
    provenance: dict[str, Any] = {
        "original_length": len(raw),
        "operations": [],
    }

    # 1. Temporal Region Slicing
    if region is not None and region.start_sample is not None and region.end_sample is not None:
        start = max(0, region.start_sample)
        end = min(len(raw), region.end_sample)
        samples = raw[start:end].copy()
        provenance["region_slice"] = {"start_sample": start, "end_sample": end}
        provenance["operations"].append("temporal_slice")
    else:
        samples = raw.copy()

    # 2. Length Clamping for Bounded Computation
    if len(samples) > cfg.max_recovery_samples:
        samples = samples[:cfg.max_recovery_samples]
        provenance["length_clamped"] = cfg.max_recovery_samples
        provenance["operations"].append("length_clamping")

    if len(samples) < 16:
        return samples.astype(np.complex64), provenance

    # 3. DC Offset Estimation & Removal
    dc_mean = complex(np.mean(samples))
    dc_mag = float(np.abs(dc_mean))
    rms_pre = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    
    # Remove DC if DC magnitude exceeds 2% of RMS
    if rms_pre > 1e-9 and (dc_mag / rms_pre) > 0.02:
        samples = samples - dc_mean
        provenance["dc_removed"] = {"dc_estimate_real": round(dc_mean.real, 6), "dc_estimate_imag": round(dc_mean.imag, 6)}
        provenance["operations"].append("dc_removal")
    else:
        provenance["dc_removed"] = None

    # 4. Coarse Baseband Frequency Translation
    f_center = region.center_freq_normalized if region is not None else 0.0
    if abs(f_center) >= 0.005 and len(samples) > 0:
        t = np.arange(len(samples), dtype=np.float32)
        mix = np.exp(-2j * np.pi * f_center * t).astype(np.complex64)
        samples = samples * mix
        provenance["coarse_frequency_translation"] = {"shift_normalized": round(float(f_center), 6)}
        provenance["operations"].append("frequency_translation")

    # 5. Unit RMS Gain Normalization
    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    if rms > 1e-12:
        gain = 1.0 / rms
        samples = (samples * gain).astype(np.complex64)
        provenance["gain_normalization"] = {"scale_factor": round(gain, 6), "original_rms": round(rms, 6)}
        provenance["operations"].append("gain_normalization")

    return samples.astype(np.complex64), provenance
