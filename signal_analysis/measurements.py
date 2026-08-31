import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from .models import SignalRecording, Diagnostic, Severity, MetadataStatus

@dataclass(frozen=True)
class TimeStatistics:
    mean_i: float
    mean_q: float
    var_i: float
    var_q: float
    rms_amplitude: float
    peak_amplitude: float
    crest_factor: float
    dynamic_range_db: float

def compute_time_statistics(samples: np.ndarray) -> TimeStatistics:
    """
    Compute time-domain statistics of the complex samples.
    
    Parameters
    ----------
    samples : np.ndarray
        complex64 numpy array.
        
    Returns
    -------
    TimeStatistics
        Computed statistics.
    """
    if samples.ndim > 1:
        samples = samples.flatten()
        
    i_samples = samples.real
    q_samples = samples.imag
    
    mean_i = float(np.mean(i_samples))
    mean_q = float(np.mean(q_samples))
    var_i = float(np.var(i_samples))
    var_q = float(np.var(q_samples))
    
    mag = np.abs(samples)
    rms_amplitude = float(np.sqrt(np.mean(mag**2)))
    peak_amplitude = float(np.max(mag))
    
    crest_factor = peak_amplitude / rms_amplitude if rms_amplitude > 0 else 0.0
    
    dynamic_range_db = 20 * np.log10(crest_factor) if crest_factor > 0 else 0.0
    
    return TimeStatistics(
        mean_i=mean_i,
        mean_q=mean_q,
        var_i=var_i,
        var_q=var_q,
        rms_amplitude=rms_amplitude,
        peak_amplitude=peak_amplitude,
        crest_factor=crest_factor,
        dynamic_range_db=float(dynamic_range_db)
    )

@dataclass(frozen=True)
class DCOffset:
    mean_i: float
    mean_q: float
    magnitude: float
    phase: float

def compute_dc_offset(samples: np.ndarray) -> DCOffset:
    """
    Compute DC offset of the complex samples.
    """
    if samples.ndim > 1:
        samples = samples.flatten()
        
    mean_i = float(np.mean(samples.real))
    mean_q = float(np.mean(samples.imag))
    
    magnitude = float(np.sqrt(mean_i**2 + mean_q**2))
    phase = float(np.arctan2(mean_q, mean_i))
    
    return DCOffset(mean_i, mean_q, magnitude, phase)

def detect_clipping(recording: SignalRecording) -> Tuple[float, Optional[Diagnostic]]:
    """
    Detect clipping in the signal.
    """
    samples = recording.samples
    dtype = recording.original_dtype
    
    # Determine extremum
    if dtype in ('int8', 'ci8', 'ci8_le'):
        extremum = 127
    elif dtype in ('uint8', 'cu8', 'cu8_le'):
        extremum = 255
    elif dtype in ('int16', 'ci16_le', 'ci16_be'):
        extremum = 32767
    elif dtype in ('int24',):
        extremum = 8388607
    elif dtype in ('int32',):
        extremum = 2147483647
    elif dtype in ('float32', 'cf32_le', 'cf32_be', 'complex64'):
        extremum = 1.0 
    else:
        extremum = 1.0 # fallback
        
    # Check within 0.1% of extremum
    threshold = extremum * 0.999
    
    if dtype in ('uint8', 'cu8', 'cu8_le'):
        # For unsigned, check near 0 and near 255
        clipped_i = np.logical_or(samples.real >= threshold, samples.real <= 255 - threshold)
        clipped_q = np.logical_or(samples.imag >= threshold, samples.imag <= 255 - threshold)
        clipped = np.logical_or(clipped_i, clipped_q) if recording.semantic_type != "mono_real" else clipped_i
    else:
        # Check absolute value for signed and floats
        clipped_i = np.abs(samples.real) >= threshold
        clipped_q = np.abs(samples.imag) >= threshold
        clipped = np.logical_or(clipped_i, clipped_q) if recording.semantic_type != "mono_real" else clipped_i
            
    clip_fraction = float(np.mean(clipped))
    
    diagnostic = None
    if clip_fraction > 0.05:
        diagnostic = Diagnostic(
            severity=Severity.ERROR,
            code="CLIPPING_DETECTED",
            message="Severe clipping detected (>5% of samples)",
            evidence=f"{clip_fraction*100:.2f}% of samples within 0.1% of {extremum}"
        )
    elif clip_fraction > 0.001:
        diagnostic = Diagnostic(
            severity=Severity.WARNING,
            code="CLIPPING_DETECTED",
            message="Clipping detected (>0.1% of samples)",
            evidence=f"{clip_fraction*100:.2f}% of samples within 0.1% of {extremum}"
        )
        
    return clip_fraction, diagnostic

@dataclass(frozen=True)
class PSDResult:
    frequencies: np.ndarray
    psd: np.ndarray
    freq_unit: str

def compute_psd(recording: SignalRecording, nperseg: int = 1024) -> PSDResult:
    """
    Compute Welch's PSD.
    Two-sided for complex, one-sided for real.
    """
    is_complex = recording.semantic_type == "complex_iq"
    
    if recording.sample_rate_hz.status == MetadataStatus.KNOWN and recording.sample_rate_hz.value is not None:
        fs = recording.sample_rate_hz.value
        unit = "Hz"
    else:
        fs = 1.0
        unit = "cycles/sample"
        
    if is_complex:
        f, pxx = signal.welch(recording.samples, fs=fs, nperseg=nperseg, return_onesided=False, window='hann')
        f = np.fft.fftshift(f)
        pxx = np.fft.fftshift(pxx)
    else:
        if recording.samples.ndim > 1:
            data = recording.samples[:, 0].real
        else:
            data = recording.samples.real
        f, pxx = signal.welch(data, fs=fs, nperseg=nperseg, return_onesided=True, window='hann')
        
    return PSDResult(f, pxx, unit)

@dataclass(frozen=True)
class SpectrogramResult:
    frequencies: np.ndarray
    times: np.ndarray
    Sxx: np.ndarray
    freq_unit: str

def compute_spectrogram(recording: SignalRecording, nperseg: int = 256) -> SpectrogramResult:
    """
    Compute STFT spectrogram.
    """
    is_complex = recording.semantic_type == "complex_iq"
    
    if recording.sample_rate_hz.status == MetadataStatus.KNOWN and recording.sample_rate_hz.value is not None:
        fs = recording.sample_rate_hz.value
        unit = "Hz"
    else:
        fs = 1.0
        unit = "cycles/sample"
        
    if is_complex:
        f, t, Sxx = signal.spectrogram(recording.samples, fs=fs, nperseg=nperseg, return_onesided=False, window='hann')
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)
    else:
        if recording.samples.ndim > 1:
            data = recording.samples[:, 0].real
        else:
            data = recording.samples.real
        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, return_onesided=True, window='hann')
        
    return SpectrogramResult(f, t, Sxx, unit)
