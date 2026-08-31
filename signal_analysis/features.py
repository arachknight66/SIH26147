from .constants import DEFAULT_MAX_ANALYSIS_SAMPLES
import numpy as np
from scipy import stats, signal
from dataclasses import dataclass, field
from typing import List
from .models import Diagnostic, Severity
from typing import Tuple, Dict, Any, Optional

from .models import FeatureValidity, SignalRecording
from .measurements import compute_psd
from .rate_estimation import estimate_symbol_rate_consensus # Will create this next

@dataclass(frozen=True)
class AmplitudeFeatures:
    validity: FeatureValidity
    mean: float
    rms: float
    normalized_variance: float
    excess_kurtosis: float
    peak_to_rms_ratio: float

def extract_amplitude_features(samples: np.ndarray) -> AmplitudeFeatures:
    n_samples = len(samples)
    if n_samples < 16:
        return AmplitudeFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    mag = np.abs(samples)
    rms = float(np.sqrt(np.mean(mag**2)))
    
    if rms < 1e-9:
        return AmplitudeFeatures(FeatureValidity.UNRELIABLE, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    mean = float(np.mean(mag))
    var = float(np.var(mag))
    norm_var = var / (rms**2)
    
    kurtosis = float(stats.kurtosis(mag, fisher=True)) # Fisher=True means excess kurtosis
    peak = float(np.max(mag))
    peak_to_rms = peak / rms
    
    return AmplitudeFeatures(
        validity=FeatureValidity.VALID,
        mean=mean,
        rms=rms,
        normalized_variance=norm_var,
        excess_kurtosis=kurtosis,
        peak_to_rms_ratio=peak_to_rms
    )

@dataclass(frozen=True)
class PhaseFeatures:
    validity: FeatureValidity
    phase_inc_circ_mean: float
    phase_inc_circ_var: float
    collapse_var_2: float
    collapse_var_4: float
    collapse_var_8: float

def extract_phase_features(samples: np.ndarray) -> PhaseFeatures:
    n_samples = len(samples)
    if n_samples < 16:
        return PhaseFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    mag = np.abs(samples)
    rms = np.sqrt(np.mean(mag**2))
    
    if rms < 1e-9:
        return PhaseFeatures(FeatureValidity.UNRELIABLE, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    # Mask samples with |x[n]| < 0.05 * RMS
    mask = mag >= 0.05 * rms
    masked_samples = samples[mask]
    
    if len(masked_samples) < 16:
        return PhaseFeatures(FeatureValidity.UNRELIABLE, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    angles = np.angle(masked_samples)
    
    # Phase increments (difference between adjacent unmasked might be weird, but let's take difference 
    # of original samples and then mask, or diff then mask?) 
    # Usually phase increment is computed on adjacent samples, so mask pairs.
    prod = samples[1:] * np.conj(samples[:-1])
    mask_pairs = (mag[1:] >= 0.05 * rms) & (mag[:-1] >= 0.05 * rms)
    phase_inc = np.angle(prod[mask_pairs])
    
    if len(phase_inc) < 8:
        return PhaseFeatures(FeatureValidity.UNRELIABLE, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    circ_mean = float(np.angle(np.mean(np.exp(1j * phase_inc))))
    circ_var = 1.0 - float(np.abs(np.mean(np.exp(1j * phase_inc))))
    
    # M-th power collapse
    def collapse_var(M):
        exp_m = np.exp(1j * (angles * M))
        return 1.0 - float(np.abs(np.mean(exp_m)))
        
    return PhaseFeatures(
        validity=FeatureValidity.VALID,
        phase_inc_circ_mean=circ_mean,
        phase_inc_circ_var=circ_var,
        collapse_var_2=collapse_var(2),
        collapse_var_4=collapse_var(4),
        collapse_var_8=collapse_var(8)
    )

@dataclass(frozen=True)
class FrequencyFeatures:
    validity: FeatureValidity
    fsk_states: int
    occupancy_ratio: float

def extract_frequency_features(samples: np.ndarray) -> FrequencyFeatures:
    n_samples = len(samples)
    if n_samples < 32:
        return FrequencyFeatures(FeatureValidity.UNAVAILABLE, 0, 0.0)
        
    mag = np.abs(samples)
    rms = np.sqrt(np.mean(mag**2))
    
    if rms < 1e-9:
        return FrequencyFeatures(FeatureValidity.UNRELIABLE, 0, 0.0)
        
    prod = samples[1:] * np.conj(samples[:-1])
    mask_pairs = (mag[1:] >= 0.05 * rms) & (mag[:-1] >= 0.05 * rms)
    
    masked_prod = prod[mask_pairs]
    if len(masked_prod) < 16:
        return FrequencyFeatures(FeatureValidity.UNRELIABLE, 0, 0.0)
        
    f_inst = np.angle(masked_prod) / (2 * np.pi)
    
    hist, bin_edges = np.histogram(f_inst, bins=64, range=(-0.5, 0.5))
    
    # Peak detection
    peaks, _ = signal.find_peaks(hist, height=np.max(hist)*0.2, distance=3)
    fsk_states = len(peaks)
    
    occupancy_ratio = float(np.sum(hist[hist > np.max(hist)*0.1]) / np.sum(hist)) if np.sum(hist) > 0 else 0.0
    
    return FrequencyFeatures(FeatureValidity.VALID, fsk_states, occupancy_ratio)

@dataclass(frozen=True)
class CumulantFeatures:
    """
    Theoretical values for classification logic:
    BPSK: f20 = 1.0, f40 = 2.0
    QPSK: f20 = 0.0, f40 = 1.0
    8PSK: f20 = 0.0, f40 = 0.0
    16QAM: f20 = 0.0, f40 = 0.68
    """
    validity: FeatureValidity
    C20: complex
    C21: float
    C40: complex
    C41: complex
    C42: float
    f20: float
    f40: float
    f41: float
    f42: float

def extract_cumulant_features(samples: np.ndarray) -> CumulantFeatures:
    n_samples = len(samples)
    if n_samples < 64:
        # Need more samples for stable higher-order moments
        return CumulantFeatures(FeatureValidity.UNAVAILABLE, 0j, 0.0, 0j, 0j, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    z = samples - np.mean(samples)
    mag2 = np.abs(z)**2
    rms2 = np.mean(mag2)
    
    if rms2 < 1e-12:
        return CumulantFeatures(FeatureValidity.UNRELIABLE, 0j, 0.0, 0j, 0j, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    # Moments
    M20 = np.mean(z**2)
    M21 = np.mean(mag2) # Same as C21
    M40 = np.mean(z**4)
    M41 = np.mean((z**3) * np.conj(z))
    M42 = np.mean(mag2**2)
    
    # Cumulants
    C20 = M20
    C21 = M21
    C40 = M40 - 3 * (M20**2)
    C41 = M41 - 3 * M20 * M21
    C42 = M42 - np.abs(M20)**2 - 2 * (M21**2)
    
    # Scale invariant ratios
    f20 = float(np.abs(C20) / C21) if C21 > 0 else 0.0
    f40 = float(np.abs(C40) / (C21**2)) if C21 > 0 else 0.0
    f41 = float(np.abs(C41) / (C21**2)) if C21 > 0 else 0.0
    f42 = float(np.abs(C42) / (C21**2)) if C21 > 0 else 0.0
    
    return CumulantFeatures(
        validity=FeatureValidity.VALID,
        C20=C20, C21=float(C21), C40=C40, C41=C41, C42=float(C42),
        f20=f20, f40=f40, f41=f41, f42=f42
    )

@dataclass(frozen=True)
class SpectralFeatures:
    validity: FeatureValidity
    centroid: float
    spread: float
    kurtosis: float
    flatness: float
    prominent_peak_count: int

def extract_spectral_features(recording: SignalRecording) -> SpectralFeatures:
    if len(recording.samples) < 64:
        return SpectralFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0, 0.0, 0.0, 0)
        
    res = compute_psd(recording, nperseg=min(1024, len(recording.samples)))
    
    pxx = res.psd
    f = res.frequencies
    
    if np.sum(pxx) == 0:
        return SpectralFeatures(FeatureValidity.UNRELIABLE, 0.0, 0.0, 0.0, 0.0, 0)
        
    # Normalize PSD to act like PDF
    p_norm = pxx / np.sum(pxx)
    
    centroid = float(np.sum(f * p_norm))
    spread = float(np.sqrt(np.sum(((f - centroid)**2) * p_norm)))
    kurtosis = float(np.sum(((f - centroid)**4) * p_norm) / (spread**4) if spread > 0 else 0.0)
    
    # Spectral flatness (geometric mean / arithmetic mean)
    # add small epsilon to avoid log(0)
    eps = 1e-12
    geom_mean = np.exp(np.mean(np.log(pxx + eps)))
    arith_mean = np.mean(pxx)
    flatness = float(geom_mean / arith_mean) if arith_mean > 0 else 0.0
    
    # Prominent peak count via MAD-thresholded PSD
    median = np.median(pxx)
    mad = np.median(np.abs(pxx - median))
    threshold = median + 5 * mad if mad > 0 else median * 1.5
    
    peaks, _ = signal.find_peaks(pxx, height=threshold)
    peak_count = len(peaks)
    
    return SpectralFeatures(FeatureValidity.VALID, centroid, spread, kurtosis, flatness, peak_count)

@dataclass(frozen=True)
class CyclostationaryFeatures:
    validity: FeatureValidity
    periodicity_score: float
    top_candidate_rate: float
    
def extract_cyclostationary_features(recording: SignalRecording) -> CyclostationaryFeatures:
    # Delegate to rate estimation consensus
    consensus = estimate_symbol_rate_consensus(recording)
    
    if consensus is None:
        return CyclostationaryFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0)
        
    val = FeatureValidity.VALID if consensus[2] == "ESTIMATED" else FeatureValidity.PARTIALLY_VALID
    
    return CyclostationaryFeatures(
        validity=val,
        periodicity_score=consensus[3], # Assuming we return confidence/score
        top_candidate_rate=consensus[0]
    )

@dataclass(frozen=True)
class ModulationFeatureVector:
    overall_validity: FeatureValidity
    amplitude: AmplitudeFeatures
    phase: PhaseFeatures
    frequency: FrequencyFeatures
    cumulant: CumulantFeatures
    spectral: SpectralFeatures
    cyclostationary: CyclostationaryFeatures
    diagnostics: List[Diagnostic] = field(default_factory=list)


def check_ofdm_plausibility(samples: np.ndarray) -> Optional[int]:
    """
    Coarse check for OFDM cyclic prefix signature by searching for strong 
    autocorrelation peaks at typical FFT sizes.
    Returns the candidate FFT size if detected, else None.
    """
    if len(samples) < 8192:
        return None
        
    import numpy as np
    mag_sq = np.abs(samples)**2
    # Standard OFDM FFT sizes: 64, 128, 256, 512, 1024, 2048, 4096
    # For a CP, we expect R(N) to have a peak. We can just correlate the original signal?
    # Actually, delay-and-multiply is standard for CP detection: sum(x[k] * conj(x[k-N]))
    # For speed, we just check a few discrete Ns
    best_N = None
    best_peak_ratio = 0.0
    
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        if len(samples) < N * 3:
            continue
            
        # compute delay-and-multiply over a window
        window = min(len(samples) - N, 10000)
        delayed = samples[N:N+window]
        orig = samples[:window]
        
        corr = np.abs(np.mean(delayed * np.conjugate(orig)))
        pwr = np.mean(mag_sq[:window])
        if pwr > 0:
            rho = corr / pwr
            if rho > 0.15 and rho > best_peak_ratio:  # Strong correlation (typically CP is 1/4 or 1/8 so rho is large if signal is clean)
                best_peak_ratio = rho
                best_N = N
                
    return best_N

def extract_all_features(recording: SignalRecording) -> ModulationFeatureVector:

    """Extract all features working on a RMS-normalized copy."""
    # Truncate to avoid massive UI freezes on giant files
    max_samples = DEFAULT_MAX_ANALYSIS_SAMPLES
    process_samples = recording.samples[:max_samples]
    
    diagnostics = []
    if len(recording.samples) > max_samples:
        frac = max_samples / len(recording.samples)
        diagnostics.append(Diagnostic(
            severity=Severity.INFO,
            code="TRUNCATED_ANALYSIS",
            message=f"Analyzed first {max_samples} of {len(recording.samples)} samples ({frac*100:.2f}% of file)",
            evidence=""
        ))

    
    # RMS Normalize
    mag = np.abs(process_samples)
    rms = np.sqrt(np.mean(mag**2))
    
    if rms > 1e-9:
        norm_samples = process_samples / rms
    else:
        norm_samples = process_samples.copy()
        
    amp = extract_amplitude_features(norm_samples)
    fre = extract_frequency_features(norm_samples)
    
    # FIX 4: OFDM plausibility check
    ofdm_n = check_ofdm_plausibility(norm_samples)
    if ofdm_n is not None:
        diagnostics.append(Diagnostic(
            severity=Severity.INFO,
            code="OFDM_PLAUSIBILITY",
            message=f"Cyclic-prefix-like periodicity detected (candidate FFT size ~{ofdm_n}); this may indicate an OFDM/multicarrier signal outside the currently supported modulation set (BPSK/QPSK/8PSK/16-QAM/2-FSK).",
            evidence=""
        ))

    # FIX 1: Real-valued gate
    if recording.semantic_type != "complex_iq":
        pha = PhaseFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0, 0.0, 0.0, 0.0)
        cum = CumulantFeatures(FeatureValidity.UNAVAILABLE, 0j, 0.0, 0j, 0j, 0j, 0.0, 0.0, 0.0, 0.0)
        diagnostics.append(Diagnostic(
            severity=Severity.WARNING,
            code="COMPLEX_FEATURES_UNAVAILABLE",
            message=f"Phase/cumulant features require complex_iq samples; got semantic_type='{recording.semantic_type}'. PSK/QAM discriminants below are not physically meaningful for this input.",
            evidence=""
        ))
    else:
        pha = extract_phase_features(norm_samples)
        cum = extract_cumulant_features(norm_samples)

    
    # Spectral and Cyclostationary use the full recording object, we can pass normalized recording
    import dataclasses
    norm_rec = dataclasses.replace(recording, samples=norm_samples)
    
    spe = extract_spectral_features(norm_rec)
    cyc = extract_cyclostationary_features(norm_rec)
    
    # Determine overall validity as the worst of constituents
    validities = [amp.validity, pha.validity, fre.validity, cum.validity, spe.validity, cyc.validity]
    
    order = {
        FeatureValidity.UNAVAILABLE: 0,
        FeatureValidity.UNRELIABLE: 1,
        FeatureValidity.PARTIALLY_VALID: 2,
        FeatureValidity.VALID: 3
    }
    
    worst_val = min(validities, key=lambda v: order[v])
    
    return ModulationFeatureVector(
        overall_validity=worst_val,
        amplitude=amp,
        phase=pha,
        frequency=fre,
        cumulant=cum,
        spectral=spe,
        cyclostationary=cyc,
        diagnostics=diagnostics
    )
