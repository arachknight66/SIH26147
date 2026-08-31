import numpy as np
from scipy import signal
from typing import List, Tuple, Optional
from .models import SignalRecording, MetadataStatus

def _detect_peaks_in_range(pxx, f, min_f=0.02, max_f=0.48):
    """Detect peaks in the specified frequency range and return best candidate rate and confidence."""
    mask = (f >= min_f) & (f <= max_f)
    if not np.any(mask):
        return None, 0.0
        
    pxx_masked = pxx[mask]
    f_masked = f[mask]
    
    # MAD-based baseline subtraction
    median = np.median(pxx_masked)
    mad = np.median(np.abs(pxx_masked - median))
    if mad == 0:
        mad = 1e-9
        
    peaks, props = signal.find_peaks(pxx_masked, height=median + 3*mad)
    if len(peaks) == 0:
        return None, 0.0
        
    # Pick the strongest peak
    best_idx = np.argmax(props['peak_heights'])
    peak_pos = peaks[best_idx]
    
    # Confidence is roughly SNR of the peak relative to MAD
    confidence = (props['peak_heights'][best_idx] - median) / mad
    
    # Parabolic interpolation for sub-bin accuracy
    if 0 < peak_pos < len(pxx_masked) - 1:
        alpha = pxx_masked[peak_pos-1]
        beta = pxx_masked[peak_pos]
        gamma = pxx_masked[peak_pos+1]
        
        # Vertex x offset
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        f_est = f_masked[peak_pos] + p * (f_masked[1] - f_masked[0])
    else:
        f_est = f_masked[peak_pos]
        
    return f_est, confidence

def estimate_rate_transition_energy(samples: np.ndarray) -> Tuple[Optional[float], float]:
    """FFT of |x[n]-x[n-1]|^2"""
    diff = np.abs(samples[1:] - samples[:-1])**2
    if len(diff) < 64:
        return None, 0.0
    f, pxx = signal.welch(diff - np.mean(diff), fs=1.0, nperseg=min(2048, len(diff)), return_onesided=True)
    return _detect_peaks_in_range(pxx, f)

def estimate_rate_squared_magnitude(samples: np.ndarray) -> Tuple[Optional[float], float]:
    """FFT of |x[n]|^2 (Gardner-style)"""
    mag2 = np.abs(samples)**2
    if len(mag2) < 64:
        return None, 0.0
    f, pxx = signal.welch(mag2 - np.mean(mag2), fs=1.0, nperseg=min(2048, len(mag2)), return_onesided=True)
    return _detect_peaks_in_range(pxx, f)

def estimate_rate_autocorrelation(samples: np.ndarray) -> Tuple[Optional[float], float]:
    """Autocorrelation secondary peak"""
    mag = np.abs(samples)
    mag_zm = mag - np.mean(mag)
    
    # Compute autocorrelation via FFT
    N = len(mag_zm)
    if N < 64:
        return None, 0.0
        
    n_fft = 2 ** int(np.ceil(np.log2(N * 2 - 1)))
    F = np.fft.fft(mag_zm, n_fft)
    R = np.fft.ifft(F * np.conj(F)).real
    
    # Normalize
    R = R[:N] / (R[0] + 1e-9)
    
    # Limit search to realistic lags (e.g., 2 to 50 samples per symbol)
    min_lag = 2
    max_lag = 50
    if N <= max_lag:
        max_lag = N - 1
        
    if max_lag <= min_lag:
        return None, 0.0
        
    lags = np.arange(N)
    mask = (lags >= min_lag) & (lags <= max_lag)
    R_search = R[mask]
    lags_search = lags[mask]
    
    peaks, props = signal.find_peaks(R_search, height=0.05, distance=2)
    if len(peaks) == 0:
        return None, 0.0
        
    best_idx = np.argmax(props['peak_heights'])
    peak_pos = peaks[best_idx]
    confidence = props['peak_heights'][best_idx] * 10 # heuristic scaling
    
    # Parabolic refinement
    if 0 < peak_pos < len(R_search) - 1:
        alpha = R_search[peak_pos-1]
        beta = R_search[peak_pos]
        gamma = R_search[peak_pos+1]
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        lag_est = lags_search[peak_pos] + p
    else:
        lag_est = lags_search[peak_pos]
        
    return 1.0 / lag_est, confidence

def estimate_symbol_rate_consensus(recording: SignalRecording) -> Optional[Tuple[float, str, str, float]]:
    """
    Returns (rate, unit, status, confidence) or None.
    status is 'ESTIMATED' or 'AMBIGUOUS'.
    """
    c1, conf1 = estimate_rate_transition_energy(recording.samples)
    c2, conf2 = estimate_rate_squared_magnitude(recording.samples)
    c3, conf3 = estimate_rate_autocorrelation(recording.samples)
    
    candidates = []
    if c1 is not None and conf1 > 3.0: candidates.append((c1, conf1, 'TE'))
    if c2 is not None and conf2 > 3.0: candidates.append((c2, conf2, 'SM'))
    if c3 is not None and conf3 > 0.5: candidates.append((c3, conf3, 'AC'))
    
    if not candidates:
        return None
        
    # Cluster candidates (tolerance ~3%)
    candidates.sort(key=lambda x: x[0])
    
    clusters = []
    for cand in candidates:
        matched = False
        for cluster in clusters:
            # check if within 3% of cluster mean
            mean_val = np.mean([c[0] for c in cluster])
            if abs(cand[0] - mean_val) / mean_val <= 0.03:
                cluster.append(cand)
                matched = True
                break
        if not matched:
            clusters.append([cand])
            
    # Resolve harmonic aliasing
    resolved_clusters = []
    skip = set()
    for i, c_a in enumerate(clusters):
        if i in skip: continue
        mean_a = np.mean([c[0] for c in c_a])
        
        merged_cluster = list(c_a)
        for j, c_b in enumerate(clusters):
            if i == j or j in skip: continue
            mean_b = np.mean([c[0] for c in c_b])
            
            # Check ratio
            ratio = mean_b / mean_a if mean_b > mean_a else mean_a / mean_b
            if any(abs(ratio - r) < 0.05 for r in [2, 3, 4]):
                if mean_a < mean_b:
                    conf_a = sum(c[1] for c in c_a)
                    conf_b = sum(c[1] for c in c_b)
                    # Only collapse if the lower frequency (fundamental candidate)
                    # is reasonably strong compared to the harmonic.
                    if conf_a > 0.1 * conf_b:
                        r_int = round(ratio)
                        adjusted_b = [(c[0]/r_int, c[1], c[2]) for c in c_b]
                        merged_cluster.extend(adjusted_b)
                        skip.add(j)
                else:
                    pass 
                    
        if i not in skip:
            resolved_clusters.append(merged_cluster)
            
    if not resolved_clusters:
        return None
        
    best_cluster = None
    best_score = -1
    best_unique_families = 0
    
    for cluster in resolved_clusters:
        unique_families = len(set(c[2] for c in cluster))
        total_conf = sum(c[1] for c in cluster)
        score = unique_families * 100 + total_conf
        if score > best_score:
            best_score = score
            best_cluster = cluster
            best_unique_families = unique_families
            
    final_rate_norm = float(np.mean([c[0] for c in best_cluster]))
    final_conf = float(np.sum([c[1] for c in best_cluster]))
    
    status = "ESTIMATED" if best_unique_families >= 2 else "AMBIGUOUS"
    
    if recording.sample_rate_hz.status == MetadataStatus.KNOWN and recording.sample_rate_hz.value is not None:
        fs = recording.sample_rate_hz.value
        rate = final_rate_norm * fs
        unit = "Hz"
    else:
        rate = final_rate_norm
        unit = "symbols/sample"
        
    return rate, unit, status, final_conf
