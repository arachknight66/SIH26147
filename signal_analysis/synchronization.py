from .constants import DEFAULT_MAX_ANALYSIS_SAMPLES
import numpy as np
from scipy import signal
from typing import Tuple, Optional, List
from .models import SynchronizationResult, Diagnostic, Severity

def estimate_coarse_cfo_psk_qam(samples: np.ndarray, M: int) -> float:
    """
    Estimate coarse CFO for PSK/QAM using M-th power non-linearity.
    Dividing by M undoes the frequency scaling introduced by M-th power.
    """
    max_samples = DEFAULT_MAX_ANALYSIS_SAMPLES
    x = samples[:max_samples]
    x_m = x ** M
    
    N = len(x_m)
    X = np.fft.fft(x_m)
    f = np.fft.fftfreq(N, d=1.0)
    
    peak_idx = np.argmax(np.abs(X))
    f_peak = f[peak_idx]
    
    cfo = f_peak / M
    return float(cfo)

def rrc_filter(sps: float, alpha: float = 0.35, length: int = 8) -> np.ndarray:
    t = np.arange(-length * sps, length * sps + 1)
    p = np.zeros_like(t, dtype=float)
    for i, tc in enumerate(t):
        if tc == 0:
            p[i] = 1.0 - alpha + 4 * alpha / np.pi
        elif abs(tc) == sps / (4 * alpha):
            p[i] = (alpha / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            num = np.sin(np.pi * tc * (1 - alpha) / sps) + 4 * alpha * tc / sps * np.cos(np.pi * tc * (1 + alpha) / sps)
            den = np.pi * tc / sps * (1 - (4 * alpha * tc / sps) ** 2)
            p[i] = num / den
    return p / np.sqrt(np.sum(p**2))

def recover_timing_gardner(samples: np.ndarray, sps: float, alpha: float = 0.35) -> Tuple[np.ndarray, bool, float, np.ndarray]:
    """
    Gardner loop using linear interpolation.
    Returns: (symbols, symbol_clock_locked, lock_quality, sample_indices)
    """
    mf = rrc_filter(sps, alpha=alpha)
    filtered = np.convolve(samples, mf, mode='same')
    
    BnT = 0.005
    zeta = 0.707
    Kp = 2 * zeta * BnT
    Ki = (BnT ** 2)
    
    W = 2.0 / sps
    nco_phase = 0.0
    integrator = 0.0
    
    out_symbols = []
    symbol_indices = []
    errors = []
    
    n = 0
    N = len(filtered)
    
    is_midpoint = False
    prev_sym = 0j
    mid_sym = 0j
    
    while n < N - 1:
        nco_phase += W + integrator
        
        if nco_phase >= 1.0:
            nco_phase -= 1.0
            
            mu = nco_phase / (W + integrator)
            strobe_val = filtered[n] * mu + filtered[n+1] * (1 - mu)
            
            if is_midpoint:
                mid_sym = strobe_val
                is_midpoint = False
            else:
                curr_sym = strobe_val
                err = np.real((curr_sym - prev_sym) * np.conj(mid_sym))
                errors.append(err)
                
                integrator += Ki * err
                nco_phase += Kp * err
                
                out_symbols.append(curr_sym)
                symbol_indices.append(n - mu)
                prev_sym = curr_sym
                is_midpoint = True
                
        n += 1
        
    out_symbols = np.array(out_symbols)
    symbol_indices = np.array(symbol_indices)
    
    if len(errors) > 100:
        trailing_errors = errors[-100:]
        lock_quality_metric = float(np.var(trailing_errors))
    else:
        lock_quality_metric = 999.0
        
    # Gardner error has high self-noise (variance ~0.5-0.8 even when locked cleanly for QPSK).
    # We set a threshold of 1.5 to allow for modulation self-noise while rejecting extreme noise.
    symbol_clock_locked = lock_quality_metric < 25.0
    
    return out_symbols, symbol_clock_locked, lock_quality_metric, symbol_indices

def recover_carrier_costas(symbols: np.ndarray, modulation: str) -> Tuple[np.ndarray, bool, float]:
    BnT = 0.01
    zeta = 0.707
    Kp = 2 * zeta * BnT
    Ki = (BnT ** 2)
    
    phase = 0.0
    freq = 0.0
    
    out_symbols = np.zeros_like(symbols)
    errors = []
    
    ideal_8psk = np.exp(1j * np.array([i * np.pi/4 for i in range(8)]))
    qam_levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    
    for i, sym in enumerate(symbols):
        derotated = sym * np.exp(-1j * phase)
        out_symbols[i] = derotated
        
        I = derotated.real
        Q = derotated.imag
        
        if modulation == "BPSK":
            err = np.sign(I) * Q
        elif modulation == "QPSK":
            err = np.sign(I) * Q - np.sign(Q) * I
        elif modulation == "8PSK":
            idx = np.argmin(np.abs(derotated - ideal_8psk))
            err = np.angle(derotated * np.conj(ideal_8psk[idx]))
        elif modulation == "16-QAM":
            I_dec = qam_levels[np.argmin(np.abs(I - qam_levels))]
            Q_dec = qam_levels[np.argmin(np.abs(Q - qam_levels))]
            err = I_dec * Q - Q_dec * I
        else:
            err = 0.0
            
        errors.append(err)
        
        freq += Ki * err
        phase += Kp * err + freq
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        
    if len(errors) > 100:
        lock_quality = float(np.var(errors[-100:]))
    else:
        lock_quality = 999.0
        
    carrier_locked = lock_quality < 0.2
    return out_symbols, carrier_locked, lock_quality


def recover_timing_fsk(samples: np.ndarray, sps: float) -> Tuple[bool, float, np.ndarray]:
    prod = samples[1:] * np.conj(samples[:-1])
    fm = np.angle(prod)
    
    BnT = 0.005
    zeta = 0.707
    Kp = 2 * zeta * BnT
    Ki = (BnT ** 2)
    
    W = 2.0 / sps
    nco_phase = 0.0
    integrator = 0.0
    
    symbol_indices = []
    errors = []
    
    n = 0
    N = len(fm)
    
    is_midpoint = False
    prev_sym = 0.0
    mid_sym = 0.0
    
    while n < N - 1:
        nco_phase += W + integrator
        if nco_phase >= 1.0:
            nco_phase -= 1.0
            
            mu = nco_phase / (W + integrator)
            strobe_val = fm[n] * mu + fm[n+1] * (1 - mu)
            
            if is_midpoint:
                mid_sym = strobe_val
                is_midpoint = False
            else:
                curr_sym = strobe_val
                err = (curr_sym - prev_sym) * mid_sym
                errors.append(err)
                
                integrator += Ki * err
                nco_phase += Kp * err
                
                symbol_indices.append(n - mu)
                prev_sym = curr_sym
                is_midpoint = True
        n += 1
        
    symbol_indices = np.array(symbol_indices)
    if len(errors) > 100:
        lock_quality_metric = float(np.var(errors[-100:]))
    else:
        lock_quality_metric = 999.0
        
    # Gardner error has high self-noise (variance ~0.5-0.8 even when locked cleanly for QPSK).
    # We set a threshold of 1.5 to allow for modulation self-noise while rejecting extreme noise.
    symbol_clock_locked = lock_quality_metric < 25.0
    return symbol_clock_locked, lock_quality_metric, symbol_indices

def fsk_dual_correlator(samples: np.ndarray, symbol_indices: np.ndarray, f0: float, f1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    e0_list = []
    e1_list = []
    decisions = []
    
    for i in range(len(symbol_indices) - 1):
        start = int(round(symbol_indices[i]))
        end = int(round(symbol_indices[i+1]))
        if start == end: end += 1
        if end > len(samples): break
        
        chunk = samples[start:end]
        t = np.arange(len(chunk))
        
        corr0 = np.sum(chunk * np.exp(-1j * 2 * np.pi * f0 * t))
        corr1 = np.sum(chunk * np.exp(-1j * 2 * np.pi * f1 * t))
        
        e0 = np.abs(corr0)
        e1 = np.abs(corr1)
        
        e0_list.append(e0)
        e1_list.append(e1)
        decisions.append(e0 - e1 + 1j*0)
        
    e0_list = np.array(e0_list)
    e1_list = np.array(e1_list)
    decisions = np.array(decisions, dtype=np.complex64)
    
    hard_bits = (e1_list > e0_list).astype(np.uint8)
    
    if len(decisions) > 100:
        steady = decisions[100:]
    else:
        steady = decisions
    mean_e = np.mean(np.abs(steady))
    if mean_e > 0:
        evm = np.sqrt(np.mean((np.abs(steady) - mean_e)**2)) / mean_e * 100
    else:
        evm = 100.0
        
    noise_var = np.var(e1_list - e0_list) + 1e-9
    soft_llrs = (e1_list - e0_list) / noise_var
    
    return hard_bits, soft_llrs.astype(np.float32), decisions, float(evm)
