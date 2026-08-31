with open('signal_analysis/synchronization.py', 'a') as f:
    f.write("""

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
        
    symbol_clock_locked = lock_quality_metric < 0.1
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
    
    mean_e = np.mean(np.abs(decisions))
    if mean_e > 0:
        evm = np.sqrt(np.mean((np.abs(decisions) - mean_e)**2)) / mean_e * 100
    else:
        evm = 100.0
        
    noise_var = np.var(e1_list - e0_list) + 1e-9
    soft_llrs = (e1_list - e0_list) / noise_var
    
    return hard_bits, soft_llrs.astype(np.float32), decisions, float(evm)
""")
print("Appended synchronization FSK")
