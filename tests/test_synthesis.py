import numpy as np
from scipy import signal

def generate_synthetic_signal(
    mod_type: str, 
    n_symbols: int = 1000, 
    sps: int = 4, 
    snr_db: float = 20.0, 
    cfo_norm: float = 0.0, # normalized CFO (cycles/sample)
    pulse_shape: str = 'rect'
) -> np.ndarray:
    
    if mod_type == "BPSK":
        syms = np.random.choice([-1, 1], size=n_symbols)
    elif mod_type == "QPSK":
        syms = np.random.choice([1+1j, -1+1j, -1-1j, 1-1j], size=n_symbols) / np.sqrt(2)
    elif mod_type == "8PSK":
        angles = np.random.randint(0, 8, size=n_symbols) * (np.pi / 4)
        syms = np.exp(1j * angles)
    elif mod_type in ("16-QAM", "16QAM"):
        levels = [-3, -1, 1, 3]
        I = np.random.choice(levels, size=n_symbols)
        Q = np.random.choice(levels, size=n_symbols)
        syms = (I + 1j * Q) / np.sqrt(10) # Normalize average power to 1
    elif mod_type in ("2-FSK", "2FSK"):
        bits = np.random.choice([-1, 1], size=n_symbols)
        h = 0.5
        phase_inc = bits * (h * np.pi / sps)
        phase_inc_up = np.repeat(phase_inc, sps)
        phase = np.cumsum(phase_inc_up)
        sig = np.exp(1j * phase)
    else:
        raise ValueError(f"Unknown mod_type: {mod_type}")
        
    if "FSK" not in mod_type:
        up = np.zeros(n_symbols * sps, dtype=np.complex128)
        up[::sps] = syms
        
        if pulse_shape == 'rect':
            p = np.ones(sps)
        elif pulse_shape == 'rrc':
            t = np.arange(-4*sps, 4*sps+1)
            alpha = 0.35
            # Proper RRC calculation to ensure excess bandwidth
            p = np.zeros_like(t, dtype=float)
            for i, tc in enumerate(t):
                if tc == 0:
                    p[i] = 1.0 - alpha + 4*alpha/np.pi
                elif abs(tc) == sps / (4*alpha):
                    p[i] = (alpha/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*alpha)) + (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
                else:
                    num = np.sin(np.pi*tc*(1-alpha)/sps) + 4*alpha*tc/sps * np.cos(np.pi*tc*(1+alpha)/sps)
                    den = np.pi*tc/sps * (1 - (4*alpha*tc/sps)**2)
                    p[i] = num / den
            
        sig = np.convolve(up, p, mode='same')
        
    # Apply CFO
    t = np.arange(len(sig))
    sig = sig * np.exp(1j * 2 * np.pi * cfo_norm * t)
    
    # Apply AWGN
    sig_power = np.mean(np.abs(sig)**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    sig += noise
    
    return sig.astype(np.complex64)
