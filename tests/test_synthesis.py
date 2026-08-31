import numpy as np
from scipy import signal

def generate_synthetic_signal(
    mod_type: str, 
    n_symbols: int = 1000, 
    sps: int = 4, 
    snr_db: float = 20.0, 
    cfo_norm: float = 0.0,
    pulse_shape: str = 'rect',
    timing_offset_frac: float = 0.0,
    return_bits: bool = False
):
    bits = None
    
    if mod_type == "BPSK":
        bits = np.random.randint(0, 2, size=n_symbols).astype(np.uint8)
        syms = np.where(bits == 1, 1, -1).astype(np.complex128)
    elif mod_type == "QPSK":
        bits = np.random.randint(0, 2, size=(n_symbols, 2)).astype(np.uint8)
        I = np.where(bits[:, 0] == 1, 1, -1)
        Q = np.where(bits[:, 1] == 1, 1, -1)
        syms = (I + 1j * Q) / np.sqrt(2)
    elif mod_type == "8PSK":
        bits = np.random.randint(0, 2, size=(n_symbols, 3)).astype(np.uint8)
        # Gray mapping inverse from demodulation.py
        # [0,0,0]=0, [0,0,1]=1, [0,1,1]=2, [0,1,0]=3
        # [1,1,0]=4, [1,1,1]=5, [1,0,1]=6, [1,0,0]=7
        idx_map = {(0,0,0):0, (0,0,1):1, (0,1,1):2, (0,1,0):3, (1,1,0):4, (1,1,1):5, (1,0,1):6, (1,0,0):7}
        angles = np.array([idx_map[tuple(b)] for b in bits]) * (np.pi / 4)
        syms = np.exp(1j * angles)
        bits = bits.flatten()
    elif mod_type in ("16-QAM", "16QAM"):
        bits = np.random.randint(0, 2, size=(n_symbols, 4)).astype(np.uint8)
        # Mapping matching demodulation.py
        # Bits: b0 b1 b2 b3.
        # Let's just generate random symbols directly and map them back to bits to ensure consistency, 
        # or use the exact map. To keep it simple, just use the exact map from demodulation:
        from signal_analysis.demodulation import CONSTELLATION_MAPS
        pts = CONSTELLATION_MAPS["16-QAM"]["points"]
        bts = CONSTELLATION_MAPS["16-QAM"]["bits"]
        chosen_idx = np.random.randint(0, 16, size=n_symbols)
        syms = pts[chosen_idx]
        bits = np.array([bts[i] for i in chosen_idx], dtype=np.uint8).flatten()
    elif mod_type in ("2-FSK", "2FSK"):
        bits = np.random.randint(0, 2, size=n_symbols).astype(np.uint8)
        # 1 -> +h, 0 -> -h
        phase_inc = np.where(bits == 1, 1, -1) * (0.5 * np.pi / sps)
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
        
    if timing_offset_frac != 0.0:
        N = len(sig)
        F = np.fft.fft(sig)
        freqs = np.fft.fftfreq(N)
        delay_samples = timing_offset_frac * sps
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_samples)
        sig = np.fft.ifft(F * phase_shift)
        
    t = np.arange(len(sig))
    sig = sig * np.exp(1j * 2 * np.pi * cfo_norm * t)
    
    sig_power = np.mean(np.abs(sig)**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    sig += noise
    
    sig = sig.astype(np.complex64)
    if return_bits:
        return sig, bits
    return sig
