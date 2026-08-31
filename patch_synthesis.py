import re
with open('tests/test_synthesis.py', 'r') as f:
    content = f.read()

# Replace definition
old_def = """def generate_synthetic_signal(
    mod_type: str, 
    n_symbols: int = 1000, 
    sps: int = 4, 
    snr_db: float = 20.0, 
    cfo_norm: float = 0.0, # normalized CFO (cycles/sample)
    pulse_shape: str = 'rect'
) -> np.ndarray:"""

new_def = """def generate_synthetic_signal(
    mod_type: str, 
    n_symbols: int = 1000, 
    sps: float = 4.0, 
    snr_db: float = 20.0, 
    cfo_norm: float = 0.0, # normalized CFO (cycles/sample)
    pulse_shape: str = 'rect',
    timing_offset_frac: float = 0.0,
    return_bits: bool = False
):"""
content = content.replace(old_def, new_def)

# To implement timing offset accurately, we can oversample, shift, then decimate.
# Or apply a phase shift in frequency domain.
# Since we already do pulse shaping, we can just evaluate the pulse shape at an offset.
# Instead of doing that, shifting in frequency domain is perfectly mathematically sound and exact for fractional delays.

old_cfo = """    # Apply CFO
    t = np.arange(len(sig))
    sig = sig * np.exp(1j * 2 * np.pi * cfo_norm * t)"""

new_cfo = """    # Apply timing offset via frequency domain phase shift
    if timing_offset_frac != 0.0:
        N = len(sig)
        F = np.fft.fft(sig)
        freqs = np.fft.fftfreq(N)
        # Shift by fractional symbols: delay in samples = timing_offset_frac * sps
        delay_samples = timing_offset_frac * sps
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_samples)
        sig = np.fft.ifft(F * phase_shift)
        
    # Apply CFO
    t = np.arange(len(sig))
    sig = sig * np.exp(1j * 2 * np.pi * cfo_norm * t)"""
content = content.replace(old_cfo, new_cfo)

# Now, we also need to capture and return bits if requested.
# I'll just write a new script to completely rewrite `test_synthesis.py` safely.
"""
