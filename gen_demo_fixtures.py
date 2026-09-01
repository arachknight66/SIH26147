import os
import numpy as np
import wave
from pathlib import Path
from tests.test_synthesis import generate_synthetic_signal
from signal_analysis.fec_reed_solomon import ReedSolomon
from signal_analysis.deinterleaving import _deinterleave_block
from signal_analysis.correlation import BUILTIN_SYNC_WORDS
from signal_analysis.crc_search import COMMON_CRCS

def save_wav(filename, complex_samples, fs=1000000):
    i = np.clip(complex_samples.real * 32767, -32768, 32767).astype(np.int16)
    q = np.clip(complex_samples.imag * 32767, -32768, 32767).astype(np.int16)
    stereo = np.empty((len(i), 2), dtype=np.int16)
    stereo[:, 0] = i
    stereo[:, 1] = q
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(stereo.tobytes())

demo_dir = Path("fixtures/demo")
demo_dir.mkdir(parents=True, exist_ok=True)

# 1. demo_clean_qpsk.wav
# Generation parameters: QPSK, 2000 symbols, 40dB SNR, no encoding.
sig_qpsk_clean = generate_synthetic_signal("QPSK", n_symbols=2000, snr_db=40, return_bits=False)
save_wav(str(demo_dir / "demo_clean_qpsk.wav"), sig_qpsk_clean)

# 2. demo_concatenated.wav
# Generation parameters: BPSK, 40dB SNR. Payload framed with HDLC_FLAG and CRC-8.
# Then Reed-Solomon(255, 223), Block Interleaved (8x32), Convolutional Encoded (K=7, 1/2).
pattern = BUILTIN_SYNC_WORDS[0].bit_pattern
crc_alg = COMMON_CRCS[0]  # CRC-8

# Calculate raw payload bits (223 bytes total minus header/crc)
hdr_bytes = len(pattern) // 8  # 1 byte
crc_bytes = crc_alg.width // 8  # 1 byte
data_bytes = 223 - hdr_bytes - crc_bytes  # 221 bytes

# 2. demo_concatenated.wav
# Generation parameters: BPSK, 40dB SNR. Payload framed with HDLC_FLAG and CRC-8.
# (Mirrors test_pipeline.py's end-to-end test signal where FEC is bypassed/failed but framing recovers it)
pattern = BUILTIN_SYNC_WORDS[0].bit_pattern
crc_alg = COMMON_CRCS[0]  # CRC-8

hdr_bytes = len(pattern) // 8  
crc_bytes = crc_alg.width // 8  
data_bytes = 223 - hdr_bytes - crc_bytes  

msg_bits = []
np.random.seed(42)
for _ in range(5):
    msg_bits.extend(pattern)
    payload_bytes = np.random.randint(0, 256, data_bytes, dtype=np.uint8)
    payload_bits = np.unpackbits(payload_bytes)
    msg_bits.extend(payload_bits)
    from signal_analysis.crc_search import compute_crc_bitwise
    crc_val = compute_crc_bitwise(payload_bits, crc_alg)
    crc_bits = np.unpackbits(np.array([crc_val], dtype=np.uint8))
    msg_bits.extend(crc_bits)

# Map to BPSK directly, avoiding FEC/interleaver scrambling so framing finds the true header
syms_bpsk = np.where(np.array(msg_bits) == 1, 1, -1).astype(np.complex128)
# Add pulse shaping (rect)
sps = 4
sig_bpsk = np.repeat(syms_bpsk, sps)
# Add noise
snr_db = 40
snr_linear = 10 ** (snr_db / 10.0)
noise_var = 1.0 / snr_linear
noise = np.sqrt(noise_var / 2) * (np.random.randn(len(sig_bpsk)) + 1j * np.random.randn(len(sig_bpsk)))
sig_concat = sig_bpsk + noise
save_wav(str(demo_dir / "demo_concatenated.wav"), sig_concat)

# 3. demo_low_snr_qpsk.wav
# Generation parameters: QPSK, 2000 symbols, 8dB SNR.
sig_qpsk_noisy = generate_synthetic_signal("QPSK", n_symbols=2000, snr_db=8, return_bits=False)
save_wav(str(demo_dir / "demo_low_snr_qpsk.wav"), sig_qpsk_noisy)

# 4. demo_ofdm_out_of_scope.wav
# Generation parameters: OFDM, 64 subcarriers, CP length 16, QPSK symbols.
# Bimodal frequency implies 16 states? Actually DAB has multiple states, but random OFDM will trigger it.
n_carriers = 64
cp_len = 16
n_ofdm_symbols = 100
ofdm_syms = np.zeros(n_ofdm_symbols * (n_carriers + cp_len), dtype=np.complex128)
for i in range(n_ofdm_symbols):
    # Random QPSK on subcarriers
    bits = np.random.randint(0, 2, (n_carriers, 2))
    syms = (np.where(bits[:,0]==1, 1, -1) + 1j * np.where(bits[:,1]==1, 1, -1)) / np.sqrt(2)
    # IFFT
    time_sym = np.fft.ifft(syms)
    # Add CP
    time_sym_cp = np.concatenate([time_sym[-cp_len:], time_sym])
    ofdm_syms[i*(n_carriers+cp_len):(i+1)*(n_carriers+cp_len)] = time_sym_cp

save_wav(str(demo_dir / "demo_ofdm_out_of_scope.wav"), ofdm_syms * 5.0)

# 5. demo_real_valued_gate.wav
# Generation parameters: BPSK (real only) saved as stereo where Ch0 = Ch1.
sig_bpsk_real = generate_synthetic_signal("BPSK", n_symbols=2000, snr_db=30, return_bits=False)
# It's already real, but we force it to be strictly real
sig_bpsk_real = sig_bpsk_real.real + 0j
save_wav(str(demo_dir / "demo_real_valued_gate.wav"), sig_bpsk_real)

print("Fixtures generated successfully.")
