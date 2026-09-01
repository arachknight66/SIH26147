import os
import numpy as np
import wave
from pathlib import Path
from tests.test_synthesis import generate_synthetic_signal
from signal_analysis.fec_reed_solomon import ReedSolomon
from signal_analysis.deinterleaving import _deinterleave_block
from signal_analysis.correlation import BUILTIN_SYNC_WORDS
from signal_analysis.crc_search import COMMON_CRCS
from signal_analysis.demodulation import CONSTELLATION_MAPS

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

demo_dir = Path(__file__).parent / "fixtures" / "demo"
demo_dir.mkdir(parents=True, exist_ok=True)

# 1. demo_qam_clean.wav
# Generation parameters: 16-QAM, 2000 symbols, 30dB SNR.
sig_qam_clean = generate_synthetic_signal("16-QAM", n_symbols=2000, snr_db=30, return_bits=False)
save_wav(str(demo_dir / "demo_qam_clean.wav"), sig_qam_clean)

# 2. demo_qam_low_snr.wav
# Generation parameters: 16-QAM, 2000 symbols, 10dB SNR.
sig_qam_low_snr = generate_synthetic_signal("16-QAM", n_symbols=2000, snr_db=10, return_bits=False)
save_wav(str(demo_dir / "demo_qam_low_snr.wav"), sig_qam_low_snr)

# 3. demo_qam_concatenated.wav
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

# Pad msg_bits to a multiple of 4 for 16-QAM
pad = (4 - (len(msg_bits) % 4)) % 4
msg_bits_padded = msg_bits + [0] * pad

pts = CONSTELLATION_MAPS["16-QAM"]["points"]
bts = CONSTELLATION_MAPS["16-QAM"]["bits"]
bit_to_pt = {tuple(b): p for b, p in zip(bts, pts)}

syms_16qam = []
for i in range(0, len(msg_bits_padded), 4):
    b = tuple(msg_bits_padded[i:i+4])
    syms_16qam.append(bit_to_pt[b])
syms_16qam = np.array(syms_16qam, dtype=np.complex128)

sps = 4
sig_qam_concat = np.repeat(syms_16qam, sps)
snr_db = 40
noise_var = 1.0 / (10 ** (snr_db / 10.0))
noise = np.sqrt(noise_var / 2) * (np.random.randn(len(sig_qam_concat)) + 1j * np.random.randn(len(sig_qam_concat)))
sig_qam_concat += noise
save_wav(str(demo_dir / "demo_qam_concatenated.wav"), sig_qam_concat)


# 4. demo_qam_unsupported_order.wav (64-QAM)
# 64-QAM points
pts_1d = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
pts_2d = [x + 1j*y for x in pts_1d for y in pts_1d]
pts_64qam = np.array(pts_2d, dtype=np.complex128) / np.sqrt(42)
np.random.seed(43)
idx = np.random.randint(0, 64, size=2000)
syms_64 = pts_64qam[idx]
sig_64 = np.repeat(syms_64, sps)
noise_var = 1.0 / (10 ** (30 / 10.0))
noise = np.sqrt(noise_var / 2) * (np.random.randn(len(sig_64)) + 1j * np.random.randn(len(sig_64)))
sig_64 += noise
save_wav(str(demo_dir / "demo_qam_unsupported_order.wav"), sig_64)

# 5. demo_qam_cfo_capture.wav
# Generation parameters: 16-QAM, 2000 symbols, 30dB SNR, cfo_norm=0.01
sig_qam_cfo = generate_synthetic_signal("16-QAM", n_symbols=2000, snr_db=30, cfo_norm=0.01, return_bits=False)
save_wav(str(demo_dir / "demo_qam_cfo_capture.wav"), sig_qam_cfo)

print("QAM fixtures generated successfully.")
