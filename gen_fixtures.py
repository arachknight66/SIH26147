import numpy as np
import wave
from tests.test_synthesis import generate_synthetic_signal
from signal_analysis.fec_reed_solomon import ReedSolomon
from signal_analysis.deinterleaving import _deinterleave_block
from signal_analysis.correlation import BUILTIN_SYNC_WORDS

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

# 1. Clean QPSK
sig_qpsk = generate_synthetic_signal("QPSK", n_symbols=2000, snr_db=40, return_bits=False)
save_wav("test_clean_qpsk.wav", sig_qpsk)

# 2. Concatenated (1 interleaver block = 8 * 255 = 2040 bits = 1 RS block)
pattern = BUILTIN_SYNC_WORDS[0].bit_pattern
rs = ReedSolomon(255, 223)

msg_bits = []
msg_bits.extend(pattern)
payload_bytes = np.random.randint(0, 256, 223 - 8, dtype=np.uint8)
msg_bits.extend(np.unpackbits(payload_bytes))

block_bytes = np.packbits(msg_bits)
enc = rs.encode(block_bytes.tolist())
encoded_bits = np.unpackbits(np.array(enc, dtype=np.uint8)).tolist()

# Interleave (Rows=8, Cols=255)
interleaved = _deinterleave_block(np.array(encoded_bits), 8, 255, read_by_row=False)

# Convolutional Encode K=7, 1/2
POLY_1 = 0o171
POLY_2 = 0o133
K = 7
conv_bits = []
state = 0
for b in interleaved:
    state = (b << (K - 1)) | state
    out1 = bin(state & POLY_1).count('1') % 2
    out2 = bin(state & POLY_2).count('1') % 2
    conv_bits.extend([out1, out2])
    state >>= 1

# Modulate BPSK
symbols = np.where(np.array(conv_bits) == 1, 1.0, -1.0) + 0j
sps = 4
up = np.zeros(len(symbols) * sps, dtype=np.complex128)
up[::sps] = symbols
p = np.ones(sps)
sig_concat = np.convolve(up, p, mode='same')
sig_concat += (np.random.randn(len(sig_concat)) + 1j * np.random.randn(len(sig_concat))) * 0.1

save_wav("test_encoded.wav", sig_concat)
print("Fixtures created.")
