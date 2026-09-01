import numpy as np

def _reflect(val: int, width: int) -> int:
    res = 0
    for i in range(width):
        if (val & (1 << i)) != 0:
            res |= (1 << (width - 1 - i))
    return res

class CRCAlgorithm:
    def __init__(self, name, width, poly, init, refin, refout, xorout):
        self.name = name
        self.width = width
        self.poly = poly
        self.init = init
        self.refin = refin
        self.refout = refout
        self.xorout = xorout

alg = CRCAlgorithm("CRC-16/IBM", 16, 0x8005, 0x0000, True, True, 0x0000)

def compute_crc_bitwise_original(bits: np.ndarray, alg: CRCAlgorithm) -> int:
    crc = alg.init
    for b in bits:
        bit_val = int(b)
        if alg.refin:
            crc_lsb = crc & 1
            crc >>= 1
            if crc_lsb ^ bit_val:
                crc ^= _reflect(alg.poly, alg.width)
        else:
            crc_msb = (crc >> (alg.width - 1)) & 1
            crc = (crc << 1) & ((1 << alg.width) - 1)
            if crc_msb ^ bit_val:
                crc ^= alg.poly
    if alg.refout != alg.refin:
        crc = _reflect(crc, alg.width)
    crc ^= alg.xorout
    return crc

MAX_LEN = 16384
T = np.zeros(MAX_LEN, dtype=np.uint32)
Z = np.zeros(MAX_LEN, dtype=np.uint32)

# Fast O(N) precomputation
# 1. We track the raw LFSR state for T and Z
if alg.refin:
    poly_ref = _reflect(alg.poly, alg.width)

def step_zero(crc):
    if alg.refin:
        crc_lsb = crc & 1
        crc >>= 1
        if crc_lsb:
            crc ^= poly_ref
    else:
        crc_msb = (crc >> (alg.width - 1)) & 1
        crc = (crc << 1) & ((1 << alg.width) - 1)
        if crc_msb:
            crc ^= alg.poly
    return crc

# T[0] is a single '1' bit (no trailing zeros)
crc_T = 0
if alg.refin:
    crc_T = poly_ref # shift in a 1 (0 ^ 1 = 1) -> 0 >> 1 ^ poly_ref
else:
    crc_T = alg.poly # shift in a 1 (0 ^ 1 = 1) -> 0 << 1 ^ poly

# Z[0] is 0 trailing zeros from init
crc_Z = alg.init

for d in range(MAX_LEN):
    # Store with refout applied if needed!
    # Wait, the final CRC applies refout if refout != refin.
    # So we must apply that mapping before storing in T and Z!
    
    val_T = crc_T
    val_Z = crc_Z
    if alg.refout != alg.refin:
        val_T = _reflect(val_T, alg.width)
        val_Z = _reflect(val_Z, alg.width)
        
    T[d] = val_T
    Z[d] = val_Z
    
    crc_T = step_zero(crc_T)
    crc_Z = step_zero(crc_Z)

def compute_crc_vectorized(bits: np.ndarray, alg: CRCAlgorithm) -> int:
    L = len(bits)
    if L == 0:
        return alg.init ^ alg.xorout
    rev = bits[::-1]
    crc_payload = 0
    if np.any(rev == 1):
        crc_payload = np.bitwise_xor.reduce(T[:L][rev == 1])
    return crc_payload ^ Z[L] ^ alg.xorout

# Test
np.random.seed(42)
for _ in range(10):
    bits = np.random.randint(0, 2, np.random.randint(10, 500)).astype(np.uint8)
    c1 = compute_crc_bitwise_original(bits, alg)
    c2 = compute_crc_vectorized(bits, alg)
    assert c1 == c2, f"Failed! {hex(c1)} != {hex(c2)}"

print("Fast Vectorized CRC works perfectly!")
