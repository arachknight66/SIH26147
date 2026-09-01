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

alg = CRCAlgorithm("CRC-16/CCITT-FALSE", 16, 0x1021, 0xFFFF, False, False, 0x0000)

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

# Precompute tables
MAX_LEN = 1000
T = np.zeros(MAX_LEN, dtype=np.uint32)
Z = np.zeros(MAX_LEN, dtype=np.uint32)

# We can precompute by just running the original algorithm!
# To find T[d], it's the CRC of a 1 followed by d zeros, with init=0, xorout=0.
alg_zero = CRCAlgorithm(alg.name, alg.width, alg.poly, 0, alg.refin, alg.refout, 0)
for d in range(MAX_LEN):
    arr = np.zeros(d + 1, dtype=np.uint8)
    arr[0] = 1
    T[d] = compute_crc_bitwise_original(arr, alg_zero)
    
    # Z[L] is the CRC of L zeros with the real init, xorout=0
    arr_z = np.zeros(d, dtype=np.uint8) # length L=d
    Z[d] = compute_crc_bitwise_original(arr_z, CRCAlgorithm(alg.name, alg.width, alg.poly, alg.init, alg.refin, alg.refout, 0))

def compute_crc_vectorized(bits: np.ndarray, alg: CRCAlgorithm) -> int:
    L = len(bits)
    if L == 0:
        return alg.init ^ alg.xorout
    
    # reverse the bits for distance indexing: bits[i] has distance L-1-i
    # so bits[::-1][d] has distance d
    rev = bits[::-1]
    
    # XOR sum of T[d] for all d where rev[d] == 1
    # plus Z[L]
    # plus xorout
    # BUT wait, the bit order for refin!
    # If refin is True, does it change the linearity? 
    # Yes, but T[d] and Z[L] were computed with the SAME refin! So it should perfectly match.
    
    # Let's compute
    crc_payload = 0
    if np.any(rev == 1):
        crc_payload = np.bitwise_xor.reduce(T[:L][rev == 1])
        
    final_crc = crc_payload ^ Z[L] ^ alg.xorout
    return final_crc

# Test
np.random.seed(42)
for _ in range(10):
    bits = np.random.randint(0, 2, np.random.randint(10, 500)).astype(np.uint8)
    c1 = compute_crc_bitwise_original(bits, alg)
    c2 = compute_crc_vectorized(bits, alg)
    assert c1 == c2, f"Failed! {hex(c1)} != {hex(c2)}"

print("Vectorized CRC works perfectly!")
