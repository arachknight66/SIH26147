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
#alg = CRCAlgorithm("CRC-32/IEEE", 32, 0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF)

def compute_crc_bitwise(bits: np.ndarray, alg: CRCAlgorithm) -> int:
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

bits = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])

crc_ref = compute_crc_bitwise(bits, alg)
print(f"Ref: {hex(crc_ref)}")

# Now using precomputed arrays!
L = len(bits)
# Let's precompute the effect of a 1 at each distance from the end.
# A 1 at distance d (0 is the last bit) means it was fed into the LFSR and shifted d times.
# BUT wait! If refin=True, the LFSR shifts right.
# Let's just track the state transition as a GF(2) linear map (matrix multiplication).
# Even simpler: Just write a vectorized byte-wise or fast bitwise in numpy?
