import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .models import CRCMatch
from .native import require_native

@dataclass(frozen=True)
class CRCAlgorithm:
    name: str
    width: int
    poly: int
    init: int
    refin: bool
    refout: bool
    xorout: int

COMMON_CRCS = [
    CRCAlgorithm("CRC-8", 8, 0x07, 0x00, False, False, 0x00),
    CRCAlgorithm("CRC-16/CCITT-FALSE", 16, 0x1021, 0xFFFF, False, False, 0x0000),
    CRCAlgorithm("CRC-16/IBM", 16, 0x8005, 0x0000, True, True, 0x0000),
    CRCAlgorithm("CRC-32/IEEE", 32, 0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF)
]


def crc_algorithm(name: str) -> CRCAlgorithm:
    """Return a shipped CRC by its stable profile name.

    Discovery code must not silently substitute a different polynomial when a
    requested protocol profile is unavailable.
    """
    requested = str(name).upper()
    for algorithm in COMMON_CRCS:
        if algorithm.name.upper() == requested:
            return algorithm
    raise ValueError(f"unsupported CRC profile: {name}")

def _reflect(val: int, width: int) -> int:
    res = 0
    for i in range(width):
        if (val & (1 << i)) != 0:
            res |= (1 << (width - 1 - i))
    return res

_CRC_CACHE = {}

def _get_crc_tables(alg: CRCAlgorithm, max_len: int = 16384) -> Tuple[np.ndarray, np.ndarray]:
    if alg.name in _CRC_CACHE:
        T, Z = _CRC_CACHE[alg.name]
        if len(T) >= max_len:
            return T, Z
            
    T = np.zeros(max_len, dtype=np.uint32)
    Z = np.zeros(max_len, dtype=np.uint32)
    
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
        
    crc_T = poly_ref if alg.refin else alg.poly
    crc_Z = alg.init
    
    for d in range(max_len):
        val_T = crc_T
        val_Z = crc_Z
        if alg.refout != alg.refin:
            val_T = _reflect(val_T, alg.width)
            val_Z = _reflect(val_Z, alg.width)
            
        T[d] = val_T
        Z[d] = val_Z
        
        crc_T = step_zero(crc_T)
        crc_Z = step_zero(crc_Z)
        
    _CRC_CACHE[alg.name] = (T, Z)
    return T, Z

def compute_crc_bitwise(bits: np.ndarray, alg: CRCAlgorithm) -> int:
    """
    Computes CRC using GF(2) linear vectorization over the entire payload array.
    This avoids Python bit-by-bit loops and runs strictly in numpy.
    """
    native = require_native()
    config = native.CrcConfig()
    config.name = alg.name
    config.width = alg.width
    config.polynomial = alg.poly
    config.initial = alg.init
    config.reflect_input = alg.refin
    config.reflect_output = alg.refout
    config.xor_output = alg.xorout
    return int(native.crc_bits(np.ascontiguousarray(bits, dtype=np.uint8), config))


def verify_crc_at_boundary(
    bits: np.ndarray,
    start_idx: int,
    payload_bits: int,
    algorithm: CRCAlgorithm,
) -> Optional[CRCMatch]:
    """Verify one predeclared CRC boundary without a payload-length sweep."""
    if start_idx < 0 or payload_bits <= 0:
        return None
    crc_start = start_idx + payload_bits
    end_idx = crc_start + algorithm.width
    if end_idx > len(bits):
        return None

    payload = np.ascontiguousarray(bits[start_idx:crc_start], dtype=np.uint8)
    received = bits[crc_start:end_idx]
    received_crc = 0
    if algorithm.refin:
        for index, bit in enumerate(received):
            received_crc |= int(bit) << index
    else:
        for index, bit in enumerate(received):
            received_crc |= int(bit) << (algorithm.width - 1 - index)
    if compute_crc_bitwise(payload, algorithm) != received_crc:
        return None
    return CRCMatch(
        polynomial_hex=hex(algorithm.poly),
        polynomial_name=algorithm.name,
        bit_range_checked=(start_idx, end_idx),
        verified=True,
    )

def search_crcs(bits: np.ndarray, start_idx: int, max_search_bytes: int = 2048) -> List[CRCMatch]:
    """
    Given a known header boundary (start_idx), search downstream for a valid CRC.
    We assume the CRC is at the end of the payload.
    For MVP, we search common payload lengths or just sweep.
    Since sweeping all bit-lengths is O(N^2), we'll do a small sweep for lengths up to max_search_bytes.
    """
    matches = []
    
    # Explicit hard cap: skip search if the window would be pathologically large
    if max_search_bytes > 4096:
        max_search_bytes = 4096
        
    end_idx = min(len(bits), start_idx + max_search_bytes * 8)
    
    # Extract the maximum possible window once to avoid copying
    window = bits[start_idx:end_idx]
    
    for alg in COMMON_CRCS:
        crc_len = alg.width
        if len(window) <= crc_len:
            continue
            
        # We sweep possible payload lengths.
        # Often payloads are byte aligned (multiples of 8 bits).
        # We search byte-aligned lengths to keep it fast.
        for payload_bits in range(8, len(window) - crc_len + 1, 8):
            match = verify_crc_at_boundary(bits, start_idx, payload_bits, alg)
            if match is not None:
                matches.append(match)
                
    return matches
