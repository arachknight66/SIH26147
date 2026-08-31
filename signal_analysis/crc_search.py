import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .models import CRCMatch

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

def _reflect(val: int, width: int) -> int:
    res = 0
    for i in range(width):
        if (val & (1 << i)) != 0:
            res |= (1 << (width - 1 - i))
    return res

def compute_crc_bitwise(bits: np.ndarray, alg: CRCAlgorithm) -> int:
    """
    Computes CRC by processing the bit array directly. 
    This avoids byte alignment assumptions which is critical for raw bitstreams 
    where the payload length might not be a multiple of 8.
    """
    crc = alg.init
    
    for b in bits:
        # If refin is True, we process the LSB of bytes first.
        # But we are taking a raw bitstream. The "refin" standard implies byte-level reflection.
        # For a pure bitstream, what does refin mean? 
        # Typically, it means the bits on the wire are sent LSB first.
        # Our `bits` array is already the order bits arrived on the wire.
        # So we process them exactly in the order they appear.
        # For standards where refin=True, the wire order IS the reflection, 
        # so we feed bits as they arrive, into the MSB or LSB of the register?
        # Actually, standard bitwise CRC:
        bit_val = int(b)
        
        # Standard unreflected:
        # MSB of CRC is tested. If CRC MSB ^ bit == 1, shift left and XOR poly.
        # Reflected: 
        # LSB of CRC is tested. If CRC LSB ^ bit == 1, shift right and XOR reversed poly.
        
        if alg.refin:
            # Shift right
            crc_lsb = crc & 1
            crc >>= 1
            if crc_lsb ^ bit_val:
                crc ^= _reflect(alg.poly, alg.width)
        else:
            # Shift left
            crc_msb = (crc >> (alg.width - 1)) & 1
            crc = (crc << 1) & ((1 << alg.width) - 1)
            if crc_msb ^ bit_val:
                crc ^= alg.poly
                
    if alg.refout != alg.refin:
        crc = _reflect(crc, alg.width)
        
    crc ^= alg.xorout
    return crc

def search_crcs(bits: np.ndarray, start_idx: int, max_search_bytes: int = 2048) -> List[CRCMatch]:
    """
    Given a known header boundary (start_idx), search downstream for a valid CRC.
    We assume the CRC is at the end of the payload.
    For MVP, we search common payload lengths or just sweep.
    Since sweeping all bit-lengths is O(N^2), we'll do a small sweep for lengths up to max_search_bytes.
    """
    matches = []
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
            payload = window[:payload_bits]
            crc_received_bits = window[payload_bits : payload_bits + crc_len]
            
            # Pack received bits into an integer to compare
            # Depending on refin, the CRC field on the wire might be LSB first.
            rx_crc = 0
            if alg.refin:
                # If refin, bits were sent LSB first. So bit 0 is LSB of byte 0.
                for i, b in enumerate(crc_received_bits):
                    rx_crc |= (int(b) << i)
            else:
                # MSB first
                for i, b in enumerate(crc_received_bits):
                    rx_crc |= (int(b) << (crc_len - 1 - i))
                    
            computed_crc = compute_crc_bitwise(payload, alg)
            
            if computed_crc == rx_crc:
                matches.append(CRCMatch(
                    polynomial_hex=hex(alg.poly),
                    polynomial_name=alg.name,
                    bit_range_checked=(start_idx, start_idx + payload_bits + crc_len),
                    verified=True
                ))
                
    return matches
