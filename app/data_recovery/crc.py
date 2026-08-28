from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .models import CRCResult

@dataclass(frozen=True)
class CRCParam:
    name: str
    width: int
    poly: int
    init: int
    xor_out: int
    reflect_in: bool
    reflect_out: bool

CRC_PRESETS: list[CRCParam] = [
    CRCParam("CRC-16-CCITT-FALSE", 16, 0x1021, 0xFFFF, 0x0000, False, False),
    CRCParam("CRC-16-IBM", 16, 0x8005, 0x0000, 0x0000, True, True),
    CRCParam("CRC-16-MODBUS", 16, 0x8005, 0xFFFF, 0x0000, True, True),
    CRCParam("CRC-16-XMODEM", 16, 0x1021, 0x0000, 0x0000, False, False),
    CRCParam("CRC-32-IEEE", 32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, True, True),
    CRCParam("CRC-32C", 32, 0x1EDC6F41, 0xFFFFFFFF, 0xFFFFFFFF, True, True),
    CRCParam("CRC-8-ATM", 8, 0x07, 0x00, 0x00, False, False),
    CRCParam("CRC-8-CCITT", 8, 0x07, 0x00, 0x55, False, False),
    CRCParam("CRC-24-LTE", 24, 0x864CFB, 0x000000, 0x000000, False, False),
]

def _reflect(val: int, width: int) -> int:
    res = 0
    for i in range(width):
        if (val >> i) & 1:
            res |= 1 << (width - 1 - i)
    return res

def compute_crc(
    data: bytes,
    width: int,
    poly: int,
    init: int,
    xor_out: int,
    reflect_in: bool = False,
    reflect_out: bool = False,
) -> int:
    """
    Compute parameterized CRC for arbitrary width (8, 16, 24, 32).

    Parameters
    ----------
    data : bytes
        Input byte stream.
    width : int
        CRC width in bits.
    poly : int
        CRC generator polynomial.
    init : int
        Initial shift register value.
    xor_out : int
        Final XOR value.
    reflect_in : bool
        Whether to reflect input bytes.
    reflect_out : bool
        Whether to reflect final CRC value.

    Returns
    -------
    crc : int
    """
    mask = (1 << width) - 1
    reg = init & mask

    for byte in data:
        if reflect_in:
            b_in = _reflect(byte, 8)
        else:
            b_in = byte

        reg ^= (b_in << (width - 8)) & mask
        for _ in range(8):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    if reflect_out:
        reg = _reflect(reg, width)

    return (reg ^ xor_out) & mask

def evaluate_frame_crc(
    payload_and_crc_bytes: bytes,
    param: CRCParam,
) -> CRCResult:
    """
    Evaluate candidate CRC parameter on a frame containing payload + trailing CRC field.

    Parameters
    ----------
    payload_and_crc_bytes : bytes
    param : CRCParam

    Returns
    -------
    CRCResult
    """
    crc_byte_len = (param.width + 7) // 8
    total_len = len(payload_and_crc_bytes)

    if total_len <= crc_byte_len:
        return CRCResult(
            crc_name=param.name,
            width=param.width,
            polynomial=param.poly,
            init_value=param.init,
            xor_out=param.xor_out,
            reflect_in=param.reflect_in,
            reflect_out=param.reflect_out,
            calculated_crc=0,
            expected_crc=0,
            is_valid=False,
            false_positive_p_value=1.0,
        )

    payload = payload_and_crc_bytes[:-crc_byte_len]
    crc_field = payload_and_crc_bytes[-crc_byte_len:]

    calc_crc = compute_crc(
        payload,
        width=param.width,
        poly=param.poly,
        init=param.init,
        xor_out=param.xor_out,
        reflect_in=param.reflect_in,
        reflect_out=param.reflect_out,
    )

    # Parse expected CRC integer from crc_field
    if param.reflect_out:
        # Little-endian field
        exp_crc = int.from_bytes(crc_field, byteorder="little")
    else:
        exp_crc = int.from_bytes(crc_field, byteorder="big")

    is_valid = bool(calc_crc == exp_crc)
    # Accidental match probability under random null
    p_val = 2.0 ** (-param.width)

    return CRCResult(
        crc_name=param.name,
        width=param.width,
        polynomial=param.poly,
        init_value=param.init,
        xor_out=param.xor_out,
        reflect_in=param.reflect_in,
        reflect_out=param.reflect_out,
        calculated_crc=calc_crc,
        expected_crc=exp_crc,
        is_valid=is_valid,
        false_positive_p_value=p_val,
    )

def search_crc_presets(
    payload_and_crc_bytes: bytes,
) -> list[CRCResult]:
    """
    Search library of standard CRC presets across a candidate frame.

    Parameters
    ----------
    payload_and_crc_bytes : bytes

    Returns
    -------
    results : list[CRCResult]
        Matching or evaluated CRC results.
    """
    results: list[CRCResult] = []
    for param in CRC_PRESETS:
        res = evaluate_frame_crc(payload_and_crc_bytes, param)
        results.append(res)

    results.sort(key=lambda r: (r.is_valid, r.width), reverse=True)
    return results
