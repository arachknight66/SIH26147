import numpy as np
import pytest
from app.data_recovery.crc import (
    CRC_PRESETS,
    compute_crc,
    evaluate_frame_crc,
    search_crc_presets,
)

def test_compute_crc_16_ccitt():
    data = b"123456789"
    # Standard CRC-16-CCITT False check value is 0x29B1
    crc = compute_crc(data, width=16, poly=0x1021, init=0xFFFF, xor_out=0x0000, reflect_in=False, reflect_out=False)
    assert crc == 0x29B1

def test_compute_crc_32_ieee():
    data = b"123456789"
    # Standard CRC-32 (IEEE 802.3) check value is 0xCBF43926
    crc = compute_crc(data, width=32, poly=0x04C11DB7, init=0xFFFFFFFF, xor_out=0xFFFFFFFF, reflect_in=True, reflect_out=True)
    assert crc == 0xCBF43926

def test_evaluate_frame_crc_valid():
    payload = b"Hello SIH26147"
    crc_val = compute_crc(payload, width=16, poly=0x1021, init=0xFFFF, xor_out=0x0000)
    frame = payload + crc_val.to_bytes(2, byteorder="big")

    param = CRC_PRESETS[0]  # CRC-16-CCITT-FALSE
    res = evaluate_frame_crc(frame, param)
    assert res.is_valid is True
    assert res.calculated_crc == crc_val

def test_search_crc_presets():
    payload = b"Telemetry Data"
    crc_val = compute_crc(payload, width=16, poly=0x1021, init=0xFFFF, xor_out=0x0000)
    frame = payload + crc_val.to_bytes(2, byteorder="big")

    results = search_crc_presets(frame)
    matching = [r for r in results if r.is_valid]
    assert len(matching) >= 1
    assert matching[0].crc_name == "CRC-16-CCITT-FALSE"
