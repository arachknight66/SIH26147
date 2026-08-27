from __future__ import annotations
import numpy as np
from .models import BitOrder, ByteStreamCandidate

_PRINTABLE_SET = set(range(32, 127)) | {9, 10, 13}

def convert_bits_to_bytes(
    bits: np.ndarray,
    bit_offset: int = 0,
    bit_order: BitOrder = BitOrder.MSB_FIRST,
) -> ByteStreamCandidate:
    """
    Convert a 1D bitstream array into bytes under a given bit offset (0-7) and bit order (MSB/LSB-first).

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary array.
    bit_offset : int
        Bit offset in 0..7.
    bit_order : BitOrder
        MSB_FIRST or LSB_FIRST.

    Returns
    -------
    ByteStreamCandidate
    """
    n_total = len(bits)
    if bit_offset >= n_total:
        return ByteStreamCandidate(
            bytes_data=b"",
            bit_offset=bit_offset,
            bit_order=bit_order,
            bit_count=0,
            entropy=0.0,
            bit_balance=0.0,
            printable_ratio=0.0,
            provenance={"error": "bit_offset exceeds length"},
        )

    sliced = bits[bit_offset:]
    n_bytes = len(sliced) // 8
    if n_bytes == 0:
        return ByteStreamCandidate(
            bytes_data=b"",
            bit_offset=bit_offset,
            bit_order=bit_order,
            bit_count=len(sliced),
            entropy=0.0,
            bit_balance=float(np.mean(sliced)) if len(sliced) > 0 else 0.0,
            printable_ratio=0.0,
        )

    usable_bits = sliced[: n_bytes * 8]
    
    if bit_order == BitOrder.LSB_FIRST:
        # Reverse bits within each 8-bit octet
        octets = usable_bits.reshape(-1, 8)[:, ::-1].ravel()
        byte_arr = np.packbits(octets)
    else:
        byte_arr = np.packbits(usable_bits)

    bytes_data = bytes(byte_arr)

    # 1. Entropy
    counts = np.bincount(byte_arr, minlength=256)
    probs = counts[counts > 0] / n_bytes
    entropy = float(-np.sum(probs * np.log2(probs)))

    # 2. Bit Balance
    bit_balance = float(np.mean(usable_bits))

    # 3. Printable ASCII Ratio
    printable_count = sum(1 for b in bytes_data if b in _PRINTABLE_SET)
    printable_ratio = float(printable_count / n_bytes) if n_bytes > 0 else 0.0

    return ByteStreamCandidate(
        bytes_data=bytes_data,
        bit_offset=bit_offset,
        bit_order=bit_order,
        bit_count=n_bytes * 8,
        entropy=round(entropy, 4),
        bit_balance=round(bit_balance, 4),
        printable_ratio=round(printable_ratio, 4),
        provenance={"n_bytes": n_bytes, "bit_offset": bit_offset, "bit_order": bit_order.value},
    )

def generate_byte_stream_candidates(
    bits: np.ndarray,
    search_lsb: bool = True,
) -> list[ByteStreamCandidate]:
    """
    Generate ByteStream candidates across all 8 bit offsets and bit orders.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary array.
    search_lsb : bool
        Whether to evaluate LSB_FIRST in addition to MSB_FIRST.

    Returns
    -------
    candidates : list[ByteStreamCandidate]
    """
    candidates: list[ByteStreamCandidate] = []
    if len(bits) < 8:
        return []

    for offset in range(8):
        cand_msb = convert_bits_to_bytes(bits, bit_offset=offset, bit_order=BitOrder.MSB_FIRST)
        if len(cand_msb.bytes_data) > 0:
            candidates.append(cand_msb)

        if search_lsb:
            cand_lsb = convert_bits_to_bytes(bits, bit_offset=offset, bit_order=BitOrder.LSB_FIRST)
            if len(cand_lsb.bytes_data) > 0:
                candidates.append(cand_lsb)

    return candidates
