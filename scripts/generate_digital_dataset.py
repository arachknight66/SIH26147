from __future__ import annotations
from typing import Any
import numpy as np
from app.data_recovery.crc import compute_crc
from app.data_recovery.fec_decode import encode_convolutional
from app.data_recovery.scrambling import descramble_lfsr

def generate_digital_stream(
    protocol: str = "PROTOCOL_A",
    num_frames: int = 5,
    payload_len_bytes: int = 32,
    ber: float = 0.0,
    bit_offset: int = 0,
    invert_polarity: bool = False,
    burst_error_len: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Generate synthetic digital streams for testing Phase 5 reconstruction and error correction.

    Parameters
    ----------
    protocol : str
        'PROTOCOL_A', 'PROTOCOL_B', 'PROTOCOL_C', 'PROTOCOL_D', 'PROTOCOL_E', 'OOD_RANDOM'
    num_frames : int
        Number of consecutive frames.
    payload_len_bytes : int
        Payload size per frame in bytes.
    ber : float
        Target bit error rate.
    bit_offset : int
        Leading dummy bits (0..7).
    invert_polarity : bool
        Whether to invert all bits.
    burst_error_len : int
        Length of a burst error to inject.
    seed : int
        Random seed.

    Returns
    -------
    rx_bits : np.ndarray
        1D uint8 received hard bits.
    rx_soft : np.ndarray
        1D float32 received soft LLRs (+/- 1.0 + noise).
    manifest : dict[str, Any]
        Ground-truth manifest.
    """
    rng = np.random.default_rng(seed)
    prot_upper = protocol.upper().strip()

    if prot_upper == "OOD_RANDOM":
        total_len = num_frames * (payload_len_bytes + 8) * 8
        raw_bits = rng.integers(0, 2, total_len, dtype=np.uint8)
        soft = np.where(raw_bits == 1, 1.0, -1.0).astype(np.float32) + rng.normal(0, 0.2, total_len).astype(np.float32)
        return raw_bits, soft, {"protocol": "OOD_RANDOM", "is_structured": False}

    frame_bit_list: list[np.ndarray] = []
    ground_truth_payloads: list[bytes] = []

    for f_idx in range(num_frames):
        # 1. Generate payload
        payload_bytes = bytes(rng.integers(32, 126, payload_len_bytes, dtype=np.uint8))
        ground_truth_payloads.append(payload_bytes)

        if prot_upper in ("PROTOCOL_A", "PROTOCOL_C", "PROTOCOL_D"):
            # Preamble: 0x2DD4 (16-bit)
            preamble_bytes = bytes.fromhex("2dd4")
            # Length field: 16-bit big-endian
            len_field = payload_len_bytes.to_bytes(2, byteorder="big")
            # CRC-16-CCITT (False) over payload
            crc_val = compute_crc(payload_bytes, width=16, poly=0x1021, init=0xFFFF, xor_out=0x0000)
            crc_bytes = crc_val.to_bytes(2, byteorder="big")

            raw_frame_bytes = preamble_bytes + len_field + payload_bytes + crc_bytes
            frame_bits = np.unpackbits(np.frombuffer(raw_frame_bytes, dtype=np.uint8))
            frame_bit_list.append(frame_bits)

        elif prot_upper in ("PROTOCOL_B", "PROTOCOL_E"):
            # Preamble: 0x1ACFFC1D (32-bit)
            preamble_bytes = bytes.fromhex("1acffc1d")
            seq_field = (f_idx + 1).to_bytes(2, byteorder="big")
            len_field = payload_len_bytes.to_bytes(2, byteorder="big")
            # CRC-32-IEEE over payload
            crc_val = compute_crc(payload_bytes, width=32, poly=0x04C11DB7, init=0xFFFFFFFF, xor_out=0xFFFFFFFF, reflect_in=True, reflect_out=True)
            crc_bytes = crc_val.to_bytes(4, byteorder="little")

            raw_frame_bytes = preamble_bytes + seq_field + len_field + payload_bytes + crc_bytes
            frame_bits = np.unpackbits(np.frombuffer(raw_frame_bytes, dtype=np.uint8))
            frame_bit_list.append(frame_bits)

        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    tx_bits = np.concatenate(frame_bit_list)

    # Apply Stream-level Scrambling & FEC transformations
    if prot_upper == "PROTOCOL_C":
        tx_bits = encode_convolutional(tx_bits, k=7, g1=0o133, g2=0o171)
    elif prot_upper == "PROTOCOL_D":
        tx_bits = descramble_lfsr(tx_bits, taps=(7, 4))
    elif prot_upper == "PROTOCOL_E":
        tx_scrambled = descramble_lfsr(tx_bits, taps=(7, 4))
        tx_bits = encode_convolutional(tx_scrambled, k=7, g1=0o133, g2=0o171)

    tx_len = len(tx_bits)

    # 2. Inject Bit Errors (AWGN-like channel)
    rx_bits = tx_bits.copy()
    if ber > 0.0:
        error_mask = rng.random(tx_len) < ber
        rx_bits[error_mask] ^= 1

    # 3. Inject Burst Errors
    if burst_error_len > 0 and tx_len > burst_error_len * 2:
        burst_start = tx_len // 2
        rx_bits[burst_start : burst_start + burst_error_len] ^= 1

    # 4. Invert Polarity if requested
    if invert_polarity:
        rx_bits = (1 - rx_bits).astype(np.uint8)

    # 5. Prepend Bit Offset
    if bit_offset > 0:
        dummy_bits = rng.integers(0, 2, bit_offset, dtype=np.uint8)
        rx_bits = np.concatenate((dummy_bits, rx_bits))

    # 6. Generate Soft LLR Proxies
    # LLR: +1.0 for 1, -1.0 for 0, with AWGN noise
    soft_clean = np.where(rx_bits == 1, 1.0, -1.0).astype(np.float32)
    noise_sigma = float(np.sqrt(max(1e-4, ber))) if ber > 0.0 else 0.1
    rx_soft = soft_clean + rng.normal(0, noise_sigma, len(rx_bits)).astype(np.float32)

    manifest: dict[str, Any] = {
        "protocol": prot_upper,
        "num_frames": num_frames,
        "payload_len_bytes": payload_len_bytes,
        "ber": ber,
        "bit_offset": bit_offset,
        "invert_polarity": invert_polarity,
        "burst_error_len": burst_error_len,
        "ground_truth_payloads": ground_truth_payloads,
        "clean_tx_bits": tx_bits,
    }

    return rx_bits, rx_soft, manifest
