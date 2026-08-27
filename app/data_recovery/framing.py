from __future__ import annotations
from typing import Any
import numpy as np
from .models import FrameBoundary, FrameCandidate, PreambleCandidate

def detect_frame_boundaries(
    bits: np.ndarray,
    preamble: PreambleCandidate | None = None,
    nominal_frame_len_bits: int = 1024,
) -> tuple[list[FrameBoundary], dict[str, Any]]:
    """
    Detect frame boundaries based on preamble repetition or periodic interval analysis.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.
    preamble : PreambleCandidate | None
        Preamble match results.
    nominal_frame_len_bits : int
        Fallback nominal frame length.

    Returns
    -------
    boundaries : list[FrameBoundary]
    periodicity_info : dict[str, Any]
    """
    n_bits = len(bits)
    if n_bits < 32:
        return [], {"is_periodic": False, "mean_interval": 0.0, "var_interval": 0.0}

    boundaries: list[FrameBoundary] = []

    if preamble is not None and len(preamble.match_indices) >= 2:
        indices = list(preamble.match_indices)
        spacings = np.diff(indices)
        mean_sp = float(np.mean(spacings))
        var_sp = float(np.var(spacings))

        for k in range(len(indices)):
            start_b = indices[k]
            if k < len(indices) - 1:
                end_b = indices[k + 1]
            else:
                end_b = min(n_bits, int(start_b + mean_sp))

            f_len = end_b - start_b
            is_valid = bool(abs(f_len - mean_sp) <= 16)
            boundaries.append(
                FrameBoundary(
                    frame_index=k + 1,
                    start_bit=start_b,
                    end_bit=end_b,
                    length_bits=f_len,
                    preamble_match=True,
                    is_valid_interval=is_valid,
                )
            )

        return boundaries, {
            "is_periodic": preamble.is_periodic,
            "mean_interval": round(mean_sp, 2),
            "var_interval": round(var_sp, 2),
            "num_frames": len(boundaries),
        }

    elif preamble is not None and len(preamble.match_indices) == 1:
        start_b = preamble.match_indices[0]
        end_b = n_bits
        boundaries.append(
            FrameBoundary(
                frame_index=1,
                start_bit=start_b,
                end_bit=end_b,
                length_bits=end_b - start_b,
                preamble_match=True,
                is_valid_interval=True,
            )
        )
        return boundaries, {
            "is_periodic": False,
            "mean_interval": float(end_b - start_b),
            "var_interval": 0.0,
            "num_frames": 1,
        }

    else:
        # Generic single chunk boundary fallback
        boundaries.append(
            FrameBoundary(
                frame_index=1,
                start_bit=0,
                end_bit=n_bits,
                length_bits=n_bits,
                preamble_match=False,
                is_valid_interval=True,
            )
        )
        return boundaries, {
            "is_periodic": False,
            "mean_interval": float(n_bits),
            "var_interval": 0.0,
            "num_frames": 1,
        }

def slice_frames(
    bits: np.ndarray,
    boundaries: list[FrameBoundary],
    header_len_bits: int = 32,
    crc_len_bits: int = 16,
) -> list[FrameCandidate]:
    """
    Slice bitstream into FrameCandidate structures with header, payload, and CRC divisions.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.
    boundaries : list[FrameBoundary]
        Frame boundary ranges.
    header_len_bits : int
        Nominal header length in bits.
    crc_len_bits : int
        Nominal CRC length in bits.

    Returns
    -------
    frames : list[FrameCandidate]
    """
    frames: list[FrameCandidate] = []

    for b in boundaries:
        frame_raw = bits[b.start_bit : b.end_bit]
        f_len = len(frame_raw)

        if f_len < (header_len_bits + crc_len_bits):
            header = frame_raw
            payload = np.array([], dtype=np.uint8)
            crc_bits = np.array([], dtype=np.uint8)
        else:
            header = frame_raw[:header_len_bits]
            crc_bits = frame_raw[-crc_len_bits:]
            payload = frame_raw[header_len_bits:-crc_len_bits]

        # Extract sequence number and length field candidates if header >= 16 bits
        seq_num: int | None = None
        len_val: int | None = None

        if len(header) >= 16:
            # Check first 16 bits as length candidate
            h_bytes = np.packbits(header)
            if len(h_bytes) >= 2:
                len_candidate = int(h_bytes[0]) * 256 + int(h_bytes[1])
                # If length candidate plausibly matches payload length in bytes
                if 0 < len_candidate <= (f_len // 8):
                    len_val = len_candidate

            if len(h_bytes) >= 4:
                seq_num = int(h_bytes[2]) * 256 + int(h_bytes[3])
            elif len(h_bytes) >= 1:
                seq_num = int(h_bytes[0])

        frames.append(
            FrameCandidate(
                frame_index=b.frame_index,
                raw_bits=frame_raw,
                header_bits=header,
                payload_bits=payload,
                crc_bits=crc_bits,
                fec_bits=np.array([], dtype=np.uint8),
                start_bit=b.start_bit,
                end_bit=b.end_bit,
                is_length_consistent=b.is_valid_interval,
                is_crc_valid=False,
                is_fec_corrected=False,
                sequence_number=seq_num,
                length_field_value=len_val,
                decoded_payload=bytes(np.packbits(payload)) if len(payload) >= 8 else None,
            )
        )

    return frames

def detect_sequence_continuity(
    frames: list[FrameCandidate],
) -> tuple[bool, list[int], list[int]]:
    """
    Evaluate sequence number continuity across frames to detect missing/lost frames.

    Parameters
    ----------
    frames : list[FrameCandidate]

    Returns
    -------
    is_continuous : bool
    observed_seqs : list[int]
    missing_seqs : list[int]
    """
    seqs = [f.sequence_number for f in frames if f.sequence_number is not None]
    if len(seqs) < 2:
        return False, seqs, []

    diffs = np.diff(seqs)
    missing: list[int] = []
    for i, d in enumerate(diffs):
        if d > 1:
            for m in range(seqs[i] + 1, seqs[i] + d):
                missing.append(m)

    is_continuous = bool(len(missing) == 0 and np.all(diffs == 1))
    return is_continuous, seqs, missing
