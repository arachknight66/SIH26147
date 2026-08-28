from __future__ import annotations
import numpy as np
from .models import FECCodeFamily, FECDecodeResult, FECHypothesis

def _parity_byte(val: int) -> int:
    """Calculate parity of integer."""
    p = 0
    while val:
        p ^= (val & 1)
        val >>= 1
    return p

def encode_convolutional(
    bits: np.ndarray,
    k: int = 7,
    g1: int = 0o133,
    g2: int = 0o171,
) -> np.ndarray:
    """
    Encode binary bits with rate 1/2 convolutional code.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.
    k : int
        Constraint length (3 or 7).
    g1 : int
        Octal generator polynomial 1.
    g2 : int
        Octal generator polynomial 2.

    Returns
    -------
    encoded : np.ndarray
        1D uint8 binary stream of length 2 * (len(bits) + k - 1).
    """
    n_bits = len(bits)
    # Add k-1 flush tail bits
    padded = np.concatenate((bits, np.zeros(k - 1, dtype=np.uint8)))
    encoded = np.zeros(len(padded) * 2, dtype=np.uint8)

    state = 0
    for i, b in enumerate(padded):
        state = ((state << 1) | int(b)) & ((1 << k) - 1)
        c0 = _parity_byte(state & g1)
        c1 = _parity_byte(state & g2)
        encoded[2 * i] = c0
        encoded[2 * i + 1] = c1

    return encoded

def viterbi_decode(
    input_bits: np.ndarray,
    soft_bits: np.ndarray | None = None,
    k: int = 7,
    g1: int = 0o133,
    g2: int = 0o171,
    max_correction_fraction: float = 0.10,
) -> FECDecodeResult:
    """
    Viterbi decoding for rate 1/2 convolutional code with soft/hard branch metrics.

    Parameters
    ----------
    input_bits : np.ndarray
        1D uint8 hard decisions.
    soft_bits : np.ndarray | None
        1D float32 soft LLR decisions (positive for 1, negative for 0).
    k : int
        Constraint length (3 or 7).
    g1 : int
        Generator 1.
    g2 : int
        Generator 2.
    max_correction_fraction : float
        Maximum allowed correction budget.

    Returns
    -------
    FECDecodeResult
    """
    num_pairs = len(input_bits) // 2
    if num_pairs < k:
        return FECDecodeResult(
            input_bits=input_bits,
            decoded_bits=input_bits.copy(),
            correction_mask=np.zeros(len(input_bits), dtype=bool),
            corrected_bit_count=0,
            correction_fraction=0.0,
            path_metric=0.0,
            normalized_path_metric=0.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.CONVOLUTIONAL,
            valid=False,
        )

    num_states = 1 << (k - 1)  # 64 for K=7, 4 for K=3
    state_mask = num_states - 1

    # Precompute branch outputs for every (state, input_bit)
    # branch_outputs[state, in_bit] = (c0, c1)
    outputs = np.zeros((num_states, 2, 2), dtype=np.uint8)
    for s in range(num_states):
        for in_bit in (0, 1):
            full_state = ((s << 1) | in_bit)
            outputs[s, in_bit, 0] = _parity_byte(full_state & g1)
            outputs[s, in_bit, 1] = _parity_byte(full_state & g2)

    # Path metrics initialized
    path_metrics = np.full(num_states, 1e9, dtype=np.float32)
    path_metrics[0] = 0.0  # Start at state 0

    # History traceback matrix: [step, state] -> (prev_state, in_bit)
    history = np.zeros((num_pairs, num_states), dtype=np.uint8)

    use_soft = (soft_bits is not None and len(soft_bits) >= num_pairs * 2)

    for t in range(num_pairs):
        r0 = input_bits[2 * t]
        r1 = input_bits[2 * t + 1]

        if use_soft:
            s0 = soft_bits[2 * t]
            s1 = soft_bits[2 * t + 1]

        new_metrics = np.full(num_states, 1e9, dtype=np.float32)

        # For each prev_state and input_bit, compute next_state and branch metric
        for s in range(num_states):
            curr_pm = path_metrics[s]
            if curr_pm > 1e8:
                continue

            for in_bit in (0, 1):
                next_state = ((s << 1) | in_bit) & state_mask
                c0 = outputs[s, in_bit, 0]
                c1 = outputs[s, in_bit, 1]

                if use_soft:
                    # Soft metric: squared distance to target (c0, c1 in {-1, +1})
                    # target: 0 -> -1.0, 1 -> +1.0
                    t0 = 1.0 if c0 == 1 else -1.0
                    t1 = 1.0 if c1 == 1 else -1.0
                    bm = (s0 - t0) ** 2 + (s1 - t1) ** 2
                else:
                    # Hard Hamming metric
                    bm = float((r0 != c0) + (r1 != c1))

                cand_metric = curr_pm + bm
                if cand_metric < new_metrics[next_state]:
                    new_metrics[next_state] = cand_metric
                    history[t, next_state] = s | (in_bit << 7)

        path_metrics = new_metrics

    # Traceback from minimum metric state (or state 0 if terminated)
    best_state = int(np.argmin(path_metrics))
    min_metric = float(path_metrics[best_state])

    decoded_bits_rev: list[int] = []
    curr_state = best_state

    for t in range(num_pairs - 1, -1, -1):
        packed = history[t, curr_state]
        prev_s = packed & 0x7F
        in_b = (packed >> 7) & 1
        decoded_bits_rev.append(in_b)
        curr_state = prev_s

    decoded_full = np.array(decoded_bits_rev[::-1], dtype=np.uint8)
    # Remove tail bits (k-1)
    info_bits = decoded_full[: num_pairs - (k - 1)] if num_pairs > (k - 1) else decoded_full

    # Re-encode to compute correction mask on input bitstream
    reencoded = encode_convolutional(info_bits, k=k, g1=g1, g2=g2)
    min_len = min(len(reencoded), len(input_bits))
    correction_mask = np.zeros(len(input_bits), dtype=bool)
    if min_len > 0:
        correction_mask[:min_len] = (reencoded[:min_len] != input_bits[:min_len])
    if len(input_bits) > min_len:
        correction_mask[min_len:] = True
    
    corrected_count = int(np.sum(correction_mask))
    correction_frac = float(corrected_count / len(input_bits)) if len(input_bits) > 0 else 0.0
    norm_metric = float(min_metric / max(1, num_pairs))

    is_over = (correction_frac > max_correction_fraction)

    return FECDecodeResult(
        input_bits=input_bits,
        decoded_bits=info_bits,
        correction_mask=correction_mask,
        corrected_bit_count=corrected_count,
        correction_fraction=round(correction_frac, 4),
        path_metric=round(min_metric, 2),
        normalized_path_metric=round(norm_metric, 4),
        is_overcorrected=is_over,
        code_family=FECCodeFamily.CONVOLUTIONAL,
        valid=not is_over,
    )

def decode_hamming_7_4(
    input_bits: np.ndarray,
    max_correction_fraction: float = 0.10,
) -> FECDecodeResult:
    """
    Syndrome decoding for (7, 4) Hamming code.

    Parameters
    ----------
    input_bits : np.ndarray
        1D uint8 binary stream.
    max_correction_fraction : float

    Returns
    -------
    FECDecodeResult
    """
    n_blocks = len(input_bits) // 7
    if n_blocks == 0:
        return FECDecodeResult(
            input_bits=input_bits,
            decoded_bits=input_bits.copy(),
            correction_mask=np.zeros(len(input_bits), dtype=bool),
            corrected_bit_count=0,
            correction_fraction=0.0,
            path_metric=0.0,
            normalized_path_metric=0.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.HAMMING,
            valid=False,
        )

    # Parity check matrix H (3x7)
    # H columns: [1..7] in binary
    h_mat = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)

    corrected_blocks = input_bits[: n_blocks * 7].copy().reshape(n_blocks, 7)
    correction_mask_blocks = np.zeros((n_blocks, 7), dtype=bool)

    info_bits_list: list[int] = []

    for b in range(n_blocks):
        block = corrected_blocks[b]
        # Syndrome S = H * block^T mod 2
        syndrome = (h_mat @ block) % 2
        syn_val = syndrome[0] * 1 + syndrome[1] * 2 + syndrome[2] * 4

        if syn_val > 0 and syn_val <= 7:
            # 1-indexed error position
            err_pos = syn_val - 1
            block[err_pos] ^= 1
            correction_mask_blocks[b, err_pos] = True

        # Extract 4 data bits: positions 2, 4, 5, 6 (0-indexed)
        info_bits_list.extend([block[2], block[4], block[5], block[6]])

    corr_mask = correction_mask_blocks.ravel()
    corr_count = int(np.sum(corr_mask))
    corr_frac = float(corr_count / (n_blocks * 7))

    return FECDecodeResult(
        input_bits=input_bits,
        decoded_bits=np.array(info_bits_list, dtype=np.uint8),
        correction_mask=corr_mask,
        corrected_bit_count=corr_count,
        correction_fraction=round(corr_frac, 4),
        path_metric=float(corr_count),
        normalized_path_metric=round(corr_frac, 4),
        is_overcorrected=bool(corr_frac > max_correction_fraction),
        code_family=FECCodeFamily.HAMMING,
        valid=bool(corr_frac <= max_correction_fraction),
    )

def decode_repetition(
    input_bits: np.ndarray,
    r: int = 3,
    max_correction_fraction: float = 0.10,
) -> FECDecodeResult:
    """
    Majority vote decoding for rate 1/R repetition code.

    Parameters
    ----------
    input_bits : np.ndarray
    r : int
        Repetition factor (3 or 5).
    max_correction_fraction : float

    Returns
    -------
    FECDecodeResult
    """
    n_blocks = len(input_bits) // r
    if n_blocks == 0:
        return FECDecodeResult(
            input_bits=input_bits,
            decoded_bits=input_bits.copy(),
            correction_mask=np.zeros(len(input_bits), dtype=bool),
            corrected_bit_count=0,
            correction_fraction=0.0,
            path_metric=0.0,
            normalized_path_metric=0.0,
            is_overcorrected=False,
            code_family=FECCodeFamily.REPETITION,
            valid=False,
        )

    blocks = input_bits[: n_blocks * r].reshape(n_blocks, r)
    sums = np.sum(blocks, axis=1)
    majority = (sums > (r / 2)).astype(np.uint8)

    reencoded = np.repeat(majority, r)
    corr_mask = (reencoded != input_bits[: n_blocks * r])
    corr_count = int(np.sum(corr_mask))
    corr_frac = float(corr_count / (n_blocks * r))

    return FECDecodeResult(
        input_bits=input_bits,
        decoded_bits=majority,
        correction_mask=corr_mask,
        corrected_bit_count=corr_count,
        correction_fraction=round(corr_frac, 4),
        path_metric=float(corr_count),
        normalized_path_metric=round(corr_frac, 4),
        is_overcorrected=bool(corr_frac > max_correction_fraction),
        code_family=FECCodeFamily.REPETITION,
        valid=bool(corr_frac <= max_correction_fraction),
    )
