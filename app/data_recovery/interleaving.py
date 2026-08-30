from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from .crc import search_crc_presets
from .framing import detect_frame_boundaries, slice_frames
from .integrity import evaluate_multi_frame_integrity
from .models import (
    DataRecoveryConfig,
    InterleaverHypothesis,
    InterleaverType,
)
from .scrambling import STANDARD_LFSR_POLYNOMIALS, berlekamp_massey, generate_lfsr_sequence
from .synchronization import detect_preamble_candidates

# Standard candidate parameterizations for bounded blind search
STANDARD_BLOCK_CONFIGS: list[tuple[int, int]] = [
    # (span/cols, depth/rows)
    (8, 4),
    (4, 8),
    (8, 8),
    (16, 8),
    (8, 16),
    (16, 16),
    (8, 32),
    (12, 17),
]

STANDARD_CONVOLUTIONAL_CONFIGS: list[tuple[int, int]] = [
    # (branches M, delay_increment D)
    (4, 1),
    (4, 2),
    (6, 2),
    (8, 1),
    (8, 2),
    (12, 17),
]

STANDARD_DIAGONAL_CONFIGS: list[tuple[int, int, int]] = [
    # (span/cols C, depth/rows R, step S)
    (8, 8, 1),
    (8, 16, 1),
    (16, 8, 1),
    (8, 8, 2),
    (16, 16, 1),
]

STANDARD_PSEUDO_RANDOM_BLOCK_SIZES: list[int] = [64, 128, 256]


# =============================================================================
# 1. PURE INVERTIBLE TRANSFORMS
# =============================================================================

# --- 1.1 Block Interleaver & De-interleaver ---

def interleave_block(bits: np.ndarray, span: int, depth: int) -> np.ndarray:
    """
    Apply generic block interleaving (matrix row-write, column-read transpose).

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.
    span : int
        Number of matrix columns (span C).
    depth : int
        Number of matrix rows (depth R).

    Returns
    -------
    interleaved : np.ndarray
    """
    n = len(bits)
    block_size = span * depth
    if n < block_size or span <= 0 or depth <= 0:
        return bits.copy()

    n_blocks = n // block_size
    usable_len = n_blocks * block_size
    out = np.zeros(n, dtype=np.uint8)

    for b in range(n_blocks):
        sub = bits[b * block_size : (b + 1) * block_size]
        # Reshape row-major (R, C) and read column-major (C, R) -> flatten
        mat = sub.reshape((depth, span))
        out[b * block_size : (b + 1) * block_size] = mat.T.reshape(-1)

    # Remainder bits pass through unchanged
    if usable_len < n:
        out[usable_len:] = bits[usable_len:]

    return out


def deinterleave_block(bits: np.ndarray, span: int, depth: int) -> np.ndarray:
    """
    Apply exact inverse block de-interleaving (matrix column-write, row-read transpose).

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.
    span : int
        Number of matrix columns (span C).
    depth : int
        Number of matrix rows (depth R).

    Returns
    -------
    deinterleaved : np.ndarray
    """
    n = len(bits)
    block_size = span * depth
    if n < block_size or span <= 0 or depth <= 0:
        return bits.copy()

    n_blocks = n // block_size
    usable_len = n_blocks * block_size
    out = np.zeros(n, dtype=np.uint8)

    for b in range(n_blocks):
        sub = bits[b * block_size : (b + 1) * block_size]
        # Inverse of row-write/col-read: reshape as (span, depth) and transpose back to (depth, span)
        mat = sub.reshape((span, depth))
        out[b * block_size : (b + 1) * block_size] = mat.T.reshape(-1)

    if usable_len < n:
        out[usable_len:] = bits[usable_len:]

    return out


def get_block_permutation_map(span: int, depth: int) -> tuple[int, ...]:
    """Return forward index permutation map for a single block."""
    block_size = span * depth
    indices = np.arange(block_size, dtype=np.int32)
    mat = indices.reshape((depth, span))
    return tuple(mat.T.reshape(-1).tolist())


# --- 1.2 Convolutional Interleaver & De-interleaver ---

def interleave_convolutional(
    bits: np.ndarray,
    branches: int = 4,
    delay_increment: int = 1,
) -> np.ndarray:
    """
    Apply Ramsey/Forney convolutional interleaver: bank of M shift registers with
    linearly increasing delays d_i = i * D, commutated in round-robin fashion.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.
    branches : int
        Number of parallel branches (M).
    delay_increment : int
        Delay increment in symbols/bits (D).

    Returns
    -------
    interleaved : np.ndarray
    """
    n = len(bits)
    if n == 0 or branches <= 1 or delay_increment <= 0:
        return bits.copy()

    # Shift registers for each branch i in 0..M-1
    # Branch i has delay i * D
    delays = [i * delay_increment for i in range(branches)]
    buffers = [np.zeros(d, dtype=np.uint8) for d in delays]
    ptrs = [0] * branches

    out = np.zeros(n, dtype=np.uint8)

    for idx in range(n):
        branch = idx % branches
        d = delays[branch]
        x_val = bits[idx]

        if d == 0:
            out[idx] = x_val
        else:
            ptr = ptrs[branch]
            out[idx] = buffers[branch][ptr]
            buffers[branch][ptr] = x_val
            ptrs[branch] = (ptr + 1) % d

    return out


def deinterleave_convolutional(
    bits: np.ndarray,
    branches: int = 4,
    delay_increment: int = 1,
) -> np.ndarray:
    """
    Apply exact complementary Ramsey/Forney convolutional de-interleaver:
    Bank of M shift registers with complementary delays d_i' = (M - 1 - i) * D.

    Mathematical Invariant:
    For every branch i, d_i + d_i' = (M - 1) * D = constant.
    Every bit experiences identical total round-trip latency Delta = M * (M - 1) * D bits.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary stream.
    branches : int
        Number of parallel branches (M).
    delay_increment : int
        Delay increment in symbols/bits (D).

    Returns
    -------
    deinterleaved : np.ndarray
    """
    n = len(bits)
    if n == 0 or branches <= 1 or delay_increment <= 0:
        return bits.copy()

    # Complementary delays: (M - 1 - i) * D
    delays = [(branches - 1 - i) * delay_increment for i in range(branches)]
    buffers = [np.zeros(d, dtype=np.uint8) for d in delays]
    ptrs = [0] * branches

    out = np.zeros(n, dtype=np.uint8)

    for idx in range(n):
        branch = idx % branches
        d = delays[branch]
        y_val = bits[idx]

        if d == 0:
            out[idx] = y_val
        else:
            ptr = ptrs[branch]
            out[idx] = buffers[branch][ptr]
            buffers[branch][ptr] = y_val
            ptrs[branch] = (ptr + 1) % d

    return out


def get_convolutional_latency(branches: int, delay_increment: int) -> int:
    """Return total end-to-end latency in bits for convolutional interleaver + de-interleaver."""
    return branches * (branches - 1) * delay_increment


# --- 1.3 Diagonal Interleaver & De-interleaver ---

def _build_diagonal_permutation(span: int, depth: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Construct forward and inverse permutation vectors for diagonal interleaver."""
    block_size = span * depth
    forward_map = np.zeros(block_size, dtype=np.int32)

    for k in range(block_size):
        r = k // span
        c = k % span
        # Diagonal shifted coordinate
        c_diag = (c + r * step) % span
        out_idx = c_diag * depth + r
        forward_map[k] = out_idx

    inv_map = np.zeros(block_size, dtype=np.int32)
    for k in range(block_size):
        inv_map[forward_map[k]] = k

    return forward_map, inv_map


def interleave_diagonal(bits: np.ndarray, span: int, depth: int, step: int = 1) -> np.ndarray:
    """
    Apply generic diagonal interleaving: bits are mapped along matrix diagonals and read out column-major.

    Parameters
    ----------
    bits : np.ndarray
    span : int
        Number of matrix columns (C).
    depth : int
        Number of matrix rows (R).
    step : int
        Diagonal step offset (S).

    Returns
    -------
    interleaved : np.ndarray
    """
    n = len(bits)
    block_size = span * depth
    if n < block_size or span <= 0 or depth <= 0:
        return bits.copy()

    fwd_map, _ = _build_diagonal_permutation(span, depth, step)
    n_blocks = n // block_size
    usable_len = n_blocks * block_size
    out = np.zeros(n, dtype=np.uint8)

    for b in range(n_blocks):
        sub = bits[b * block_size : (b + 1) * block_size]
        out[b * block_size + fwd_map] = sub

    if usable_len < n:
        out[usable_len:] = bits[usable_len:]

    return out


def deinterleave_diagonal(bits: np.ndarray, span: int, depth: int, step: int = 1) -> np.ndarray:
    """
    Apply exact inverse diagonal de-interleaving.

    Parameters
    ----------
    bits : np.ndarray
    span : int
    depth : int
    step : int

    Returns
    -------
    deinterleaved : np.ndarray
    """
    n = len(bits)
    block_size = span * depth
    if n < block_size or span <= 0 or depth <= 0:
        return bits.copy()

    _, inv_map = _build_diagonal_permutation(span, depth, step)
    n_blocks = n // block_size
    usable_len = n_blocks * block_size
    out = np.zeros(n, dtype=np.uint8)

    for b in range(n_blocks):
        sub = bits[b * block_size : (b + 1) * block_size]
        out[b * block_size + inv_map] = sub

    if usable_len < n:
        out[usable_len:] = bits[usable_len:]

    return out


# --- 1.4 Pseudo-Random Interleaver & De-interleaver ---

def _build_pseudorandom_permutation(
    taps: Sequence[int],
    block_size: int,
    init_state: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct deterministic bijective forward and inverse permutation vectors from an LFSR sequence.
    """
    # Generate sufficient PRBS sequence to produce unique integer rankings
    prbs = generate_lfsr_sequence(taps, block_size * 16, init_state=init_state)
    # Form 16-bit pseudo-random words for each position
    words = np.zeros(block_size, dtype=np.uint32)
    for i in range(block_size):
        chunk = prbs[i * 16 : (i + 1) * 16]
        val = 0
        for bit_idx, b in enumerate(chunk):
            val |= (int(b) << bit_idx)
        words[i] = val

    # Stable argsort defines deterministic bijective permutation of 0..block_size-1
    fwd_map = np.argsort(words, kind="stable").astype(np.int32)
    inv_map = np.zeros(block_size, dtype=np.int32)
    for i in range(block_size):
        inv_map[fwd_map[i]] = i

    return fwd_map, inv_map


def interleave_pseudorandom(
    bits: np.ndarray,
    taps: Sequence[int],
    init_state: Sequence[int] | None = None,
    block_size: int = 128,
) -> np.ndarray:
    """
    Apply deterministic pseudo-random permutation interleaving generated via LFSR sequence.

    Parameters
    ----------
    bits : np.ndarray
    taps : Sequence[int]
    init_state : Sequence[int] | None
    block_size : int

    Returns
    -------
    interleaved : np.ndarray
    """
    n = len(bits)
    if n < block_size or block_size <= 1:
        return bits.copy()

    fwd_map, _ = _build_pseudorandom_permutation(taps, block_size, init_state)
    n_blocks = n // block_size
    usable_len = n_blocks * block_size
    out = np.zeros(n, dtype=np.uint8)

    for b in range(n_blocks):
        sub = bits[b * block_size : (b + 1) * block_size]
        out[b * block_size : (b + 1) * block_size] = sub[fwd_map]

    if usable_len < n:
        out[usable_len:] = bits[usable_len:]

    return out


def deinterleave_pseudorandom(
    bits: np.ndarray,
    taps: Sequence[int],
    init_state: Sequence[int] | None = None,
    block_size: int = 128,
) -> np.ndarray:
    """
    Apply exact inverse pseudo-random de-interleaving.

    Parameters
    ----------
    bits : np.ndarray
    taps : Sequence[int]
    init_state : Sequence[int] | None
    block_size : int

    Returns
    -------
    deinterleaved : np.ndarray
    """
    n = len(bits)
    if n < block_size or block_size <= 1:
        return bits.copy()

    _, inv_map = _build_pseudorandom_permutation(taps, block_size, init_state)
    n_blocks = n // block_size
    usable_len = n_blocks * block_size
    out = np.zeros(n, dtype=np.uint8)

    for b in range(n_blocks):
        sub = bits[b * block_size : (b + 1) * block_size]
        out[b * block_size : (b + 1) * block_size] = sub[inv_map]

    if usable_len < n:
        out[usable_len:] = bits[usable_len:]

    return out


# =============================================================================
# 2. BOUNDED HYPOTHESIS GENERATION & MULTI-EVIDENCE SCORING
# =============================================================================

def evaluate_deinterleaved_stream_metrics(
    deint_bits: np.ndarray,
    reference_preambles: list[np.ndarray] | None = None,
) -> tuple[float, bool, int, float, bool, int]:
    """
    Evaluate structural evidence on a de-interleaved bitstream.

    Returns
    -------
    (preamble_conf, is_periodic, valid_crc_count, crc_valid_fraction, is_crc_valid, linear_complexity)
    """
    if len(deint_bits) < 16:
        return 0.0, False, 0, 0.0, False, 0

    # 1. Preamble & Framing Detection
    preambles = detect_preamble_candidates(deint_bits)
    best_p = preambles[0] if preambles else None
    boundaries, p_info = detect_frame_boundaries(deint_bits, preamble=best_p)
    is_per = bool(p_info.get("is_periodic", False))
    p_conf = float(best_p.confidence) if best_p else 0.0

    # Also check if deinterleaving reveals any explicit reference preambles
    if reference_preambles:
        for ref_p in reference_preambles:
            if len(ref_p) <= len(deint_bits):
                for i in range(len(deint_bits) - len(ref_p) + 1):
                    if np.sum(deint_bits[i : i + len(ref_p)] != ref_p) == 0:
                        p_conf = max(p_conf, 0.90)
                        is_per = True
                        break

    # 2. CRC Integrity Verification
    frames = slice_frames(deint_bits, boundaries)
    integ = evaluate_multi_frame_integrity(frames)
    crc_count = integ.valid_frame_count
    crc_frac = integ.crc_valid_fraction
    crc_valid = integ.valid

    # 3. Linear Complexity (Berlekamp-Massey)
    # Note: Pure permutations do not inherently alter linear complexity significantly.
    # This metric serves as a secondary sanity/null-hypothesis discriminator.
    lc = berlekamp_massey(deint_bits[: min(512, len(deint_bits))])

    return p_conf, is_per, crc_count, crc_frac, crc_valid, lc


def generate_interleaver_hypotheses(
    bits: np.ndarray,
    config: DataRecoveryConfig | None = None,
    reference_preambles: list[np.ndarray] | None = None,
) -> list[InterleaverHypothesis]:
    """
    Generate and evaluate bounded candidate de-interleaving hypotheses across all four families
    plus the null (no-interleaving) baseline.

    Epistemic Multi-Evidence Contract:
    A de-interleaving hypothesis is NEVER accepted on a single improved metric alone.
    Promotion to valid=True and confidence >= 0.70 requires corroboration across at least
    two independent structural signals (e.g. periodic preamble AND valid CRC).

    Parameters
    ----------
    bits : np.ndarray
        Input binary stream.
    config : DataRecoveryConfig | None
        Configuration and search limits.
    reference_preambles : list[np.ndarray] | None
        Optional reference sync words.

    Returns
    -------
    hypotheses : list[InterleaverHypothesis]
    """
    cfg = config or DataRecoveryConfig()
    n = len(bits)

    if n < 32:
        return [
            InterleaverHypothesis(
                interleaver_type=InterleaverType.NONE,
                parameters={},
                permutation_map=None,
                confidence=1.0,
                entropy_improvement=0.0,
                structural_improvement=0.0,
                valid=True,
                assumptions=("Stream too short for interleaver analysis; null hypothesis retained.",),
            )
        ]

    hyps: list[InterleaverHypothesis] = []

    # -------------------------------------------------------------
    # 0. Baseline Null Hypothesis (No Interleaving)
    # -------------------------------------------------------------
    base_p_conf, base_is_per, base_crc, base_crc_frac, base_crc_valid, base_lc = evaluate_deinterleaved_stream_metrics(
        bits, reference_preambles=reference_preambles
    )

    null_conf = 0.90 if (base_is_per and base_crc_valid) else (0.70 if base_is_per else 0.50)
    hyps.append(
        InterleaverHypothesis(
            interleaver_type=InterleaverType.NONE,
            parameters={},
            permutation_map=None,
            confidence=null_conf,
            entropy_improvement=0.0,
            structural_improvement=0.0,
            valid=True,
            assumptions=("Baseline null hypothesis: transmission has no bit-level interleaving.",),
        )
    )

    if not cfg.enable_deinterleaver:
        return hyps

    def _score_and_create_hypothesis(
        deint_bits: np.ndarray,
        i_type: InterleaverType,
        params: dict[str, Any],
        perm_map: tuple[int, ...] | None,
        descr_str: str,
    ) -> InterleaverHypothesis:
        cand_p_conf, cand_is_per, cand_crc, cand_crc_frac, cand_crc_valid, cand_lc = evaluate_deinterleaved_stream_metrics(
            deint_bits, reference_preambles=reference_preambles
        )

        p_gain = cand_p_conf - base_p_conf
        crc_gain = float(cand_crc - base_crc)
        struct_gain = float(np.clip(p_gain * 0.5 + min(crc_gain, 3.0) * 0.25, -1.0, 1.0))
        entropy_gain = float(np.clip((base_lc - cand_lc) / max(1, base_lc), -1.0, 1.0))

        # Multi-evidence corroboration rule:
        # Require BOTH periodic framing (with high confidence) AND multi-frame statistically valid CRC
        # (valid CRC fraction >= 50% or multi_frame_p_value < 0.05) with significant gain over baseline.
        is_corroborated = (
            cand_is_per
            and cand_p_conf >= 0.70
            and cand_crc >= 2
            and cand_crc_frac >= 0.50
            and cand_crc_valid
            and (cand_crc > base_crc or not base_is_per)
        )
        is_partial = (cand_is_per and not base_is_per) or (cand_crc > base_crc)

        if is_corroborated:
            conf = float(np.clip(0.80 + 0.15 * min(cand_crc / 3.0, 1.0), 0.75, 0.98))
            is_valid = True
            assump_text = (
                f"De-interleaving via {descr_str} restored both periodic framing (conf={cand_p_conf:.2f}) and valid CRC matches ({cand_crc}/{int(cand_crc / max(1e-3, cand_crc_frac))}).",
                "Cross-validated across 2 independent structural evidence signals.",
            )
        elif is_partial:
            # Single-metric improvement is capped at ambiguous to prevent look-elsewhere false positives
            conf = float(np.clip(0.35 + 0.15 * max(p_gain, crc_gain / 5.0), 0.30, 0.55))
            is_valid = False
            assump_text = (
                f"Candidate {descr_str} showed partial metric gain without dual-evidence corroboration; capped at ambiguous.",
            )
        else:
            conf = 0.10
            is_valid = False
            assump_text = (
                f"Candidate {descr_str} produced no measurable structural or integrity improvement.",
            )

        return InterleaverHypothesis(
            interleaver_type=i_type,
            parameters=params,
            permutation_map=perm_map,
            confidence=round(conf, 3),
            entropy_improvement=round(entropy_gain, 3),
            structural_improvement=round(struct_gain, 3),
            valid=is_valid,
            assumptions=assump_text,
        )

    # -------------------------------------------------------------
    # 1. Block Interleaver Search
    # -------------------------------------------------------------
    for span, depth in STANDARD_BLOCK_CONFIGS:
        if span * depth > n:
            continue
        deint = deinterleave_block(bits, span=span, depth=depth)
        perm = get_block_permutation_map(span, depth)
        hyp = _score_and_create_hypothesis(
            deint,
            InterleaverType.BLOCK,
            {"span": span, "depth": depth, "block_size": span * depth},
            perm,
            f"block interleaver (span={span}, depth={depth})",
        )
        if hyp.confidence > 0.20 or hyp.structural_improvement > 0.0:
            hyps.append(hyp)

    # -------------------------------------------------------------
    # 2. Convolutional Interleaver Search
    # -------------------------------------------------------------
    for branches, delay_inc in STANDARD_CONVOLUTIONAL_CONFIGS:
        lat = get_convolutional_latency(branches, delay_inc)
        if lat >= n:
            continue
        deint = deinterleave_convolutional(bits, branches=branches, delay_increment=delay_inc)
        # Sliced after latency alignment
        aligned_deint = deint[lat:] if len(deint) > lat + 32 else deint
        hyp = _score_and_create_hypothesis(
            aligned_deint,
            InterleaverType.CONVOLUTIONAL,
            {"branches": branches, "delay_increment": delay_inc, "latency_bits": lat},
            None,
            f"convolutional interleaver (branches={branches}, delay={delay_inc})",
        )
        if hyp.confidence > 0.20 or hyp.structural_improvement > 0.0:
            hyps.append(hyp)

    # -------------------------------------------------------------
    # 3. Diagonal Interleaver Search
    # -------------------------------------------------------------
    for span, depth, step in STANDARD_DIAGONAL_CONFIGS:
        if span * depth > n:
            continue
        deint = deinterleave_diagonal(bits, span=span, depth=depth, step=step)
        fwd_map, _ = _build_diagonal_permutation(span, depth, step)
        hyp = _score_and_create_hypothesis(
            deint,
            InterleaverType.DIAGONAL,
            {"span": span, "depth": depth, "step": step, "block_size": span * depth},
            tuple(fwd_map.tolist()),
            f"diagonal interleaver (span={span}, depth={depth}, step={step})",
        )
        if hyp.confidence > 0.20 or hyp.structural_improvement > 0.0:
            hyps.append(hyp)

    # -------------------------------------------------------------
    # 4. Pseudo-Random Interleaver Search
    # -------------------------------------------------------------
    for name, taps, deg in STANDARD_LFSR_POLYNOMIALS[:3]:
        for b_size in STANDARD_PSEUDO_RANDOM_BLOCK_SIZES:
            if b_size > n:
                continue
            deint = deinterleave_pseudorandom(bits, taps=taps, block_size=b_size)
            fwd_map, _ = _build_pseudorandom_permutation(taps, b_size)
            hyp = _score_and_create_hypothesis(
                deint,
                InterleaverType.PSEUDO_RANDOM,
                {"polynomial_name": name, "taps": taps, "block_size": b_size},
                tuple(fwd_map.tolist()),
                f"pseudo-random interleaver ({name}, block_size={b_size})",
            )
            if hyp.confidence > 0.20 or hyp.structural_improvement > 0.0:
                hyps.append(hyp)

    # Sort hypotheses: valid (corroborated) first, then by confidence and structural improvement
    hyps.sort(key=lambda h: (1 if h.valid and h.interleaver_type != InterleaverType.NONE else 0, h.confidence, h.structural_improvement), reverse=True)

    return hyps[: cfg.max_interleaver_hypotheses]