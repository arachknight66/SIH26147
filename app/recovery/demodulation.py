from __future__ import annotations
import numpy as np
from .models import BitStreamStatus, DemodulationResult, ModulationFamily

def demodulate_bpsk(symbols: np.ndarray) -> DemodulationResult:
    """
    Demodulate BPSK symbols: hard bits, soft LLR decisions, and constellation indices.

    Parameters
    ----------
    symbols : np.ndarray
        1-SPS normalized symbol samples.

    Returns
    -------
    DemodulationResult
    """
    n = len(symbols)
    if n == 0:
        return DemodulationResult(
            hard_bits=np.array([], dtype=np.uint8),
            soft_decisions=np.array([], dtype=np.float32),
            symbol_indices=np.array([], dtype=np.int32),
            bit_stream_status=BitStreamStatus.UNAVAILABLE,
            valid=False,
        )

    re_vals = np.real(symbols)
    # Binary decision: Re(z) > 0 -> 1, Re(z) <= 0 -> 0
    hard_bits = (re_vals > 0.0).astype(np.uint8)
    symbol_indices = hard_bits.astype(np.int32)
    soft_decisions = (2.0 * re_vals).astype(np.float32)

    return DemodulationResult(
        hard_bits=hard_bits,
        soft_decisions=soft_decisions,
        symbol_indices=symbol_indices,
        bit_stream_status=BitStreamStatus.AVAILABLE,
        bit_polarity="unresolved",
        fec_status="not_applied",
        mapping_scheme="natural",
        valid=True,
    )

def demodulate_qpsk(
    symbols: np.ndarray,
    mapping: str = "gray",
) -> DemodulationResult:
    """
    Demodulate QPSK symbols: 2 bits per symbol, Gray mapping, and soft decisions.

    Parameters
    ----------
    symbols : np.ndarray
        1-SPS normalized symbol samples.
    mapping : str
        'gray' or 'natural'.

    Returns
    -------
    DemodulationResult
    """
    n = len(symbols)
    if n == 0:
        return DemodulationResult(
            hard_bits=np.array([], dtype=np.uint8),
            soft_decisions=np.array([], dtype=np.float32),
            symbol_indices=np.array([], dtype=np.int32),
            bit_stream_status=BitStreamStatus.UNAVAILABLE,
            valid=False,
        )

    re_vals = np.real(symbols)
    im_vals = np.imag(symbols)

    # 2 bits per symbol: b0 from I axis, b1 from Q axis
    # Gray mapping:
    # Quadrant 1 (+, +) -> 00 (index 0)
    # Quadrant 2 (-, +) -> 01 (index 1)
    # Quadrant 3 (-, -) -> 11 (index 2)
    # Quadrant 4 (+, -) -> 10 (index 3)
    b0 = (re_vals < 0.0).astype(np.uint8)
    b1 = (im_vals < 0.0).astype(np.uint8)

    hard_bits = np.column_stack((b0, b1)).ravel().astype(np.uint8)
    soft_b0 = (-2.0 * re_vals).astype(np.float32)
    soft_b1 = (-2.0 * im_vals).astype(np.float32)
    soft_decisions = np.column_stack((soft_b0, soft_b1)).ravel().astype(np.float32)

    # Constellation index 0..3
    sym_indices = (b0 * 2 + b1).astype(np.int32)

    return DemodulationResult(
        hard_bits=hard_bits,
        soft_decisions=soft_decisions,
        symbol_indices=sym_indices,
        bit_stream_status=BitStreamStatus.AVAILABLE,
        bit_polarity="unresolved",
        fec_status="not_applied",
        mapping_scheme=mapping,
        valid=True,
    )

def demodulate_8psk(
    symbols: np.ndarray,
    mapping: str = "gray",
) -> DemodulationResult:
    """
    Demodulate 8-PSK symbols: 3 bits per symbol, nearest phase sector slicing.

    Parameters
    ----------
    symbols : np.ndarray
        1-SPS normalized symbol samples.
    mapping : str
        'gray'.

    Returns
    -------
    DemodulationResult
    """
    n = len(symbols)
    if n == 0:
        return DemodulationResult(
            hard_bits=np.array([], dtype=np.uint8),
            soft_decisions=np.array([], dtype=np.float32),
            symbol_indices=np.array([], dtype=np.int32),
            bit_stream_status=BitStreamStatus.UNAVAILABLE,
            valid=False,
        )

    angles = np.mod(np.angle(symbols), 2.0 * np.pi)
    # 8 sectors centered at k * 2*pi / 8
    sector_idx = np.mod(np.round(angles * 8.0 / (2.0 * np.pi)), 8).astype(np.int32)

    # Standard 8-PSK Gray Code Table: 0->000, 1->001, 2->011, 3->010, 4->110, 5->111, 6->101, 7->100
    gray_table = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
        [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]
    ], dtype=np.uint8)

    bits_matrix = gray_table[sector_idx]
    hard_bits = bits_matrix.ravel().astype(np.uint8)

    # Soft metric based on distance to nearest sector boundary
    sector_center = sector_idx * (2.0 * np.pi / 8.0)
    phase_offset = np.abs(np.angle(symbols * np.exp(-1j * sector_center)))
    soft_conf = np.repeat((np.pi / 8.0 - phase_offset) / (np.pi / 8.0), 3).astype(np.float32)

    return DemodulationResult(
        hard_bits=hard_bits,
        soft_decisions=soft_conf,
        symbol_indices=sector_idx,
        bit_stream_status=BitStreamStatus.AVAILABLE,
        bit_polarity="unresolved",
        fec_status="not_applied",
        mapping_scheme=mapping,
        valid=True,
    )

def demodulate_16qam(
    symbols: np.ndarray,
    mapping: str = "gray",
) -> DemodulationResult:
    """
    Demodulate 16-QAM symbols: 4 bits per symbol with rectangular Gray grid slicing.

    Parameters
    ----------
    symbols : np.ndarray
        1-SPS normalized symbol samples.
    mapping : str
        'gray'.

    Returns
    -------
    DemodulationResult
    """
    n = len(symbols)
    if n == 0:
        return DemodulationResult(
            hard_bits=np.array([], dtype=np.uint8),
            soft_decisions=np.array([], dtype=np.float32),
            symbol_indices=np.array([], dtype=np.int32),
            bit_stream_status=BitStreamStatus.UNAVAILABLE,
            valid=False,
        )

    # Scale to standard integer grid [-3, -1, +1, +3]
    scale = np.sqrt(10.0)
    re_scaled = np.real(symbols) * scale
    im_scaled = np.imag(symbols) * scale

    # 1D Gray slicing:
    # x < -2: 00
    # -2 <= x < 0: 01
    # 0 <= x < 2: 11
    # x >= 2: 10
    def slice_1d(val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b_msb = (val >= 0.0).astype(np.uint8)
        b_lsb = (np.abs(val) < 2.0).astype(np.uint8)
        # Sliced level: -3, -1, +1, +3
        level = np.where(val < -2.0, -3.0, np.where(val < 0.0, -1.0, np.where(val < 2.0, 1.0, 3.0)))
        return b_msb, b_lsb, level

    i_b0, i_b1, i_level = slice_1d(re_scaled)
    q_b0, q_b1, q_level = slice_1d(im_scaled)

    # 4 bits per symbol: [i_b0, i_b1, q_b0, q_b1]
    bits_matrix = np.column_stack((i_b0, i_b1, q_b0, q_b1))
    hard_bits = bits_matrix.ravel().astype(np.uint8)

    # Soft LLR decisions
    llr_i0 = (2.0 * re_scaled / scale).astype(np.float32)
    llr_i1 = (2.0 - np.abs(re_scaled)).astype(np.float32)
    llr_q0 = (2.0 * im_scaled / scale).astype(np.float32)
    llr_q1 = (2.0 - np.abs(im_scaled)).astype(np.float32)
    soft_decisions = np.column_stack((llr_i0, llr_i1, llr_q0, llr_q1)).ravel().astype(np.float32)

    # Symbol index 0..15
    i_idx = np.where(i_level == -3.0, 0, np.where(i_level == -1.0, 1, np.where(i_level == 1.0, 2, 3)))
    q_idx = np.where(q_level == -3.0, 0, np.where(q_level == -1.0, 1, np.where(q_level == 1.0, 2, 3)))
    sym_indices = (i_idx * 4 + q_idx).astype(np.int32)

    return DemodulationResult(
        hard_bits=hard_bits,
        soft_decisions=soft_decisions,
        symbol_indices=sym_indices,
        bit_stream_status=BitStreamStatus.AVAILABLE,
        bit_polarity="unresolved",
        fec_status="not_applied",
        mapping_scheme=mapping,
        valid=True,
    )

def demodulate_bfsk(
    samples: np.ndarray,
    f0: float,
    f1: float,
    sps: float = 8.0,
) -> DemodulationResult:
    """
    Demodulate BFSK using dual-tone matched correlation filter bank over symbol intervals.

    Parameters
    ----------
    samples : np.ndarray
        Complex baseband samples.
    f0 : float
        Space frequency (cycles/sample).
    f1 : float
        Mark frequency (cycles/sample).
    sps : float
        Samples per symbol.

    Returns
    -------
    DemodulationResult
    """
    sps_int = max(2, int(round(sps)))
    n_syms = len(samples) // sps_int
    if n_syms == 0:
        return DemodulationResult(
            hard_bits=np.array([], dtype=np.uint8),
            soft_decisions=np.array([], dtype=np.float32),
            symbol_indices=np.array([], dtype=np.int32),
            bit_stream_status=BitStreamStatus.UNAVAILABLE,
            valid=False,
        )

    hard_bits = np.zeros(n_syms, dtype=np.uint8)
    soft_decisions = np.zeros(n_syms, dtype=np.float32)
    t = np.arange(sps_int, dtype=np.float64)

    tone0 = np.exp(-2j * np.pi * f0 * t)
    tone1 = np.exp(-2j * np.pi * f1 * t)

    for k in range(n_syms):
        block = samples[k * sps_int:(k + 1) * sps_int]
        m0 = float(np.abs(np.sum(block * tone0)))
        m1 = float(np.abs(np.sum(block * tone1)))
        
        # Decision: m1 > m0 -> bit 1, else bit 0
        hard_bits[k] = 1 if m1 > m0 else 0
        soft_decisions[k] = float(m1 - m0) / (m1 + m0 + 1e-12)

    return DemodulationResult(
        hard_bits=hard_bits,
        soft_decisions=soft_decisions,
        symbol_indices=hard_bits.astype(np.int32),
        bit_stream_status=BitStreamStatus.AVAILABLE,
        bit_polarity="unresolved",
        fec_status="not_applied",
        mapping_scheme="natural",
        valid=True,
    )
