from __future__ import annotations
import math
from typing import Any
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity
from app.recovery.models import RecoveredSignal, RecoveryAnalysis
from .models import BitOrder, BitPolarity, BitStream, DataRecoveryConfig

def extract_bitstream_from_recovery(
    recovery: RecoveryAnalysis | RecoveredSignal,
    config: DataRecoveryConfig | None = None,
) -> tuple[BitStream, list[Diagnostic]]:
    """
    Extract canonical BitStream from Phase 4 RecoveryAnalysis or RecoveredSignal.

    Parameters
    ----------
    recovery : RecoveryAnalysis | RecoveredSignal
    config : DataRecoveryConfig | None

    Returns
    -------
    bitstream : BitStream
    diagnostics : list[Diagnostic]
    """
    diagnostics: list[Diagnostic] = []

    if isinstance(recovery, RecoveryAnalysis):
        if not recovery.is_recovered or recovery.recovered_signal is None:
            # Check if any candidate has demodulated bits
            cand_bits = None
            cand_soft = None
            cand_sym_idx = None
            cand_name = "inconclusive"
            for cand in recovery.candidates:
                if cand.demodulation and len(cand.demodulation.hard_bits) > 0:
                    cand_bits = cand.demodulation.hard_bits
                    cand_soft = cand.demodulation.soft_decisions
                    cand_sym_idx = cand.demodulation.symbol_indices
                    cand_name = cand.label
                    break

            if cand_bits is None:
                diagnostics.append(
                    Diagnostic(
                        code="NO_BITSTREAM_AVAILABLE",
                        message="No demodulated bitstream available from Phase 4 recovery.",
                        severity=DiagnosticSeverity.ERROR,
                    )
                )
                return BitStream(
                    hard_bits=np.array([], dtype=np.uint8),
                    soft_bits=None,
                    symbol_indices=None,
                    bit_order=BitOrder.UNKNOWN,
                    bit_polarity=BitPolarity.UNRESOLVED,
                    diagnostics=diagnostics,
                ), diagnostics

            rec_sig = None
            hard_bits = cand_bits
            soft_bits = cand_soft
            symbol_indices = cand_sym_idx
            sample_indices = None
            source_cand = cand_name
            polarity_str = "unresolved"
        else:
            rec_sig = recovery.recovered_signal
            hard_bits = rec_sig.hard_bits
            soft_bits = rec_sig.soft_bits
            symbol_indices = rec_sig.symbol_indices
            sample_indices = rec_sig.sample_indices
            source_cand = f"{rec_sig.modulation_family.value}_{rec_sig.modulation_order or ''}"
            polarity_str = rec_sig.bit_polarity_status
    else:
        rec_sig = recovery
        hard_bits = rec_sig.hard_bits
        soft_bits = rec_sig.soft_bits
        symbol_indices = rec_sig.symbol_indices
        sample_indices = rec_sig.sample_indices
        source_cand = f"{rec_sig.modulation_family.value}_{rec_sig.modulation_order or ''}"
        polarity_str = rec_sig.bit_polarity_status

    if len(hard_bits) < 16 or np.all(hard_bits == 0) or np.all(hard_bits == 1):
        diagnostics.append(
            Diagnostic(
                code="TRIVIAL_OR_SHORT_BITSTREAM",
                message="Bitstream is all-zero, all-one, or too short for data recovery.",
                severity=DiagnosticSeverity.WARNING,
            )
        )
    elif soft_bits is None or len(soft_bits) == 0:
        diagnostics.append(
            Diagnostic(
                code="NO_SOFT_INFORMATION",
                message="Soft decisions / LLRs unavailable; downstream decoders will fall back to hard decisions.",
                severity=DiagnosticSeverity.INFO,
            )
        )

    polarity = BitPolarity.NORMAL if polarity_str == "normal" else (
        BitPolarity.INVERTED if polarity_str == "inverted" else BitPolarity.UNRESOLVED
    )

    stats = compute_digital_statistics(hard_bits)

    bs = BitStream(
        hard_bits=hard_bits.astype(np.uint8),
        soft_bits=soft_bits.astype(np.float32) if soft_bits is not None else None,
        symbol_indices=symbol_indices.astype(np.int32) if symbol_indices is not None else None,
        bit_order=BitOrder.UNKNOWN,
        bit_polarity=polarity,
        bit_offset=0,
        source_candidate=source_cand,
        sample_indices=sample_indices,
        diagnostics=diagnostics,
        provenance={"stats": stats, "length": len(hard_bits)},
    )
    return bs, diagnostics

def compute_digital_statistics(bits: np.ndarray) -> dict[str, float]:
    """
    Calculate comprehensive digital statistics for a binary sequence.

    Parameters
    ----------
    bits : np.ndarray
        1D uint8 binary array.

    Returns
    -------
    stats : dict[str, float]
    """
    n = len(bits)
    if n == 0:
        return {
            "bit_balance": 0.0,
            "transition_probability": 0.0,
            "mean_run_length": 0.0,
            "byte_entropy": 0.0,
            "conditional_entropy": 0.0,
        }

    # 1. Bit Balance (fraction of 1s)
    balance = float(np.mean(bits))

    # 2. Transition Probability
    if n >= 2:
        trans_count = np.sum(bits[1:] != bits[:-1])
        trans_prob = float(trans_count / (n - 1))
    else:
        trans_prob = 0.0

    # 3. Run-length distribution
    if n >= 2:
        diffs = np.diff(bits)
        run_starts = np.where(diffs != 0)[0] + 1
        runs = np.diff(np.concatenate(([0], run_starts, [n])))
        mean_run = float(np.mean(runs)) if len(runs) > 0 else 1.0
    else:
        mean_run = 1.0

    # 4. Byte Entropy (8-bit non-overlapping chunks)
    n_bytes = n // 8
    if n_bytes > 0:
        byte_vals = np.packbits(bits[: n_bytes * 8])
        counts = np.bincount(byte_vals, minlength=256)
        probs = counts[counts > 0] / n_bytes
        byte_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        byte_entropy = 0.0

    # 5. Conditional Entropy H(X_i | X_{i-1})
    if n >= 2:
        # P(00), P(01), P(10), P(11)
        pairs = bits[:-1] * 2 + bits[1:]
        pair_counts = np.bincount(pairs, minlength=4)
        p_pairs = pair_counts / (n - 1)
        # H(X_{i-1}, X_i)
        p_p_valid = p_pairs[p_pairs > 0]
        h_joint = float(-np.sum(p_p_valid * np.log2(p_p_valid)))
        # H(X_{i-1})
        p0 = 1.0 - balance
        p1 = balance
        h_prev = 0.0
        if 0 < p0 < 1.0:
            h_prev = float(-p0 * np.log2(p0) - p1 * np.log2(p1))
        cond_entropy = max(0.0, h_joint - h_prev)
    else:
        cond_entropy = 0.0

    return {
        "bit_balance": round(balance, 4),
        "transition_probability": round(trans_prob, 4),
        "mean_run_length": round(mean_run, 2),
        "byte_entropy": round(byte_entropy, 4),
        "conditional_entropy": round(cond_entropy, 4),
    }
