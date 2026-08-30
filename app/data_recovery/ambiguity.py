from __future__ import annotations
import numpy as np
from .models import (
    BitHypothesis,
    BitOrder,
    BitPolarity,
    BitStream,
    DataRecoveryConfig,
    EpistemicStatus,
    LineCodeType,
)

def generate_ambiguity_hypotheses(
    bitstream: BitStream,
    config: DataRecoveryConfig | None = None,
) -> list[BitHypothesis]:
    """
    Generate bounded candidate bitstream hypotheses across phase rotations and bit polarity.

    Parameters
    ----------
    bitstream : BitStream
        Input bitstream from Phase 4.
    config : DataRecoveryConfig | None
        Configuration limits.

    Returns
    -------
    hypotheses : list[BitHypothesis]
    """
    cfg = config or DataRecoveryConfig()
    hypotheses: list[BitHypothesis] = []
    hard = bitstream.hard_bits
    soft = bitstream.soft_bits

    if len(hard) == 0:
        return []

    # 1. Normal Polarity / 0 deg Rotation
    hypotheses.append(
        BitHypothesis(
            hypothesis_id=1,
            bitstream=bitstream,
            phase_rotation_deg=0.0,
            polarity=BitPolarity.NORMAL,
            line_code=LineCodeType.NONE,
            bit_order=BitOrder.UNKNOWN,
            bit_offset=0,
            epistemic_status=EpistemicStatus.OBSERVED,
        )
    )

    # 2. Inverted Polarity / 180 deg Rotation
    inv_hard = 1 - hard
    inv_soft = -soft if soft is not None else None
    inv_stream = BitStream(
        hard_bits=inv_hard.astype(np.uint8),
        soft_bits=inv_soft.astype(np.float32) if inv_soft is not None else None,
        symbol_indices=bitstream.symbol_indices,
        bit_order=BitOrder.UNKNOWN,
        bit_polarity=BitPolarity.INVERTED,
        bit_offset=0,
        source_candidate=bitstream.source_candidate,
        sample_indices=bitstream.sample_indices,
        diagnostics=bitstream.diagnostics,
        provenance={"inverted_from_hypothesis": 1},
    )
    if cfg.evaluate_polarity_inversion:
        hypotheses.append(
            BitHypothesis(
                hypothesis_id=2,
                bitstream=inv_stream,
                phase_rotation_deg=180.0,
                polarity=BitPolarity.INVERTED,
                line_code=LineCodeType.NONE,
                bit_order=BitOrder.UNKNOWN,
                bit_offset=0,
                epistemic_status=EpistemicStatus.INFERRED,
            )
        )

    # 3. 90 deg and 270 deg rotations for 2-bit per symbol (QPSK-like streams)
    if cfg.evaluate_rotational_ambiguities and len(hard) % 2 == 0 and len(hard) >= 32 and cfg.max_bit_hypotheses >= 4:
        # Reshape to paired bits [b_I, b_Q]
        b_pairs = hard.reshape(-1, 2)
        b_I = b_pairs[:, 0]
        b_Q = b_pairs[:, 1]

        # 90 deg rotation: (1 - b_Q, b_I)
        rot90_I = 1 - b_Q
        rot90_Q = b_I
        rot90_hard = np.column_stack((rot90_I, rot90_Q)).ravel()
        
        rot90_soft = None
        if soft is not None and len(soft) == len(hard):
            s_pairs = soft.reshape(-1, 2)
            rot90_soft = np.column_stack((-s_pairs[:, 1], s_pairs[:, 0])).ravel()

        rot90_stream = BitStream(
            hard_bits=rot90_hard.astype(np.uint8),
            soft_bits=rot90_soft.astype(np.float32) if rot90_soft is not None else None,
            symbol_indices=bitstream.symbol_indices,
            bit_order=BitOrder.UNKNOWN,
            bit_polarity=BitPolarity.NORMAL,
            bit_offset=0,
            source_candidate=bitstream.source_candidate,
            provenance={"rotation_deg": 90.0},
        )
        hypotheses.append(
            BitHypothesis(
                hypothesis_id=3,
                bitstream=rot90_stream,
                phase_rotation_deg=90.0,
                polarity=BitPolarity.NORMAL,
                line_code=LineCodeType.NONE,
                bit_order=BitOrder.UNKNOWN,
                bit_offset=0,
                epistemic_status=EpistemicStatus.INFERRED,
            )
        )

        # 270 deg rotation: (b_Q, 1 - b_I)
        rot270_I = b_Q
        rot270_Q = 1 - b_I
        rot270_hard = np.column_stack((rot270_I, rot270_Q)).ravel()

        rot270_soft = None
        if soft is not None and len(soft) == len(hard):
            s_pairs = soft.reshape(-1, 2)
            rot270_soft = np.column_stack((s_pairs[:, 1], -s_pairs[:, 0])).ravel()

        rot270_stream = BitStream(
            hard_bits=rot270_hard.astype(np.uint8),
            soft_bits=rot270_soft.astype(np.float32) if rot270_soft is not None else None,
            symbol_indices=bitstream.symbol_indices,
            bit_order=BitOrder.UNKNOWN,
            bit_polarity=BitPolarity.NORMAL,
            bit_offset=0,
            source_candidate=bitstream.source_candidate,
            provenance={"rotation_deg": 270.0},
        )
        hypotheses.append(
            BitHypothesis(
                hypothesis_id=4,
                bitstream=rot270_stream,
                phase_rotation_deg=270.0,
                polarity=BitPolarity.NORMAL,
                line_code=LineCodeType.NONE,
                bit_order=BitOrder.UNKNOWN,
                bit_offset=0,
                epistemic_status=EpistemicStatus.INFERRED,
            )
        )

    # Bit alignment is an explicit hypothesis, not an implicit byte conversion.
    # Each offset retains the source transformation and trims only leading bits.
    if cfg.evaluate_all_bit_offsets:
        base_hypotheses = list(hypotheses)
        next_id = len(hypotheses) + 1
        for offset in range(1, 8):
            for base in base_hypotheses:
                if len(hypotheses) >= cfg.max_bit_hypotheses:
                    break
                source = base.bitstream
                shifted = BitStream(
                    hard_bits=source.hard_bits[offset:].copy(),
                    soft_bits=source.soft_bits[offset:].copy() if source.soft_bits is not None else None,
                    symbol_indices=source.symbol_indices,
                    bit_order=source.bit_order,
                    bit_polarity=source.bit_polarity,
                    bit_offset=offset,
                    source_candidate=source.source_candidate,
                    sample_indices=source.sample_indices,
                    diagnostics=source.diagnostics,
                    provenance={**source.provenance, "bit_offset": offset, "derived_from_hypothesis": base.hypothesis_id},
                )
                hypotheses.append(BitHypothesis(
                    hypothesis_id=next_id,
                    bitstream=shifted,
                    phase_rotation_deg=base.phase_rotation_deg,
                    polarity=base.polarity,
                    line_code=base.line_code,
                    bit_order=base.bit_order,
                    bit_offset=offset,
                    epistemic_status=EpistemicStatus.INFERRED,
                ))
                next_id += 1
            if len(hypotheses) >= cfg.max_bit_hypotheses:
                break

    return hypotheses[: cfg.max_bit_hypotheses]
