import numpy as np
import pytest
from app.data_recovery.ambiguity import generate_ambiguity_hypotheses
from app.data_recovery.models import BitOrder, BitPolarity, BitStream, DataRecoveryConfig

def test_ambiguity_generation_polarity_and_rotations():
    # 32 bits (16 symbols for QPSK)
    hard = np.random.randint(0, 2, 32, dtype=np.uint8)
    soft = np.where(hard == 1, 1.2, -1.2).astype(np.float32)

    bs = BitStream(
        hard_bits=hard,
        soft_bits=soft,
        symbol_indices=None,
        bit_order=BitOrder.UNKNOWN,
        bit_polarity=BitPolarity.UNRESOLVED,
    )

    hyps = generate_ambiguity_hypotheses(bs, config=DataRecoveryConfig(evaluate_all_bit_offsets=False))
    # Should produce: 0 deg (normal), 180 deg (inverted), 90 deg, 270 deg
    assert len(hyps) == 4
    assert hyps[0].phase_rotation_deg == 0.0
    assert hyps[1].phase_rotation_deg == 180.0
    assert np.array_equal(hyps[1].bitstream.hard_bits, 1 - hard)
    assert np.allclose(hyps[1].bitstream.soft_bits, -soft)
