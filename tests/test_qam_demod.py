import numpy as np
import pytest
from app.recovery.constellation import get_ideal_constellation
from app.recovery.demodulation import demodulate_16qam
from app.recovery.models import BitStreamStatus, ModulationFamily

def test_16qam_demodulation_ideal():
    ideal_pts = get_ideal_constellation(ModulationFamily.QAM, 16)
    res = demodulate_16qam(ideal_pts)
    assert res.valid is True
    assert res.bit_stream_status == BitStreamStatus.AVAILABLE
    # 16 symbols * 4 bits = 64 bits
    assert len(res.hard_bits) == 64
    assert len(res.soft_decisions) == 64
    assert len(res.symbol_indices) == 16

def test_16qam_demodulation_empty():
    res = demodulate_16qam(np.array([], dtype=np.complex64))
    assert res.valid is False
    assert res.bit_stream_status == BitStreamStatus.UNAVAILABLE
