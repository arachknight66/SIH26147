import numpy as np
import pytest
from app.recovery.demodulation import demodulate_bfsk
from app.recovery.models import BitStreamStatus
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_fsk_demodulation_clean():
    samples, manifest = generate_modulated_signal("BFSK", n_symbols=128, samples_per_symbol=8, snr_db=30.0, seed=42)
    res = demodulate_bfsk(samples, f0=-0.125, f1=+0.125, sps=8.0)
    assert res.valid is True
    assert res.bit_stream_status == BitStreamStatus.AVAILABLE
    assert len(res.hard_bits) == 128
    
    # Ground truth bit comparison
    tx_bits = manifest.get("tx_bits")
    if tx_bits is not None:
        ber = np.mean(res.hard_bits != tx_bits[:128])
        assert ber < 0.05

def test_fsk_demodulation_short():
    res = demodulate_bfsk(np.array([1.0], dtype=np.complex64), f0=-0.125, f1=0.125, sps=8.0)
    assert res.valid is False
