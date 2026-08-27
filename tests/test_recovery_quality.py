import numpy as np
import pytest
from app.recovery.models import ModulationFamily, RecoveryConfig
from app.recovery.psk_receiver import run_psk_receiver
from app.recovery.quality import evaluate_windowed_stability
from scripts.generate_modulated_dataset import generate_modulated_signal

def test_windowed_stability_clean_signal():
    samples, _ = generate_modulated_signal("QPSK", n_symbols=1024, samples_per_symbol=8, snr_db=25.0, seed=42)
    
    def runner(s_slice: np.ndarray):
        return run_psk_receiver(s_slice, order=4, sps=8.0)

    score, diags = evaluate_windowed_stability(samples, runner, num_windows=4)
    assert score >= 0.75
    assert len(diags) == 0

def test_windowed_stability_short_signal():
    short_samples = np.ones(256, dtype=np.complex64)
    score, diags = evaluate_windowed_stability(short_samples, lambda s: None, num_windows=4)
    assert score == 1.0
