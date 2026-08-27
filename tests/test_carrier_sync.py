import numpy as np
import pytest
from app.recovery.carrier_sync import costas_carrier_recovery
from app.recovery.models import LockStatus, ModulationFamily

def test_costas_loop_bpsk_lock():
    # Ideal BPSK symbols rotated by 0.25 rad
    symbols = np.random.choice([-1.0, 1.0], size=1024).astype(np.complex64)
    phi_rot = 0.25
    rotated = symbols * np.exp(1j * phi_rot)
    
    corrected, res = costas_carrier_recovery(rotated, family=ModulationFamily.PSK, order=2)
    assert res.lock_status == LockStatus.LOCKED
    assert res.phase_error_var < 0.05
    assert len(corrected) == len(symbols)

def test_costas_loop_qpsk_lock():
    # Ideal QPSK symbols rotated by 0.3 rad + small residual CFO 0.001
    const_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    indices = np.random.randint(0, 4, 1024)
    symbols = const_pts[indices]
    
    t = np.arange(1024)
    rotated = symbols * np.exp(1j * (0.30 + 2.0 * np.pi * 0.001 * t))
    
    corrected, res = costas_carrier_recovery(rotated, family=ModulationFamily.PSK, order=4)
    assert res.lock_status == LockStatus.LOCKED
    assert res.phase_error_var < 0.08
    assert res.lock_duration_fraction > 0.60
