import numpy as np
import pytest
from app.recovery.constellation import analyze_constellation, get_ideal_constellation
from app.recovery.models import ModulationFamily

def test_ideal_constellations():
    bpsk = get_ideal_constellation(ModulationFamily.PSK, 2)
    assert len(bpsk) == 2
    assert np.isclose(np.mean(np.abs(bpsk) ** 2), 1.0)

    qpsk = get_ideal_constellation(ModulationFamily.PSK, 4)
    assert len(qpsk) == 4
    assert np.isclose(np.mean(np.abs(qpsk) ** 2), 1.0)

    qam16 = get_ideal_constellation(ModulationFamily.QAM, 16)
    assert len(qam16) == 16
    assert np.isclose(np.mean(np.abs(qam16) ** 2), 1.0)

def test_constellation_analysis_clean_qpsk():
    const_pts = get_ideal_constellation(ModulationFamily.PSK, 4)
    indices = np.random.randint(0, 4, 1024)
    clean_syms = const_pts[indices]
    
    res = analyze_constellation(clean_syms, family=ModulationFamily.PSK, order=4)
    assert res.valid is True
    assert res.evm_percent < 0.1
    assert res.decision_margin > 0.95
    assert len(res.cluster_centroids) == 4
    assert res.rotational_ambiguity_deg == (0.0, 90.0, 180.0, 270.0)

def test_constellation_analysis_noisy():
    const_pts = get_ideal_constellation(ModulationFamily.PSK, 4)
    indices = np.random.randint(0, 4, 1024)
    noisy_syms = const_pts[indices] + (np.random.randn(1024) + 1j * np.random.randn(1024)) * 0.1
    
    res = analyze_constellation(noisy_syms, family=ModulationFamily.PSK, order=4)
    assert res.valid is True
    assert res.evm_percent > 5.0
    assert res.evm_percent < 30.0
