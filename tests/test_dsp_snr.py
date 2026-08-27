import numpy as np
import pytest
from app.dsp.noise import estimate_noise_floor
from app.dsp.psd import compute_psd
from app.dsp.snr import compute_all_snr_estimates, estimate_snr_m2m4, estimate_snr_spectral

@pytest.mark.parametrize("target_snr_db", [5.0, 10.0, 20.0])
def test_snr_estimation_accuracy(target_snr_db):
    np.random.seed(42)
    n_samples = 32768
    t = np.arange(n_samples)
    # Unit power complex sinusoid
    signal = np.exp(2j * np.pi * 0.125 * t).astype(np.complex64)
    # Noise with power N = 10^(-SNR/10)
    noise_power = 1.0 / (10.0 ** (target_snr_db / 10.0))
    noise = (np.random.normal(0, np.sqrt(noise_power / 2), n_samples) + 1j * np.random.normal(0, np.sqrt(noise_power / 2), n_samples)).astype(np.complex64)
    x = signal + noise

    psd_res = compute_psd(x, segment_length=4096, is_complex=True)
    noise_est = estimate_noise_floor(psd_res.psd, method="iterative_sigma_clip")
    
    spectral_snr = estimate_snr_spectral(psd_res, noise_est)
    assert spectral_snr.snr_db is not None
    assert np.isclose(spectral_snr.snr_db, target_snr_db, atol=0.5)

    m2m4_snr = estimate_snr_m2m4(x)
    assert m2m4_snr.snr_db is not None
    assert np.isclose(m2m4_snr.snr_db, target_snr_db, atol=1.0)
