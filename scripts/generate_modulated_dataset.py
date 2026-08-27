from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import scipy.signal as signal

@dataclass(frozen=True)
class GroundTruthManifest:
    modulation_name: str
    family: str
    order: int | None
    symbol_rate_normalized: float
    samples_per_symbol: int
    snr_db: float
    cfo_normalized: float
    timing_offset: float
    pulse_shape: str
    fading: str

def _rrc_filter(sps: int, alpha: float, num_symbols: int = 8) -> np.ndarray:
    """Generate Root Raised Cosine (RRC) pulse shaping filter."""
    n_taps = num_symbols * sps + 1
    t = np.arange(-(n_taps // 2), (n_taps // 2) + 1, dtype=np.float64) / sps
    h = np.zeros(len(t), dtype=np.float64)
    
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-6:
            h[i] = (alpha / np.sqrt(2.0)) * (((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))) + ((1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))))
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + (4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha)))
            denom = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / denom

    h /= np.sqrt(np.sum(h ** 2))
    return h

def generate_modulated_signal(
    modulation: str,
    *,
    n_symbols: int = 1024,
    samples_per_symbol: int = 8,
    snr_db: float = 20.0,
    cfo_normalized: float = 0.0,
    timing_offset: float = 0.0,
    phase_offset_rad: float = 0.0,
    pulse_shape: str = "rrc",
    rrc_alpha: float = 0.35,
    fading: str = "none",
    iq_imbalance_db: float = 0.0,
    dc_offset: complex = 0.0j,
    clipping_ratio: float = 1.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Generate synthetic digitally modulated signals with realistic channel impairments.

    Parameters
    ----------
    modulation : str
        Target modulation ('BFSK', 'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM',
        or OOD: 'AM', 'FM', 'GMSK', 'OFDM', 'NOISE').
    n_symbols : int
        Number of symbols to generate.
    samples_per_symbol : int
        Oversampling factor (SPS).
    snr_db : float
        Target Signal-to-Noise Ratio in dB.
    cfo_normalized : float
        Carrier Frequency Offset in cycles/sample.
    timing_offset : float
        Fractional timing offset in samples.
    phase_offset_rad : float
        Initial carrier phase offset in radians.
    pulse_shape : str
        Pulse shaping filter ('rrc', 'rect', 'hann').
    rrc_alpha : float
        RRC roll-off factor (0.2 to 0.5).
    fading : str
        Channel fading model ('none', 'rayleigh', 'rician').
    iq_imbalance_db : float
        Amplitude imbalance between I and Q in dB.
    dc_offset : complex
        Additive complex DC offset.
    clipping_ratio : float
        Amplitude threshold ratio for clipping (1.0 = no clipping).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    samples : np.ndarray
        Generated complex64 signal samples.
    manifest : dict[str, Any]
        Ground truth parameter manifest.
    """
    rng = np.random.default_rng(seed)
    sps = samples_per_symbol
    mod_upper = modulation.upper().strip()

    # 1. Baseband Symbol Generation
    order: int | None = None
    family = "UNKNOWN"

    if mod_upper in ("BFSK", "2FSK", "FSK"):
        family = "FSK"
        order = 2
        bits = rng.integers(0, 2, n_symbols)
        # FSK frequencies: +- delta_f
        delta_f = 0.125
        freq_seq = np.where(bits == 0, -delta_f, +delta_f)
        freq_upsampled = np.repeat(freq_seq, sps)
        phase_accum = 2.0 * np.pi * np.cumsum(freq_upsampled)
        tx_signal = np.exp(1j * phase_accum).astype(np.complex64)

    elif mod_upper == "BPSK":
        family = "PSK"
        order = 2
        bits = rng.integers(0, 2, n_symbols)
        const_pts = np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
        syms = const_pts[bits]
        upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
        upsampled[::sps] = syms
        h = _rrc_filter(sps, rrc_alpha) if pulse_shape == "rrc" else np.ones(sps, dtype=np.float64) / np.sqrt(sps)
        tx_signal = signal.convolve(upsampled, h, mode="same").astype(np.complex64)

    elif mod_upper == "QPSK":
        family = "PSK"
        order = 4
        indices = rng.integers(0, 4, n_symbols)
        const_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
        syms = const_pts[indices]
        upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
        upsampled[::sps] = syms
        h = _rrc_filter(sps, rrc_alpha) if pulse_shape == "rrc" else np.ones(sps, dtype=np.float64) / np.sqrt(sps)
        tx_signal = signal.convolve(upsampled, h, mode="same").astype(np.complex64)

    elif mod_upper in ("8PSK", "8-PSK"):
        family = "PSK"
        order = 8
        indices = rng.integers(0, 8, n_symbols)
        angles = np.arange(8) * (2.0 * np.pi / 8.0)
        const_pts = np.exp(1j * angles).astype(np.complex64)
        syms = const_pts[indices]
        upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
        upsampled[::sps] = syms
        h = _rrc_filter(sps, rrc_alpha) if pulse_shape == "rrc" else np.ones(sps, dtype=np.float64) / np.sqrt(sps)
        tx_signal = signal.convolve(upsampled, h, mode="same").astype(np.complex64)

    elif mod_upper in ("16QAM", "16-QAM"):
        family = "QAM"
        order = 16
        re_vals = np.array([-3, -1, 1, 3], dtype=np.float32)
        im_vals = np.array([-3, -1, 1, 3], dtype=np.float32)
        re = rng.choice(re_vals, size=n_symbols)
        im = rng.choice(im_vals, size=n_symbols)
        syms = (re + 1j * im) / np.sqrt(10.0)
        upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
        upsampled[::sps] = syms
        h = _rrc_filter(sps, rrc_alpha) if pulse_shape == "rrc" else np.ones(sps, dtype=np.float64) / np.sqrt(sps)
        tx_signal = signal.convolve(upsampled, h, mode="same").astype(np.complex64)

    elif mod_upper in ("64QAM", "64-QAM"):
        family = "QAM"
        order = 64
        grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
        re = rng.choice(grid, size=n_symbols)
        im = rng.choice(grid, size=n_symbols)
        syms = (re + 1j * im) / np.sqrt(42.0)
        upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
        upsampled[::sps] = syms
        h = _rrc_filter(sps, rrc_alpha) if pulse_shape == "rrc" else np.ones(sps, dtype=np.float64) / np.sqrt(sps)
        tx_signal = signal.convolve(upsampled, h, mode="same").astype(np.complex64)

    elif mod_upper == "AM":
        # Amplitude modulated analog audio tone (OOD)
        t = np.arange(n_symbols * sps, dtype=np.float32)
        audio = np.sin(2.0 * np.pi * 0.005 * t)
        tx_signal = ((1.0 + 0.8 * audio) * np.exp(1j * 0.0)).astype(np.complex64)

    elif mod_upper == "FM":
        # Frequency modulated analog chirp/tone (OOD)
        t = np.arange(n_symbols * sps, dtype=np.float32)
        mod_tone = np.sin(2.0 * np.pi * 0.002 * t)
        inst_phi = np.cumsum(0.10 * mod_tone)
        tx_signal = np.exp(1j * inst_phi).astype(np.complex64)

    elif mod_upper == "OFDM":
        # Multicarrier OFDM-like symbol block (OOD)
        n_sub = 64
        n_blocks = (n_symbols * sps) // (n_sub + 16)
        blocks = []
        for _ in range(max(1, n_blocks)):
            sub_syms = rng.choice([-1, 1], size=n_sub) + 1j * rng.choice([-1, 1], size=n_sub)
            time_sym = np.fft.ifft(sub_syms)
            cp = time_sym[-16:]
            blocks.append(np.concatenate([cp, time_sym]))
        tx_signal = np.concatenate(blocks).astype(np.complex64)[: n_symbols * sps]

    elif mod_upper == "GMSK":
        # GMSK continuous phase (OOD)
        bits = rng.integers(0, 2, n_symbols)
        diff_bits = 2 * bits - 1.0
        upsampled = np.repeat(diff_bits, sps)
        # Gaussian filter
        t_g = np.linspace(-2, 2, sps * 4)
        g_filter = np.exp(- (t_g ** 2) / (2 * (0.3 ** 2)))
        g_filter /= np.sum(g_filter)
        freq_filt = signal.convolve(upsampled, g_filter, mode="same")
        phase_g = np.cumsum(freq_filt * (np.pi / (2.0 * sps)))
        tx_signal = np.exp(1j * phase_g).astype(np.complex64)

    elif mod_upper == "NOISE":
        # Pure AWGN (OOD)
        tx_signal = (rng.normal(0, 1.0, n_symbols * sps) + 1j * rng.normal(0, 1.0, n_symbols * sps)).astype(np.complex64)

    else:
        raise ValueError(f"Unsupported modulation '{modulation}'. Supported: BFSK, BPSK, QPSK, 8PSK, 16QAM, 64QAM, AM, FM, GMSK, OFDM, NOISE.")

    total_len = len(tx_signal)

    # 2. Channel Fading
    if fading == "rayleigh":
        h_fading = (rng.normal(0, 1.0, total_len) + 1j * rng.normal(0, 1.0, total_len)) / np.sqrt(2.0)
        # Low-pass filter fading envelope to simulate realistic Doppler spread
        b_fade = signal.firwin(31, 0.01)
        h_fading = signal.lfilter(b_fade, [1.0], h_fading).astype(np.complex64)
        tx_signal = tx_signal * h_fading
    elif fading == "rician":
        k_factor = 4.0  # Linear K-factor
        los = np.sqrt(k_factor / (k_factor + 1.0))
        nlos = (rng.normal(0, 1.0, total_len) + 1j * rng.normal(0, 1.0, total_len)) / np.sqrt(2.0 * (k_factor + 1.0))
        b_fade = signal.firwin(31, 0.01)
        nlos_filt = signal.lfilter(b_fade, [1.0], nlos).astype(np.complex64)
        tx_signal = tx_signal * (los + nlos_filt)

    # 3. Carrier Frequency Offset (CFO) and Phase Offset
    t_idx = np.arange(total_len, dtype=np.float32)
    cfo_rot = np.exp(1j * (2.0 * np.pi * cfo_normalized * t_idx + phase_offset_rad)).astype(np.complex64)
    tx_signal = tx_signal * cfo_rot

    # 4. Fractional Timing Offset
    if abs(timing_offset) > 1e-4:
        # Fractional delay via sinc interpolation / FIR filter
        t_shift = np.arange(-8, 9, dtype=np.float64)
        sinc_filter = np.sinc(t_shift - timing_offset)
        tx_signal = signal.lfilter(sinc_filter, [1.0], tx_signal).astype(np.complex64)

    # 5. I/Q Imbalance
    if abs(iq_imbalance_db) > 1e-4:
        gain_i = 10.0 ** (iq_imbalance_db / 20.0)
        gain_q = 1.0 / gain_i
        tx_signal = (tx_signal.real * gain_i + 1j * tx_signal.imag * gain_q).astype(np.complex64)

    # 6. DC Offset
    if abs(dc_offset) > 0.0:
        tx_signal = tx_signal + dc_offset

    # 7. Additive White Gaussian Noise (AWGN)
    sig_power = float(np.mean(np.abs(tx_signal) ** 2))
    if sig_power > 0 and snr_db < 90.0:
        noise_p = sig_power / (10.0 ** (snr_db / 10.0))
        noise = (rng.normal(0, np.sqrt(noise_p / 2.0), total_len) + 1j * rng.normal(0, np.sqrt(noise_p / 2.0), total_len)).astype(np.complex64)
        rx_signal = tx_signal + noise
    else:
        rx_signal = tx_signal

    # 8. Clipping
    if clipping_ratio < 0.999:
        max_val = np.max(np.abs(rx_signal))
        threshold = max_val * clipping_ratio
        mag = np.abs(rx_signal)
        scale = np.where(mag > threshold, threshold / (mag + 1e-12), 1.0)
        rx_signal = (rx_signal * scale).astype(np.complex64)

    manifest: dict[str, Any] = {
        "modulation_name": mod_upper,
        "family": family,
        "order": order,
        "symbol_rate_normalized": round(1.0 / sps, 6),
        "samples_per_symbol": sps,
        "snr_db": snr_db,
        "cfo_normalized": cfo_normalized,
        "timing_offset": timing_offset,
        "pulse_shape": pulse_shape,
        "fading": fading,
        "sample_count": total_len,
        "tx_symbols": syms if "syms" in locals() else None,
        "tx_bits": bits if "bits" in locals() else None,
    }

    return rx_signal, manifest
