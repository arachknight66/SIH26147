from __future__ import annotations
import numpy as np
from .models import CarrierSyncResult, LockStatus, ModulationFamily

def costas_carrier_recovery(
    symbols: np.ndarray,
    family: ModulationFamily = ModulationFamily.PSK,
    order: int = 4,
    loop_bw: float = 0.015,
    damping: float = 0.707,
) -> tuple[np.ndarray, CarrierSyncResult]:
    """
    Execute decision-directed Costas carrier phase and fine frequency tracking on 1-SPS symbol samples.

    Parameters
    ----------
    symbols : np.ndarray
        1-SPS complex baseband symbol samples.
    family : ModulationFamily
        Modulation family (PSK, QAM).
    order : int
        Modulation order (2, 4, 8, 16).
    loop_bw : float
        Normalized loop bandwidth (Bn * T).
    damping : float
        Loop damping factor (zeta).

    Returns
    -------
    corrected_symbols : np.ndarray
        Phase-corrected 1-SPS symbol samples.
    result : CarrierSyncResult
    """
    n_syms = len(symbols)
    if n_syms < 16:
        return symbols.copy().astype(np.complex64), CarrierSyncResult(
            phase_estimate_rad=0.0,
            phase_error_var=1.0,
            phase_error_rms_rad=1.0,
            residual_cfo_normalized=0.0,
            lock_status=LockStatus.UNLOCKED,
            lock_duration_fraction=0.0,
            loop_bandwidth=loop_bw,
            damping_factor=damping,
            settling_symbols=0,
            valid=False,
        )

    # 2nd-Order Proportional-Integral (PI) Loop Gains
    theta_norm = loop_bw / (damping + 0.25 / damping)
    d = 1.0 + 2.0 * damping * theta_norm + theta_norm ** 2
    kp = (4.0 * damping * theta_norm) / d
    ki = (4.0 * theta_norm ** 2) / d

    corrected = np.zeros(n_syms, dtype=np.complex64)
    phase_errors = np.zeros(n_syms, dtype=np.float64)

    phi = 0.0           # Carrier phase estimate in radians
    freq_integrator = 0.0  # Normalized frequency offset (radians per symbol)

    # Grid slicing helper for 16-QAM
    qam_levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)

    for k, sym in enumerate(symbols):
        # Apply current phase correction
        z_k = sym * np.exp(-1j * phi)
        corrected[k] = z_k

        # Phase Error Detector (PED)
        if family == ModulationFamily.PSK:
            if order == 2:
                # BPSK Costas: e = Im(z) * sign(Re(z))
                error = float(np.imag(z_k) * np.sign(np.real(z_k)))
            elif order == 4:
                # QPSK Costas: e = Im(z)*sign(Re(z)) - Re(z)*sign(Im(z))
                error = float(np.imag(z_k) * np.sign(np.real(z_k)) - np.real(z_k) * np.sign(np.imag(z_k)))
            elif order == 8:
                # 8PSK Decision Directed: e = Im(z * conj(a_hat))
                ang = float(np.angle(z_k))
                sector = round(ang * 8.0 / (2.0 * np.pi))
                a_hat = np.exp(1j * sector * (2.0 * np.pi / 8.0))
                error = float(np.imag(z_k * np.conj(a_hat)))
            else:
                ang = float(np.angle(z_k))
                sector = round(ang * order / (2.0 * np.pi))
                a_hat = np.exp(1j * sector * (2.0 * np.pi / order))
                error = float(np.imag(z_k * np.conj(a_hat)))

        elif family == ModulationFamily.QAM:
            # 16-QAM Decision Directed Grid Slicing
            re_slice = float(qam_levels[np.argmin(np.abs(np.real(z_k) - qam_levels))])
            im_slice = float(qam_levels[np.argmin(np.abs(np.imag(z_k) - qam_levels))])
            a_hat = complex(re_slice, im_slice)
            error = float(np.imag(z_k * np.conj(a_hat)))
        else:
            error = float(np.imag(z_k) * np.sign(np.real(z_k)))

        # Normalize error to avoid extreme swings
        error = float(np.clip(error, -np.pi / 2.0, np.pi / 2.0))
        phase_errors[k] = error

        # 2nd-order loop filter update
        freq_integrator += ki * error
        phi += freq_integrator + kp * error

    # Statistical Evaluation over steady-state (last 60% of symbols)
    steady_start = int(0.40 * n_syms)
    steady_errors = phase_errors[steady_start:]
    
    pe_var = float(np.var(steady_errors))
    pe_rms = float(np.sqrt(np.mean(steady_errors ** 2)))
    residual_cfo = float(freq_integrator / (2.0 * np.pi))
    phase_est = float(np.mod(phi, 2.0 * np.pi))

    # Lock criteria based on phase error variance and sector boundaries
    max_tol_var = 0.08 if order <= 4 else (0.04 if order == 8 else 0.06)
    in_lock_mask = np.abs(steady_errors) < (np.pi / (order if family == ModulationFamily.PSK else 4))
    lock_fraction = float(np.mean(in_lock_mask))

    is_locked = (pe_var < max_tol_var and lock_fraction > 0.65)
    lock_status = LockStatus.LOCKED if is_locked else (LockStatus.AMBIGUOUS if lock_fraction > 0.45 else LockStatus.UNLOCKED)

    result = CarrierSyncResult(
        phase_estimate_rad=round(phase_est, 4),
        phase_error_var=round(pe_var, 6),
        phase_error_rms_rad=round(pe_rms, 4),
        residual_cfo_normalized=round(residual_cfo, 6),
        lock_status=lock_status,
        lock_duration_fraction=round(lock_fraction, 4),
        loop_bandwidth=loop_bw,
        damping_factor=damping,
        settling_symbols=steady_start,
        valid=True,
    )
    return corrected, result
