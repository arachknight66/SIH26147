from __future__ import annotations
import numpy as np
from .fractional_delay import interpolate_sample_cubic
from .models import LockStatus, TimingSyncResult

def gardner_timing_recovery(
    samples: np.ndarray,
    sps: float = 8.0,
    loop_bw: float = 0.015,
    damping: float = 0.707,
    interpolation_method: str = "cubic",
) -> tuple[np.ndarray, np.ndarray, TimingSyncResult]:
    """
    Execute symbol timing synchronization using Gardner Timing Error Detector and 2nd-order loop filter.

    Parameters
    ----------
    samples : np.ndarray
        Complex baseband samples (typically matched-filtered).
    sps : float
        Nominal samples per symbol.
    loop_bw : float
        Normalized loop bandwidth (Bn * T).
    damping : float
        Loop damping factor (zeta).
    interpolation_method : str
        Interpolation method ('cubic' or 'linear').

    Returns
    -------
    symbols : np.ndarray
        Extracted 1-SPS symbol samples.
    strobes : np.ndarray
        Fractional sample index positions of extracted symbols.
    result : TimingSyncResult
    """
    n_samples = len(samples)
    if n_samples < int(sps * 8):
        return np.array([], dtype=np.complex64), np.array([], dtype=np.float64), TimingSyncResult(
            estimated_sps=sps,
            timing_offset_samples=0.0,
            timing_drift=0.0,
            ted_variance=1.0,
            ted_mean=0.0,
            lock_status=LockStatus.UNLOCKED,
            eye_opening_proxy=0.0,
            interpolation_method=interpolation_method,
            valid=False,
        )

    # 2nd-Order Proportional-Integral (PI) Loop Gains
    theta = loop_bw / (damping + 0.25 / damping)
    d = 1.0 + 2.0 * damping * theta + theta ** 2
    kp = (4.0 * damping * theta) / d
    ki = (4.0 * theta ** 2) / d

    # State variables
    symbols_list: list[complex] = []
    strobes_list: list[float] = []
    ted_errors: list[float] = []

    t_k = float(sps)                 # Initial strobe index
    half_sps = sps / 2.0
    integrator = 0.0
    prev_sym = interpolate_sample_cubic(samples, max(0.0, t_k - sps))

    while t_k + half_sps < (n_samples - 2):
        # Sample on-time symbol x(t_k) and half-time midpoint x(t_k - T/2)
        curr_sym = interpolate_sample_cubic(samples, t_k)
        mid_sym = interpolate_sample_cubic(samples, t_k - half_sps)

        # Gardner Timing Error: e_k = Re{ x(t_k - T/2) * [ conj(x(t_k - T)) - conj(x(t_k)) ] }
        diff = np.conj(prev_sym) - np.conj(curr_sym)
        error = float(np.real(mid_sym * diff))
        # Power normalization
        norm_factor = float(abs(curr_sym) ** 2 + abs(prev_sym) ** 2 + 1e-6) / 2.0
        error = float(np.clip(error / norm_factor, -2.0, 2.0))

        ted_errors.append(error)
        symbols_list.append(curr_sym)
        strobes_list.append(t_k)

        # Loop Filter Update
        integrator += ki * error
        v_k = kp * error + integrator

        # Next Strobe Position with NCO adjustment
        step = sps + v_k
        step = float(np.clip(step, 0.5 * sps, 1.5 * sps))
        t_k += step
        prev_sym = curr_sym

    sym_arr = np.array(symbols_list, dtype=np.complex64)
    strobe_arr = np.array(strobes_list, dtype=np.float64)

    if len(ted_errors) < 8:
        return sym_arr, strobe_arr, TimingSyncResult(
            estimated_sps=sps,
            timing_offset_samples=0.0,
            timing_drift=0.0,
            ted_variance=1.0,
            ted_mean=0.0,
            lock_status=LockStatus.UNLOCKED,
            eye_opening_proxy=0.0,
            interpolation_method=interpolation_method,
            valid=False,
        )

    # Statistical Evaluation over steady-state (last 60% of symbols)
    steady_start = int(0.40 * len(ted_errors))
    steady_errors = np.array(ted_errors[steady_start:])
    
    ted_mean = float(np.mean(steady_errors))
    ted_var = float(np.var(steady_errors))
    
    # Timing offset relative to nominal sample grid: median fractional phase
    timing_offset = float(np.median(np.mod(strobe_arr[steady_start:], sps)))

    # Timing drift rate
    if len(strobe_arr) > 10:
        drift = float(np.polyfit(np.arange(len(strobe_arr[steady_start:])), strobe_arr[steady_start:], 1)[0] - sps)
    else:
        drift = 0.0

    # Eye Opening Proxy: 1.0 - (symbol dispersion / mean radius)
    radii = np.abs(sym_arr[steady_start:])
    mean_r = float(np.mean(radii)) if len(radii) > 0 else 1.0
    std_r = float(np.std(radii)) if len(radii) > 0 else 1.0
    eye_proxy = float(np.clip(1.0 - (std_r / (mean_r + 1e-6)), 0.0, 1.0))

    # Lock Declaration (supports multi-amplitude QAM with higher variance if eye opening is strong)
    is_locked = ((ted_var < 0.18 and eye_proxy > 0.30) or (ted_var < 0.55 and eye_proxy > 0.50)) and len(sym_arr) >= 16
    lock_status = LockStatus.LOCKED if is_locked else (LockStatus.AMBIGUOUS if ted_var < 0.60 else LockStatus.UNLOCKED)

    result = TimingSyncResult(
        estimated_sps=round(float(sps + drift), 4),
        timing_offset_samples=round(timing_offset, 4),
        timing_drift=round(drift, 6),
        ted_variance=round(ted_var, 6),
        ted_mean=round(ted_mean, 6),
        lock_status=lock_status,
        eye_opening_proxy=round(eye_proxy, 4),
        interpolation_method=interpolation_method,
        valid=True,
    )
    return sym_arr, strobe_arr, result
