from __future__ import annotations
from typing import Callable
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity
from .models import LockStatus, RecoveryCandidate, RecoveryQuality, RecoveryQualityLevel

def evaluate_windowed_stability(
    samples: np.ndarray,
    receiver_runner: Callable[[np.ndarray], RecoveryCandidate],
    num_windows: int = 4,
) -> tuple[float, list[Diagnostic]]:
    """
    Evaluate multi-window temporal stability of receiver synchronization across non-overlapping sub-windows.

    Parameters
    ----------
    samples : np.ndarray
        Prepared baseband complex IQ samples.
    receiver_runner : Callable[[np.ndarray], RecoveryCandidate]
        Function executing candidate receiver on sample slice.
    num_windows : int
        Number of sub-windows to evaluate (default 4).

    Returns
    -------
    consistency_score : float
        Stability score in [0.0, 1.0].
    diagnostics : list[Diagnostic]
    """
    n = len(samples)
    min_window_len = 1024
    if n < min_window_len * 2:
        return 1.0, []

    win_len = n // num_windows
    if win_len < 512:
        return 1.0, []

    diagnostics: list[Diagnostic] = []
    cfos: list[float] = []
    evms: list[float] = []
    lock_flags: list[bool] = []

    for w in range(num_windows):
        seg = samples[w * win_len:(w + 1) * win_len]
        cand = receiver_runner(seg)
        
        if cand.synchronization:
            cfos.append(cand.synchronization.frequency.coarse_cfo_normalized)
            lock_flags.append(cand.synchronization.is_locked)
        else:
            lock_flags.append(False)

        if cand.constellation:
            evms.append(cand.constellation.evm_percent)

    if not lock_flags:
        return 0.0, diagnostics

    # Metrics
    lock_persistence = float(np.mean(lock_flags))
    cfo_drift = float(np.ptp(cfos)) if len(cfos) >= 2 else 0.0
    evm_std = float(np.std(evms)) if len(evms) >= 2 else 0.0

    if lock_persistence < 0.50:
        diagnostics.append(
            Diagnostic(
                code="UNSTABLE_LOCK_PERSISTENCE",
                message=f"Receiver only achieved lock in {int(lock_persistence * 100)}% of analysis windows.",
                severity=DiagnosticSeverity.WARNING,
            )
        )

    if cfo_drift > 0.02:
        diagnostics.append(
            Diagnostic(
                code="SIGNAL_NONSTATIONARY_CFO",
                message=f"Significant carrier frequency drift detected across windows (drift={cfo_drift:.4f} cycles/sample).",
                severity=DiagnosticSeverity.INFO,
            )
        )

    cfo_score = float(np.clip(1.0 - (cfo_drift / 0.03), 0.0, 1.0))
    evm_score = float(np.clip(1.0 - (evm_std / 15.0), 0.0, 1.0))
    consistency = 0.50 * lock_persistence + 0.25 * cfo_score + 0.25 * evm_score

    return round(float(np.clip(consistency, 0.0, 1.0)), 3), diagnostics
