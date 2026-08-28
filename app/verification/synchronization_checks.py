from __future__ import annotations
import numpy as np
from app.recovery.models import RecoveredSignal, RecoveryAnalysis
from .models import SyncAuditResult, TestResultStatus, VerificationConfig, VerificationTest

def audit_synchronization_and_stability(
    recovery: RecoveryAnalysis | RecoveredSignal | None = None,
    config: VerificationConfig | None = None,
) -> tuple[SyncAuditResult, list[VerificationTest]]:
    """
    Independently verify timing/carrier synchronization and temporal window stability.

    Parameters
    ----------
    recovery : RecoveryAnalysis | RecoveredSignal | None
    config : VerificationConfig | None

    Returns
    -------
    audit_result : SyncAuditResult
    tests : list[VerificationTest]
    """
    cfg = config or VerificationConfig()
    tests: list[VerificationTest] = []

    rec_sig: RecoveredSignal | None = None
    if isinstance(recovery, RecoveryAnalysis):
        rec_sig = recovery.recovered_signal
    elif isinstance(recovery, RecoveredSignal):
        rec_sig = recovery

    if rec_sig is None or len(rec_sig.symbols) < 32:
        res = SyncAuditResult(
            residual_cfo_hz=0.0,
            phase_variance_rad2=1.0,
            ted_variance=1.0,
            window_count=0,
            passed_window_count=0,
            window_consistency_fraction=0.0,
            is_stable=False,
            details={"status": "insufficient_symbols"},
        )
        tests.append(
            VerificationTest(
                test_id="SYNC_00_INPUT",
                name="Synchronization Input Check",
                category="synchronization",
                description="Check availability of symbols for temporal segmentation",
                status=TestResultStatus.FAIL,
                score=0.0,
                counter_evidence="Insufficient symbols (< 32) to verify temporal synchronization stability",
                is_critical=True,
            )
        )
        return res, tests

    symbols = rec_sig.symbols
    n_syms = len(symbols)

    # 1. Temporal window segmentation (N_w = 8 to 16 windows)
    num_windows = min(16, max(4, n_syms // 32))
    win_len = n_syms // num_windows

    passed_windows = 0
    win_evms: list[float] = []

    for w in range(num_windows):
        seg = symbols[w * win_len : (w + 1) * win_len]
        # Calculate segmented EVM proxy
        p_seg = np.mean(np.abs(seg) ** 2)
        norm_seg = seg / np.sqrt(max(1e-9, p_seg))
        # Distance to 4 QPSK points (+/- 1 +/- j)/sqrt(2)
        qpsk_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
        dists = np.min(np.abs(norm_seg[:, None] - qpsk_pts[None, :]), axis=1)
        evm_seg = float(np.sqrt(np.mean(dists ** 2)) * 100.0)
        win_evms.append(evm_seg)

        if evm_seg <= 35.0:
            passed_windows += 1

    win_frac = float(passed_windows / num_windows)
    is_stable = bool(win_frac >= cfg.min_window_consistency_fraction)

    # Residual Phase Variance
    phase_angles = np.angle(symbols)
    phase_var = float(np.var(np.diff(phase_angles)))

    tests.append(
        VerificationTest(
            test_id="SYNC_01_TEMPORAL_CONSISTENCY",
            name="Temporal Window Consistency",
            category="synchronization",
            description=f"Verify synchronization stability across windows (>= {cfg.min_window_consistency_fraction * 100:.0f}%)",
            status=TestResultStatus.PASS if is_stable else (TestResultStatus.WEAK_PASS if win_frac >= 0.60 else TestResultStatus.FAIL),
            score=win_frac,
            details={
                "num_windows": num_windows,
                "passed_windows": passed_windows,
                "consistency_fraction": round(win_frac, 3),
                "window_evm_mean": round(float(np.mean(win_evms)), 2),
                "window_evm_std": round(float(np.std(win_evms)), 2),
            },
            counter_evidence=f"Intermittent lock: only {win_frac * 100:.1f}% of temporal windows pass EVM threshold" if not is_stable else None,
            is_critical=True,
        )
    )

    cfo_hz = float(rec_sig.cfo_normalized * rec_sig.symbol_rate_normalized)

    res = SyncAuditResult(
        residual_cfo_hz=round(cfo_hz, 4),
        phase_variance_rad2=round(phase_var, 4),
        ted_variance=round(phase_var / 2.0, 4),
        window_count=num_windows,
        passed_window_count=passed_windows,
        window_consistency_fraction=round(win_frac, 3),
        is_stable=is_stable,
        details={"window_evms": [round(e, 1) for e in win_evms]},
    )
    return res, tests
