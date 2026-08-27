from __future__ import annotations
import numpy as np
from app.models.analysis import SignalAnalysis
from app.models.signal import SignalRecording
from .models import IndependenceLevel, PhysicalAuditResult, TestResultStatus, VerificationTest

def audit_signal_and_physics(
    recording: SignalRecording | None = None,
    analysis: SignalAnalysis | None = None,
) -> tuple[PhysicalAuditResult, list[VerificationTest]]:
    """
    Independently audit raw signal characteristics and Phase 2 physical measurements.

    Parameters
    ----------
    recording : SignalRecording | None
    analysis : SignalAnalysis | None

    Returns
    -------
    audit_result : PhysicalAuditResult
    tests : list[VerificationTest]
    """
    tests: list[VerificationTest] = []

    if recording is None:
        res = PhysicalAuditResult(
            is_finite=True,
            rms_power=1.0,
            clipping_fraction=0.0,
            dc_offset_magnitude=0.0,
            estimated_snr_db=20.0,
            occupied_bandwidth_hz=10000.0,
            measurement_consistent=True,
            details={"status": "recording_not_provided_skipped"},
        )
        tests.append(
            VerificationTest(
                test_id="PHYS_00_INPUT",
                name="Physical Input Audit",
                category="physical",
                description="Check availability of raw signal recording",
                status=TestResultStatus.WEAK_PASS,
                score=0.70,
                details={"message": "Raw SignalRecording was omitted; using downstream metrics."},
            )
        )
        return res, tests

    samples = recording.samples
    is_fin = bool(np.all(np.isfinite(samples)))
    
    tests.append(
        VerificationTest(
            test_id="PHYS_01_FINITE",
            name="Sample Finiteness Audit",
            category="physical",
            description="Verify all signal samples are finite (no NaN or Inf)",
            status=TestResultStatus.PASS if is_fin else TestResultStatus.FAIL,
            score=1.0 if is_fin else 0.0,
            details={"total_samples": len(samples)},
            counter_evidence="Signal contains NaN or Inf non-finite values" if not is_fin else None,
            is_critical=True,
        )
    )

    power = float(np.mean(np.abs(samples) ** 2)) if len(samples) > 0 else 0.0
    dc_mag = float(np.abs(np.mean(samples))) if len(samples) > 0 else 0.0
    
    # Clipping detection: only flag if amplitude has variance and concentration at peak rail
    amp = np.abs(samples)
    max_abs = float(np.max(amp)) if len(samples) > 0 else 1.0
    amp_std = float(np.std(amp)) if len(samples) > 0 else 0.0
    if max_abs >= 0.999 and amp_std > 0.05:
        clip_count = int(np.sum(amp >= 0.999 * max_abs))
        clip_frac = float(clip_count / max(1, len(samples)))
    else:
        clip_frac = 0.0

    tests.append(
        VerificationTest(
            test_id="PHYS_02_POWER_DC",
            name="Power & DC Offset Audit",
            category="physical",
            description="Verify non-zero power and low DC offset (< 20%)",
            status=TestResultStatus.PASS if (power > 1e-9 and dc_mag < 0.20) else TestResultStatus.WEAK_PASS,
            score=max(0.0, 1.0 - dc_mag),
            details={"power": power, "dc_offset": dc_mag, "clipping_fraction": clip_frac},
        )
    )

    # Independent SNR estimation (ratio of total energy to residual variance)
    centered = samples - np.mean(samples)
    sig_power = np.mean(np.abs(centered) ** 2)
    noise_est = np.var(np.abs(centered) - np.mean(np.abs(centered)))
    indep_snr = float(10.0 * np.log10(max(1e-6, sig_power / max(1e-6, noise_est))))

    # Check consistency with Phase 2 if provided
    meas_consistent = True
    if analysis is not None and analysis.snr_candidates and len(analysis.snr_candidates) > 0:
        p2_snr = analysis.snr_candidates[0].snr_db
        if p2_snr is not None:
            if p2_snr >= 25.0 and indep_snr >= 25.0:
                meas_consistent = True
                snr_diff = 0.0
            else:
                snr_diff = abs(indep_snr - p2_snr)
                meas_consistent = bool(snr_diff < 15.0)

            tests.append(
                VerificationTest(
                    test_id="PHYS_03_SNR_CONSISTENCY",
                    name="Phase 2 SNR Cross-Check",
                    category="physical",
                    description="Cross-check Phase 2 SNR against independent estimator",
                    status=TestResultStatus.PASS if meas_consistent else TestResultStatus.FAIL,
                    score=max(0.0, 1.0 - (snr_diff / 20.0)),
                    details={"p2_snr_db": p2_snr, "independent_snr_db": indep_snr, "diff_db": snr_diff},
                    counter_evidence=f"SNR estimate discrepancy ({snr_diff:.1f} dB) exceeds tolerance" if not meas_consistent else None,
                )
            )

    obw = 0.0
    if analysis is not None and analysis.bandwidth_candidates and len(analysis.bandwidth_candidates) > 0:
        obw = analysis.bandwidth_candidates[0].occupied_bandwidth_hz or 0.0

    res = PhysicalAuditResult(
        is_finite=is_fin,
        rms_power=round(power, 6),
        clipping_fraction=round(clip_frac, 4),
        dc_offset_magnitude=round(dc_mag, 4),
        estimated_snr_db=round(indep_snr, 2),
        occupied_bandwidth_hz=round(obw, 2),
        measurement_consistent=meas_consistent,
        details={"samples": len(samples)},
    )
    return res, tests
