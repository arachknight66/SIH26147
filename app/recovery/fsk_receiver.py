from __future__ import annotations
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity
from .constellation import ConstellationResult
from .demodulation import demodulate_bfsk
from .frequency_sync import estimate_bfsk_frequencies
from .models import (
    CarrierSyncResult,
    ConstellationResult,
    LockStatus,
    ModulationFamily,
    RecoveryCandidate,
    RecoveryConfig,
    RecoveryQuality,
    RecoveryQualityLevel,
    RecoveryStatus,
    SynchronizationResult,
    TimingSyncResult,
)

def run_fsk_receiver(
    samples: np.ndarray,
    candidate_id: int = 1,
    sps: float = 8.0,
    phase3_score: float = 0.5,
    config: RecoveryConfig | None = None,
) -> RecoveryCandidate:
    """
    Execute end-to-end 2-FSK / BFSK receiver synchronization, tone correlation, and recovery.

    Parameters
    ----------
    samples : np.ndarray
        Prepared complex64 baseband samples.
    candidate_id : int
        Candidate identifier.
    sps : float
        Estimated samples per symbol.
    phase3_score : float
        Prior score from Phase 3.
    config : RecoveryConfig | None
        Recovery configuration.

    Returns
    -------
    RecoveryCandidate
    """
    cfg = config or RecoveryConfig()
    diagnostics: list[Diagnostic] = []

    if len(samples) < int(sps * 8):
        diagnostics.append(
            Diagnostic(
                code="INSUFFICIENT_SAMPLES_FOR_FSK",
                message="Signal length too short for reliable FSK tone correlation.",
                severity=DiagnosticSeverity.WARNING,
            )
        )
        return RecoveryCandidate(
            candidate_id=candidate_id,
            family=ModulationFamily.FSK,
            order=2,
            symbol_rate_normalized=1.0 / sps,
            samples_per_symbol=sps,
            phase3_score=phase3_score,
            status=RecoveryStatus.RECOVERY_INCONCLUSIVE,
            quality=RecoveryQuality(
                composite_score=0.0,
                evm_score=0.0,
                timing_lock_score=0.0,
                carrier_lock_score=0.0,
                constellation_score=0.0,
                decision_margin_score=0.0,
                window_consistency_score=0.0,
                quality_level=RecoveryQualityLevel.REJECTED,
            ),
            diagnostics=diagnostics,
        )

    # 1. Frequency Acquisition
    f0, f1, delta_f, freq_sync = estimate_bfsk_frequencies(samples, sps=sps)
    is_valid_separation = (0.03 <= delta_f <= 0.45)

    if not is_valid_separation:
        diagnostics.append(
            Diagnostic(
                code="FSK_SEPARATION_INVALID",
                message=f"Estimated FSK tone separation delta_f={delta_f:.4f} is outside physical limits.",
                severity=DiagnosticSeverity.WARNING,
            )
        )

    # 2. Dual-Tone Demodulation
    demod_result = demodulate_bfsk(samples, f0=f0, f1=f1, sps=sps)

    # 3. FSK Decision Margin & Quality Metrics
    soft_margins = np.abs(demod_result.soft_decisions)
    mean_margin = float(np.mean(soft_margins)) if len(soft_margins) > 0 else 0.0

    # 4. Timing & Carrier Representation
    is_locked = (is_valid_separation and mean_margin >= 0.40 and phase3_score >= 0.20)
    
    timing_sync = TimingSyncResult(
        estimated_sps=sps,
        timing_offset_samples=0.0,
        timing_drift=0.0,
        ted_variance=0.05 if is_locked else 0.50,
        ted_mean=0.0,
        lock_status=LockStatus.LOCKED if is_locked else LockStatus.UNLOCKED,
        eye_opening_proxy=float(np.clip(mean_margin, 0.0, 1.0)),
        interpolation_method="block_correlation",
        valid=True,
    )

    carrier_sync = CarrierSyncResult(
        phase_estimate_rad=0.0,
        phase_error_var=0.01 if is_locked else 0.50,
        phase_error_rms_rad=0.10 if is_locked else 0.70,
        residual_cfo_normalized=freq_sync.coarse_cfo_normalized,
        lock_status=LockStatus.LOCKED if is_locked else LockStatus.UNLOCKED,
        lock_duration_fraction=1.0 if is_locked else 0.0,
        loop_bandwidth=cfg.loop_bandwidth,
        damping_factor=cfg.damping_factor,
        settling_symbols=0,
        valid=True,
    )

    sync_result = SynchronizationResult(
        frequency=freq_sync,
        carrier=carrier_sync,
        timing=timing_sync,
        is_locked=is_locked,
    )

    const_result = ConstellationResult(
        symbols=demod_result.soft_decisions.astype(np.complex64),
        cluster_centroids=np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex64),
        cluster_variances=np.array([float(np.var(demod_result.soft_decisions))], dtype=np.float64),
        rms_radius=1.0,
        evm_linear=float(np.clip(1.0 - mean_margin, 0.0, 1.0)),
        evm_percent=float(np.clip(1.0 - mean_margin, 0.0, 1.0) * 100.0),
        evm_db=float(20.0 * np.log10(max(1e-4, 1.0 - mean_margin))),
        decision_margin=round(mean_margin, 4),
        phase_error_rms_rad=0.0,
        amplitude_error_rms=0.0,
        rotational_ambiguity_deg=(0.0,),
        valid=True,
    )

    # Quality scoring
    if is_locked and mean_margin >= 0.50:
        status = RecoveryStatus.RECOVERED
        q_level = RecoveryQualityLevel.HIGH
        comp_score = 0.80 + 0.20 * mean_margin
    elif is_locked and mean_margin >= 0.40:
        status = RecoveryStatus.RECOVERED
        q_level = RecoveryQualityLevel.MODERATE
        comp_score = 0.60 + 0.20 * mean_margin
    else:
        status = RecoveryStatus.RECOVERY_INCONCLUSIVE
        q_level = RecoveryQualityLevel.REJECTED
        comp_score = 0.15 * mean_margin

    quality = RecoveryQuality(
        composite_score=round(float(comp_score), 3),
        evm_score=round(float(mean_margin), 3),
        timing_lock_score=1.0 if timing_sync.lock_status == LockStatus.LOCKED else 0.0,
        carrier_lock_score=1.0 if carrier_sync.lock_status == LockStatus.LOCKED else 0.0,
        constellation_score=round(float(mean_margin), 3),
        decision_margin_score=round(float(mean_margin), 3),
        window_consistency_score=1.0,
        quality_level=q_level,
    )

    return RecoveryCandidate(
        candidate_id=candidate_id,
        family=ModulationFamily.FSK,
        order=2,
        symbol_rate_normalized=1.0 / sps,
        samples_per_symbol=sps,
        phase3_score=phase3_score,
        status=status,
        quality=quality,
        synchronization=sync_result,
        constellation=const_result,
        demodulation=demod_result,
        diagnostics=diagnostics,
    )
