from __future__ import annotations
import numpy as np
from app.models.metadata import Diagnostic, DiagnosticSeverity
from .carrier_sync import costas_carrier_recovery
from .constellation import analyze_constellation
from .demodulation import demodulate_16qam
from .frequency_sync import correct_frequency_offset, estimate_coarse_cfo_mth_power
from .matched_filter import apply_matched_filter
from .models import (
    LockStatus,
    ModulationFamily,
    RecoveryCandidate,
    RecoveryConfig,
    RecoveryQuality,
    RecoveryQualityLevel,
    RecoveryStatus,
    SynchronizationResult,
)
from .timing_sync import gardner_timing_recovery

def run_qam_receiver(
    samples: np.ndarray,
    order: int = 16,
    candidate_id: int = 1,
    sps: float = 8.0,
    phase3_score: float = 0.5,
    rrc_alpha: float = 0.35,
    config: RecoveryConfig | None = None,
) -> RecoveryCandidate:
    """
    Execute complete receiver synchronization and demodulation for 16-QAM signals.

    Parameters
    ----------
    samples : np.ndarray
        Prepared baseband complex IQ samples.
    order : int
        Modulation order (16).
    candidate_id : int
        Candidate identifier.
    sps : float
        Samples per symbol.
    phase3_score : float
        Prior score from Phase 3.
    rrc_alpha : float
        RRC roll-off factor.
    config : RecoveryConfig | None
        Recovery configuration.

    Returns
    -------
    RecoveryCandidate
    """
    cfg = config or RecoveryConfig()
    diagnostics: list[Diagnostic] = []

    # 1. Coarse CFO Estimation (4th power)
    freq_sync = estimate_coarse_cfo_mth_power(samples, order=order, family=ModulationFamily.QAM)
    cfo_corrected = correct_frequency_offset(samples, freq_sync.coarse_cfo_normalized)

    # 2. Matched Filtering (RRC)
    mf_samples, _ = apply_matched_filter(
        cfo_corrected,
        sps=sps,
        alpha=rrc_alpha,
        span_symbols=cfg.filter_span_symbols,
    )

    # 3. Timing Synchronization (Gardner TED)
    sym_timed, strobes, timing_sync = gardner_timing_recovery(
        mf_samples,
        sps=sps,
        loop_bw=cfg.loop_bandwidth,
        damping=cfg.damping_factor,
    )

    if len(sym_timed) < cfg.min_recovery_symbols:
        diagnostics.append(
            Diagnostic(
                code="INSUFFICIENT_SYMBOLS",
                message="Insufficient symbols extracted for 16-QAM receiver.",
                severity=DiagnosticSeverity.WARNING,
            )
        )
        return RecoveryCandidate(
            candidate_id=candidate_id,
            family=ModulationFamily.QAM,
            order=order,
            symbol_rate_normalized=1.0 / sps,
            samples_per_symbol=sps,
            phase3_score=phase3_score,
            status=RecoveryStatus.TIMING_UNLOCKED,
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

    # 4. Carrier Synchronization (Decision-Directed 16QAM Loop)
    sym_carrier, carrier_sync = costas_carrier_recovery(
        sym_timed,
        family=ModulationFamily.QAM,
        order=order,
        loop_bw=cfg.loop_bandwidth,
        damping=cfg.damping_factor,
    )

    is_sync_locked = (
        timing_sync.lock_status in (LockStatus.LOCKED, LockStatus.AMBIGUOUS)
        and carrier_sync.lock_status in (LockStatus.LOCKED, LockStatus.AMBIGUOUS)
    )

    sync_result = SynchronizationResult(
        frequency=freq_sync,
        carrier=carrier_sync,
        timing=timing_sync,
        is_locked=is_sync_locked,
    )

    # 5. Constellation & EVM Analysis on 16 Grid Points
    const_result = analyze_constellation(
        sym_carrier,
        family=ModulationFamily.QAM,
        order=order,
    )

    # 6. Demodulation
    demod_result = demodulate_16qam(const_result.symbols)

    # 7. Receiver Diagnostics
    is_high_evm = (const_result.evm_percent > (cfg.evm_threshold_max * 100.0))
    if is_high_evm:
        diagnostics.append(
            Diagnostic(
                code="HIGH_EVM",
                message=f"16-QAM EVM is high ({const_result.evm_percent:.1f}%), suggesting distortion or wrong modulation hypothesis.",
                severity=DiagnosticSeverity.WARNING,
            )
        )

    if carrier_sync.lock_status == LockStatus.UNLOCKED:
        diagnostics.append(
            Diagnostic(
                code="CARRIER_UNLOCKED",
                message=f"16-QAM carrier tracking did not achieve phase lock (phase error var={carrier_sync.phase_error_var:.4f}).",
                severity=DiagnosticSeverity.WARNING,
            )
        )

    if timing_sync.lock_status == LockStatus.UNLOCKED:
        diagnostics.append(
            Diagnostic(
                code="TIMING_UNLOCKED",
                message=f"16-QAM timing recovery did not achieve stable lock (TED var={timing_sync.ted_variance:.4f}).",
                severity=DiagnosticSeverity.WARNING,
            )
        )

    # 8. Composite Recovery Quality Scoring
    evm_score = float(np.clip(1.0 - (const_result.evm_linear / 0.35), 0.0, 1.0))
    timing_score = 1.0 if timing_sync.lock_status == LockStatus.LOCKED else (0.6 if timing_sync.lock_status == LockStatus.AMBIGUOUS else 0.0)
    carrier_score = 1.0 if carrier_sync.lock_status == LockStatus.LOCKED else (0.6 if carrier_sync.lock_status == LockStatus.AMBIGUOUS else 0.0)
    margin_score = const_result.decision_margin

    comp_score = 0.35 * evm_score + 0.25 * timing_score + 0.25 * carrier_score + 0.15 * margin_score

    # Rejection if EVM > 20% or decision margin < 0.35 or carrier unlocked
    if is_high_evm or const_result.evm_percent > 20.0 or const_result.decision_margin < 0.35 or carrier_sync.lock_status == LockStatus.UNLOCKED:
        status = RecoveryStatus.RECOVERY_INCONCLUSIVE
        q_level = RecoveryQualityLevel.REJECTED
        comp_score = min(comp_score, 0.20)
    elif comp_score >= 0.60 and const_result.decision_margin >= 0.40 and const_result.evm_percent <= 20.0:
        status = RecoveryStatus.RECOVERED
        q_level = RecoveryQualityLevel.HIGH if comp_score >= 0.70 else RecoveryQualityLevel.MODERATE
    elif comp_score >= 0.45 and const_result.decision_margin >= 0.35 and const_result.evm_percent <= 20.0:
        status = RecoveryStatus.RECOVERED
        q_level = RecoveryQualityLevel.MODERATE
    elif comp_score >= 0.25:
        status = RecoveryStatus.RECOVERY_INCONCLUSIVE
        q_level = RecoveryQualityLevel.LOW
    else:
        status = RecoveryStatus.RECOVERY_INCONCLUSIVE
        q_level = RecoveryQualityLevel.REJECTED

    quality = RecoveryQuality(
        composite_score=round(float(comp_score), 3),
        evm_score=round(float(evm_score), 3),
        timing_lock_score=round(float(timing_score), 3),
        carrier_lock_score=round(float(carrier_score), 3),
        constellation_score=round(float(evm_score), 3),
        decision_margin_score=round(float(margin_score), 3),
        window_consistency_score=1.0,
        quality_level=q_level,
        post_sync_snr_db=round(float(-const_result.evm_db), 2),
    )

    return RecoveryCandidate(
        candidate_id=candidate_id,
        family=ModulationFamily.QAM,
        order=order,
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
