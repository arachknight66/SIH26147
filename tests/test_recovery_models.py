import numpy as np
import pytest
from app.modulation.models import ModulationFamily
from app.recovery.models import (
    BitStreamStatus,
    CarrierSyncResult,
    ConstellationResult,
    DemodulationResult,
    FrequencySyncResult,
    LockStatus,
    RecoveredSignal,
    RecoveryAnalysis,
    RecoveryCandidate,
    RecoveryConfig,
    RecoveryQuality,
    RecoveryQualityLevel,
    RecoveryStatus,
    SynchronizationResult,
    TimingSyncResult,
)

def test_recovery_models_instantiation():
    cfg = RecoveryConfig(max_candidates=3, loop_bandwidth=0.02)
    assert cfg.max_candidates == 3
    assert cfg.loop_bandwidth == 0.02

    freq_sync = FrequencySyncResult(
        coarse_cfo_normalized=0.005,
        residual_cfo_normalized=0.0001,
        cfo_variance=1e-5,
        capture_bandwidth=0.125,
        method="4th_power_fft",
        ambiguity_set=(0.005, 0.255),
        is_ambiguous=False,
        valid=True,
    )
    assert freq_sync.coarse_cfo_normalized == 0.005

    carrier_sync = CarrierSyncResult(
        phase_estimate_rad=0.12,
        phase_error_var=0.01,
        phase_error_rms_rad=0.10,
        residual_cfo_normalized=0.0001,
        lock_status=LockStatus.LOCKED,
        lock_duration_fraction=0.95,
        loop_bandwidth=0.015,
        damping_factor=0.707,
        settling_symbols=128,
        valid=True,
    )
    assert carrier_sync.lock_status == LockStatus.LOCKED

    timing_sync = TimingSyncResult(
        estimated_sps=8.0,
        timing_offset_samples=0.25,
        timing_drift=0.0,
        ted_variance=0.02,
        ted_mean=0.0,
        lock_status=LockStatus.LOCKED,
        eye_opening_proxy=0.85,
        interpolation_method="cubic",
        valid=True,
    )
    assert timing_sync.lock_status == LockStatus.LOCKED

    sync_res = SynchronizationResult(
        frequency=freq_sync,
        carrier=carrier_sync,
        timing=timing_sync,
        is_locked=True,
    )
    assert sync_res.is_locked

    cand = RecoveryCandidate(
        candidate_id=1,
        family=ModulationFamily.PSK,
        order=4,
        symbol_rate_normalized=0.125,
        samples_per_symbol=8.0,
        phase3_score=0.88,
        status=RecoveryStatus.RECOVERED,
        quality=RecoveryQuality(
            composite_score=0.92,
            evm_score=0.90,
            timing_lock_score=1.0,
            carrier_lock_score=1.0,
            constellation_score=0.90,
            decision_margin_score=0.85,
            window_consistency_score=1.0,
            quality_level=RecoveryQualityLevel.HIGH,
        ),
        synchronization=sync_res,
    )
    assert cand.label == "QPSK"
    assert cand.status == RecoveryStatus.RECOVERED
