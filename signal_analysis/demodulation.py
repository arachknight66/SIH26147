from .constants import DEFAULT_MAX_ANALYSIS_SAMPLES
import numpy as np
from typing import List, Dict, Tuple, Optional
from .models import SignalRecording, ModulationHypothesis, SynchronizationResult, DemodulationResult, Diagnostic, Severity
from .synchronization import (
    estimate_coarse_cfo_psk_qam, 
    recover_timing_gardner, 
    recover_carrier_costas,
    recover_timing_fsk,
    fsk_dual_correlator
)

CONSTELLATION_MAPS = {
    "BPSK": {
        "points": np.array([-1, 1], dtype=np.complex64),
        "bits": [[0], [1]],
        "bits_per_symbol": 1
    },
    "QPSK": {
        "points": np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2),
        "bits": [[1, 1], [0, 1], [0, 0], [1, 0]],
        "bits_per_symbol": 2
    },
    "8PSK": {
        "points": np.exp(1j * np.array([0, 1, 3, 2, 6, 7, 5, 4]) * np.pi/4).astype(np.complex64),
        # 8PSK Gray code
        "bits": [
            [0,0,0], [0,0,1], [0,1,1], [0,1,0],
            [1,1,0], [1,1,1], [1,0,1], [1,0,0]
        ],
        "bits_per_symbol": 3
    },
    "16-QAM": {
        "points": np.array([
            -3+3j, -1+3j, 1+3j, 3+3j,
            -3+1j, -1+1j, 1+1j, 3+1j,
            -3-1j, -1-1j, 1-1j, 3-1j,
            -3-3j, -1-3j, 1-3j, 3-3j
        ], dtype=np.complex64) / np.sqrt(10),
        "bits": [
            [0,0,0,0], [0,0,0,1], [0,1,0,1], [0,1,0,0],
            [0,0,1,0], [0,0,1,1], [0,1,1,1], [0,1,1,0],
            [1,0,1,0], [1,0,1,1], [1,1,1,1], [1,1,1,0],
            [1,0,0,0], [1,0,0,1], [1,1,0,1], [1,1,0,0]
        ],
        "bits_per_symbol": 4
    }
}

def psk_qam_demodulate(symbols: np.ndarray, modulation: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (hard_bits, soft_llrs, evm)
    """
    if modulation not in CONSTELLATION_MAPS:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32), 100.0
        
    cmap = CONSTELLATION_MAPS[modulation]
    points = cmap["points"]
    bits = cmap["bits"]
    bps = cmap["bits_per_symbol"]
    
    # Pre-calculate bit arrays
    bits_arr = np.array(bits, dtype=np.uint8)
    
    # Calculate EVM
    # EVM is RMS error normalized to max constellation amplitude or RMS amplitude.
    # Here, points are already normalized to unit RMS power.
    distances = np.abs(symbols[:, None] - points[None, :])
    nearest_idx = np.argmin(distances, axis=1)
    error_vectors = symbols - points[nearest_idx]
    
    # Ignore first 100 symbols for EVM calculation to avoid loop lock transients
    if len(error_vectors) > 100:
        evm = float(np.sqrt(np.mean(np.abs(error_vectors[100:])**2)) * 100.0)
    else:
        evm = float(np.sqrt(np.mean(np.abs(error_vectors)**2)) * 100.0)

    
    # Derive noise variance from EVM
    # EVM = sqrt(N0 / Es) * 100
    # N0 = (EVM / 100)^2 (since Es=1)
    noise_var = (evm / 100.0)**2 + 1e-9
    
    # LLR calculation (max-log approximation)
    hard_bits = bits_arr[nearest_idx].flatten()
    
    soft_llrs = []
    distances_sq = distances ** 2
    for b in range(bps):
        # find min dist for bit=0 and bit=1
        idx0 = np.where(bits_arr[:, b] == 0)[0]
        idx1 = np.where(bits_arr[:, b] == 1)[0]
        
        min_d0 = np.min(distances_sq[:, idx0], axis=1)
        min_d1 = np.min(distances_sq[:, idx1], axis=1)
        
        # LLR > 0 means bit=1 is more likely.
        # If min_d1 < min_d0, point is closer to 1-set, so (min_d0 - min_d1) > 0.
        llr = (min_d0 - min_d1) / noise_var
        soft_llrs.append(llr)
        
    soft_llrs = np.column_stack(soft_llrs).flatten().astype(np.float32)
    return hard_bits, soft_llrs, evm

def attempt_synchronization(recording: SignalRecording, hyp: ModulationHypothesis, config: dict) -> DemodulationResult:
    """Run one modulation hypothesis through the native Phase 4 receiver."""
    c_params = hyp.candidate_parameters
    if c_params.symbol_rate is None or c_params.samples_per_symbol is None:
        diag = Diagnostic(Severity.ERROR, "SYNC_MISSING_PARAMS", "Hypothesis lacks required symbol rate.", "")
        sync_res = SynchronizationResult(0.0, "cycles/sample", 0.0, False, False, 999.0, 100.0, [diag])
        return DemodulationResult(np.array([]), np.array([]), 1, np.array([]), sync_res, hyp.label, False)
        
    from .native import demodulate as native_demodulate, require_native

    native = require_native()
    native_config = native.ReceiverConfig()
    native_config.modulation = hyp.label
    native_config.samples_per_symbol = float(c_params.samples_per_symbol)
    native_config.sample_rate_hz = recording.sample_rate_hz.value
    native_config.pulse_shape = str(config.get("pulse_shape", "auto"))
    native_config.rrc_rolloff = float(config.get("rrc_rolloff", 0.35))
    native_config.acquisition_symbols = int(config.get("acquisition_symbols", 16))
    if c_params.carrier_offset is not None:
        if c_params.carrier_offset_unit == "Hz" and recording.sample_rate_hz.value:
            native_config.coarse_cfo_cycles_per_sample = c_params.carrier_offset / recording.sample_rate_hz.value
        elif c_params.carrier_offset_unit == "cycles/sample":
            native_config.coarse_cfo_cycles_per_sample = c_params.carrier_offset
    if "phase_reference_radians" in config:
        native_config.phase_reference_radians = float(config["phase_reference_radians"])
    if "carrier_reference_cycles_per_sample" in config:
        native_config.carrier_reference_cycles_per_sample = float(config["carrier_reference_cycles_per_sample"])
    if "noise_variance" in config:
        native_config.noise_variance = float(config["noise_variance"])

    samples = recording.samples[:DEFAULT_MAX_ANALYSIS_SAMPLES]
    if samples.ndim > 1:
        samples = samples[:, 0]
    samples = np.ascontiguousarray(samples, dtype=np.complex64)
    native_result = native_demodulate(samples, config=native_config)
    diagnostics = [
        Diagnostic(
            getattr(Severity, item.severity.name),
            item.code,
            item.message,
            item.evidence,
        )
        for item in native_result.diagnostics
    ]
    hard_bits = np.asarray(native_result.hard_bits)
    soft_llrs = np.asarray(native_result.soft_llrs)
    requested_backend = config.get("compute_backend", "cpu")
    from .acceleration import should_use_gpu
    if should_use_gpu(requested_backend) and native_result.acquisition_status.name == "LOCKED":
        try:
            from .gpu_dsp import gpu_demap_symbols, gpu_demap_fsk
            name = hyp.label.upper()
            if name in {"BPSK", "QPSK", "OQPSK", "8PSK", "16-QAM", "64-QAM", "256-QAM"}:
                hard_bits, soft_llrs = gpu_demap_symbols(
                    np.asarray(native_result.symbols), name, float(native_result.noise_variance)
                )
            elif name in {"2-FSK", "4-FSK", "8-FSK", "MSK"}:
                tone_count = {"2-FSK": 2, "4-FSK": 4, "8-FSK": 8, "MSK": 2}[name]
                hard_bits, soft_llrs = gpu_demap_fsk(
                    np.asarray(native_result.symbols), tone_count, float(native_result.noise_variance)
                )
            else:
                raise ValueError(f"no GPU demapper for {name}")
            diagnostics.append(Diagnostic(
                Severity.INFO, "GPU_RECEIVER_DECISIONS",
                "CUDA recomputed post-lock hard decisions and max-log LLRs; native acquisition remains authoritative.",
                f"modulation={name}; symbols={len(native_result.symbols)}",
            ))
        except ValueError as exc:
            diagnostics.append(Diagnostic(
                Severity.INFO, "GPU_RECEIVER_STAGE_NOT_APPLICABLE",
                "This receiver profile has no equivalent CUDA decision kernel; native CPU decisions were retained.", str(exc),
            ))
    locked = native_result.acquisition_status.name == "LOCKED"
    # Receiver evidence is combined with the independent upstream ranking; low EVM alone is not confirmation.
    hypothesis_confirmed = locked and hyp.score >= float(config.get("receiver_hypothesis_threshold", 0.55))
    if not hypothesis_confirmed and not diagnostics:
        diagnostics.append(Diagnostic(Severity.WARNING, "SYNC_FAILED", "Native receiver did not lock this hypothesis.", f"status={native_result.acquisition_status.name}"))
    sync_res = SynchronizationResult(
        cfo_estimate=float(native_result.carrier_offset),
        cfo_unit=native_result.carrier_offset_unit,
        timing_offset_fractional_symbols=float(native_result.timing_offset_samples / c_params.samples_per_symbol),
        symbol_clock_locked=locked,
        carrier_locked=locked,
        lock_quality_metric=float(native_result.timing_error),
        evm_percent=float(native_result.evm_percent),
        diagnostics=diagnostics,
        acquisition_status=native_result.acquisition_status.name,
        mapping_status=native_result.mapping_status.name,
        unresolved_phase_rotations=list(native_result.unresolved_phase_rotations),
        unresolved_carrier_offsets=list(native_result.unresolved_carrier_offsets),
        timing_offset_samples=float(native_result.timing_offset_samples),
    )
    return DemodulationResult(
        hard_bits=hard_bits,
        soft_llrs=soft_llrs,
        bits_per_symbol=int(native_result.bits_per_symbol),
        symbol_decisions=np.asarray(native_result.symbols),
        sync_result=sync_res,
        source_hypothesis_label=hyp.label,
        hypothesis_confirmed=hypothesis_confirmed,
        mapping_verified=native_result.mapping_status.name == "VERIFIED",
        sample_offsets=np.asarray(native_result.sample_offsets),
    )

def attempt_synchronization_multi_hypothesis(recording: SignalRecording, hypotheses: List[ModulationHypothesis], config: dict) -> List[DemodulationResult]:
    """Attempt sync on multiple hypotheses."""
    results = []
    for hyp in hypotheses:
        if hyp.status in [hyp.status.HYPOTHESIS_UNVERIFIED, hyp.status.AMBIGUOUS]:
            res = attempt_synchronization(recording, hyp, config)
            results.append(res)
    return results
