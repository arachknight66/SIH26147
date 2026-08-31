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
    """Attempt sync on a single hypothesis."""
    c_params = hyp.candidate_parameters
    if c_params.symbol_rate is None or c_params.samples_per_symbol is None:
        diag = Diagnostic(Severity.ERROR, "SYNC_MISSING_PARAMS", "Hypothesis lacks required symbol rate.", "")
        sync_res = SynchronizationResult(0.0, "cycles/sample", 0.0, False, False, 999.0, 100.0, [diag])
        return DemodulationResult(np.array([]), np.array([]), 1, np.array([]), sync_res, hyp.label, False)
        
    sps = c_params.samples_per_symbol
    mod = hyp.label
    
    max_samples = DEFAULT_MAX_ANALYSIS_SAMPLES
    samples = recording.samples[:max_samples]
    if samples.ndim > 1:
        samples = samples[:, 0]
        
    diagnostics = []
    
    if mod == "2-FSK":
        # Extract features from Phase 2 frequency extractor
        # Coarse CFO is done by picking the peaks. 
        # But we can just use the evidence f0 and f1 if available, otherwise estimate from samples
        prod = samples[1:] * np.conj(samples[:-1])
        f_inst = np.angle(prod) / (2 * np.pi)
        hist, bin_edges = np.histogram(f_inst, bins=64, range=(-0.5, 0.5))
        peaks = []
        for i in range(1, 63):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((hist[i], bin_edges[i]))
        peaks = sorted(peaks, key=lambda x: x[0], reverse=True)
        if len(peaks) >= 2:
            f0 = peaks[0][1]
            f1 = peaks[1][1]
        else:
            f0, f1 = -0.1, 0.1 # fallback
            
        clock_locked, lock_quality, sym_idx = recover_timing_fsk(samples, sps)
        hard_bits, soft_llrs, decisions, evm = fsk_dual_correlator(samples, sym_idx, f0, f1)
        carrier_locked = True # FSK non-coherent correlator doesn't need explicit carrier lock
        cfo = 0.0
        cfo_unit = "cycles/sample"
        bps = 1
        
    else:
        # PSK/QAM
        M_map = {"BPSK": 2, "QPSK": 4, "16-QAM": 4, "8PSK": 8}
        if mod not in M_map:
            diag = Diagnostic(Severity.ERROR, "SYNC_UNSUPPORTED", f"Unsupported modulation {mod}", "")
            sync_res = SynchronizationResult(0.0, "cycles/sample", 0.0, False, False, 999.0, 100.0, [diag])
            return DemodulationResult(np.array([]), np.array([]), 1, np.array([]), sync_res, mod, False)
            
        M = M_map[mod]
        cfo = estimate_coarse_cfo_psk_qam(samples, M)
        cfo_unit = "cycles/sample"
        
        # Check CFO bounds against candidate bandwidth
        # Rough check: bandwidth is often roughly 1/SPS in cycles/sample
        candidate_bw = 1.0 / sps if c_params.bandwidth_hz is None else (1.0 / sps) # normalized
        if abs(cfo) > candidate_bw * 0.5:
            diag = Diagnostic(Severity.ERROR, "CFO_OUT_OF_BOUNDS", f"CFO {cfo:.4f} exceeds max expected {candidate_bw*0.5:.4f}", "")
            diagnostics.append(diag)
            # Proceed anyway, but we will likely fail lock
            
        # Correct coarse CFO
        t = np.arange(len(samples))
        corrected_samples = samples * np.exp(-1j * 2 * np.pi * cfo * t)
        
        # AGC for timing recovery
        rms = np.sqrt(np.mean(np.abs(corrected_samples)**2))
        if rms > 1e-9:
            corrected_samples /= rms

        
        # Timing recovery
        syms_timed, clock_locked, lock_quality, sym_idx = recover_timing_gardner(corrected_samples, sps)
        
        # Carrier recovery
        # AGC again to normalize output symbols for EVM and Costas
        rms_timed = np.sqrt(np.mean(np.abs(syms_timed)**2))
        if rms_timed > 1e-9:
            syms_timed /= rms_timed
        
        decisions, carrier_locked, phase_var = recover_carrier_costas(syms_timed, mod)
        
        # Demodulate
        hard_bits, soft_llrs, evm = psk_qam_demodulate(decisions, mod)
        bps = CONSTELLATION_MAPS[mod]["bits_per_symbol"]
        
    # Lock thresholds
    evm_threshold = 35.0 # Percent. Configurable.
    hypothesis_confirmed = clock_locked and carrier_locked and evm < evm_threshold
    
    if not hypothesis_confirmed:
        diagnostics.append(Diagnostic(Severity.WARNING, "SYNC_FAILED", "Failed to confirm hypothesis", f"clock={clock_locked}, carrier={carrier_locked}, evm={evm:.2f}"))
        
    # Convert CFO to Hz if possible
    if c_params.symbol_rate_unit == "Hz" and c_params.symbol_rate is not None:
        fs = c_params.symbol_rate * sps
        cfo_hz = cfo * fs
        cfo_final = cfo_hz
        cfo_unit_final = "Hz"
    else:
        cfo_final = cfo
        cfo_unit_final = cfo_unit
        
    sync_res = SynchronizationResult(
        cfo_estimate=float(cfo_final),
        cfo_unit=cfo_unit_final,
        timing_offset_fractional_symbols=0.0, # Not strictly tracked cleanly in our Gardner loop
        symbol_clock_locked=clock_locked,
        carrier_locked=carrier_locked,
        lock_quality_metric=lock_quality,
        evm_percent=evm,
        diagnostics=diagnostics
    )
    
    return DemodulationResult(
        hard_bits=hard_bits,
        soft_llrs=soft_llrs,
        bits_per_symbol=bps,
        symbol_decisions=decisions,
        sync_result=sync_res,
        source_hypothesis_label=mod,
        hypothesis_confirmed=hypothesis_confirmed
    )

def attempt_synchronization_multi_hypothesis(recording: SignalRecording, hypotheses: List[ModulationHypothesis], config: dict) -> List[DemodulationResult]:
    """Attempt sync on multiple hypotheses."""
    results = []
    for hyp in hypotheses:
        if hyp.status in [hyp.status.HYPOTHESIS_UNVERIFIED, hyp.status.AMBIGUOUS]:
            res = attempt_synchronization(recording, hyp, config)
            results.append(res)
    return results
