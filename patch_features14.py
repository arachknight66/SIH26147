import re

with open('signal_analysis/features.py', 'r') as f:
    content = f.read()

ofdm_func = '''
def check_ofdm_plausibility(samples: np.ndarray) -> Optional[int]:
    """
    Coarse check for OFDM cyclic prefix signature by searching for strong 
    autocorrelation peaks at typical FFT sizes.
    Returns the candidate FFT size if detected, else None.
    """
    if len(samples) < 8192:
        return None
        
    import numpy as np
    mag_sq = np.abs(samples)**2
    # Standard OFDM FFT sizes: 64, 128, 256, 512, 1024, 2048, 4096
    # For a CP, we expect R(N) to have a peak. We can just correlate the original signal?
    # Actually, delay-and-multiply is standard for CP detection: sum(x[k] * conj(x[k-N]))
    # For speed, we just check a few discrete Ns
    best_N = None
    best_peak_ratio = 0.0
    
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        if len(samples) < N * 3:
            continue
            
        # compute delay-and-multiply over a window
        window = min(len(samples) - N, 10000)
        delayed = samples[N:N+window]
        orig = samples[:window]
        
        corr = np.abs(np.mean(delayed * np.conjugate(orig)))
        pwr = np.mean(mag_sq[:window])
        if pwr > 0:
            rho = corr / pwr
            if rho > 0.15 and rho > best_peak_ratio:  # Strong correlation (typically CP is 1/4 or 1/8 so rho is large if signal is clean)
                best_peak_ratio = rho
                best_N = N
                
    return best_N

def extract_all_features(recording: SignalRecording) -> ModulationFeatureVector:
'''
content = content.replace('def extract_all_features(recording: SignalRecording) -> ModulationFeatureVector:', ofdm_func)

fix1_and_4_patch = '''
    # FIX 4: OFDM plausibility check
    ofdm_n = check_ofdm_plausibility(process_samples)
    if ofdm_n is not None:
        diagnostics.append(Diagnostic(
            code="OFDM_PLAUSIBILITY",
            message=f"Cyclic-prefix-like periodicity detected (candidate FFT size ~{ofdm_n}); this may indicate an OFDM/multicarrier signal outside the currently supported modulation set (BPSK/QPSK/8PSK/16-QAM/2-FSK).",
            severity=Severity.INFO
        ))

    overall_val = FeatureValidity.VALID

    # FIX 1: Real-valued gate
    if recording.semantic_type != "complex_iq":
        phase = PhaseFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0)
        cumulant = CumulantFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0, 0.0, 0.0)
        diagnostics.append(Diagnostic(
            code="COMPLEX_FEATURES_UNAVAILABLE",
            message=f"Phase/cumulant features require complex_iq samples; got semantic_type='{recording.semantic_type}'. PSK/QAM discriminants below are not physically meaningful for this input.",
            severity=Severity.WARNING
        ))
        overall_val = FeatureValidity.PARTIALLY_VALID
    else:
        phase = extract_phase_features(process_samples)
        cumulant = extract_cumulant_features(process_samples)
        
    if phase.validity != FeatureValidity.VALID or cumulant.validity != FeatureValidity.VALID:
        overall_val = FeatureValidity.PARTIALLY_VALID
'''
# Replace the extraction calls for phase and cumulant
content = re.sub(
    r'    phase = extract_phase_features\(process_samples\)\n    cumulant = extract_cumulant_features\(process_samples\)\n    overall_val = FeatureValidity\.VALID\n    if phase\.validity != FeatureValidity\.VALID or cumulant\.validity != FeatureValidity\.VALID:\n        overall_val = FeatureValidity\.PARTIALLY_VALID',
    fix1_and_4_patch,
    content,
    flags=re.DOTALL
)

with open('signal_analysis/features.py', 'w') as f:
    f.write(content)
