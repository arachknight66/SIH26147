import re
with open('signal_analysis/features.py', 'r') as f:
    content = f.read()

# I will just replace the `amp = extract_...` block.
patch = '''    amp = extract_amplitude_features(norm_samples)
    fre = extract_frequency_features(norm_samples)
    
    # FIX 4: OFDM plausibility check
    ofdm_n = check_ofdm_plausibility(norm_samples)
    if ofdm_n is not None:
        diagnostics.append(Diagnostic(
            code="OFDM_PLAUSIBILITY",
            message=f"Cyclic-prefix-like periodicity detected (candidate FFT size ~{ofdm_n}); this may indicate an OFDM/multicarrier signal outside the currently supported modulation set (BPSK/QPSK/8PSK/16-QAM/2-FSK).",
            severity=Severity.INFO
        ))

    # FIX 1: Real-valued gate
    if recording.semantic_type != "complex_iq":
        pha = PhaseFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0)
        cum = CumulantFeatures(FeatureValidity.UNAVAILABLE, 0.0, 0.0, 0.0, 0.0)
        diagnostics.append(Diagnostic(
            code="COMPLEX_FEATURES_UNAVAILABLE",
            message=f"Phase/cumulant features require complex_iq samples; got semantic_type='{recording.semantic_type}'. PSK/QAM discriminants below are not physically meaningful for this input.",
            severity=Severity.WARNING
        ))
    else:
        pha = extract_phase_features(norm_samples)
        cum = extract_cumulant_features(norm_samples)
'''

content = re.sub(r'    amp = extract_amplitude_features\(norm_samples\)\n    pha = extract_phase_features\(norm_samples\)\n    fre = extract_frequency_features\(norm_samples\)\n    cum = extract_cumulant_features\(norm_samples\)', patch, content, flags=re.DOTALL)

with open('signal_analysis/features.py', 'w') as f:
    f.write(content)
