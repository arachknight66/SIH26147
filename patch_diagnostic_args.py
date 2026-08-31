import re
with open('signal_analysis/features.py', 'r') as f:
    content = f.read()

content = content.replace(
    'code="OFDM_PLAUSIBILITY",\n            message=f"Cyclic-prefix-like periodicity detected (candidate FFT size ~{ofdm_n}); this may indicate an OFDM/multicarrier signal outside the currently supported modulation set (BPSK/QPSK/8PSK/16-QAM/2-FSK).",\n            severity=Severity.INFO',
    'severity=Severity.INFO,\n            code="OFDM_PLAUSIBILITY",\n            message=f"Cyclic-prefix-like periodicity detected (candidate FFT size ~{ofdm_n}); this may indicate an OFDM/multicarrier signal outside the currently supported modulation set (BPSK/QPSK/8PSK/16-QAM/2-FSK).",\n            evidence=""'
)

content = content.replace(
    'code="COMPLEX_FEATURES_UNAVAILABLE",\n            message=f"Phase/cumulant features require complex_iq samples; got semantic_type=\'{recording.semantic_type}\'. PSK/QAM discriminants below are not physically meaningful for this input.",\n            severity=Severity.WARNING',
    'severity=Severity.WARNING,\n            code="COMPLEX_FEATURES_UNAVAILABLE",\n            message=f"Phase/cumulant features require complex_iq samples; got semantic_type=\'{recording.semantic_type}\'. PSK/QAM discriminants below are not physically meaningful for this input.",\n            evidence=""'
)

content = content.replace(
    'code="TRUNCATED_ANALYSIS",\n            message=f"Analyzed first {max_samples} of {len(recording.samples)} samples ({frac*100:.2f}% of file)",\n            severity=Severity.INFO',
    'severity=Severity.INFO,\n            code="TRUNCATED_ANALYSIS",\n            message=f"Analyzed first {max_samples} of {len(recording.samples)} samples ({frac*100:.2f}% of file)",\n            evidence=""'
)

with open('signal_analysis/features.py', 'w') as f:
    f.write(content)
