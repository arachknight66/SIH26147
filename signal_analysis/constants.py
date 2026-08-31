
# Shared constants across signal_analysis

# Maximum samples to process for computationally intensive operations
# (e.g. feature extraction, cyclic prefix search).
# Justification: 262144 samples (2^18) provides sufficient statistical
# significance for SNR/cumulant estimation down to ~3dB without hanging
# the UI thread during exploratory analysis.
DEFAULT_MAX_ANALYSIS_SAMPLES = 262144
