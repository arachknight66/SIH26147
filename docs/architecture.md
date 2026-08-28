# SIH26147 — System Architecture

## End-to-End Computational Pipeline Architecture

```text
+-----------------------------------------------------------------------------------+
|                                   SIH26147                                       |
+-----------------------------------------------------------------------------------+
|  Phase 1: Input Ingestion & Canonical complex64 Validation                        |
|    • WAV / Raw IQ / SigMF                                                         |
|    • Endian / IQ Order Resolution                                                 |
|    • Provenance Metadata                                                          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  Phase 2: Quantitative Physical Measurement & Signal Region Detection             |
|    • Welch PSD, Spectrogram, Noise Floor                                          |
|    • Occupied Bandwidth, SNR, Autocorrelation                                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  Phase 3: Modulation Hypothesis Generation & Feature Extraction                   |
|    • Cumulants (C_40, C_42), Cyclic Moments                                       |
|    • PSK / QAM / FSK / MSK Classifier                                             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  Phase 4: Carrier/Timing Synchronization & Demodulation                           |
|    • Coarse/Fine CFO Correction, Costas Loop                                      |
|    • Gardner Timing Recovery, 1-SPS Constellation                                 |
|    • Soft LLR & Hard Bit Decisions                                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  Phase 5: Data-Stream Reconstruction, Framing & FEC Error Correction              |
|    • Rotational / Polarity Ambiguity Resolution                                   |
|    • Byte Alignment, Sync Preambles, Framing                                      |
|    • LFSR Descrambling, Viterbi FEC Decoding, CRC Check                           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  Phase 6: Independent Scientific Verification & Falsification Engine              |
|    • 7-Claim Verification Matrix                                                  |
|    • Boundary Perturbation, Leave-One-Out Robustness                              |
|    • 70/30 Held-Out Cross-Validation, Bonferroni False-Discovery Control          |
|    • Composite Error Budget & SHA-256 Reproducibility Hash                        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  Phase 7: Productization, GUI, Orchestrator, Reporting & Replay                   |
|    • PySide6 Desktop GUI & Unified CLI                                            |
|    • Schema v1.0 JSON, HTML, CSV Exports                                          |
|    • Deterministic Replay & Differential Run Comparison                           |
+-----------------------------------------------------------------------------------+
```
