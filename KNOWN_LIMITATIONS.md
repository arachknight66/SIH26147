# Known Limitations and Explicit Non-Goals

To maintain strict epistemic integrity, this codebase explicitly refuses to silently handle scenarios it cannot mathematically prove. The following are architectural non-goals and known limitations of the MVP.

## 1. OFDM and Multicarrier Signals
**Limitation:** Signals such as DAB, DVB-T, LTE, and Wi-Fi are **unsupported**.
**Behavior:** Phase 2 includes a cyclostationary plausibility detector that will correctly flag cyclic-prefix periodicity. However, the classifier cannot identify specific subcarrier mappings, and Phase 3 will completely abort rather than attempt to lock a single-carrier PLL to a multicarrier waveform.

## 2. Magic Metadata Inference
**Limitation:** It is a physical impossibility to infer sample rate, center frequency, or timestamp natively from a flat array of `float32` complex IQ bytes.
**Behavior:** The Phase 1 loaders will not guess. If a raw `.iq` file is provided without an accompanying `RawIQConfig` (or if a WAV file lacks standard header chunks), these values are marked `MISSING` and downstream calculations that require true time (like baud rate in Hz) will degrade gracefully to fractional units.

## 3. Blind Pseudo-Random De-interleaving
**Limitation:** The system explicitly refuses to blindly recover pseudo-random convolutional interleavers without a known generator polynomial.
**Behavior:** Attempting to brute-force a pseudo-random permutation is computationally unfalsifiable without knowing the exact frame payload. 

## 4. Bounded Block Interleaver Search
**Limitation:** Block interleaver dimension discovery is constrained to a predefined, finite search grid (e.g., `8, 12, 16, 32, 64, 128, 255`).
**Behavior:** Interleavers with `rows` or `cols` outside this exact grid are invisible to the search. If a signal uses an unmapped dimension, Phase 4 will exhaust the search grid, report a failure diagnostic, and halt.

## 5. LDPC Decoding and Systematic Extraction
**Limitation:** Low-Density Parity-Check (LDPC) coding remains explicitly out of scope for the MVP.
**Behavior:** While convolutional (Viterbi) and Reed-Solomon blocks are robustly verified, LDPC systematic bit-position extraction has not been implemented or verified against known ground-truth, and will be ignored in the FEC cascade.

## 6. Analysis Window Truncation
**Limitation:** The pipeline operates only on a truncated prefix of the file.
**Behavior:** To ensure responsive analysis times (especially in the GUI), processing is capped by `DEFAULT_MAX_ANALYSIS_SAMPLES = 100_000` (adjustable in `constants.py`). For a 1 Msps signal, this means only the first 0.1 seconds of the recording are ever evaluated. Deep-file anomalies or late-arriving packets will not be detected unless the user explicitly slices the file prior to ingestion.

## 7. Known GUI Divergences
*Currently, all identified GUI-vs-pipeline wiring gaps have been successfully patched as of the Phase 5 verification phase (specifically, the `QInputDialog` stereo prompt race condition and the Phase 4/5 `PipelineResult` attribute mapping errors).* No other wiring divergence is known, but the GUI code explicitly relies on exact field matches to the `PipelineResult` dataclasses and must be updated in lockstep if those models change.

### Classifier Limitations

1. **Unsupported Constellation Orders (e.g. 64-QAM):** The Phase 2 classification stage relies on scale-invariant phase and cumulant discriminants which are vulnerable to misidentifying unsupported higher-order QAMs. When fed a 64-QAM signal, the classifier does not cleanly refuse it as UNKNOWN, but instead confidently mislabels it as 16-QAM (score 1.0). This propagates an incorrect hypothesis to the sync and demodulation stages resulting in high EVM rather than an early rejection.
2. **CFO Sensitivity in 16-QAM Classification:** The Phase 2 cumulant-based feature extraction (specifically C_40 and C_42) is extremely brittle to Carrier Frequency Offset for constellations with multiple amplitude levels like 16-QAM. A very small CFO (e.g. 0.01 cycles/sample) spins the constellation during feature extraction, destroying the amplitude/phase discriminants. Consequently, the classifier confidently mislabels the 16-QAM signal as QPSK. This creates a gatekeeping vulnerability where the robust decision-directed Costas loop in Phase 3 never gets the opportunity to lock because Phase 2 feeds it the wrong hypothesis.
