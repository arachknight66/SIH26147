# Changelog

All notable changes to the SIH26147 Signal Analysis project will be documented in this file.

## [Unreleased] - 2026-09-30

### Added

* Shared ordinary GUI/CLI/Demo preprocessing through an external GNU Radio
  sidecar, with pass-through, frequency translation, low-pass filtering, and
  rational resampling. The 10 kS/s raw-IQ default is explicitly `ASSUMED`.
* Optional CUDA kernels for selected PSD/STFT, post-lock receiver decisions,
  fixed K=7 Viterbi, RS zero-syndrome screening, and bitstream correlation.
* Standalone experimental ExtraTrees training for six modulation classes,
  with a committed model and synthetic holdout report. It is not connected to
  the production classifier.
* WAV I/Q assets, a raw complex-float `.iq` copy, and
  [dataset provenance and encoding notes](docs/dataset_assets.md).
* Phase 3 fractional timing interpolation and low-offset robustness changes;
  the source-labeled BPSK segment now ranks BPSK first but remains an
  unverified, single-segment result.

### Validation limits

* Synthetic ML holdout balanced accuracy is 0.8389. The real-data check covers
  one source-labeled BPSK recording only; it is not representative multi-class
  accuracy. See [the complete report](data/dataset_batches/ml/training_report.json).
* Focused GPU vectors passed, but the local single-stream Viterbi timing did
  not beat CPU. The frozen v1 negative suite's two L5 events remain a failed
  PRD gate until its S6 and full-suite post-mitigation reruns complete.
* GNU Radio is installed in a separate runtime. Windows execution, representative
  real-RF evidence, and full ordinary-workflow 1-GB processing remain open.

## [0.2.0.dev1 / Native Phase 7] - 2026-09-18

### Added

* Versioned JSON run metadata for CLI reports, including engine/API version, selected profile configuration, source semantics, execution status, and processing coverage.
* Repeatable native spectral/Viterbi microbenchmark output and a JSON-producing validation runner.
* Native sanitizer CMake preset and Linux CI gate, plus focused release integration tests and a small deterministic negative-noise smoke corpus.

### Limits

* These tools record local evidence; they do not establish the plan's 10,000-window negative corpus, reference-laptop performance thresholds, 1-GiB ordinary workflow, representative RF corpus, or both-platform clean-machine installation.

## [0.2.0.dev1 / Native Phase 6] - 2026-09-18

### Added

* A shared `AnalysisRequest` and production workflow for GUI, CLI, and Demo Mode, including explicit raw-IQ CLI import options.
* Qt-polled background analysis with coarse progress and cooperative cancellation; plotting and metadata rendering remain on the Qt main thread.
* A deterministic Demo Mode catalog and an explicit Reveal Ground Truth action backed by evaluation-only metadata stored separately from analysis requests.

### Changed

* Demo labels no longer claim that legacy structured recordings demonstrate validated concatenated FEC.

### Limits

* The shared workflow still creates an in-memory recording before pipeline execution. Streaming source jobs and a genuinely encoded FEC demonstration corpus remain open.

## [0.2.0.dev1 / Native Phase 5] - 2026-09-17

### Added

* Native configured block, convolutional, diagonal, and seeded pseudo-random bit transforms with hard-bit/LLR provenance and incomplete-block diagnostics.
* Native RS encoder/decoder, correlation, CRC, normalized-min-sum LDPC, sparse matrix input, and `.alist` import through API version 5.
* Native Phase 5 C++ and Python tests covering transform round-trips, RS correction/failure bounds, LDPC syndrome behavior, CRC, and correlation.

### Changed

* Production Viterbi, RS, CRC, and correlation adapters now call C++; uncoded input is explicit by default rather than forcing a speculative concatenated FEC chain.
* Incomplete Viterbi/RS input is rejected with diagnostics instead of padded.
* A lone CRC-8 match no longer outranks repeated header evidence during frame ranking.

### Limits

* No validated bundled `MACKAY_504_1008` matrix, punctured Viterbi catalogue, erasure-aware RS, or genuine concatenated capture is available yet; Phase 5 remains in progress.

## [0.2.0.dev1 / Native Phases 3–4] - 2026-09-15

### Added

* Native SNR, occupied-bandwidth, carrier-offset, symbol-rate candidate, temporal-consistency, and modulation-ranking contracts.
* CFO-tolerant PSK/QAM constellation fitting, FSK tone/persistence evidence, explicit unknown/ambiguous/unsupported results, and JSON-safe pipeline summaries.
* Native configured BPSK/QPSK/8PSK, square 16/64/256-QAM, 2/4/8-FSK, DBPSK/DQPSK, OQPSK, and MSK receiver paths.
* Bounded `ReceiverSession`, matched filtering, fractional timing search, carrier-line refinement, max-log LLRs, sample offsets, and phase/frequency ambiguity reporting.

### Changed

* The production pipeline now uses native estimation and receiver adapters instead of the hard-coded 20 dB SNR and legacy Python synchronization path.
* The GUI displays measured parameters, acquisition state, mapping state, and unresolved phase count.
* Native API version is 4.

### Validation

* Standalone Linux CTest covers Phases 1–4, including exact noiseless QPSK and receiver chunk equivalence.
* Native Python tests cover held-out family/rate estimates, mapped noiseless recovery, independent QPSK reference BER, FSK/differential/continuous-phase profiles, ambiguity, low-SNR, and negative inputs.
* Representative-capture calibration, Windows execution, continuous tracking, channel equalization, and complete specialized-profile gates remain open.

## [0.2.0.dev1 / Native Phase 2] - 2026-09-14

### Added

* C++20 chunked raw-IQ and RIFF/WAVE PCM/float readers with explicit conversion metadata.
* Stateful DC blocking, AGC, mixing, FIR design/filtering, rational resampling, and streaming statistics.
* Native Welch PSD, bounded STFT, normalized axes, and spectral-band grouping.
* Python recording-source preview/full orchestration with offsets, coverage, cancellation, and progress.

### Changed

* Existing PSD and waterfall providers now use the native API version 2 spectral core.
* GUI plots display frequency/time/power units and processed sample coverage.
* The stereo hint reuses the samples it actually read instead of checking an exhausted WAV stream.

### Validation

* Linux Release CTest passed; the Qt-enabled Python suite passed 108 tests.
* A logical sparse 1-GiB source completed a bounded preview at 45,488 KiB maximum RSS.
* Windows and complete 1-GiB full-mode execution remain pending gates.

## [MVP / Phase 5 Verification Pass] - 2026-08-31

This release marks the unification of the 6-layer pipeline and extensive GUI/pipeline stability hardening.

### Fixed
*   **Real-Valued Feature Gating:** Added strict structural gating in Phase 2 (`features.py`) to prevent cumulant and phase discriminant extraction on real-valued signals (e.g., `stereo_real` WAV files). Attempting to extract angle-based features from real arrays previously yielded statistically invalid confidences; the pipeline now cleanly detects non-complex domains, emits `COMPROMISED` warnings, and gracefully halts downstream single-carrier processing.
*   **RS Re-encode Verification:** Reinforced the Reed-Solomon root generation logic and fixed Chien search diagnostic mismatches inside `fec_reed_solomon.py` to ensure block decoding degrades gracefully on out-of-scope parity structures.
*   **OFDM Plausibility Diagnostic:** Implemented `check_ofdm_plausibility` to explicitly catch and warn on cyclic-prefix periodicity, cleanly marking multicarrier waveforms (like DAB) as `UNKNOWN` rather than forcing false positive single-carrier categorizations.
*   **Truncation Transparency:** Enforced explicit warning indicators when files exceed `DEFAULT_MAX_ANALYSIS_SAMPLES`, ensuring the user is visually notified that deep-file anomalies are being truncated for performance.
*   **Stereo-Dialog Event Loop Bug (Multiple Passes):** Addressed a deeply hidden race condition in `gui.py` where a native Windows `QFileDialog` event was instantly closing the subsequent static `QInputDialog.getItem` prompt. This bug survived two earlier validation passes because it failed silently by defaulting the signal to `stereo_real` under the hood. Fixed by explicitly instantiating a `QInputDialog`, making it `ApplicationModal`, and flushing the event loop with `QApplication.processEvents()` before execution.
*   **Phase 4/5 GUI Wiring Divergence:** Traced and fixed a silent GUI failure where `gui.py`'s `update_metadata` method was calling outdated `PipelineResult` attributes (`fec.scheme_name` instead of `codec_name`, `fec.success` instead of `decode_success`, etc.). The resulting `AttributeError` was swallowed by a general exception handler, leaving the UI permanently rendering `NOT_ATTEMPTED` or `N/A`. The exact attribute paths were re-mapped to match the hardened dataclasses.

### Added
*   **Comprehensive Test Artifacts:** Consolidated `test_gui_pipeline_integration.py` containing end-to-end regression tests verifying that the exact visual state of the GUI updates correctly against real, disk-backed WAV files. 
