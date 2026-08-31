# Changelog

All notable changes to the SIH26147 Signal Analysis project will be documented in this file.

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
