# SIH26147 — Phase 7 Documentation

## Scientific Productization, GUI Integration, End-to-End Orchestration, Reproducibility, Benchmarking & Deployment Layer

---

## 1. Overview & Architectural Role

Phase 7 forms the operational productization, orchestration, and explainability layer of the SIH26147 prototype. Governed by the invariant:

> **Phase 2 measures. Phase 3 hypothesizes. Phase 4 recovers. Phase 5 corrects. Phase 6 verifies. Phase 7 communicates, reproduces, benchmarks and operationalizes.**

Phase 7 coordinates execution of Phases 1 through 6 without modifying, falsifying, or smoothing their scientific conclusions.

---

## 2. Core Subsystems

### 2.1 Unified Pipeline Orchestrator (`app.orchestration`)
* **`run_pipeline(recording_or_path, config)`**: Thread-safe, single entry-point coordinating stages 1 to 6.
* **`PipelineStateMachine`**: Enforces strict state transitions (`IDLE` $\to$ `LOADING` $\to$ `VALIDATING` $\to$ `ANALYZING` $\to \dots \to$ `COMPLETED`).
* **`CancellationToken`**: Provides safe, cooperative cancellation across stage boundaries.
* **`PipelineCache`**: Deterministic cache keyed by `SHA-256(source_hash : stage : config_hash)`.
* **`FailureCategory` Taxonomy**: Explicitly classifies failures as `LOW_SNR`, `MODULATION_AMBIGUITY`, `SYNCHRONIZATION_FAILURE`, `FRAME_FAILURE`, `CRC_FAILURE`, `RESOURCE_LIMIT`, etc.

### 2.2 Desktop User Interface (`app.ui`)
* **PySide6 + PyQtGraph**: Responsive local desktop application with scientific dark theme.
* **13 Interactive Pages**:
  1. Input Metadata & Forensics
  2. Signal Spectrum & Waveform
  3. Signal Detection & ROIs
  4. Extracted Parameters Table
  5. Modulation Hypotheses
  6. Recovery & 1-SPS Constellation
  7. Data Reconstruction & Frames
  8. FEC Bit-Modification Mask
  9. 7-Claim Verification Matrix
  10. Adversarial Falsification Log
  11. Executive Summary & "WHY?" Explainability
  12. Forensic Data Lineage Graph
  13. System Health & Diagnostics

### 2.3 Reporting & Provenance (`app.reporting`, `app.deployment`)
* **Schema v1.0 JSON Report**: Standardized, machine-readable pipeline output.
* **Responsive HTML Report**: Publication-quality standalone report with KaTeX math and CSS formatting.
* **Tabular CSV Exports**: Parameter summary and frame hierarchy export.
* **Reproducibility Manifest**: Complete provenance metadata with SHA-256 reproducibility hash.

### 2.4 Replay & Differential Engine (`app.replay`)
* **Deterministic Replay**: Rerun saved experiments with frozen random seeds and verify hash identity.
* **Differential Comparator (`compare_runs`)**: Evaluates two runs stage-by-stage and identifies the exact point of divergence.
