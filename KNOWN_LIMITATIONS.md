# Known Limitations and Explicit Non-Goals

Current as of 2026-09-30. The following are architectural non-goals and observed
limitations of the implementation. A stage completing, lock being reported,
or frame candidate appearing is not proof that the inferred signal family or
payload is correct.

## 1. OFDM and Multicarrier Signals
**Limitation:** Signals such as DAB, DVB-T, LTE, and Wi-Fi are **unsupported**.
**Behavior:** The legacy feature extractor has a coarse cyclic-prefix plausibility diagnostic, but it is not a reliable hard gate on the production native classifier/receiver. In the frozen v1 synthetic S6 OFDM-like negative stratum, 572/1,250 windows produced L1 modulation claims, 1,017/1,250 L2 receiver claims, 676/1,250 L4 frame claims, and **2/1,250 L5 strict frame claims**. The fixed-suite zero-confirmation PRD gate therefore **fails**; the repeated-symbol 64-point generator is not representative real RF. No specific OFDM mapping is decoded or validated.

## 2. Magic Metadata Inference
**Limitation:** Sample rate, center frequency, and timestamp are underdetermined from a flat array of complex IQ samples without external acquisition information.
**Behavior:** Raw IQ requires an explicit `RawIQConfig`; the application does not infer the byte format. The GUI/CLI 10 kS/s default is an `ASSUMED` value, not measured or source metadata. For other absent values, metadata is `MISSING`; calculations must not present guessed absolute units as known. A malformed WAV header is an import error.

## 3. Blind Pseudo-Random De-interleaving
**Limitation:** Seeded pseudo-random, diagonal, and convolutional transforms require an explicit profile or permutation; block dimensions may also be searched within a bounded grid as described below.
**Behavior:** The native engine applies a supplied seed/permutation deterministically but does not brute-force arbitrary pseudo-random permutations or unknown convolutional delay-line state.

## 4. Bounded Block Interleaver Search
**Limitation:** Blind block-interleaver discovery uses a finite default dimension grid (`8, 12, 16, 32, 64, 128, 255`), which callers may override through `deinterleaver_test_dims`; explicit profiles are a separate native path.
**Behavior:** Dimensions outside the *configured* search grid are not discovered automatically. If none beats the no-interleaver baseline, the Python candidate path emits `DEINTERLEAVER_SEARCH_EXHAUSTED` and retains a `NONE` candidate; it does not necessarily halt the whole pipeline.

## 5. LDPC Profiles and Systematic Extraction
**Limitation:** The native normalized min-sum decoder accepts supplied sparse matrices and `.alist` files, but no validated bundled `MACKAY_504_1008` matrix is present in this repository.
**Behavior:** An LDPC profile without an explicit matrix returns `LDPC_MATRIX_REQUIRED`; the engine does not invent systematic information-bit positions or claim a successful decode merely because iterations finished.

## 6. In-Memory Production Workflow
**Limitation:** GUI, CLI, and Demo Mode now share one request/job workflow, but that workflow still imports a complete `SignalRecording` before the pipeline runs.
**Behavior:** Native `RecordingSource` supports explicit regions, bounded previews, and chunked full-source analysis with reported coverage. Its full streaming path is not yet the ordinary GUI/CLI workflow, so large captures can still require full input memory and several analysis operations retain prefix limits.

## 7. Demo Fixture Scope
**Limitation:** The legacy structured BPSK and 16-QAM fixtures do not contain validated advertised concatenated FEC chains.
**Behavior:** Demo Mode labels them as structured fixtures, keeps evaluation metadata separate from production requests, and never uses ground truth to select an analysis outcome. A genuine end-to-end FEC fixture corpus remains required before beta validation.

## 8. GUI/Pipeline Field Wiring
The 2026-09-29 [field audit](docs/gui_pipeline_field_audit.md) found **0 missing/renamed dataclass accesses** in the audited GUI render paths. A pre-render contract guard prevents partial sidebar mutation on field drift, and the actual asynchronous `AnalysisJob` completion path surfaces `GUI_PIPELINE_CONTRACT` distinctly; seven focused GUI tests passed. This is structural regression evidence, not exhaustive interactive GUI or Windows evidence. The audit also notes a low-severity `HeaderMatch.pattern` annotation/runtime discrepancy that the GUI currently handles safely.

## 9. Negative-Suite Scope and False Confirmation
The frozen version-1 synthetic release suite contains 10,000 windows and produced **L5=2/10,000** under its historical confirmed-frame predicate, both in S6 (`index=949` and `1041`), so its fixed-suite zero-L5 criterion **fails**. [Forensic reproduction](docs/negative_suite_s6_l5_root_cause.md) traced both to repeated 80-sample OFDM blocks, an unsupported-detector crest-factor miss, wrong 64-QAM lock with unverified bit mapping, three-offset matches to the short HDLC flag, and CRC-16/IBM hits found by sweeping thousands of header/polynomial/payload-length combinations. Both accepted CRC spans cross later HDLC flags. In 1041 the three matching 720-bit words are *identical repeats* and the top frame is `AMBIGUOUS`, yet historical L5 still fires. The working tree now constrains CRC validation to explicit profiles and fixed boundaries, rejects repeated words and crossing spans, and requires verified mapping before `CONFIRMED`; direct reruns of both triggers and the smoke suite produce no L5 claims. The historical gate remains FAIL until the frozen S6 and complete 10,000-window suite are rerun once with the revised predicate. L1=3,073, L2=3,518, and L4=2,937 are also nonzero; L3 non-uncoded FEC=0. A separate generic 1% worst-stratum rate-bound verdict happens to say PASS for L5, but it is not the zero-event gate. Moreover, `prd.md` requires held-out evidence for frame confirmation; the revised implementation still needs that broader validation. The v1 S7/S8 generator has uint8 PSK-symbol underflow, S8 mutates sync-filtered bits after modulation, and S6 uses only repeated 64-point OFDM symbols. These limit interpretation; they do not erase the observed L5 positives or authorize post-hoc suite edits. The corpus now contains one real T2 capture, but it has no independently established negative/family truth, so **T2 real-RF negatives remain 0** and no real-RF bound is claimed. See the [release findings](docs/negative_suite_findings.md) and the full per-stratum JSON/Markdown report under `build/negative/`.

## 10. Windows Execution
The `_native` CMake **MODULE** target already has the correct `LIBRARY DESTINATION signal_analysis` install rule under [CMake's artifact classification](https://cmake.org/cmake/help/latest/command/install.html#installing-targets). This is a static check only: no Windows build, wheel install, or GUI import transcript has been captured, and the cross-platform packaging gate remains open. The Windows runner's actual optional-package probe results also remain unobserved; the current probes log availability but do not link alternate implementations. See the [risk review](docs/windows_known_risks.md) and [CI runbook](docs/windows_ci_runbook.md).

## 11. GNU Radio runtime

**Limitation:** GNU Radio is an external runtime, not a dependency installed by
the Python project extras. Ordinary GUI, CLI, and Demo imports require a
compatible GNU Radio Python interpreter.
**Behavior:** The adapter discovers a runtime or uses
`SIH_GNURADIO_PYTHON`. Without a working runtime, ordinary processing fails
with an explicit preprocessing error; there is no silent bypass. The current
flowgraph supports pass-through, frequency translation, low-pass filtering,
and rational resampling. Platform-specific installation and clean-machine
validation remain open; see the README and Windows runbook.

## 12. Experimental ML model

**Limitation:** The ExtraTrees model was trained on generated signals, not a
representative measured dataset, and is not used by the production classifier.
**Evidence:** A seeded synthetic holdout scored 0.8389 balanced accuracy across
six classes. The separate real-data check predicted BPSK for 32 windows from a
single source-labeled capture. This does not establish real multi-class
accuracy or independent capture performance. The training script, model, and
report are under `tools/train_modulation_ml.py` and
`data/dataset_batches/ml/`.

## 13. IQ WAV assets

The dataset WAV files in `data/` store I and Q as separate channels and are
intended for signal-processing tools, not audio playback. The BPSK source rate
is unknown; its WAV uses a 10 kS/s container placeholder documented in its
sidecar. Use the accompanying raw `.iq` file when the consumer can accept
complex float32 samples and an externally supplied rate. Source attribution
and formats are listed in [dataset assets](docs/dataset_assets.md).

### Native estimation and receiver limitations

1. Phase 3 family classification and rate/CFO calibration has deterministic synthetic coverage plus a corpus manifest, hash-checked ingestion, T1 impairment instrument, and split/firewall evaluator. The committed T2 over-the-air corpus is **PARTIAL (1 capture)**: a checksum-verified CC-BY-4.0 Zenodo SigMF downlink recording. Its published provenance does not independently establish family, symbol rate, SNR, FEC, or transmitted bits, so it supports ingestion evidence only—not representative accuracy or a PRD gate. Exact 16/64/256-QAM order ranking can remain ambiguous even when the QAM family is correct.
2. The PSD-floor SNR estimate is marked unreliable when the recording does not expose a separable noise-only band. Wideband FSK and multicarrier captures are particularly difficult.
3. `ReceiverSession` preserves arbitrary chunk equivalence but buffers a bounded acquisition window and emits on `flush`; continuous incremental tracking is not implemented yet.
4. DBPSK/DQPSK, OQPSK, and MSK paths are present. GMSK/GFSK, generic CPM, pi/4-DQPSK, adaptive multipath equalization, and finite-memory CPM sequence detection remain unsupported.
5. Carrier acquisition explicitly reports rotational and nonlinear-frequency aliases. A lock does not verify absolute bit mapping unless a carrier/phase reference or later frame evidence resolves it.

### Reed--Solomon bounded-distance invariant

For `t=(n-k)/2`, RS minimum distance is `d=2t+1`, so radius-`t` decoding is unique: those Hamming balls are disjoint. The Python re-encode check is redundant after a correct zero-syndrome result and protects only against internal decoder inconsistency, not a reachable decode-time ambiguity; no native gap exists. Native and Python regressions instead protect the real dependency of this argument: the generator polynomial's consecutive BCH-root structure.
