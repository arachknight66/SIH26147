# SIH26147 progress

Last updated: 2026-09-30.

## Current state

**Native Phases 1–5, the Phase 6 shared GUI/CLI/Demo workflow, and Phase 7 validation tooling are integrated and locally validated on Linux. The fixed 10,000-window synthetic negative suite is complete but FAILS the PRD's zero-confirmed-frame criterion (L5=2); representative capture calibration, performance gates, cross-platform execution, and full 1-GB ordinary-workflow evidence remain open.**

The PySide GUI remains active. C++20 now provides recording conversion, preprocessing, statistics, PSD/STFT, spectral-band analysis, measured parameter/modulation analysis, configured receiver paths, bitstream transforms, Viterbi, RS, sparse-matrix LDPC, correlation, and CRC. The production Python pipeline calls native Phase 3–5 adapters; Python retains profile/orchestration and frame presentation.

## Completed

- [x] Inspected the Python application, compute modules, CLI, GUI, fixtures, generators, tests, and historical patch scripts.
- [x] Traced the executable pipeline and distinguished working primitives from weak or hard-coded behavior.
- [x] Ran the ten Demo recordings through the existing production pipeline.
- [x] Exercised the existing GUI with an offscreen Qt application and confirmed populated waveform, PSD, waterfall, and synchronized constellation data.
- [x] Ran direct fixed-code Viterbi and RS roundtrip/error-correction checks.
- [x] Profiled the existing BPSK Demo pipeline to identify migration priorities.
- [x] Agreed Linux/Windows support, offline laptop workload, GPL-compatible dependencies, and broad configured decoding with bounded inference.
- [x] Recorded the migration plan in [plan.md](plan.md), requirements in [prd.md](prd.md), and implementation guidance in [agent.md](agent.md), with [AGENTS.md](AGENTS.md) as its discovery entry point.
- [x] Added a CMake/scikit-build-core/pybind11 build with a reproducible `uv.lock`, optional dependency probes, and Linux/Windows CI configuration.
- [x] Added native execution/evidence status, diagnostics, cancellation, progress, and shared-lifetime buffer contracts.
- [x] Migrated the fixed K=7, rate-1/2 `(171,133)` soft Viterbi pilot to C++ with the established positive-LLR-means-one convention.
- [x] Added an explicit Python native adapter with no silent Python fallback and a Qt-independent one-shot background job abstraction.
- [x] Built and installed a clean Linux wheel, verified the native result in an isolated environment, and confirmed the existing Qt main window still constructs offscreen.
- [x] Added normalized, random-access native raw-IQ and WAV sources for signed/unsigned integer and floating-point representations, including PCM8/16/24/32 and IEEE-float WAV.
- [x] Added Python `RecordingSource` adapters for raw IQ, WAV, and SigMF with explicit metadata status, conversion provenance, chunk ranges, preview/full modes, and processed coverage.
- [x] Added stateful native DC blocking, AGC, complex mixing, FIR design/filtering, and rational polyphase resampling with arbitrary-partition equivalence.
- [x] Added native streaming statistics, clipping applicability, radix-2 FFT, Welch PSD, bounded STFT retention, normalized/absolute axes, and spectral-band grouping.
- [x] Migrated `compute_psd` and `compute_spectrogram` to native computation and added units/coverage to the existing GUI plot labels.
- [x] Repaired the stereo hint's exhausted-read bug and moved the GUI negative-path fixture into pytest temporary storage.
- [x] Replaced the production pipeline's fixed 20 dB SNR and repeated legacy estimator calls with one native parameter/modulation analysis result.
- [x] Added native noise-floor SNR, occupied bandwidth, CFO/rate candidates, CFO-tolerant constellation fitting, FSK clustering/persistence, temporal consistency, and explicit unknown/ambiguous/unsupported results.
- [x] Added a JSON-safe Python result adapter and displayed measured estimates in the existing GUI.
- [x] Added native carrier-line acquisition, optional RRC matched filtering, fractional timing search, PSK/QAM max-log demapping, FSK tone receivers, sample offsets, and bounded chunk-invariant `ReceiverSession` orchestration.
- [x] Added configured BPSK/QPSK/8PSK, square 16/64/256-QAM, 2/4/8-FSK, DBPSK/DQPSK, OQPSK, and MSK paths with explicit unsupported results for unimplemented profiles.
- [x] Preserved unresolved phase/frequency aliases and separated acquisition lock, upstream hypothesis confirmation, and bit-mapping verification.
- [x] Added native configured block, convolutional, diagonal, and seeded pseudo-random interleaver transforms with residual-bit preservation and no blind seed recovery.
- [x] Moved production Viterbi, RS, CRC, and sync-word correlation paths to the native extension; added configured normalized-min-sum LDPC and `.alist` matrix import.
- [x] Made uncoded data an explicit default candidate instead of forcing unknown recordings through a concatenated FEC chain; FEC residuals and decode failures are now explicit.
- [x] Tightened frame ranking so a single CRC-8 coincidence cannot outrank repeated header evidence.
- [x] Routed ordinary GUI imports, headless CLI imports, and Demo Mode through one `AnalysisRequest`/`run_production_analysis` workflow, with explicit raw-IQ CLI parameters.
- [x] Moved GUI loading and pipeline execution to the existing Qt-independent `AnalysisJob`; Qt updates are polled and applied only on the main thread, with visible coarse progress and cancellation controls.
- [x] Replaced Demo Mode's hard-coded labels and invalid concatenated-FEC claims with a deterministic fixture manifest; evaluation truth is isolated in `fixtures/demo/truth.json` and read only by the explicit Reveal Ground Truth action.
- [x] Added versioned CLI run metadata containing native API/runtime identity, selected profiles, source semantics, execution status, and coverage without sample data, paths, or Demo truth.
- [x] Added reproducible JSON validation and benchmark tools, sanitizer CMake configuration, Linux CI sanitizer/release gates, and a deterministic small noise regression corpus.
- [x] Verified public over-the-air recordings through ordinary import paths: the Zenodo Intelsat 37e SigMF recording (and its identical raw `.iq` byte stream with explicit import metadata) passed import/provenance checks; the CC0 BPSK31 WAV completed bounded spectral and production-pipeline analysis. Long recordings now use an explicit 262,144-sample parameter-analysis window and emit `ANALYSIS_WINDOW_LIMITED` rather than implying full-recording inference. The BPSK31 recording remains a real-valued audio/passband input outside the configured complex-baseband receiver path, so it is correctly not claimed as demodulated.
- [x] Downloaded and processed two additional CC-BY-NC-SA Polish 11 off-air WAVs with declared 400 Bd FSK and 100 Bd QPSK formats. Both completed ordinary WAV import and bounded analysis; their real passband representation is explicitly gated as `NON_COMPLEX_PIPELINE`, not misreported as complex-baseband demodulation. Added optional CUDA/CuPy capability discovery and a strict `--compute-backend` contract: `auto` reports hardware readiness while retaining CPU-native DSP, and forced `gpu` fails instead of silently using the CPU. See [gpu acceleration readiness](docs/gpu_acceleration.md).
- [x] Installed and validated the optional CuPy CUDA 13 backend on the local RTX 3050. GUI PSD and STFT/waterfall computation now select the GPU automatically when available; CPU/GPU peak-equivalence and GUI tests pass. A warm 262,144-sample, 256-point STFT measured 0.001542 s on GPU versus 0.014118 s through the CPU baseline on this host; this is a local visualization microbenchmark, not a receiver/FEC throughput claim.

## Implementation phases

| Phase | Status | Completion evidence required |
|---|---|---|
| 1. Native architecture and bindings | In progress | Implementation and Linux wheel pass; Windows wheel/build execution is pending CI |
| 2. Input, preprocessing, spectral core | In progress | Linux implementation/tests pass; complete 1-GB full-mode and Windows execution evidence remain pending |
| 3. Estimation and modulation analysis | In progress | Native implementation and small deterministic held-out tests pass; representative corpus calibration and the full negative-confidence gate remain open |
| 4. Synchronization and demodulation | In progress | Existing/core configured profiles, exact mapping tests, ambiguity and chunk tests pass; continuous tracking, multipath equalization, generic CPM and complete Gaussian-profile matrix remain open |
| 5. Bitstream, interleaving, FEC | In progress | Configured native transforms/codecs and synthetic vectors pass; validated bundled LDPC matrix, configurable puncturing, genuine concatenated captures, bounded discovery, and corpus/negative gates remain open |
| 6. Pipeline, GUI, Demo integration | In progress | Shared production jobs and manifest-backed Demo Mode pass local workflow/Qt tests; native streaming-source job execution, validated advertised FEC demo fixtures, and broader interactive evidence remain open |
| 7. Validation, optimization, release | In progress | The frozen 10,000-window synthetic negative suite completed with two strict L5 claims, so the PRD zero-confirmation gate fails; hardware performance, representative T2 negatives, full ordinary-workflow streaming, clean-machine installation, and Windows evidence remain open |

See [plan.md](plan.md) for the full acceptance criteria. Phases 1 and 2 have local Linux evidence; their remaining platform/scale gates and all later phase gates remain open.

## Audit evidence

| Check | Observed result | Limits |
|---|---|---|
| Clean QPSK Demo | QPSK score approximately 1.0; synchronization completed; EVM approximately 7.68%; 4,000 demodulated bits | Does not establish payload correctness or general receiver robustness |
| Clean 16-QAM Demo | 16-QAM score 1.0; synchronization completed; EVM approximately 8.28% | Narrow synthetic input |
| Unsupported 64-QAM Demo | Mislabeled 16-QAM with score 1.0; synchronization completed at EVM approximately 21.53% | Demonstrates false modulation confirmation |
| 16-QAM with CFO Demo | Top QPSK hypothesis approximately 0.583 and ambiguous; synchronization failed | Demonstrates classifier/acquisition weakness |
| Clean QPSK downstream results | FEC stage completed despite RS decode failure; framing completed with an unverified HDLC/CRC-8 candidate | Stage completion is not proof of correct FEC or payload |
| Demo corpus FEC | All seven Demo cases reaching FEC returned RS decode failure while the FEC stage was marked completed | Existing “concatenated” Demo generators directly modulate framed bits without the advertised FEC chain |
| OFDM Demo | 8,000 samples; classification unknown | Cannot trigger the OFDM plausibility check, which requires at least 8,192 samples |
| GUI execution | Two waveform traces, one PSD trace, waterfall data of shape (35, 256), and a synchronized constellation were populated | Offscreen functional exercise, not interactive usability or platform validation |
| Raw-IQ dialog defaults | int8, IQ, little endian, 1 MHz sample rate, 0 Hz center frequency | Loader marks entered/default values known; no actual parameter inference |
| Viterbi direct check | Noiseless fixed K=7, rate-1/2 sequence recovered with zero bit errors | Self-consistent local encoder/decoder check, not independent golden-vector certification |
| RS direct checks | Zero-error roundtrip succeeded; two injected symbol errors corrected exactly | Limited local roundtrip tests |
| Profiling | 35,680-sample BPSK Demo: approximately 2.997 s total; Viterbi 1.915 s; sync attempts 0.546 s; framing/CRC 0.409 s | cProfile overhead included; not the unprofiled beta performance baseline |

Audit runs also emitted NumPy FFT overflow warnings. Numerical range and normalization need investigation during migration.

## Phase 1 verification

| Check | Result |
|---|---|
| Standalone C++20 configure/build/CTest on Linux | Passed `sih_core_viterbi` |
| Binding and job contract tests | 12 passed: golden vector, dtype/rank/stride rejection, ownership, diagnostic types, GIL, progress, cancellation, and Qt-free import |
| Existing non-GUI Python tests | 76 passed in the initial complete run; after adding the diagnostic test, 76 passed and one pre-existing randomized deinterleaver test failed, then passed immediately in isolation |
| Linux wheel | CPython 3.13 wheel built, installed into a clean environment, and decoded the fixed vector correctly |
| Existing GUI | `MainWindow` constructed and closed with Qt's offscreen platform using the system PySide6 installation |
| Dependency probes | FFTW3, libsndfile, and AFF3CT were absent locally; the native build reports this and uses its portable Phase 2 baselines |

## Phase 2 verification

| Check | Result |
|---|---|
| Standalone native tests | Release C++20 build passed both `sih_core_viterbi` and `sih_phase2_core` |
| Python/GUI suite | 108 passed with Qt offscreen; four warnings remain in unmigrated legacy feature/pipeline paths |
| Format matrix | Raw s8/u8/s16/s24/s32/f32/complex-f32, endian/order cases, PCM8/16/24/32 WAV, float WAV, stereo interpretation, SigMF, and malformed sizes passed |
| Numeric behavior | Unsigned centering, full-scale scaling, missing-rate units, PSD peak/energy, band detection, clipping semantics, and resampler pass/reject behavior passed |
| Stateful behavior | Preprocessing and spectral results matched whole-input results across irregular chunk partitions |
| Large-source preview | A sparse logical 1-GiB raw capture opened without whole-file allocation; a 65,536-sample preview completed at 45,488 KiB maximum RSS and reported incomplete source coverage |
| Full streaming/cancellation | Full mode covered a 10,000-sample source across 333-sample chunks; a background full-source job over the logical 1-GiB source cancelled with partial coverage |
| Linux wheel | CPython 3.13 `0.2.0.dev1` wheel built; clean-environment raw-source/full-analysis smoke test passed with native API version 2 |

## Phase 3 verification

| Check | Result |
|---|---|
| Native contracts | API version 4 returns measured estimates with units, uncertainty, validity/evidence, ranked rate/modulation candidates, temporal consistency, and explicit inference status |
| Held-out synthetic profiles | Deterministic seeds pass PSK/QAM/FSK family checks at 15 dB for the current test profiles; symbol rate is within the 2% gate on those fixtures |
| CFO/gain behavior | CFO-tolerant family ranking passes configured synthetic offsets; uncertain high-order estimates are checked against their reported uncertainty rather than treated as exact |
| Negative behavior | Targeted noise/silence/short/real/multicarrier fixtures produced bounded or explicit unknown/unsupported results; the larger frozen Phase 7 suite below found counterexamples, including two L5 claims on OFDM-like windows |
| Pipeline/CLI/GUI | Production pipeline uses one native analysis call; summaries serialize through CLI and measured rate/bandwidth/SNR appear in the existing sidebar |

## Phase 4 verification

| Check | Result |
|---|---|
| Standalone native tests | Linux Release CTest passes all four executables, including exact noiseless QPSK after acquisition and arbitrary receiver chunk partitions |
| Linear profiles | Independent mappings pass exact noiseless recovery for BPSK/QPSK/8PSK and square 16/64/256-QAM when carrier and phase references resolve inherent ambiguities |
| Receiver evidence | RRC/CFO acquisition, fractional timing offsets, low-SNR unlock, positive-LLR semantics, sample offsets, and owned result buffers pass targeted Python tests |
| FSK and variants | Configured 2/4/8-FSK, DBPSK/DQPSK, OQPSK, and MSK paths have direct tests; unresolved carrier/tone mapping remains visible; Gaussian and generic CPM profiles remain open |
| Independent comparison | Native noisy QPSK BER matches an independent known-clock NumPy reference within the recorded tolerance |
| Complete Python/GUI suite | 150 passed with Qt offscreen; one pre-existing precision-loss warning remains in the legacy OFDM plausibility feature path |
| Carrier acquisition regression | A 40-seed, 25 dB RRC-QPSK sweep with 0.05 cycles/sample CFO locked and reported CFO within the existing 0.01 Hz-at-4-Hz fixture tolerance |
| Distribution build | Clean CPython 3.13 Linux wheel built successfully with the Phase 4 bindings |

## Phase 5 verification

| Check | Result |
|---|---|
| Native contracts | API version 5 exposes configured block/convolutional/diagonal/seeded-pseudo-random transforms, correlation, CRC, RS, and sparse-matrix LDPC results with shared-lifetime NumPy views |
| Interleavers | All four supplied-profile families round-trip hard bits and LLRs; incomplete blocks retain their exact residual bits and emit evidence |
| RS profiles | Native RS(255,223) and RS(255,239) correct their exact symbol budgets, reject over-budget words, and reject incomplete trailing codewords without padding |
| RS minimum-distance invariant | Python generator-root and native encoder-span regressions check the exact consecutive BCH roots for RS(255,223), RS(255,239), and RS(15,11). The previously considered native re-encode “gap” was not recorded here because it does not exist: zero syndrome already identifies a systematic codeword and radius-t RS decoding is unique. |
| LDPC | Normalized min-sum validates syndrome convergence and reports iteration-budget exhaustion; direct sparse checks and `.alist` import are supported |
| Production integration | Pipeline defaults to the explicit uncoded candidate; configured convolutional, RS, concatenated, and LDPC profiles use native code paths. Native correlation/CRC feed existing frame presentation |
| CRC-8 ranking regression | GUI regression showed a lone CRC-8 collision could outrank periodic HDLC evidence; ranking now requires repetition before CRC-8 boosts a candidate. This does not pass the Phase 7 false-confirmation gate. |
| Complete local suite | 159 passed with Qt offscreen; one legacy OFDM feature precision-loss warning remains |
| Distribution build | Clean CPython 3.13 Linux wheel built successfully with the API version 5 bindings |

## Phase 6 verification

| Check | Result |
|---|---|
| Shared workflow | Four deterministic workflow tests passed: WAV import/analysis, renamed-capture invariance, pre-load cancellation, and Demo request/truth separation |
| GUI/CLI | Qt-offscreen GUI workflow tests passed (7 tests across GUI suites); CLI completed the same production workflow for `test_clean_qpsk.wav` |
| GUI field wiring | A live cross-reference found 0 missing/renamed dataclass accesses. A pre-render contract guard and narrow `AnalysisJob` completion-path handler surface `GUI_PIPELINE_CONTRACT` distinctly; `uv run pytest tests/test_gui.py -q` passed 7 tests. Details: [field audit](docs/gui_pipeline_field_audit.md). |

## Phase 7 verification

| Check | Result |
|---|---|
| Quick release checks | `tools/run_validation.py --quick --skip-native` passed 104 focused tests in 16.34 s and wrote a JSON transcript; schema and benchmark-report tests passed |
| Sanitizer | Clang 22 sanitizer build compiled all five native test executables. Tests 1–2 passed; Phase 3 sanitizer execution exceeded the local 30-second command window. The host GCC sanitizer linker is unavailable because `/usr/lib64/libasan.so.8.0.0` is missing. CI retains the complete sanitizer gate. |
| Linux wheel | `uv build --wheel` produced the CPython 3.13 Linux wheel; a clean temporary environment installed it with declared NumPy/SciPy dependencies and imported the release metadata/native API contract |

### Fixed negative-suite result (2026-09-29)

The existing `build/negative/release.jsonl` checkpoint was complete at 10,000 unique version-1 windows (1,250 in each of eight strata), with zero ERROR outcomes; no resume or new full-suite invocation was needed. Its SHA-256 is `17df428de04776f0ad6c212a28c1973e2528e9ad69f2fe932269e5f4def94ed6`. All eight regenerated concatenated-window-byte SHA-256 values matched the report. All statistical fields were recomputed from the JSONL with identical values; the final JSON additionally records the PRD fixed-suite zero-L5 FAIL, distinct from the generic 1% rate verdict. `uv run pytest tests/acceptance/test_negative_suite.py -q` passed 3 tests. The execution report records **one original full-run audit invocation** at commit `c8ad052c8bfe0dd02acf65ab61b63604523c2764` with a dirty worktree; five later forensic invocations of the same two L5 IDs are separately logged (current audit count: six), not pooled as new windows. Checkpoint timing records 17,338.124 summed per-window seconds but no whole-run start/end, so wall-clock runtime is not recoverable. See `build/negative/release_report.json`, `build/negative/release_report.md`, and [all triggering windows](docs/negative_suite_findings.md).

| Claim level | Pooled k/10,000; rate (two-sided exact CP 95%) | Worst stratum; k/1,250; rate (two-sided exact CP 95%) | Generic 1% worst-stratum rate verdict |
|---|---|---|---|
| L1 modulation | 3,073; 30.73% (29.826–31.645%) | S7/S8; 1,250 each; 100% (99.705–100%) | FAIL |
| L2 receiver | 3,518; 35.18% (34.243–36.125%) | S7/S8; 1,250 each; 100% (99.705–100%) | FAIL |
| L3 non-uncoded FEC | 0; 0% (0–0.03688%) | All; 0; 0% (0–0.29468%) | PASS |
| L4 frame | 2,937; 29.37% (28.478–30.274%) | S7; 1,139; 91.12% (89.405–92.639%) | FAIL |
| **L5 strict confirmed frame** | **2; 0.02% (0.002422–0.072228%)** | **S6 OFDM-like; 2; 0.16% (0.019383–0.576768%)** | **PASS for the separate 1% rate bound, but FAIL for the PRD fixed-suite zero-event gate** |

L1/L2/L4 are nonzero, as expected for a suite containing structured negatives, short sync words, and supported-family lookalikes; chance matching and partial acquisition are hypotheses requiring per-window triage, not explanations that erase the failures. Critically, the v1 S7/S8 generator has uint8 PSK-symbol underflow, and S8 removes sync patterns only after modulation, so the nominal low-SNR and sync-free interpretations are invalid. S6 contains a repeated 64-point OFDM symbol rather than diverse FFT sizes. The suite remains frozen; corrections require a separately versioned run. The synthetic-only L5=2 result fails the fixed suite's zero-L5 criterion. One T2 capture is now registered, but it has no independently established negative/family truth and is not pooled with this synthetic-negative result. The JSON's generic 1% PASS must not be read as a PRD zero-event PASS. Separately, `prd.md` requires held-out evidence for actual frame confirmation, while the suite's operational L5 predicate also allows repetition plus verified non-CRC-8; this semantic deviation needs a release decision rather than a claim of full PRD confirmation semantics.

**S6 L5 root cause and mitigation (2026-09-29):** [Both windows were reproduced and traced](docs/negative_suite_s6_l5_root_cause.md). Their repeated 80-sample OFDM block evades the native unsupported gate because normalized crest factor is below 3, then 64-QAM is ranked CANDIDATE and receiver lock is mistaken for mapping evidence. Exact 8-bit HDLC flags form repeated-offset triples; byte-length CRC-16/IBM sweeps find 46- and 88-byte apparent payloads that cross later HDLC flags. In 1041, three CRC-valid 720-bit words are byte-identical repetitions of one accidental match, and the top frame is `AMBIGUOUS` even as operational L5 counts it. The eligible L5 search budgets are 2,792 and 9,423 (header, polynomial, length) trials respectively. The working tree now uses boundary-constrained, predeclared-profile CRC verification, rejects crossing and duplicate-word observations, and requires verified mapping before a `CONFIRMED` frame/L5 claim. Focused framing/pipeline/release tests (15) pass; the 400-window accumulated smoke suite and direct reruns of S6 949/1041 report L5=0. No full frozen S6 or 10,000-window post-fix run is claimed, so the historical gate remains FAIL.

## Verification limits and test hazards

- Windows build and wheel installation are configured in `.github/workflows/native.yml` but have not executed in this workspace, so the Phase 1 cross-platform acceptance gate remains open. The `_native` MODULE install destination is statically correct under CMake's `LIBRARY` rule; this is not Windows execution evidence. See [runbook](docs/windows_ci_runbook.md) and [risk review](docs/windows_known_risks.md).
- The `gui` extra is now installed in the project environment and its tests run offscreen.
- The complete 1-GiB source was not processed end to end in this task. Full mode is chunk bounded and cancellation was exercised on the logical 1-GiB source, while the measured completed large-file run was a bounded preview with explicit incomplete coverage.
- FFTW3 and libsndfile were unavailable locally. Phase 2 currently uses a portable radix-2 FFT and a bounded RIFF/WAVE PCM/float reader; RF64 and compressed WAV formats are unsupported.
- Preview mode currently covers one explicit contiguous region. Multi-region representative preview selection remains future work.
- GUI, CLI, and Demo now use one production job request, but that request still creates an in-memory `SignalRecording`; native `RecordingSource` full-streaming execution is not yet the ordinary workflow.
- The legacy randomized deinterleaver test is nondeterministic because it uses unseeded random data and a score threshold; one broad-suite invocation failed and its isolated rerun passed. This is unrelated to the native Phase 1 path.
- The legacy Demo test still emits a NumPy FFT overflow warning, and the OFDM feature test emits a precision-loss warning. These numerical issues remain for later repair.
- Some current tests assert UI labels or mock the decoder rather than verifying end-to-end payload recovery. Treat them as limited regression checks.
- No validated `MACKAY_504_1008` matrix or genuine encoded concatenated Demo capture exists in this repository. LDPC therefore requires an explicit supplied matrix; the Demo catalog labels legacy recordings as structured fixtures instead of claiming validated concatenated FEC.
- The native Viterbi profile is currently the fixed K=7, rate-1/2 `(171,133)` implementation. Puncturing/profile catalogues, erasure-aware RS, and bounded profile discovery remain Phase 5 work.
- The GUI integration test now writes its negative fixture under pytest's temporary directory. Root `test_qpsk_cfo.py` still writes tracked `test_16qam_cfo.wav` when imported and remains outside configured `tests/` collection.
- Existing docs contain stale implementation claims. Use source/runtime evidence and this progress record to distinguish current behavior from the planned beta.
- Representative-corpus status (2026-09-29): added `tools/corpus` schema/validator, hash-checked production-path ingestion, seeded T1 impairment instrument, calibration/held-out invocation audit firewall, and schema-versioned evaluator/tuning proposal path. Baseline `python -m tools.corpus.evaluate --tier T1 --split calibration` completed with no committed T1 captures; all applicable gates were `INSUFFICIENT_POWER` or `NOT_MEASURABLE`. The T2 corpus is **PARTIAL (1 capture)**: a CC-BY-4.0 Zenodo SigMF downlink recording (record 13371136), checksum-verified and successfully loaded through the ordinary production path. Its source does not independently establish family, symbol rate, SNR, FEC, or transmitted bits, so it contributes no representative accuracy, negative-confidence, or PRD gate metric. T0/T1/T2 are never merged; T1 reports `NOT_REPRESENTATIVE`. No native estimator or receiver threshold/constant was changed. Representative Phase 3/4 validation remains blocked on a diverse set of independently documented T2 captures.

## GPU stage acceleration (2026-09-30)

- Added optional CuPy CUDA kernels behind the existing `compute_backend`
  contract. A CUDA request now performs post-lock PSK/QAM and configured FSK
  decisions/LLRs, K=7 rate-1/2 `(171,133)` Viterbi ACS/traceback, zero-syndrome
  RS screening, and sliding bit/LLR correlation. GUI PSD/STFT remains CUDA
  accelerated as before. A forced GPU request still fails when CUDA is absent;
  `auto` retains CPU only for stages without a validated CUDA implementation.
- Native carrier/timing acquisition remains authoritative; RS nonzero-syndrome
  correction remains native CPU; LDPC, deinterleaving, CRC/framing, differential
  PSK, and GMSK/GFSK are not represented as GPU accelerated.
- On the local RTX 3050 Laptop GPU, focused CUDA vectors passed: noiseless K=7
  recovery, receiver LLR sign contract, CPU/GPU correlation evidence parity,
  and clean/corrupt RS syndrome controls. `pytest` completed 67 focused tests
  in 6.92 s: acceleration, GPU measurement/stage, receiver, FEC, native Phase
  5, framing, pipeline, and CLI tests. An exploratory 65,536-symbol single-
  stream Viterbi timing measured CPU 0.030760 s and CUDA 0.038264 s; the CUDA
  kernel executes correctly but is not a speedup for this inherently serial,
  single-stream workload. No general FEC speedup claim is made.

## GNU Radio shared processing and 10 kS/s default (2026-09-30)

- The shared production workflow now routes every ordinary GUI, CLI, and Demo
  import (WAV, SigMF, or explicitly described raw IQ) through GNU Radio. It
  normalizes the selected analysis channel to complex64, runs File Source,
  optional rotator/FIR low-pass/rational resampler, and File Sink, then passes
  the result to the existing analysis pipeline. With no configured transform,
  it remains an auditable GNU Radio pass-through. GNU Radio runs in its own
  compatible Python runtime and is a required normal-processing dependency.
- Raw-IQ GUI/CLI defaults now present 10,000 samples/s. This value is recorded
  as `ASSUMED` (`cli_default_10ksps`/GUI default), never as measured metadata;
  CLI users can set `--sample-rate-status known` for documented acquisition
  metadata.
- Local GNU Radio 3.10.12 sidecar validation resampled a 20 kS/s complex64
  control to approximately 10 kS/s and confirmed provenance/rate status through
  the shared workflow.
- Optimized ordinary GNU Radio import without bypassing the flowgraph: runtime
  discovery is cached per configured interpreter, scheduler buffer requests are
  131,072 complex items, and little-endian `complex64` or `int16` I/Q input is
  streamed directly into GNU Radio rather than first being staged as a temporary
  complex64 input. On this host, the runtime probe measured 0.139 s cold and
  13 microseconds cached; a 128 MB complex64 pass-through measured 309 MB/s and
  the direct-CF32 workflow (including output materialization) measured 255 MB/s.
  These are local microbenchmarks, not the PRD's end-to-end performance gate.

## Phase 3 fractional timing and low-offset robustness (2026-09-30)

- Constellation sampling now interpolates at the estimated fractional samples
  per symbol rather than rounding each interval to an integer. Timing-phase
  selection also considers constellation occupancy when fit errors are close,
  avoiding selections supported by just one occupied point. Estimated carrier
  offsets producing fewer than four cycles across the analysis window are not
  applied as an exact derotation; they are below this preview's usable
  frequency resolution and were smearing the observed constellation.
- Added a deterministic BPSK regression at 6.4 samples/symbol with AWGN and
  carrier offset. Focused `tests/test_phase3_native.py` and
  `tests/test_phase4_native.py`: **43 passed**.
- On the downloaded, source-labeled indoor-jamming BPSK segment, the leading
  production candidate changed from 8PSK (score 0.773359) to BPSK (score
  0.993749). It remains `HYPOTHESIS_UNVERIFIED`; sync and FEC ran, but frame
  recovery failed and no payload was verified. This is a single-segment
  calibration result, not a representative accuracy claim. CAMRAS stays
  unlabeled and is not included in accuracy metrics.
- Phase 3 remains **In progress**: independent multi-class labeled captures
  and representative held-out evidence are still missing.

## Experimental modulation ML training (2026-09-30)

- Added optional `ml` dependencies and `tools/train_modulation_ml.py`. The
  offline trainer generates six classes (BPSK, QPSK, 8PSK, 16QAM, 64QAM,
  2FSK), extracts normalized amplitude, phase-increment, spectral, and moment
  features, and fits a 400-tree ExtraTrees model. Synthetic data include
  randomized SNR, CFO, phase, and samples-per-symbol. The source-labeled real
  BPSK segment is excluded from fitting and used only for a separate transfer
  check. The production hand-scored classifier has not been replaced.
- Run: `uv sync --extra ml`, then
  `uv run --extra ml python tools/train_modulation_ml.py --per-class 1200`.
  Training produced 7,200 synthetic windows with a stratified 75/25 split.
  Held-out synthetic balanced accuracy: **0.8389**. Per-class F1: BPSK .910,
  QPSK .778, 8PSK .761, 16QAM .795, 64QAM .793, 2FSK 1.000. Confusions cluster
  between neighboring PSK/QAM orders. The separate real-capture check predicted
  BPSK on all 32 sampled windows; these are one recording/session, not 32
  independent captures.
- Model artifacts are committed under `data/dataset_batches/ml/`: model
  `synthetic_modulation_extratrees.joblib` and full metrics
  `training_report.json`. This is experimental synthetic-set performance only;
  no independent real multi-class accuracy, payload evidence, or production
  integration is claimed. Real-world transfer and new independently labeled
  captures remain required.

## WAV exports and documentation refresh (2026-09-30)

- Added stereo WAV exports for the labeled BPSK segment, two CAMRAS recordings,
  and the Intelsat 37e recording. Left/right channels carry I/Q. CAMRAS rates
  (500 kS/s) and the Intelsat rate (500 samples/s) come from SigMF metadata. The
  BPSK source rate is unknown; its WAV header uses an explicitly documented
  10 kS/s placeholder, with the raw `.iq` file retained alongside it.
- The WAVs, per-file JSON sidecars, `.iq` file, trained model, and training
  report are committed. The incomplete 947 MB `w1.mat` download remains local
  and was not committed.
- Refreshed README, architecture, plan status, PRD status, changelog, known
  limitations, and pip compatibility requirements to match the shared GNU
  Radio workflow, selected CUDA coverage, ML experiment, committed dataset
  assets, and current open release gates. Dated forensic/CI reports remain
  historical records and are not rewritten as new validation results.
- Changed the trainer's default real-capture input to the committed `.iq` copy
  so the documented training command also performs its real BPSK transfer
  check in a clean clone.
- `git diff --check` and a local Markdown-link scan passed. No test suite was
  run for this documentation refresh.

## Next implementation task

The scoped structural L5 mitigation is implemented and has focused positive/negative evidence; the next release-validation step is to run the frozen S6 stratum and then the full frozen suite before changing the historical gate outcome. Investigate the v1 suite-generator defects separately; a corrected suite requires a new version and fixed run, not post-hoc edits to v1. Obtain independently documented T2 captures across supported families and unsupported negatives for representative Phase 3/4 validation. Continue Phase 4 tracking/equalization/profile work and Phase 5 matrix/puncturing/erasure/profile work. Run the reference-laptop performance suite, full ordinary-workflow streaming test, clean-machine installs, and Windows CI before closing the phase gates.

## Documentation reconciliation (2026-09-29)

- Moved Phase 6 and Phase 7 evidence out of the Phase 4 verification table, and merged the duplicate GUI audit note into Phase 6 verification; the underlying observations and counts were retained.
- Qualified the Phase 3 targeted-negative row and renamed the Phase 5 CRC-8 item so neither implies the now-measured Phase 7 zero-confirmation gate passed.
- Replaced the stale next-step instruction to execute an already-complete 10,000-window run with the observed L5 failures, frozen-generator findings, and versioned follow-up requirement.
- Kept Windows execution pending and the RS no-gap statement. The corpus now has one provenance-verified but truth-incomplete T2 capture; it must not be read as representative gate evidence. `KNOWN_LIMITATIONS.md` now replaces the stale absolute OFDM/metadata/interleaver claims, including its block-search-vs-explicit-profile contradiction, with behavior supported by source and the release checkpoint. Both documents flag the difference between the suite's operational L5 and `prd.md`'s held-out frame-confirmation wording.

## Update rules

- Record completed changes, exact checks and outcomes, remaining issues, and next steps after each implementation task.
- Use `Not started`, `In progress`, `Blocked`, or `Complete` for phase status. Include the specific cause for any blocker.
- Mark completion only with evidence for the phase acceptance criteria; do not count planned files or documented interfaces as implemented.
- Keep profiled timings separate from unprofiled benchmarks, local roundtrips separate from independent verification, and synthetic coverage separate from representative capture validation.
