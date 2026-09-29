# SIH26147 progress

Last updated: 2026-09-18.

## Current state

**Native Phases 1–5, the Phase 6 shared GUI/CLI/Demo workflow, and Phase 7 validation tooling are integrated and locally validated on Linux; representative capture calibration, full negative/performance gates, cross-platform execution, and full 1-GB ordinary-workflow evidence remain open.**

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

## Implementation phases

| Phase | Status | Completion evidence required |
|---|---|---|
| 1. Native architecture and bindings | In progress | Implementation and Linux wheel pass; Windows wheel/build execution is pending CI |
| 2. Input, preprocessing, spectral core | In progress | Linux implementation/tests pass; complete 1-GB full-mode and Windows execution evidence remain pending |
| 3. Estimation and modulation analysis | In progress | Native implementation and small deterministic held-out tests pass; representative corpus calibration and the full negative-confidence gate remain open |
| 4. Synchronization and demodulation | In progress | Existing/core configured profiles, exact mapping tests, ambiguity and chunk tests pass; continuous tracking, multipath equalization, generic CPM and complete Gaussian-profile matrix remain open |
| 5. Bitstream, interleaving, FEC | In progress | Configured native transforms/codecs and synthetic vectors pass; validated bundled LDPC matrix, configurable puncturing, genuine concatenated captures, bounded discovery, and corpus/negative gates remain open |
| 6. Pipeline, GUI, Demo integration | In progress | Shared production jobs and manifest-backed Demo Mode pass local workflow/Qt tests; native streaming-source job execution, validated advertised FEC demo fixtures, and broader interactive evidence remain open |
| 7. Validation, optimization, release | In progress | Local release-gate transcript, benchmark schema, native sanitizer build, and focused negative smoke corpus pass; hardware performance thresholds, 10,000-window/representative corpus, full ordinary-workflow streaming, clean-machine installation, and Windows evidence remain open |

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
| Negative behavior | Noise, silence, short data, real-only data, and unsupported multicarrier evidence produce bounded or explicit unknown/unsupported results in targeted tests |
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
| Phase 6 shared workflow | Four deterministic workflow tests passed: WAV import/analysis, renamed-capture invariance, pre-load cancellation, and Demo request/truth separation |
| Phase 6 GUI/CLI | Qt-offscreen GUI workflow tests passed (7 tests across GUI suites); CLI completed the same production workflow for `test_clean_qpsk.wav` |
| Phase 7 release gate | `tools/run_validation.py --quick --skip-native` passed 104 focused tests in 16.34 s and wrote a JSON transcript; schema and benchmark-report tests passed |
| Phase 7 sanitizer | Clang 22 sanitizer build compiled all five native test executables. Tests 1–2 passed; Phase 3 sanitizer execution exceeded the local 30-second command window. The host GCC sanitizer linker is unavailable because `/usr/lib64/libasan.so.8.0.0` is missing. CI retains the complete sanitizer gate. |
| Phase 7 wheel | `uv build --wheel` produced the CPython 3.13 Linux wheel; a clean temporary environment installed it with declared NumPy/SciPy dependencies and imported the release metadata/native API contract |
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
| Frame false-positive gate | GUI regression showed a lone CRC-8 collision could outrank periodic HDLC evidence; ranking now requires repetition before CRC-8 boosts a candidate |
| Complete local suite | 159 passed with Qt offscreen; one legacy OFDM feature precision-loss warning remains |
| Distribution build | Clean CPython 3.13 Linux wheel built successfully with the API version 5 bindings |

## Verification limits and test hazards

- Windows build and wheel installation are configured in `.github/workflows/native.yml` but have not executed in this workspace, so the Phase 1 cross-platform acceptance gate remains open.
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
- Representative-corpus status (2026-09-28): added `tools/corpus` schema/validator, hash-checked production-path ingestion, seeded T1 impairment instrument, calibration/held-out invocation audit firewall, and schema-versioned evaluator/tuning proposal path. Baseline `python -m tools.corpus.evaluate --tier T1 --split calibration` completed with no committed T1 captures; all applicable gates were `INSUFFICIENT_POWER` or `NOT_MEASURABLE`. The T2 corpus is **EMPTY (0 captures)**. T0/T1/T2 are never merged; T1 reports `NOT_REPRESENTATIVE`. No native estimator or receiver threshold/constant was changed. Representative Phase 3/4 validation remains blocked on user-supplied, independently documented T2 captures.
- GUI/pipeline field-wiring audit (2026-09-29): completed a live mechanical cross-reference in `docs/gui_pipeline_field_audit.md`; zero missing/renamed dataclass accesses were found. Added a pre-mutation dataclass contract guard plus a narrow async-completion handler that surfaces `GUI_PIPELINE_CONTRACT` distinctly. `uv run pytest tests/test_gui.py -q` passed 7 tests, including real-dataclass happy path, atomic drift rejection, and the actual `AnalysisJob -> _poll_analysis_job` contract-error path.

## Next implementation task

Complete the representative Phase 3 calibration/negative corpus and Phase 4 continuous tracking, multipath equalization, and remaining specialized-profile matrix. For Phase 5, add a validated bundled LDPC matrix, punctured/profiled Viterbi, erasure-aware RS, genuine concatenated captures, and bounded profile discovery. Then execute the 10,000-window negative corpus, reference-laptop performance suite, full ordinary-workflow streaming test, clean-machine installs, and Windows CI before closing the phase gates.

## Update rules

- Record completed changes, exact checks and outcomes, remaining issues, and next steps after each implementation task.
- Use `Not started`, `In progress`, `Blocked`, or `Complete` for phase status. Include the specific cause for any blocker.
- Mark completion only with evidence for the phase acceptance criteria; do not count planned files or documented interfaces as implemented.
- Keep profiled timings separate from unprofiled benchmarks, local roundtrips separate from independent verification, and synthetic coverage separate from representative capture validation.
