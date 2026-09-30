# Incremental migration to a C++-accelerated SIH26147 beta

Preserve the Qt application, plotting workflow, Demo Mode, and Python-facing data models. Replace weak decision logic while migrating useful computation into an independently testable C++ core. A full application rewrite is unnecessary.

**Release baseline:** Linux and Windows x86-64; C++20, CMake, pybind11; CPython 3.12; existing PySide6 GUI. Target offline recordings up to 1 GB on a four-core, 16-GB laptop. GPL-compatible dependencies are acceptable.

**Coverage:** configured decoding across all required families, with automatic inference restricted to documented profiles. Broader radio variants remain ambiguous when the evidence cannot distinguish them.

## Starting baseline versus target architecture

The `Current` column below records the repository state when this migration
plan was written. It is historical context, not a description of the present
checkout. Use the dated snapshot and phase gates that follow for current
implementation status.

| Current | Target |
|---|---|
| GUI directly invokes synchronous Python pipeline | Existing GUI submits cancellable background jobs |
| Python loops plus NumPy/SciPy computation | Python orchestration → pybind11 → stateful C++ modules |
| Whole-file loading; most analysis silently limited to a prefix | Chunked sources, explicit preview coverage, bounded-memory full-file processing |
| Five handcrafted modulation scores; fixed 20-dB SNR | Measured parameters, ranked candidates, rejection and ambiguity handling |
| All five demodulators run; only first result is accepted | Bounded candidate evaluation with evidence-based arbitration |
| Unconditional Viterbi → block search → RS | Configurable processing chains, including uncoded and unknown alternatives |
| Demo fixtures contain misleading “concatenated” examples | Deterministic captures processed through the production path; separate ground truth |

## Current implementation snapshot (2026-09-30)

| Area | Implemented now | Still open |
|---|---|---|
| Production workflow | GUI, CLI, and Demo share a Python request/job path; GNU Radio preprocessing is used for ordinary imports | Ordinary jobs still materialize an in-memory `SignalRecording`; native chunked-source execution is not the ordinary path |
| Native compute | C++20/pybind11 API v5 covers source primitives, preprocessing/spectra, estimation, configured receivers, bitstream transforms, Viterbi, RS, LDPC, correlation, and CRC | Advertised profile coverage, full streaming scale, platform packaging, and several specialized decoders remain incomplete |
| GPU | Optional CuPy CUDA kernels cover the operations listed in [GPU acceleration](docs/gpu_acceleration.md) | GPU is not a general speedup; many stages remain CPU-native |
| ML experiment | Standalone ExtraTrees trainer and committed model/report; synthetic six-class holdout balanced accuracy 0.8389 | Synthetic score only; one real BPSK capture is a transfer check, and ML is not used by production classification |
| Dataset evidence | Several WAV/IQ assets and provenance notes are committed | Independent labeled multi-class T2 captures and real-RF negative evidence are missing |
| Release validation | Linux native/functional evidence and a completed frozen v1 synthetic suite exist | Frozen v1 had 2 L5 events; post-fix full rerun, Windows run, reference-laptop gates, clean installs, and ordinary-workflow 1-GB evidence remain open |

All seven phases remain **In progress**. Exact evidence and caveats are in
[progress.md](progress.md); do not infer phase completion from the presence of
planned files or interfaces.

**Measured bottlenecks:** profiling the existing 35,680-sample BPSK demo took approximately 3.0 seconds: Viterbi 1.9 seconds, synchronization attempts 0.55 seconds, and framing/CRC 0.41 seconds. These are indicative profiled timings, not release benchmarks. Additional scaling problems include repeated rate estimation, repeated interleaver FFTs, repeated CRC prefix calculations, and full-array copies during CLI serialization. FFTs already execute natively through NumPy/SciPy.

## Preserve / migrate / replace

| Existing components | Disposition |
|---|---|
| `gui.py`, `run_gui.py`, pyqtgraph views | Preserve layout and interactions; add job controls, configuration, and trustworthy result rendering |
| `models.py`, `pipeline.py`, `cli.py` | Retain Python-facing entry points; revise contracts, scheduling, branching, and serialization |
| `loaders.py` | Preserve metadata/provenance and format choices; move sample decoding/conversion to native streaming readers |
| `measurements.py` | Migrate PSD/STFT/statistics; correct units and coverage reporting |
| `features.py`, `rate_estimation.py` | Migrate useful estimators; remove duplicate work and validate robustness |
| `classifier.py`, `hypotheses.py` | Replace fixed confidence logic; retain candidate/evidence concepts |
| `synchronization.py`, `demodulation.py` | Migrate validated operations; repair acquisition, mapping, ambiguity resolution, and lock assessment |
| `fec_convolutional.py`, `fec_reed_solomon.py` | Migrate and generalize tested primitives; replace unconditional success and incomplete-block handling |
| `deinterleaving.py`, `fec_concatenated.py` | Preserve block permutation primitive; replace discovery scoring and forced cascade |
| `correlation.py`, `crc_search.py`, `framing.py` | Migrate matching/checksums; replace repeated-prefix computation and weak frame validation |
| Demo generators, fixtures, tests | Preserve useful regression cases; regenerate mislabeled fixtures and remove mocked end-to-end claims |
| Root `append*`, `fix_gui*`, `patch*`, `gui_copy.txt` | Remove from the active project after recording their historical purpose; Git retains history |

## Native libraries and build tools

| Selection | Purpose |
|---|---|
| [pybind11](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) | Typed NumPy buffers and C++ ownership exposure; explicitly release the GIL around computation |
| [scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/) + CMake | Python wheel builds around the native library |
| [FFTW3](https://www.fftw.org/) | Single-precision FFT backend with reusable plans; verify [Windows builds](https://www.fftw.org/install/windows.html) in Phase 1 |
| [libsndfile](https://libsndfile.github.io/libsndfile/) | Native chunked PCM/float WAV reading and conversion |
| [AFF3CT](https://aff3ct.github.io/) | LDPC implementation behind a narrow adapter; use its supported [AList/QC matrix formats](https://aff3ct.readthedocs.io/en/latest/user/simulation/parameters/codec/ldpc/decoder.html) |
| CTest + Catch2; pytest | Independent native tests and Python/binding/integration tests |

Pin dependency revisions and include their notices. Keep the existing Viterbi/RS mathematics only after independent verification. Avoid adding a complete radio framework as an application runtime dependency.

## Shared interfaces and defaults

- Retain `run_full_pipeline(recording, config)` as a small-recording compatibility adapter. Add chunked `RecordingSource` and `AnalysisJob` abstractions for production jobs.
- Native interfaces operate on recording chunks, analysis windows, receiver sessions, and decoder frames—not individual samples crossing Python boundaries.
- Use `complex64` samples, `float32` LLRs, `uint8` unpacked bits, and 64-bit offsets. Preserve the existing **positive LLR means bit 1** convention; adapt external libraries explicitly.
- Borrow compatible contiguous buffers without copying. Make conversion copies explicit. Keep borrowed inputs alive and unmodified; return native-owned arrays with shared lifetime ownership.
- Preserve state and absolute offsets across chunks. Separate execution status from evidence status: finishing a decoder does not confirm its hypothesis.
- Default chunks/previews to 262,144 samples; evaluate at most three modulation candidates and 256 automatic decoder/interleaver candidate tests per analysis window. Budget exhaustion remains visible.
- Initial profiles cover BPSK/QPSK/8PSK, DBPSK/DQPSK/OQPSK, square 16/64/256-QAM, 2/4/8-FSK, and MSK/GMSK/GFSK. Gaussian profiles initially use documented BT values 0.3/0.5/1.0 and modulation indices 0.5/1.0. Automatic inference may return a family or ambiguity.

## Phase 1 — Native architecture and bindings foundation

- **Goal:** Establish a working native vertical slice without disrupting the application.
- **Existing components affected:** Build configuration, models, pipeline adapter, convolutional decoder, tests.
- **C++ modules to create/migrate:** Core types, buffer ownership, diagnostics, cancellation/progress, fixed-code Viterbi pilot.
- **Python components retained/changed:** Retain GUI and entry points; add native adapter and background-job infrastructure.
- **Main algorithms:** Existing K=7, rate-1/2 soft Viterbi, verified before porting.
- **Python↔C++ interfaces:** Typed arrays, result objects, versioned configuration, GIL-released calls; native progress queue polled by Python.
- **Tests:** Linux/Windows builds, independent Viterbi vectors, buffer lifetime/stride/dtype checks, GIL responsiveness, cancellation, import without Qt.
- **Acceptance criteria:** Native wheel installs on both platforms; Viterbi returns correct bits; invalid buffers fail cleanly; existing GUI launches.
- **Dependencies:** CMake, pybind11, scikit-build-core, native/Python test dependencies; build probes for selected native libraries.

## Phase 2 — Input, preprocessing, and spectral core

- **Goal:** Provide correct, bounded-memory signal ingestion and spectral analysis.
- **Existing components affected:** Loaders, measurements, constants, GUI plot data providers.
- **C++ modules to create/migrate:** WAV/raw sample readers, conversion, preprocessing, FIR/resampling, FFT, PSD/STFT, band detection.
- **Python components retained/changed:** Retain file dialogs, SigMF JSON/provenance, and plotting; introduce chunked source adapters and explicit preview/full-file modes.
- **Main algorithms:** Signed/unsigned conversion, DC/clipping statistics, optional DC removal/AGC, complex mixing, polyphase resampling, Welch PSD, STFT, robust spectral band grouping.
- **Python↔C++ interfaces:** `read_chunk`, preprocessing sessions, and spectral results with units, offsets, scaling, and coverage.
- **Tests:** PCM8/16/24/32 and float WAV; raw endian/order variants; unsigned centering; malformed files; PSD energy/axes; chunk versus whole-signal equivalence.
- **Acceptance criteria:** No fabricated sampling rate; missing metadata uses normalized units. A 1-GB capture is analyzed without whole-file allocation. Real passband conversion requires explicit channel/band interpretation.
- **Dependencies:** Phase 1, FFTW, libsndfile.

## Phase 3 — Parameter estimation and modulation analysis

- **Goal:** Replace fixed confidence with measured parameters and calibrated candidate ranking.
- **Existing components affected:** Features, rate estimation, classifier, hypotheses.
- **C++ modules to create/migrate:** Noise/SNR and bandwidth estimators, CFO/rate candidates, feature extraction, modulation scoring, temporal consistency.
- **Python components retained/changed:** Retain configuration and result presentation; remove hard-coded SNR and duplicate feature/rate calls.
- **Main algorithms:** Noise-floor-based SNR where identifiable, occupied bandwidth, normalized power spectra, transition/cyclic rate estimates, CFO-tolerant cumulants, constellation fitting, FSK tone clustering, held-out candidate scoring.
- **Python↔C++ interfaces:** `analyze_window` returns ranked candidates, parameter estimates, uncertainty, validity, and supporting evidence.
- **Tests:** Seeded SNR/CFO/rate sweeps, pulse-shaping variation, gain/DC changes, bursts, noise, OFDM, and unsupported orders.
- **Acceptance criteria:** No measured-looking default values; unsupported cases can remain unknown. On separable held-out cases at ≥15 dB, target ≥95% family accuracy, <1% high-confidence unsupported-family errors, and ≤2% symbol-rate error.
- **Dependencies:** Phase 2 and a labeled corpus separated into calibration and held-out sets.

## Phase 4 — Synchronization and demodulation

- **Goal:** Recover trustworthy bits and soft information for the agreed modulation profiles.
- **Existing components affected:** Synchronization, demodulation, candidate selection, constellation rendering.
- **C++ modules to create/migrate:** Acquisition, timing/carrier tracking, equalization, PSK/QAM/FSK demodulators, differential/staggered/CPM receivers.
- **Python components retained/changed:** Retain synchronized constellation display; expose profile selection and diagnostic results.
- **Main algorithms:** Coarse CFO correction, matched filtering, polyphase timing interpolation, Gardner/M&M loops, Costas/decision-directed carrier tracking, adaptive equalization, max-log demapping, FSK correlators, differential decoding, OQPSK branch alignment, finite-memory CPM sequence detection.
- **Python↔C++ interfaces:** Stateful `ReceiverSession.process/flush` returns symbols, bits, LLRs, acquisition validity, and sample/bit offsets.
- **Tests:** Independent symbol mappings; CFO/timing/phase ambiguity; BER curves; channel distortion; chunk boundaries; explicit specialized profiles.
- **Acceptance criteria:** Exact noiseless recovery after declared acquisition/tail regions; BER agrees with independent reference receivers. Select candidates using downstream evidence, retain unresolved rotations, and never confirm solely from low EVM.
- **Dependencies:** Phases 2–3; broader variants follow validation of the existing five modulations.

## Phase 5 — Bitstream, interleaving, and FEC

- **Goal:** Implement every required decoding family and bounded profile discovery.
- **Existing components affected:** All interleaving, FEC, correlation, CRC, and framing modules.
- **C++ modules to create/migrate:** Bit transforms, pattern/periodicity analysis, incremental CRC, four deinterleavers, configurable Viterbi/RS, LDPC adapter, chain evaluation.
- **Python components retained/changed:** Retain hypothesis presentation; add profile configuration/import and frame/report models.
- **Main algorithms:** Hard/soft correlation, repeated-header masks, frame periodicity, incremental CRC; block permutations, convolutional delay lines, explicit diagonal permutations, versioned seeded permutations; punctured Viterbi, GF(256) RS errors/erasures, LDPC normalized min-sum with syndrome stopping.
- **Python↔C++ interfaces:** `BitstreamView`, `DecoderProfile`, and stateful chain sessions; carry bit order, alignment, interleaving unit, puncturing, matrix/seed provenance, and information-bit mapping.
- **Tests:** Independent golden vectors; burst errors; wrong profiles; partial frames; all interleaver inverses; actual encoded concatenated captures; LDPC convergence/failure and LLR-sign checks.
- **Acceptance criteria:** All four deinterleavers work with supplied profiles. Ship K=7 convolutional, RS(255,223)/(255,239), concatenated, and a validated bundled `MACKAY_504_1008` LDPC profile plus matrix import. Uncoded data remains a candidate. No silent padding, discarded residual data, fabricated post-decoder LLRs, or unconditional success.
- **Dependencies:** Phase 4, AFF3CT, versioned profile catalogue. Frame confirmation requires repeated consistent structure and held-out validation, not a single CRC-8 coincidence.

## Phase 6 — Pipeline, GUI, and Demo Mode integration

- **Goal:** Deliver one production execution path through the retained GUI and CLI.
- **Existing components affected:** Pipeline, GUI, CLI, models, Demo Mode, fixture generators.
- **C++ modules to create/migrate:** Production session assembly, bounded result summaries, cancellation checkpoints, streaming result delivery.
- **Python components retained/changed:** Keep current plots/sidebar/Demo selector; add start/cancel/progress, profile selection, coverage, export, and optional Reveal Ground Truth.
- **Main algorithms:** Python schedules native stages and candidate branches; C++ retains receiver/decoder state. No heavy Python DSP fallback in beta.
- **Python↔C++ interfaces:** `AnalysisJob` publishes progress and typed results through queued Qt signals; GUI updates occur only on the main thread.
- **Tests:** Identical capture/config produces equivalent CLI, GUI, and Demo results; cancellation and stale-job handling; headless operation; native-package absence.
- **Acceptance criteria:** Demo uses the same source adapter and job path as ordinary files. Regenerated deterministic fixtures contain genuine advertised encoding. Ground truth is separate, loaded only for reveal/evaluation, and never supplied to analysis. Renaming fixtures cannot change results.
- **Dependencies:** Phases 1–5; early integration occurs throughout migration, with final consolidation here.

## Phase 7 — Validation, optimization, and beta release

- **Goal:** Establish reproducible correctness, performance, and installability.
- **Existing components affected:** Tests, benchmarks, packaging, documentation, development artifacts.
- **C++ modules to create/migrate:** Optimize measured hotspots, FFT-plan reuse, bounded traceback, memory reuse, and safe SIMD dispatch.
- **Python components retained/changed:** Finalize reports and diagnostics; remove legacy compute from runtime imports; retain useful Python reference implementations only in tests.
- **Main algorithms:** Profile-guided optimization without altering validated receiver/decoder behavior.
- **Python↔C++ interfaces:** Freeze the beta configuration/result schema; record engine version, profile revision, processing coverage, and benchmark metadata.
- **Tests:** Native sanitizers, malformed-input fuzzing, independent vectors, held-out captures, 10,000 seeded negative windows, 1-GB streaming jobs, clean-machine installation on both platforms.
- **Acceptance criteria:** No confirmed frames in the negative suite; every promised profile passes noiseless recovery and reference BER tests. On the reference laptop: spectral preview ≤2 seconds, cancellation ≤500 ms, GUI event stalls <100 ms, incremental analysis memory ≤512 MiB, and fixed-code Viterbi ≥5× faster than the original unprofiled baseline.
- **Dependencies:** Phase 6; recorded reference hardware and representative held-out captures. Synthetic-only validation is insufficient for the broader near-beta claim.

## Proposed repository structure

```text
CMakeLists.txt
pyproject.toml
cmake/                       # dependency and platform configuration
cpp/
  include/sih/               # stable native contracts
  src/
    io/ preprocessing/ spectral/ estimation/
    modulation/ synchronization/ demodulation/
    bitstream/ interleaving/ fec/ pipeline/
  bindings/
  tests/
signal_analysis/             # retained Python package
  gui.py  cli.py  models.py
  loaders.py  pipeline.py     # compatibility adapters
  native/  jobs/  config/  reporting/  demo/
profiles/                    # receiver, interleaver, FEC definitions
fixtures/
  demo/captures/
  demo/truth/                # never passed to analysis
tests/
  reference/  bindings/  integration/  acceptance/
tools/                       # deterministic fixture generation
benchmarks/
docs/
```

## Migration risks

| Risk | Control |
|---|---|
| Porting existing scientific errors faster | Independent vectors and negative tests before accepting native equivalents |
| NumPy lifetime, GIL, or cancellation faults | Explicit ownership, coarse binding calls, native checkpoints, sanitizer tests |
| State loss at chunk boundaries | Persist filter/loop/decoder state and test arbitrary partitions |
| FEC/profile search combinatorial growth | Finite catalogue, candidate budgets, staged rejection, visible exhaustion |
| Mapping, phase, byte-order, or LLR mismatches | Versioned profiles and cross-language golden vectors |
| Windows dependency/SIMD incompatibility | Build dependencies on both platforms immediately; retain portable CPU baseline |
| Broad modulation scope delaying beta | Existing orders first, then named profile groups; advertise only profiles passing their gates |
| Demo success leaking into production inference | Separate truth assets, filename-independent analysis, identical production job path |

## Final implementation order

**Foundation + Viterbi pilot → streaming input and spectral core → estimation and candidate ranking → existing demodulators → broader configured receivers → interleavers/FEC/framing → complete GUI/Demo integration → optimization and release validation.**

Maintain a runnable GUI after each phase. Remove each Python compute implementation from production only after its native replacement passes independent tests and the shared pipeline integration checks.
