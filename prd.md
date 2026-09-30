# Product requirements — SIH26147 beta

## Purpose and users

Build a desktop signal-recording analysis application for SIH26147, “Automated model for analysis of .IQ and .wav files along with signal parameter extraction” by NTRO. Intended users are analysts inspecting unfamiliar recordings and SIH evaluators exercising a reproducible demonstration.

The application must help users inspect signals, identify plausible modulation and coding profiles, recover bits where supported, and inspect possible frame/header/payload structure. Results must distinguish evidence-backed findings from assumptions and unresolved hypotheses.

The existing Python prototype is the starting point. This document defines the target beta, not current implemented capability. Implementation details and phase gates are in [plan.md](plan.md); current evidence is in [progress.md](progress.md).

**Status (2026-09-30):** the native migration is active, but all seven plan
phases remain in progress. The frozen v1 negative suite recorded two L5
confirmed-frame claims and therefore failed the PRD's zero-event gate. A
scoped mitigation has passed focused tests; the frozen S6 and full-suite
reruns, representative real-RF validation, Windows execution, reference
hardware performance gates, and complete ordinary-workflow 1-GB run remain
open. See [known limitations](KNOWN_LIMITATIONS.md) for the current boundaries.

## Agreed release constraints

| Area | Requirement |
|---|---|
| Platforms | Linux and Windows x86-64 |
| Workload | Offline recordings up to 1 GB on a four-core laptop with 16 GB RAM and SSD storage |
| GUI | Preserve the existing Qt application, plots, and Demo Mode; extend controls only where needed |
| Native engine | C++20, CMake, pybind11; independently testable modules |
| Python | GUI, orchestration, plotting, configuration, reporting, and bindings; CPython 3.12 baseline |
| Dependencies | Reliable existing libraries where justified; GPL-compatible distribution is acceptable |
| Inference | Configured decoding plus bounded profile discovery; exact blind recovery of arbitrary codes, seeds, or radio variants is not a beta promise |

Live SDR acquisition, real-time processing guarantees, OFDM decoding, arbitrary protocol semantics, and macOS packaging are outside this beta scope.

## User workflow

1. Open a raw IQ, WAV, or supported SigMF recording, or select a Demo fixture.
2. Confirm channel interpretation and available metadata. Raw IQ format choices and missing sample rate remain explicit.
3. View a fast spectral/waveform preview and its processed coverage; select an analysis region, band, or supported profile where needed.
4. Start a cancellable analysis job and inspect ranked parameter/modulation hypotheses.
5. Inspect synchronization, recovered constellation, bits, deinterleaver/FEC candidates, and possible frames with their evidence and diagnostics.
6. Export a reproducible report and available bit/frame artifacts. In Demo Mode, optionally reveal separate ground truth for comparison.

## Functional requirements

| ID | Capability | Beta behavior |
|---|---|---|
| FR-01 | Recording import | Raw IQ dtype, endianness, and IQ/QI order; PCM8/16/24/32 and float WAV; existing SigMF metadata workflow; bounded-memory reading |
| FR-02 | Metadata and provenance | Preserve source, original representation, conversion, units, sample offsets, and coverage; distinguish known, assumed, estimated, and missing parameters |
| FR-03 | Visualization | Existing waveform, spectrum/PSD, waterfall/STFT, and raw/synchronized constellation views with correct units and clear coverage |
| FR-04 | Preprocessing | Explicit normalization/DC handling, clipping diagnostics, band selection, mixing, filtering, and resampling; real passband conversion requires valid channel/band interpretation |
| FR-05 | Parameter estimation | Band detection, occupied bandwidth, SNR where identifiable, CFO, symbol-rate candidates, uncertainty, and temporal consistency |
| FR-06 | Modulation inference | Rank supported families/profiles using measured evidence; reject unsupported signals and retain ambiguity rather than forcing exact labels |
| FR-07 | PSK/QAM decoding | Configured BPSK/QPSK/8PSK, DBPSK/DQPSK/OQPSK, and square 16/64/256-QAM receivers with tested mapping and synchronization |
| FR-08 | FSK/CPM decoding | Configured 2/4/8-FSK, MSK, GMSK, and GFSK receivers; initial Gaussian profiles use documented BT 0.3/0.5/1.0 and modulation indices 0.5/1.0 |
| FR-09 | Deinterleaving | Functional block, convolutional, diagonal, and pseudo-random deinterleavers with explicit parameters/permutations; bounded discovery over a documented catalogue |
| FR-10 | FEC | Convolutional/Viterbi, RS, concatenated chains, and LDPC with actual decode validation; initial shipped profiles and matrix import as specified in the plan |
| FR-11 | Bitstream/frame analysis | Hard/soft pattern correlation, periodicity, repeated-header candidates, CRC verification, and possible payload boundaries; confidence must account for chance matches |
| FR-12 | GUI and CLI parity | Same engine and configuration semantics for GUI, CLI, and Demo; raw-IQ CLI import included in the target workflow |
| FR-13 | Jobs and reporting | Responsive progress/cancel controls; explicit incomplete/failed/exhausted outcomes; reports include engine/profile revisions, assumptions, diagnostics, and processed coverage |
| FR-14 | Demo Mode | Deterministic genuine captures, production pipeline, no hard-coded analysis outputs, and optional separate Reveal Ground Truth |

## Evidence and output requirements

- Stage execution status and scientific/decode validity must be separate. Successful execution alone must not imply correct modulation, FEC, or framing.
- Uncoded, unknown, ambiguous, unsupported, and budget-exhausted outcomes are valid results.
- Incomplete blocks, acquisition regions, residual data, bit mappings, and phase ambiguities remain visible; no silent padding or invented post-decoder LLRs.
- Sampling rate must come from metadata or explicit user input. When unavailable, report normalized frequency/rate units rather than guessed absolute values.
- Frame confirmation requires consistent repeated structure and held-out evidence. An isolated short sync word or CRC-8 match is insufficient.
- Exported results must avoid treating undecoded or unverified bytes as a confirmed payload. Separate recovered data from confirmed frame interpretations.

## Demo acceptance

- A Demo selection supplies the recording and legitimate import settings through the ordinary source/job path. It does not inject modulation/FEC ground truth into automatic analysis.
- Capture-generation parameters, seeds, expected bits, and expected payloads live in separate truth assets used only by reveal/evaluation tooling.
- The same capture and user configuration yield equivalent results through GUI, CLI, and Demo. Renaming a fixture does not alter results.
- Include positive, noisy, CFO-impaired, genuinely concatenated, unsupported, and real-valued examples. Narration must match the encoded contents and demonstrated limits.

## Quality and release gates

These are acceptance targets, not measured current performance.

| Area | Gate |
|---|---|
| Import/spectral correctness | Format conversion and units verified; chunked and whole-signal reference results agree within declared numeric tolerances |
| Estimation/inference | On separable held-out cases at ≥15 dB: ≥95% family accuracy, <1% high-confidence unsupported-family errors, and ≤2% symbol-rate error |
| Demodulation | Every advertised profile passes exact noiseless recovery after declared acquisition/tail regions and independent reference BER checks |
| Interleaving/FEC | Independent vectors, error/burst/failure tests, actual concatenated captures, and LDPC syndrome/information-bit checks |
| False confirmation | No confirmed frames in the fixed suite of 10,000 seeded negative windows |
| Preview/responsiveness | Spectral preview ≤2 seconds, cancellation ≤500 ms, GUI event stalls <100 ms on the recorded reference laptop |
| Memory | Incremental analysis memory ≤512 MiB for a 1-GB capture; no whole-file sample allocation |
| Native performance | Fixed-code Viterbi ≥5× faster than the original unprofiled baseline on identical reference hardware/workload |
| Packaging | Clean-machine install and execution on Linux and Windows; no Python DSP fallback hidden behind a native result |
| Validation coverage | Independent and representative held-out captures in addition to deterministic synthetic fixtures |

The reference corpus, numeric comparison tolerances, hardware identity, and benchmark commands must be recorded with the corresponding implementation tests. Do not claim the broader near-beta coverage from synthetic roundtrips alone.
