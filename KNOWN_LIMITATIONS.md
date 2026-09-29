# Known Limitations and Explicit Non-Goals

To maintain strict epistemic integrity, this codebase explicitly refuses to silently handle scenarios it cannot mathematically prove. The following are architectural non-goals and known limitations of the MVP.

## 1. OFDM and Multicarrier Signals
**Limitation:** Signals such as DAB, DVB-T, LTE, and Wi-Fi are **unsupported**.
**Behavior:** Phase 2 includes a cyclostationary plausibility detector that will correctly flag cyclic-prefix periodicity. However, the classifier cannot identify specific subcarrier mappings, and Phase 3 will completely abort rather than attempt to lock a single-carrier PLL to a multicarrier waveform.

## 2. Magic Metadata Inference
**Limitation:** It is a physical impossibility to infer sample rate, center frequency, or timestamp natively from a flat array of `float32` complex IQ bytes.
**Behavior:** The Phase 1 loaders will not guess. If a raw `.iq` file is provided without an accompanying `RawIQConfig` (or if a WAV file lacks standard header chunks), these values are marked `MISSING` and downstream calculations that require true time (like baud rate in Hz) will degrade gracefully to fractional units.

## 3. Blind Pseudo-Random De-interleaving
**Limitation:** Seeded pseudo-random, diagonal, convolutional, and block transforms require an explicit profile or permutation.
**Behavior:** The native engine applies a supplied seed/permutation deterministically but does not brute-force arbitrary pseudo-random permutations or unknown convolutional delay-line state.

## 4. Bounded Block Interleaver Search
**Limitation:** Block interleaver dimension discovery is constrained to a predefined, finite search grid (e.g., `8, 12, 16, 32, 64, 128, 255`).
**Behavior:** Interleavers with `rows` or `cols` outside this exact grid are invisible to the search. If a signal uses an unmapped dimension, Phase 4 will exhaust the search grid, report a failure diagnostic, and halt.

## 5. LDPC Profiles and Systematic Extraction
**Limitation:** The native normalized min-sum decoder accepts supplied sparse matrices and `.alist` files, but no validated bundled `MACKAY_504_1008` matrix is present in this repository.
**Behavior:** An LDPC profile without an explicit matrix returns `LDPC_MATRIX_REQUIRED`; the engine does not invent systematic information-bit positions or claim a successful decode merely because iterations finished.

## 6. In-Memory Production Workflow
**Limitation:** GUI, CLI, and Demo Mode now share one request/job workflow, but that workflow still imports a complete `SignalRecording` before the pipeline runs.
**Behavior:** Native `RecordingSource` supports explicit regions, bounded previews, and chunked full-source analysis with reported coverage. Its full streaming path is not yet the ordinary GUI/CLI workflow, so large captures can still require full input memory and several analysis operations retain prefix limits.

## 7. Demo Fixture Scope
**Limitation:** The legacy structured BPSK and 16-QAM fixtures do not contain validated advertised concatenated FEC chains.
**Behavior:** Demo Mode labels them as structured fixtures, keeps evaluation metadata separate from production requests, and never uses ground truth to select an analysis outcome. A genuine end-to-end FEC fixture corpus remains required before beta validation.

## 8. Known GUI Divergences
*Currently, all identified GUI-vs-pipeline wiring gaps have been successfully patched through the Phase 6 verification phase.* No other wiring divergence is known, but the GUI code explicitly relies on exact field matches to the `PipelineResult` dataclasses and must be updated in lockstep if those models change.

### Native estimation and receiver limitations

1. Phase 3 family classification and rate/CFO calibration has deterministic synthetic coverage plus a corpus manifest, hash-checked ingestion, T1 impairment instrument, and split/firewall evaluator. The committed T2 over-the-air corpus is **EMPTY (0 captures)**, so no representative accuracy or PRD gate is claimed. Exact 16/64/256-QAM order ranking can remain ambiguous even when the QAM family is correct.
2. The PSD-floor SNR estimate is marked unreliable when the recording does not expose a separable noise-only band. Wideband FSK and multicarrier captures are particularly difficult.
3. `ReceiverSession` preserves arbitrary chunk equivalence but buffers a bounded acquisition window and emits on `flush`; continuous incremental tracking is not implemented yet.
4. DBPSK/DQPSK, OQPSK, and MSK paths are present. GMSK/GFSK, generic CPM, pi/4-DQPSK, adaptive multipath equalization, and finite-memory CPM sequence detection remain unsupported.
5. Carrier acquisition explicitly reports rotational and nonlinear-frequency aliases. A lock does not verify absolute bit mapping unless a carrier/phase reference or later frame evidence resolves it.

### Reed--Solomon bounded-distance invariant

For `t=(n-k)/2`, RS minimum distance is `d=2t+1`, so radius-`t` decoding is unique: those Hamming balls are disjoint. The Python re-encode check is redundant after a correct zero-syndrome result and protects only against internal decoder inconsistency, not a reachable decode-time ambiguity; no native gap exists. Native and Python regressions instead protect the real dependency of this argument: the generator polynomial's consecutive BCH-root structure.
