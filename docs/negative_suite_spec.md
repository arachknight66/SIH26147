# Fixed negative suite specification (v1)

The Phase 7 negative suite is a frozen set of 10,000 generated windows. It runs the ordinary `SignalRecording` → `run_full_pipeline` path; it does not inject truth, mocks, or an analysis shortcut.

## Claim ladder

L1 is a modulation claim: the top hypothesis maps to `CANDIDATE` (`HYPOTHESIS_UNVERIFIED` in the current public model) and its score is at least the production `unknown_threshold` (0.55). L2 is a receiver claim: `sync_status == COMPLETED`, the receiver acquisition is `LOCKED`, and `hypothesis_confirmed` is true. L3 is a FEC claim: FEC decode succeeded and its codec is not `UNCODED`; uncoded success is reported separately. L4 is a frame claim: framing is completed and its structure is not `UNKNOWN`. L5, the PRD “confirmed frame” gate, requires an L4 structure with periodicity-consistent headers and a verified CRC stronger than CRC-8. Held-out corroboration is an alternative L5 route but is not emit-able by the current pipeline, so it is always false here. L5 is intentionally a pure reporting predicate and does not alter pipeline behavior.

The fixed strata are S1 white noise, S2 silence/near-silence, S3 short input, S4 colored/real noise, S5 tones/interference, S6 unsupported modulation, S7 low-SNR supported lookalikes, and S8 random framed PSK bits with no intentionally inserted built-in sync word. Release uses 1,250 windows each; smoke uses 25 each.

Seeds derive from `SeedSequence([suite_version, stratum_id, index])` and PCG64. Every report records package versions and each stratum’s concatenated-byte SHA-256. Calibration uses a disjoint suite identifier and is never pooled with release.

For every level and stratum the report gives n, k, rate, exact two-sided 95% Clopper–Pearson interval, one-sided 95% upper bound, and Jeffreys interval. The verdict uses the worst stratum. The synthetic and T2 sets are never pooled.

**Negatives here are SYNTHETIC (T0/T1 provenance). The bound applies to THIS generative distribution only and does not certify behavior on real RF.**
