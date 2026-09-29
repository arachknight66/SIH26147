# S6 release-suite L5 false confirmations: windows 949 and 1041

Investigation date: 2026-09-29. The historical fixed version-1 suite remains **FAIL** at L5=2/10,000 (both in S6: 2/1,250, 0.16%; exact two-sided Clopper–Pearson 95% interval 0.01938–0.57677%). A later working-tree framing change implements the scoped structural mitigation below, but does not rewrite the historical result or establish a new release outcome: only the two triggering S6 windows and the smoke corpus have been rerun. Real-RF T2 negatives remain 0.

## Reproduction and provenance

Both windows use `SeedSequence([1, 5, index])` with PCG64; S6 is a single random 64-point OFDM symbol with a 16-sample cyclic prefix, tiled every **80 complex samples** into a 2,048-sample window. The generated sample arrays satisfy `samples[:-80] == samples[80:]` exactly. Before pipeline inspection, two independent regenerations of each window yielded identical bytes, and regeneration of all 1,250 S6 windows reproduced the report's concatenated-byte SHA-256 `346b5d1552c91442a789d5db6c283f5e66090847cbb60a6e873785210611c86f`.

| Window | Seed entropy | Sample-byte SHA-256 |
|---|---|---|
| `S6_unsupported/949` | `(1, 5, 949)` | `c3e315a0f104bb8bc7f17d3e2b61f04dbb6f593a62e4e8389fbb43e0156aa252` |
| `S6_unsupported/1041` | `(1, 5, 1041)` | `9254c349905f0e4148e3cd7c84b6da3ae7b6437f64d1e06328e4eb323bc4cf00` |

The production `run_full_pipeline(generate(spec('release', 'S6_unsupported', index)))` path was run for both windows. Every diagnostic invocation was appended to `corpus/evaluation_invocations.jsonl` through the existing held-out audit. That log now contains **six** release-suite entries: one original full-run invocation recorded by `build/negative/release_report.json`, followed by five forensic invocations of these same two IDs. These reruns are not new independent observations and were not pooled into the fixed-suite statistics. A compact audit-logged reproduction command for either index is in [the findings](negative_suite_findings.md); substitute its stratum and index.

## Window-by-window trace

| Evidence | 949 | 1041 |
|---|---|---|
| Native analysis outcome | `CANDIDATE`; wrong top `64-QAM`, score `0.7100848`; runner-up `16-QAM`, `0.5866498` | `CANDIDATE`; wrong top `64-QAM`, score `0.7381342`; runner-up `256-QAM`, `0.6110376` |
| Selected-rate evidence | 8 samples/symbol; 125,000 symbols/s at synthetic 1 MHz | 2 samples/symbol; 500,000 symbols/s at synthetic 1 MHz |
| Why not UNKNOWN/AMBIGUOUS | Top score exceeds production `unknown_threshold=0.55`; gap `0.123435` exceeds `ambiguity_margin=0.06` | Top score exceeds `0.55`; gap `0.127097` exceeds `0.06` |
| Why OFDM/UNSUPPORTED did not win | Delayed-correlation peak `1.0`, normalized occupied bandwidth `0.988281`, but normalized crest factor **1.9255**, below the native `>3.0` multicarrier gate | Peak `1.0`, bandwidth `0.976563`, crest factor **1.9905**, below `>3.0` |
| Receiver | 64-QAM `LOCKED`, `hypothesis_confirmed=True`, mapping `UNVERIFIED`; 1,440 recovered bits | 64-QAM `LOCKED`, `hypothesis_confirmed=True`, mapping `UNVERIFIED`; 6,048 recovered bits |
| FEC | `Uncoded`, `decode_success=True`; not L3 | `Uncoded`, `decode_success=True`; not L3 |
| Top frame state | `HYPOTHESIS_UNVERIFIED` | **`AMBIGUOUS`**, nevertheless counted by L5 |
| Top sync/periodicity | Exact, zero-error 8-bit `HDLC_FLAG` at bit offset **814**. The native periodicity predicate finds the equal-spacing triple **560, 814, 1068** (period 254 bits). Recovered bits agree at lag 254 in 90.56% of comparisons. | Exact `HDLC_FLAG` at bit offset **4844**. The equal-spacing triple **3884, 4844, 5804** has period 960 bits. Recovered bits agree at lag 960 in 98.88% of comparisons. |
| CRC match | `CRC-16/IBM`, start bit **822**, **368 payload bits (46 bytes)**, 16 CRC bits, end bit **1206**; computed and received CRC both `0x7b16` | `CRC-16/IBM`, start **4852**, **704 payload bits (88 bytes)**, 16 CRC bits, end **5572**; computed and received CRC both `0x14a9` |
| Header flags *inside* accepted CRC span | 902, 1068, 1156 | 5050, 5215 |

The complete recovered-bitstream HDLC offsets are:

- 949: `52, 140, 306, 394, 560, 648, 814, 902, 1068, 1156, 1327` (11 matches; no Barker-11 or CCSDS-ASM match).
- 1041: `44, 250, 415, 1004, 1210, 1375, 1964, 2170, 2335, 2924, 3130, 3295, 3884, 4090, 4255, 4844, 5050, 5215, 5804` (19 matches; no Barker-11 or CCSDS-ASM match).

For 1041, `CRC-16/IBM` also verifies at payload starts **2932** and **3892**, each for the same 704 payload bits and 16 CRC bits. Starts 2932, 3892, and 4852 are exactly 960 bits apart. All three 720-bit payload-plus-CRC words have SHA-256 `f72fcd0d72ab5a94f639afa7f4fdc6aaa994694923abe155ee7dfc7e418f4408` and pairwise Hamming distance **0**. They are one repeated accidental CRC-valid template, not three independent frame confirmations.

The causal chain is thus: repeated OFDM samples evade the multicarrier gate because crest factor is below its conjunction threshold; a 64-QAM fit with sufficient score/gap proceeds to a receiver lock that does not verify bit mapping; exact short HDLC flags in the recovered periodic bits satisfy a three-offset *header-only* periodicity test; the unrestricted byte-length CRC sweep finds an IBM-16 equality even when the candidate span crosses later HDLC flags. In 1041, `assemble_frames` marks the top structure `AMBIGUOUS`, but `evaluate_claims`'s L4 condition excludes only `UNKNOWN`, so L5 still fires. There is no evidence that the native `crc_bits` arithmetic itself miscomputed the reported CRCs.

## Actual coincidence budget

Native `correlate` tests every possible bit offset for three built-in patterns (8, 11, and 32 bits); patterns shorter than 16 bits have **zero** tolerated errors. It finds 11 and 19 exact HDLC matches respectively. `assemble_frames` keeps the 10 highest-confidence headers and calls `search_crcs` once per header. That function tries four CRC algorithms (`CRC-8`, two CRC-16 variants, `CRC-32`) at **every byte-aligned payload length** to the end of the recovered bitstream or 2,048 bytes. The measured, loop-exact trial counts are:

| Budget | 949 | 1041 |
|---|---:|---:|
| Raw (pattern, bit-offset) comparisons | 4,272 | 18,096 |
| Sync matches / headers searched for CRC | 11 / 10 | 19 / 10 |
| All (header, polynomial, payload-length) CRC trials | **4,034** | **12,982** |
| Trials eligible for L5: periodic header + non-CRC-8 | **2,792** | **9,423** |
| Eligible CRC-16 trials (both polynomials) | 1,872 | 6,294 |
| Eligible CRC-32 trials | 920 | 3,129 |
| Distinct eligible exact test words after deduplication | 2,769 | 6,945 |

If each CRC-16 comparison were an independent uniform trial, the rough per-window probability of at least one eligible hit would be `1-exp(-1872/2^16 - 920/2^32) ≈ 2.82%` for 949 and `1-exp(-6294/2^16 - 3129/2^32) ≈ 9.16%` for 1041. These are **conditional opportunity calculations on two selected positive windows**, not a calibrated prediction for all S6 windows: their tested prefixes overlap heavily, many tests are exact duplicates, and the input/demodulated bits are structured. The dominant multiplicity is a thousands-long CRC-16 sweep, not an anomalously powerful individual CRC match. Observed S6 L5 frequency is 2/1,250; the present evidence supports a structural multiple-testing/repeated-template explanation, not a demonstrated CRC primitive bug. Calling this a simple “birthday bound” would obscure the dependence and post-selection.

## Scoped structural fix — implemented; release validation pending

1. Treat an HDLC candidate as bounded by the **next** same-pattern flag (after protocol-appropriate unstuffing), rather than allowing an arbitrary CRC end beyond later flags. Neither observed L5 span satisfies that boundary. For unknown framing, leave the result a candidate; do not silently infer a payload length from a broad sweep.
2. Separate exploratory CRC discovery from confirmation. Choose protocol/CRC polynomial, byte order, and payload length using a documented profile or calibration evidence, then test those **predeclared** fields on held-out data. This reduces thousands of selected trials to a fixed small number and directly follows `prd.md`'s repeated-structure **and held-out evidence** requirement.
3. Require at least two non-overlapping frame observations with CRC validity at their predicted boundaries, and deduplicate byte-identical repeated payload-plus-CRC words. Repeating the same 720-bit accidental word at three offsets in 1041 must count as **one** observation. Reject `AMBIGUOUS` frame status as a confirmation candidate, but do not rely on that alone: 949 is unverified rather than ambiguous. Keep receiver mapping verification separate from mere lock.
4. Preserve a genuinely framed positive control and explicit unknown/unverified outcomes, so the new L5 path can still fire for independent evidence rather than becoming vacuous. Audit and freeze any revised predicate before looking at release IDs. With an implemented fix, run the **same frozen S6 1,250 windows first**, then the **full frozen 10,000 once**, and report both before/after counts and runtime.

The implementation now keeps exploratory framing header-only; production CRC validation requires a predeclared header, CRC algorithm, and payload boundary. It rejects a candidate whose CRC spans the next same-pattern header, requires at least two CRC-valid observations with distinct payload-plus-CRC words, and permits `CONFIRMED` only when the receiver reports a verified bit mapping. The release claim ladder now treats only that `CONFIRMED` state as L5. The direct reruns of S6 indices 949 and 1041 both produce no CRC candidate and `L5=false`; the smoke suite has zero L5 cases. The complete frozen S6 and release-suite reruns remain required before changing the historical FAIL gate.
