# GUI/pipeline field-wiring audit

Audit date: 2026-09-29. Scope: every `PipelineResult`/`SignalRecording`-reachable access in `MetadataSidebar.update_metadata`, plus `update_synced_constellation` and `update_plots`. The live source was checked against `signal_analysis/models.py`, not historical patch scripts. Result: **0 missing/renamed-field mismatches**.

`learnings-and-workflow.md` and `overview.md`, requested historical sources, are absent from this checkout; this audit therefore relies on live code plus `CHANGELOG.md`'s documented historical divergence.

## update_metadata cross-reference

| GUI access (line) | Current contract | None safety / verdict |
|---|---|---|
| `pipe_res.diagnostics`, `d.code` (185) | `PipelineResult.diagnostics: List[Diagnostic]`; `Diagnostic.code` | Always list; **OK** |
| `recording.source_format.name`, `semantic_type`, `samples` (192–194) | `SignalRecording.source_format: SourceFormat`, `semantic_type`, `samples` | Required fields; **OK** |
| `recording.sample_rate_hz.value/status.value`, `center_frequency_hz.value/status.value` (196–200) | `MetadataValue.value/status`; status is `MetadataStatus` | Value may be `None`, formatter accepts it; **OK** |
| `pipe_res.parameter_analysis` (201) | `Optional[Any]` | `isinstance(dict)` guard; dictionary keys use safe `.get`; **OK** |
| `recording.diagnostics`, `d.severity.name/value`, `code`, `message` (213–217) | `List[Diagnostic]`, `Severity`, fields above | Empty list guard; **OK** |
| `pipe_res.all_hypotheses`; `h.label/status.value/score/quality_tier` (227–232) | `List[ModulationHypothesis]`; fields current | Empty list safe; **OK** |
| `pipe_res.hypothesis_status` (234) | `PipelineStageStatus` | Required enum; **OK** |
| `sync_status`, `demod_result` (237–239) | `PipelineStageStatus`, `Optional[DemodulationResult]` | Explicit non-None guard; **OK** |
| `dm.sync_result/hypothesis_confirmed/hard_bits` (239, 242, 248) | `DemodulationResult` fields; `sync_result: SynchronizationResult` | Guarded by `demod_result`; **OK** |
| `sync.acquisition_status/mapping_status/cfo_estimate/cfo_unit/lock_quality_metric/evm_percent/unresolved_phase_rotations` (243–249) | `SynchronizationResult` fields | `sync_result` required; list supports `len`; **OK** |
| `fec_status`, `deint_result`, `fec_result` (261–263) | `PipelineResult` fields | Pipeline only reaches COMPLETED while assigning both at 82–87; guard will enforce structural fields, invariant remains runtime-owned; **OK** |
| `deint.hypothesis.family.name` (266) | `DeinterleavingResult.hypothesis: DeinterleaverHypothesis`; `family: DeinterleaverFamily` | Required nested fields; **OK** |
| `fec.codec_name/corrected_bit_count/corrected_bit_fraction/decode_success` (267–269) | `FECDecodeResult` fields | Required; **OK** |
| `framing_status`, `frame_structure` (277–278) | `PipelineResult` fields | Explicit non-None guard; **OK** |
| `fs.header_match.pattern.name` (279) | `FrameStructure.header_match`; `HeaderMatch.pattern` annotated `SyncWordPattern` | GUI checks `pattern` despite `framing.py` fallback assigning `None`; safe runtime handling; **OK, annotation discrepancy noted** |
| `fs.crc_candidate.polynomial_name` (280) | `Optional[CRCMatch]`, `CRCMatch.polynomial_name` | Explicit non-None guard; **OK** |
| `fs.header_length_bits/payload_length_bits` (281, 285–286) | `FrameStructure` fields | Optional payload handled with `or 0`; **OK** |
| `frame_structure.payload_start_bit/payload_length_bits`, `fec_result.decoded_bits` (297–302) | `FrameStructure`, `FECDecodeResult` fields | `frame_structure` and `fec_result` checks precede access; **OK** |
| `demod_result.hard_bits` (303–304) | `DemodulationResult.hard_bits` | Explicit non-None guard; **OK** |
| `framing_status` (315) | `PipelineStageStatus` | Required enum; **OK** |

## Other live render paths

| Method/access | Contract and verdict |
|---|---|
| `update_synced_constellation`: `res.symbol_decisions`, `source_hypothesis_label` (663–676) | Called with a guarded `DemodulationResult` or `None`; both fields exist. **OK**. |
| `update_plots`: `recording.samples`, `semantic_type` (683–741) | Required `SignalRecording` fields. **OK**. PSD/STFT fields belong to their returned result models, not `PipelineResult`. |

## Reverse-direction awareness

The sidebar does not currently render several available fields (for example `PipelineResult.deint_result.diagnostics`, `FECDecodeResult.pre_correction_metric`, `DemodulationResult.mapping_verified/sample_offsets`, `SynchronizationResult.symbol_clock_locked/carrier_locked`, and `FrameStructure.status`). This is informational only; no display change is made by this audit.

## Divergence classification

No live field-name divergence was found. The only finding is the low-severity annotation/runtime inconsistency for fallback `HeaderMatch.pattern=None`; it is already safely handled at GUI line 279 and needs a separate type-contract task if annotation correction is desired.

## Async completion-path addendum

Before the guard-specific handler was added, `_poll_analysis_job` caught only `job.result(timeout=0)` at lines 526–531: `except Exception as exc: ... QMessageBox.critical(...)`. The later render calls, quoted from lines 538–541, were outside that block: `self.update_plots(outcome.recording)` followed by `self.sidebar.update_metadata(outcome.recording, outcome.pipeline_result)`. Thus a `GuiPipelineContractError` propagated uncaught out of the Qt timer slot. This was a medium-severity guard-delivery finding: pre-render validation prevented partial sidebar mutation, but the mismatch was not distinctly surfaced to the user. The narrowly scoped completion-path handler now reports `GUI_PIPELINE_CONTRACT` in a dedicated dialog and marks analysis failed; it does not catch other rendering failures.
