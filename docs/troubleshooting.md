# SIH26147 — Troubleshooting Guide

## Diagnostic Taxonomy and Resolution Strategies

---

## 1. Common Pipeline Failures

### `LOW_SNR`
* **Symptoms**: Signal power is within 3 dB of the noise floor.
* **Resolution**: Re-examine signal region selection or collect higher gain recording.

### `SYNCHRONIZATION_FAILURE`
* **Symptoms**: Costas loop or Gardner TED unable to achieve phase/timing lock.
* **Resolution**: Verify sample rate assumption, adjust CFO search range, or inspect for non-linear phase distortions.

### `FRAME_FAILURE` / `CRC_FAILURE`
* **Symptoms**: Preamble sync word not detected or CRC verification fails.
* **Resolution**: Inspect bit polarity, byte alignment offset (0..7), or verify whether scrambling/FEC was applied.

---

## 2. Running Diagnostic Self-Test

```bash
python -c "from app.deployment.diagnostics import run_self_diagnostics; print(run_self_diagnostics())"
```
