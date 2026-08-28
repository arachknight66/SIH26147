# SIH26147 — User Guide

## Complete Guide to the SIH26147 Desktop Application & CLI

---

## 1. Quick Start

### Launch the GUI
```bash
python -m scripts.sih26147 gui
# or simply
python -m scripts.sih26147
```

### Run Analysis via CLI
```bash
# Standard analysis
python -m scripts.sih26147 analyze recording.iq

# Fast screening with HTML/JSON artifact export
python -m scripts.sih26147 analyze recording.iq --preset fast --export output_dir/

# Fully deterministic reproducible run
python -m scripts.sih26147 analyze recording.iq --reproducible
```

### Verify Independent Scientific Claims
```bash
python -m scripts.sih26147 verify recording.iq --strict
```

### Replay Saved Experiment Bundle
```bash
python -m scripts.sih26147 replay experiment.json
```

### Compare Two Analyses
```bash
python -m scripts.sih26147 compare run_a.json run_b.json
```

---

## 2. Supported Signal Formats

| Extension | Canonical Representation | Format Description |
| :--- | :--- | :--- |
| `.iq`, `.raw`, `.bin` | `complex64` interleaved IQ | Interleaved float32 or int16 complex samples. |
| `.wav` | Canonical `complex64` IQ | Stereo WAV file where Left = I and Right = Q. |
| `.sigmf-meta` | SigMF Standard | SigMF metadata paired with binary dataset `.sigmf-data`. |

---

## 3. Presets Reference

* **`FAST_SCREENING`**: Rapid spectral inspection, 4 temporal windows, quick SNR.
* **`STANDARD_ANALYSIS`**: Full 6-phase scientific pipeline (Default).
* **`DEEP_ANALYSIS`**: Expanded modulation search space, 16 temporal windows, extended bootstrap trials.
* **`FORENSIC_ANALYSIS`**: Maximum evidence capture, detailed diagnostic logging, boundary perturbation sweeps.
