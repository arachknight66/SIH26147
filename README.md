# SIH26147 — Scientific Signal Recovery & Verification Engine

**A research-grade signal intelligence system for blind signal recovery, modulation identification, carrier/timing synchronization, data stream reconstruction, and independent scientific verification.**

---

## The Governing Invariant

> **Phase 2 measures. Phase 3 hypothesizes. Phase 4 recovers. Phase 5 corrects. Phase 6 verifies. Phase 7 communicates, reproduces, benchmarks and operationalizes.**

```text
Phase 1 → Canonical signal ingestion, endian & I/Q ordering, SigMF & WAV support
Phase 2 → Quantitative physical measurements (Welch PSD, noise floor, SNR, OBW, ROI)
Phase 3 → Modulation hypothesis generation & cumulant/spectral feature extraction
Phase 4 → Carrier frequency/phase PLL, Gardner TED, 1-SPS constellation & soft LLR
Phase 5 → Rotational ambiguity resolution, framing, LFSR descrambler, Viterbi FEC, CRC
Phase 6 → 7-claim verification matrix, boundary perturbation, 70/30 held-out cross-validation
Phase 7 → PySide6 desktop GUI, unified CLI, Schema v1.0 JSON/HTML/CSV reporting, deterministic replay
```

---

## 1. Quick Start

### Launch the Desktop GUI
```bash
python -m scripts.sih26147 gui
# or simply
python -m scripts.sih26147
```

### Analyze a Signal via Unified CLI
```bash
# Standard analysis
python -m scripts.sih26147 analyze recording.iq

# Fast screening with full artifact export (HTML, JSON, CSV, Manifest)
python -m scripts.sih26147 analyze recording.iq --preset fast --export output_dir/

# Fully deterministic reproducible run
python -m scripts.sih26147 analyze recording.iq --reproducible
```

### Independent Scientific Verification
```bash
python -m scripts.sih26147 verify recording.iq --strict
```

### Replay Saved Experiment
```bash
python -m scripts.sih26147 replay experiment.json
```

### Compare Two Analyses
```bash
python -m scripts.sih26147 compare run_a.json run_b.json
```

---

## 2. Run Comprehensive Benchmarks & Tests

```bash
# Run all 292 unit, regression, and quality gate tests
python -m pytest

# Run comprehensive end-to-end benchmark
python -m scripts.run_full_benchmark

# Generate demonstration dataset in examples/
python -m scripts.generate_demo_dataset
```

---

## 3. Documentation

* [Phase 1: Signal Ingestion](docs/phase1.md)
* [Phase 2: Quantitative Physical Measurement](docs/phase2.md)
* [Phase 3: Modulation Hypothesis Generation](docs/phase3.md)
* [Phase 4: Synchronization & Demodulation](docs/phase4.md)
* [Phase 5: Data Recovery & Error Correction](docs/phase5.md)
* [Phase 6: Independent Verification](docs/phase6.md)
* [Phase 7: Productization & GUI](docs/phase7.md)
* [Architecture Guide](docs/architecture.md)
* [User Guide](docs/user_guide.md)
* [Benchmarking Guide](docs/benchmarking.md)
* [Troubleshooting Guide](docs/troubleshooting.md)
